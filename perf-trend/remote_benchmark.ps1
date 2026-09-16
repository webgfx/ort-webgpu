[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('Probe', 'Deploy', 'Start', 'Status', 'Collect')][string]$Action,
    [Parameter(Mandatory)][string]$ComputerName,
    [ValidateSet('AMD', 'Intel')][string]$Vendor,
    [string]$ArchiveRoot,
    [string]$Manifest,
    [string]$RemoteRoot = 'C:\ort-perf-trend',
    [string]$LocalResults,
    [string]$RemotePython = 'python',
    [string]$ArtifactBaseUrl,
    [System.Management.Automation.PSCredential]$Credential
)
$ErrorActionPreference = 'Stop'
try {
    [Net.Dns]::GetHostAddresses($ComputerName) | Out-Null
} catch {
    throw "Cannot resolve $ComputerName. Supply a reachable IP address or fully qualified hostname. $($_.Exception.Message)"
}
$connection = @{
    ComputerName = $ComputerName
    SessionOption = New-PSSessionOption -OpenTimeout 15000 -OperationTimeout 60000 -IdleTimeout 43200000 -OutputBufferingMode Drop
}
if ($Credential) { $connection.Credential = $Credential }
if (-not [IO.Path]::IsPathRooted($RemoteRoot) -or $RemoteRoot.TrimEnd('\') -match '^[A-Za-z]:$') {
    throw 'RemoteRoot must name a dedicated absolute subdirectory.'
}
$session = New-PSSession @connection
try {
    if ($Action -eq 'Probe') {
        Invoke-Command -Session $session -ScriptBlock {
            [pscustomobject]@{
                hostname = $env:COMPUTERNAME
                user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
                gpus = @(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID)
                disks = @(Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace,Size)
                python = (Get-Command python -ErrorAction SilentlyContinue).Source
                powerScheme = (& powercfg /getactivescheme | Out-String).Trim()
                processes = @(Get-CimInstance Win32_Process | Where-Object Name -match '^(python|model_benchmark)\.exe$' |
                    Select-Object ProcessId,ParentProcessId,Name,CommandLine,CreationDate)
            } | ConvertTo-Json -Depth 6
        }
    } elseif ($Action -eq 'Deploy') {
        if (-not $ArchiveRoot -or -not $Manifest) { throw 'Deploy requires ArchiveRoot and Manifest.' }
        $sourceRoot = (Resolve-Path -LiteralPath $ArchiveRoot).Path
        $package = Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json
        $fileList = ConvertTo-Json -InputObject @($package.files) -Depth 5 -Compress
        $pending = @(Invoke-Command -Session $session -ArgumentList $RemoteRoot,$fileList -ScriptBlock {
            param($root, $json)
            $root = [IO.Path]::GetFullPath($root)
            New-Item -ItemType Directory -Path $root -Force | Out-Null
            foreach ($file in (ConvertFrom-Json $json)) {
                $path = [IO.Path]::GetFullPath((Join-Path $root $file.path))
                if (-not $path.StartsWith($root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe manifest path.' }
                $same = (Test-Path -LiteralPath $path) -and (Get-Item -LiteralPath $path).Length -eq $file.bytes
                if ($same) { $same = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -eq $file.sha256 }
                if (-not $same) {
                    New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($path)) -Force | Out-Null
                    $file.path
                }
            }
        })
        $byPath = @{}
        $byHash = @{}
        foreach ($file in $package.files) {
            $byPath[$file.path] = $file
            if ($pending -notcontains $file.path) { $byHash[$file.sha256] = $file.path }
        }
        foreach ($relative in $pending) {
            $source = [IO.Path]::GetFullPath((Join-Path $sourceRoot $relative))
            if (-not $source.StartsWith($sourceRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe source path.' }
            $digest = $byPath[$relative].sha256
            if ($byHash.ContainsKey($digest)) {
                Write-Host "Reusing identical artifact on $ComputerName : $relative"
                Invoke-Command -Session $session -ArgumentList $RemoteRoot,$byHash[$digest],$relative -ScriptBlock {
                    param($root, $existing, $relative)
                    # Copy on the remote disk; the two archive entries keep independent files.
                    Copy-Item -LiteralPath (Join-Path $root $existing) -Destination (Join-Path $root $relative) -Force
                }
            } else {
                Write-Host "Copying $ComputerName : $relative"
                if ($ArtifactBaseUrl) {
                    $url = $ArtifactBaseUrl.TrimEnd('/') + '/' + (($relative -split '/' | ForEach-Object { [Uri]::EscapeDataString($_) }) -join '/')
                    Invoke-Command -Session $session -ArgumentList $url,(Join-Path $RemoteRoot $relative),$byPath[$relative].bytes,$digest -ScriptBlock {
                        param($url, $destination, $expectedBytes, $expectedHash)
                        for ($attempt = 0; $attempt -lt 2; $attempt++) {
                            $offset = if ($attempt -eq 0 -and (Test-Path -LiteralPath $destination)) { (Get-Item -LiteralPath $destination).Length } else { 0 }
                            if ($offset -ge $expectedBytes) { $offset = 0 }
                            $request = [Net.HttpWebRequest]::Create($url)
                            $request.Proxy = $null
                            $request.Timeout = 30000
                            $request.ReadWriteTimeout = 120000
                            if ($offset) { $request.AddRange([long]$offset) }
                            $response = $request.GetResponse()
                            if ($offset -and [int]$response.StatusCode -ne 206) { $offset = 0 }
                            $mode = if ($offset) { [IO.FileMode]::Append } else { [IO.FileMode]::Create }
                            $output = [IO.File]::Open($destination, $mode, [IO.FileAccess]::Write, [IO.FileShare]::Read)
                            try { $response.GetResponseStream().CopyTo($output, 1048576) }
                            finally { $output.Dispose(); $response.Dispose() }
                            if ((Get-Item -LiteralPath $destination).Length -eq $expectedBytes -and
                                (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -eq $expectedHash) { return }
                        }
                        throw "Downloaded artifact failed verification: $destination"
                    }
                } else {
                    Copy-Item -LiteralPath $source -Destination (Join-Path $RemoteRoot $relative) -ToSession $session -Force
                }
                $byHash[$digest] = $relative
            }
        }
        foreach ($name in 'cross_device_benchmark.py','milestone_benchmark.py') {
            Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination (Join-Path $RemoteRoot $name) -ToSession $session -Force
        }
        Copy-Item -LiteralPath $Manifest -Destination (Join-Path $RemoteRoot 'transfer-manifest.json') -ToSession $session -Force
        Invoke-Command -Session $session -ArgumentList $RemoteRoot,$RemotePython -ScriptBlock {
            param($root, $python)
            & $python (Join-Path $root 'cross_device_benchmark.py') verify-files --archive-root $root --manifest (Join-Path $root 'transfer-manifest.json')
            if ($LASTEXITCODE -ne 0) { throw 'Transferred artifact verification failed.' }
        }
    } elseif ($Action -eq 'Start') {
        if (-not $Vendor) { throw 'Start requires Vendor.' }
        Invoke-Command -Session $session -ArgumentList $RemoteRoot -ScriptBlock {
            param($root)
            $active = @(Get-CimInstance Win32_Process | Where-Object {
                $_.Name -eq 'model_benchmark.exe' -or ($_.Name -eq 'python.exe' -and $_.CommandLine -like '*cross_device_benchmark.py*run*')
            })
            if ($active.Count) { throw "Benchmark process already active: $($active.ProcessId -join ', ')" }
            New-Item -ItemType Directory -Path (Join-Path $root 'collection') -Force | Out-Null
        }
        $name = 'perf-trend-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
        # Disconnected WinRM sessions survive the client invocation and can be inspected/reconnected.
        Invoke-Command @connection -InDisconnectedSession -SessionName $name -ArgumentList $RemoteRoot,$RemotePython,$Vendor -ScriptBlock {
            param($root, $python, $vendor)
            & $python -u (Join-Path $root 'cross_device_benchmark.py') run --archive-root $root --manifest (Join-Path $root 'transfer-manifest.json') --results (Join-Path $root 'collection\results.json') --vendor $vendor
            if ($LASTEXITCODE -ne 0) { throw "Collector exited with $LASTEXITCODE; inspect collection/results.json and logs." }
        } | Select-Object ComputerName,Name,InstanceId,State,Availability | ConvertTo-Json
    } elseif ($Action -eq 'Status') {
        Invoke-Command -Session $session -ArgumentList $RemoteRoot -ScriptBlock {
            param($root)
            $results = Join-Path $root 'collection\results.json'
            [pscustomobject]@{
                processes = @(Get-CimInstance Win32_Process | Where-Object {
                    $_.Name -eq 'model_benchmark.exe' -or ($_.Name -eq 'python.exe' -and $_.CommandLine -like '*cross_device_benchmark.py*')
                } | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine)
                results = if (Test-Path -LiteralPath $results) { Get-Content -Raw -LiteralPath $results | ConvertFrom-Json } else { $null }
            } | ConvertTo-Json -Depth 20
        }
    } elseif ($Action -eq 'Collect') {
        if (-not $LocalResults) { throw 'Collect requires LocalResults.' }
        New-Item -ItemType Directory -Path $LocalResults -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $RemoteRoot 'collection\results.json') -Destination (Join-Path $LocalResults 'results.json') -FromSession $session -Force
        Copy-Item -LiteralPath (Join-Path $RemoteRoot 'collection\logs') -Destination $LocalResults -FromSession $session -Recurse -Force
        Copy-Item -LiteralPath (Join-Path $RemoteRoot 'transfer-manifest.json') -Destination $LocalResults -FromSession $session -Force
    }
} finally {
    Remove-PSSession -Session $session
}

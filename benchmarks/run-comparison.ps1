param(
  [Parameter(Mandatory=$true)][string]$Request,
  [Parameter(Mandatory=$true)][string]$OutputDirectory,
  [Parameter(Mandatory=$true)][string]$WebRuntime,
  [string]$NativeExecutable = '',
  [string]$Browser = 'edge',
  [switch]$NativeOnly
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONPYCACHEPREFIX=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../gitignore/benchmarks/python-cache'))
if (-not $NativeExecutable) { $NativeExecutable = Join-Path $PSScriptRoot '../gitignore/benchmarks/build/native/ort_webgpu_benchmark.exe' }
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Preserve old results; choose a new output directory' }
if (Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @('model_benchmark.exe','matched_native.exe','ort_webgpu_benchmark.exe','llama-bench.exe') -or
    ($_.Name -match '^(python|node)' -and $_.CommandLine -match 'ai-test|native/run.py|web/run.py|host.mjs')
  }) { throw 'A competing benchmark is running' }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$environment = @{
  at = (Get-Date -Format o)
  hostname = $env:COMPUTERNAME
  gpu = @(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID)
  cpu = @(Get-CimInstance Win32_Processor | Select-Object Name)
  power = (powercfg /getactivescheme)
  request = (Get-FileHash -LiteralPath $Request -Algorithm SHA256).Hash
}
$environment | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'environment.json') -Encoding UTF8
$control = Join-Path $OutputDirectory 'native-cpu.json'
$native = Join-Path $OutputDirectory 'native-gpu.json'
$web = Join-Path $OutputDirectory 'web.json'
Write-Output 'PHASE native CPU-sampling control'
& python native/run.py --request $Request --executable $NativeExecutable --gpu-sampling cpu --output $control
if ($LASTEXITCODE -ne 0) { throw 'Native CPU control failed' }
Write-Output 'PHASE optimized native'
& python native/run.py --request $Request --executable $NativeExecutable --output $native
if ($LASTEXITCODE -ne 0) { throw 'Optimized native failed' }
$validation = Join-Path $OutputDirectory 'native-optimization.json'
& python compare.py $control $native --output $validation
if ($LASTEXITCODE -ne 0) { throw 'Native validation failed' }
if (-not (Get-Content -LiteralPath $validation -Raw | ConvertFrom-Json).allOutputsMatch) { throw 'Optimized native changed generated tokens; do not accept its performance' }
if ($NativeOnly) { Write-Output 'Native validation completed; Web comparison was explicitly deferred.'; exit 0 }
Write-Output 'PHASE Web'
$webArguments = @('web/run.py','--request',('"'+(Resolve-Path -LiteralPath $Request).Path+'"'),
  '--runtime',('"'+(Resolve-Path -LiteralPath $WebRuntime).Path+'"'),'--browser',$Browser,'--output',('"'+$web+'"'))
# OS-level redirection keeps benign child-process stderr from becoming a
# terminating NativeCommandError under Windows PowerShell's Stop preference.
$webProcess = Start-Process -FilePath (Get-Command python.exe).Source -ArgumentList $webArguments `
  -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -Wait -PassThru `
  -RedirectStandardOutput (Join-Path $OutputDirectory 'web.stdout.log') `
  -RedirectStandardError (Join-Path $OutputDirectory 'web.stderr.log')
if ($webProcess.ExitCode -ne 0) { throw 'Web failed; inspect web.stderr.log' }
& python compare.py $native $web --output (Join-Path $OutputDirectory 'native-vs-web.json')
if ($LASTEXITCODE -ne 0) { throw 'Comparison validation failed' }
Write-Output 'COMPLETED: inspect output agreement and configuration differences before drawing conclusions.'

param(
  [Parameter(Mandatory=$true)][string]$ComputerName,
  [Parameter(Mandatory=$true)][string]$Destination
)
$ErrorActionPreference = 'Stop'
$session = New-PSSession -ComputerName $ComputerName
try {
  Invoke-Command -Session $session -ArgumentList $Destination -ScriptBlock {
    param($destination)
    if (Test-Path -LiteralPath $destination) { throw 'Destination exists; use a new experiment directory' }
    New-Item -ItemType Directory -Path $destination | Out-Null
  }
  $source = (Resolve-Path -LiteralPath $PSScriptRoot).Path
  $files = Get-ChildItem -LiteralPath $source -File -Recurse | Where-Object {
    $relative = $_.FullName.Substring($source.Length + 1)
    $relative -notmatch '^(node_modules|results|build|__pycache__)\\' -and $relative -notmatch '\\__pycache__\\'
  }
  foreach ($file in $files) {
    $relative = $file.FullName.Substring($source.Length + 1)
    $target = Join-Path $Destination $relative
    Invoke-Command -Session $session -ArgumentList (Split-Path -Parent $target) -ScriptBlock {
      param($parent)
      New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    Copy-Item -LiteralPath $file.FullName -Destination $target -ToSession $session
  }
  $binaryTarget = Join-Path $Destination 'build/native'
  Invoke-Command -Session $session -ArgumentList $binaryTarget -ScriptBlock { param($folder) New-Item -ItemType Directory -Path $folder -Force | Out-Null }
  Get-ChildItem -LiteralPath (Join-Path $source '../gitignore/benchmarks/build/native') -File | Where-Object {
    $_.Extension -in @('.exe', '.dll') -or $_.Name -in @('build-metadata.json', 'benchmark-build.json')
  } | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $binaryTarget $_.Name) -ToSession $session
  }
} finally { Remove-PSSession $session }

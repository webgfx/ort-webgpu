param(
  [Parameter(Mandatory=$true)][string]$OrtSource,
  [Parameter(Mandatory=$true)][string]$OrtBuild,
  [Parameter(Mandatory=$true)][string]$Runtime,
  [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
if (-not $Output) { $Output = Join-Path $PSScriptRoot '../../gitignore/benchmarks/build/native' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio C++ x64 tools are required' }
$command = '"' + $vs + '\Common7\Tools\VsDevCmd.bat" -host_arch=x64 -arch=x64 >nul && set'
$compilerEnvironment = & $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw 'Compiler environment setup failed' }
foreach ($line in $compilerEnvironment) {
  if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$ortLibrary = Join-Path $OrtBuild 'onnxruntime.lib'
if (-not (Test-Path -LiteralPath $ortLibrary)) { $ortLibrary = Join-Path $OrtBuild 'Release/onnxruntime.lib' }
$builtDll = Join-Path (Split-Path -Parent $ortLibrary) 'onnxruntime.dll'
$runtimeDll = Join-Path $Runtime 'onnxruntime.dll'
if ((Get-FileHash -LiteralPath $builtDll -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $runtimeDll -Algorithm SHA256).Hash) {
  throw 'Use the ORT DLL from this exact build: the embedded Dawn procedure-table ABI must match'
}
& cmake -S $PSScriptRoot -B $Output -G Ninja -DCMAKE_BUILD_TYPE=Release "-DORT_INCLUDE_DIR=$OrtSource/include/onnxruntime/core/session" "-DORT_LIBRARY=$ortLibrary" "-DORT_BUILD_DIR=$OrtBuild" "-DJSON_INCLUDE_DIR=$OrtBuild/_deps/nlohmann_json-src/include"
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
& cmake --build $Output --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'C++ build failed' }
Get-ChildItem -LiteralPath $Runtime -Filter '*.dll' -File | Copy-Item -Destination $Output
if (Test-Path -LiteralPath (Join-Path $Runtime 'build-metadata.json')) {
  Copy-Item -LiteralPath (Join-Path $Runtime 'build-metadata.json') -Destination $Output
} else {
  $commit = (& git -C $OrtSource rev-parse HEAD).Trim()
  $metadata = @{
    runtime = 'ort'
    version = ('source-' + $commit.Substring(0,12))
    repositories = @{ onnxruntime = @{ commit = $commit } }
    sourceStatus = @(& git -C $OrtSource status --porcelain)
    note = 'Local source-build runtime, not an archived daily package'
  }
  $metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Output 'build-metadata.json') -Encoding UTF8
}
$sources = @{}
foreach ($file in @('main.cpp','gpu.h','probe.h','emit-shader.mjs','../web/gpu-greedy.mjs')) {
  $sources[$file] = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $file) -Algorithm SHA256).Hash.ToLowerInvariant()
}
@{
  sourceFiles = $sources
  ortSource = (& git -C $OrtSource rev-parse HEAD).Trim()
  runtimeSha256 = (Get-FileHash -LiteralPath $runtimeDll -Algorithm SHA256).Hash.ToLowerInvariant()
  dawnDependency = (Select-String -LiteralPath (Join-Path $OrtSource 'cmake/deps.txt') -Pattern '^dawn;').Line
  compiler = (Get-Item (Get-Command cl.exe).Source).VersionInfo.FileVersion
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Output 'benchmark-build.json') -Encoding UTF8

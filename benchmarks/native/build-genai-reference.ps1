param(
  [Parameter(Mandatory=$true)][string]$GenaiSource,
  [Parameter(Mandatory=$true)][string]$GenaiBuild,
  [Parameter(Mandatory=$true)][string]$OrtRuntime
)
$ErrorActionPreference='Stop'
$output=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../gitignore/benchmarks/build/genai-reference'))
New-Item -ItemType Directory -Path $output -Force|Out-Null
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$command='"'+$vs+'\Common7\Tools\VsDevCmd.bat" -host_arch=x64 -arch=x64 >nul && set'
foreach($line in (& $env:ComSpec /d /s /c $command)){if($line -match '^([^=]+)=(.*)$'){[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}}
$jsonInclude=Join-Path $OrtRuntime '_deps/nlohmann_json-src/include'
Push-Location $output
try{
 & cl.exe /nologo /std:c++20 /O2 /EHsc ('/I'+$GenaiSource+'/src') ('/I'+$jsonInclude) (Join-Path $PSScriptRoot 'genai-reference.cpp') /Fe:genai_reference.exe /link ($GenaiBuild+'/onnxruntime-genai.lib')
 if($LASTEXITCODE -ne 0){throw 'GenAI reference build failed'}
 & cl.exe /nologo /std:c++20 /O2 /EHsc ('/I'+$GenaiSource+'/src') ('/I'+$jsonInclude) (Join-Path $PSScriptRoot 'diagnose-genai.cpp') /Fe:genai_logits_diagnostic.exe /link ($GenaiBuild+'/onnxruntime-genai.lib')
 if($LASTEXITCODE -ne 0){throw 'GenAI diagnostic build failed'}
 Copy-Item -LiteralPath ($GenaiBuild+'/onnxruntime-genai.dll') -Destination $output
 Copy-Item -LiteralPath ($GenaiBuild+'/benchmark/c/model_benchmark.exe') -Destination $output
 Get-ChildItem -LiteralPath $OrtRuntime -Filter '*.dll' -File|Copy-Item -Destination $output
 @{
   genaiCommit=(& git -C $GenaiSource rev-parse HEAD).Trim()
   genaiDllSha256=(Get-FileHash -LiteralPath ($output+'/onnxruntime-genai.dll')).Hash.ToLowerInvariant()
   ortDllSha256=(Get-FileHash -LiteralPath ($output+'/onnxruntime.dll')).Hash.ToLowerInvariant()
   harnessSha256=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'genai-reference.cpp')).Hash.ToLowerInvariant()
 }|ConvertTo-Json|Set-Content -LiteralPath ($output+'/genai-build.json') -Encoding UTF8
}finally{Pop-Location}

param(
  [string]$ModelsRoot = 'D:/workspace/project/agents/ai-models',
  [string]$Models = 'aion',
  [string]$PromptLengths = '128,512,1024',
  [int]$GenerationLength = 128,
  [int]$Repetitions = 5,
  [ValidateSet('auto','cpu','gpu')][string]$Sampling = 'auto'
)
$ErrorActionPreference='Stop'
$benchmarkRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$repoRoot=(Resolve-Path -LiteralPath (Join-Path $benchmarkRoot '..')).Path
$env:PYTHONPYCACHEPREFIX=Join-Path $repoRoot 'gitignore/benchmarks/python-cache'
$executable=Join-Path $repoRoot 'gitignore/benchmarks/build/native/ort_webgpu_benchmark.exe'
if(-not(Test-Path -LiteralPath $executable)){throw 'Build the native benchmark with native/build.ps1 first'}
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -in @('ort_webgpu_benchmark.exe','model_benchmark.exe','llama-bench.exe') -or ($_.Name -eq 'node.exe' -and $_.CommandLine -match 'host.mjs')}){throw 'A competing benchmark is running'}
$output=Join-Path $repoRoot ('gitignore/benchmarks/results/native-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'-'+[guid]::NewGuid().ToString('N').Substring(0,6))
New-Item -ItemType Directory -Path $output | Out-Null
$request=Join-Path $output 'request.json'
$result=Join-Path $output 'result.json'
& python (Join-Path $benchmarkRoot 'prepare.py') --models-root $ModelsRoot --models $Models --prompt-lengths $PromptLengths --generation-length $GenerationLength --repetitions $Repetitions --max-length 8192 --output $request
if($LASTEXITCODE -ne 0){throw 'Could not prepare benchmark request'}
& python (Join-Path $PSScriptRoot 'run.py') --request $request --executable $executable --gpu-sampling $Sampling --output $result
if($LASTEXITCODE -ne 0){throw "Benchmark failed; see $result and its .artifacts logs"}
& python (Join-Path $PSScriptRoot 'summary.py') $result
if($LASTEXITCODE -ne 0){throw 'Performance validation failed'}
Write-Output "Saved raw samples and generated tokens: $result"

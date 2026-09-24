param([string]$Root)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $Root
if($env:COMPUTERNAME -ne 'WEBGFX-32'){throw 'This experiment is designated for webgfx-32'}
if(Get-CimInstance Win32_Process|Where-Object{$_.Name -in @('ort_webgpu_benchmark.exe','genai_reference.exe','model_benchmark.exe','llama-bench.exe')}){throw 'Conflicting benchmark'}
$python=Join-Path $Root '.venv/Scripts/python.exe'
if(-not(Test-Path -LiteralPath $python)){
 & python -m venv (Join-Path $Root '.venv')
 if($LASTEXITCODE -ne 0){throw 'Could not create experiment-local Python environment'}
 & $python -m pip install --disable-pip-version-check -r web/requirements.txt
 if($LASTEXITCODE -ne 0){throw 'Local benchmark dependency installation failed'}
}
New-Item -ItemType Directory -Path results -ErrorAction Stop|Out-Null
@{at=(Get-Date -Format o);device=$env:COMPUTERNAME;gpu=@(Get-CimInstance Win32_VideoController|Select-Object Name,DriverVersion,PNPDeviceID);cpu=@(Get-CimInstance Win32_Processor|Select-Object Name);power=(powercfg /getactivescheme)}|ConvertTo-Json -Depth 4|Set-Content -LiteralPath results/environment.json -Encoding UTF8
& $python native/test_sampler.py --executable build/native/ort_webgpu_benchmark.exe --output-dir results/sampler-check
if($LASTEXITCODE -ne 0){throw 'GPU selector correctness failed'}
& $python prepare.py --models-root D:/workspace/project/agents/ai-models --prompt-lengths 1024 --generation-length 128 --repetitions 1 --output results/correctness-request.json
if($LASTEXITCODE -ne 0){throw 'Model verification failed'}
Write-Output 'PHASE correctness: CPU control'
& $python native/run.py --request results/correctness-request.json --executable build/native/ort_webgpu_benchmark.exe --gpu-sampling cpu --output results/correctness-cpu.json
if($LASTEXITCODE -ne 0){throw 'CPU correctness control failed'}
Write-Output 'PHASE correctness: GPU feedback'
& $python native/run.py --request results/correctness-request.json --executable build/native/ort_webgpu_benchmark.exe --output results/correctness-gpu.json
if($LASTEXITCODE -ne 0){throw 'GPU correctness run failed'}
& $python compare.py results/correctness-cpu.json results/correctness-gpu.json --output results/cpu-gpu-correctness.json
if($LASTEXITCODE -ne 0 -or -not(Get-Content results/cpu-gpu-correctness.json -Raw|ConvertFrom-Json).allOutputsMatch){throw 'CPU and GPU outputs differ; performance runs were not started'}
Write-Output 'PHASE correctness: GenAI reference'
& $python native/run-genai.py --request results/correctness-request.json --binary-dir build/genai --output results/correctness-genai.json
if($LASTEXITCODE -ne 0){throw 'GenAI reference failed'}
& $python compare.py results/correctness-genai.json results/correctness-gpu.json --output results/genai-correctness.json
if($LASTEXITCODE -ne 0 -or -not(Get-Content results/genai-correctness.json -Raw|ConvertFrom-Json).allOutputsMatch){throw 'GenAI and native outputs differ; performance runs were not started'}
Write-Output 'CORRECTNESS PASSED: all five full 128-token sequences match'
& $python prepare.py --models-root D:/workspace/project/agents/ai-models --prompt-lengths 1024 --generation-length 128 --repetitions 3 --output results/perf-request.json
if($LASTEXITCODE -ne 0){throw 'Perf request preparation failed'}
Write-Output 'PHASE performance: GenAI matched'
& $python native/run-genai.py --request results/perf-request.json --binary-dir build/genai --output results/perf-genai.json
if($LASTEXITCODE -ne 0){throw 'GenAI performance failed'}
Write-Output 'PHASE performance: optimized native'
& $python native/run.py --request results/perf-request.json --executable build/native/ort_webgpu_benchmark.exe --output results/perf-native.json
if($LASTEXITCODE -ne 0){throw 'Native performance failed'}
& $python compare.py results/perf-genai.json results/perf-native.json --output results/perf-comparison.json
if($LASTEXITCODE -ne 0 -or -not(Get-Content results/perf-comparison.json -Raw|ConvertFrom-Json).allOutputsMatch){throw 'Performance output validation failed'}
Write-Output 'PHASE performance: stock daily-style GenAI benchmark (different random prompts)'
& $python native/run-genai.py --request results/perf-request.json --binary-dir build/genai --output results/perf-genai-stock.json --stock --validation disabled
if($LASTEXITCODE -ne 0){throw 'Stock GenAI run failed'}
Write-Output 'COMPLETED'

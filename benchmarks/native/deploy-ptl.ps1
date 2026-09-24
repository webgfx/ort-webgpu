$ErrorActionPreference='Stop'
$benchmarkRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$repoRoot=(Resolve-Path -LiteralPath (Join-Path $benchmarkRoot '..')).Path
$remoteRoot='D:/workspace/project/ort-webgpu/gitignore/benchmarks/ptl-20260924'
$session=New-PSSession -ComputerName webgfx-32.guest.corp.microsoft.com
try {
 Invoke-Command -Session $session -ArgumentList $remoteRoot -ScriptBlock {param($root) if(Test-Path -LiteralPath $root){throw 'Experiment exists; preserve it'}; foreach($folder in @('native','web','build/native','build/genai')){New-Item -ItemType Directory -Path (Join-Path $root $folder) -Force|Out-Null}}
 foreach($file in @('prepare.py','compare.py','web/run.py','web/model_manifest.py','web/requirements.txt','native/run.py','native/test_sampler.py','native/run-genai.py','native/ptl-comparison.ps1')){
  Copy-Item -LiteralPath (Join-Path $benchmarkRoot $file) -Destination (Join-Path $remoteRoot $file) -ToSession $session
 }
 foreach($variant in @('native','genai-reference')){
  $dest=if($variant -eq 'native'){'native'}else{'genai'}
  Get-ChildItem -LiteralPath (Join-Path $repoRoot ('gitignore/benchmarks/build/'+$variant)) -File|Where-Object{$_.Extension -in @('.exe','.dll','.json')}|ForEach-Object{
   Copy-Item -LiteralPath $_.FullName -Destination (Join-Path ($remoteRoot+'/build/'+$dest) $_.Name) -ToSession $session
  }
 }
 Invoke-Command -Session $session -ArgumentList $remoteRoot -ScriptBlock {param($root) & powershell -NoProfile -File ($root+'/native/ptl-comparison.ps1') -Root $root; if($LASTEXITCODE -ne 0){throw 'PTL experiment failed; see retained results and logs'} }
}finally{Remove-PSSession $session}

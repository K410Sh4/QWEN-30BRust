param(
    [string]$Model = '',
    [string]$Prompt = 'Escreva uma funcao Rust que verifica se um numero e primo.',
    [int]$Context = 4096,
    [int]$MaxTokens = 128,
    [int]$GpuCacheMiB = 4096,
    [int]$GpuKvMiB = 1024,
    [switch]$Reference,
    [string]$Adapter = ''
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Push-Location $PSScriptRoot
try {
    . (Join-Path $PSScriptRoot 'scripts\Common.ps1')
    $Model = Resolve-QwenModel -Model $Model
    $exe = Build-QwenEngine
    & $exe inspect --model $Model --ctx $Context
    if ($LASTEXITCODE -ne 0) { throw 'GGUF incompativel. Veja a mensagem acima.' }
    & $exe devices
    if ($LASTEXITCODE -ne 0) { throw 'GPU Vulkan nao encontrada. Atualize o driver AMD.' }
    $tokenizerPath = Join-Path $PSScriptRoot 'models\tokenizer.json'
    $expectedHash = '19564A48C4F71A2A1B937CCE34C737A1E662B171C5F5D7EDF641A15CD896F07D'
    if (-not (Test-Path -LiteralPath $tokenizerPath)) {
        New-Item -ItemType Directory -Force -Path (Split-Path $tokenizerPath) | Out-Null
        $tempPath = $tokenizerPath + '.download'
        try {
            Invoke-WebRequest -UseBasicParsing -Uri 'https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/resolve/b2cff646eb4bb1d68355c01b18ae02e7cf42d120/tokenizer.json' -OutFile $tempPath
            if ((Get-FileHash -LiteralPath $tempPath -Algorithm SHA256).Hash -ne $expectedHash) { throw 'Hash do tokenizer diferente do esperado.' }
            Move-Item -LiteralPath $tempPath -Destination $tokenizerPath
        } finally { if (Test-Path -LiteralPath $tempPath) { Remove-Item -LiteralPath $tempPath } }
    }
    if ((Get-FileHash -LiteralPath $tokenizerPath -Algorithm SHA256).Hash -ne $expectedHash) { throw 'Tokenizer local difere da versao oficial fixada. Use a CLI com --tokenizer para outra versao compativel.' }
    $runArgs = @('chat','--model',$Model,'--tokenizer',$tokenizerPath,'--prompt',$Prompt,'--ctx',"$Context",'--max-tokens',"$MaxTokens",'--gpu-cache-mib',"$GpuCacheMiB",'--device','vulkan','--gpu-kv-mib',"$GpuKvMiB")
    if ($Reference) { $runArgs += '--vulkan-reference' }
    if ($Adapter) { $runArgs += @('--adapter',$Adapter) }
    & $exe @runArgs
    if ($LASTEXITCODE -ne 0) { throw 'Falha na inferencia. Guarde a mensagem de erro para diagnostico.' }
} finally { Pop-Location }

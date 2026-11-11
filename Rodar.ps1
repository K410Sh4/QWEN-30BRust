param(
    [string]$Model = '',
    [string]$Prompt = 'Escreva uma funcao Rust que verifica se um numero e primo.',
    [int]$Context = 4096,
    [int]$MaxTokens = 128,
    [int]$GpuCacheMiB = 4096,
    [string]$Adapter = ''
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Push-Location $PSScriptRoot
try {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw 'Instale Rust stable com rustup (rustup.rs) e as ferramentas MSVC de C++ do Visual Studio Build Tools. Depois reabra o PowerShell.'
    }
    if (-not $Model) {
        $modelsRoot = $env:OLLAMA_MODELS
        if (-not $modelsRoot) { $modelsRoot = Join-Path $env:USERPROFILE '.ollama\models' }
        $manifestPath = Join-Path $modelsRoot 'manifests\registry.ollama.ai\library\qwen3-coder\30b'
        if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'Modelo nao encontrado. Informe -Model C:\caminho\modelo.gguf (arquivo GGUF unico).' }
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $layer = @($manifest.layers | Where-Object { $_.mediaType -eq 'application/vnd.ollama.image.model' })
        if ($layer.Count -ne 1) { throw 'Manifesto Ollama sem camada unica de modelo.' }
        $Model = Join-Path $modelsRoot ('blobs\' + $layer[0].digest.Replace(':', '-'))
    }
    if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "Arquivo de modelo ausente: $Model" }
    $Model = (Resolve-Path -LiteralPath $Model).Path
    cargo build --release --locked
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao compilar.' }
    $exe = Join-Path $PSScriptRoot 'target\release\qwen-rust.exe'
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
    $runArgs = @('chat','--model',$Model,'--tokenizer',$tokenizerPath,'--prompt',$Prompt,'--ctx',"$Context",'--max-tokens',"$MaxTokens",'--gpu-cache-mib',"$GpuCacheMiB",'--device','vulkan')
    if ($Adapter) { $runArgs += @('--adapter',$Adapter) }
    & $exe @runArgs
    if ($LASTEXITCODE -ne 0) { throw 'Falha na inferencia. Guarde a mensagem de erro para diagnostico.' }
} finally { Pop-Location }

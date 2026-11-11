Set-StrictMode -Version Latest
function Assert-QwenRust {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        $cargoDir = Join-Path $env:USERPROFILE '.cargo\bin'
        if (Test-Path -LiteralPath (Join-Path $cargoDir 'cargo.exe')) { $env:Path = "$cargoDir;$env:Path" }
    }
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw 'Cargo nao encontrado. Instale Rust stable e MSVC/Windows SDK; reabra o PowerShell. Instrucoes no README.md.'
    }
}
function Resolve-QwenModel {
    param([string]$Model)
    if (-not $Model) {
        $modelsRoot = $env:OLLAMA_MODELS
        if (-not $modelsRoot) { $modelsRoot = Join-Path $env:USERPROFILE '.ollama\models' }
        $manifestPath = Join-Path $modelsRoot 'manifests\registry.ollama.ai\library\qwen3-coder\30b'
        if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'Modelo nao encontrado. Informe -Model C:\caminho\modelo.gguf (GGUF unico).' }
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $layer = @($manifest.layers | Where-Object { $_.mediaType -eq 'application/vnd.ollama.image.model' })
        if ($layer.Count -ne 1) { throw 'Manifesto Ollama sem camada unica de modelo.' }
        $Model = Join-Path $modelsRoot ('blobs\' + $layer[0].digest.Replace(':', '-'))
    }
    if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "Arquivo de modelo ausente: $Model" }
    return (Resolve-Path -LiteralPath $Model).Path
}
function Build-QwenEngine {
    Assert-QwenRust
    # Native compiler diagnostics remain visible; suppress only stdout from the build.
    cargo build --release --locked | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao compilar. Verifique Rust stable, MSVC e Windows SDK.' }
    $engine = Join-Path (Get-Location).Path 'target\release\qwen-rust.exe'
    if (-not (Test-Path -LiteralPath $engine)) { throw "Executavel ausente: $engine" }
    return $engine
}

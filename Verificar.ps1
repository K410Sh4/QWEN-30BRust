param([switch]$GPU)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    . (Join-Path $PSScriptRoot 'scripts\Common.ps1')
    Assert-QwenRust
    cargo fmt --check
    if ($LASTEXITCODE -ne 0) { throw 'Falha na formatacao.' }
    cargo clippy --locked --all-targets -- -D warnings
    if ($LASTEXITCODE -ne 0) { throw 'Falha no Clippy.' }
    cargo test --locked
    if ($LASTEXITCODE -ne 0) { throw 'Falha nos testes CPU.' }
    if ($GPU) {
        cargo test --locked -- --ignored
        if ($LASTEXITCODE -ne 0) { throw 'Falha nos testes Vulkan.' }
    }
} finally { Pop-Location }

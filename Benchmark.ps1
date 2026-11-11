param(
    [string]$Model = '',
    [string]$Tokens = '151644,894,198,2683',
    [int]$Context = 4096,
    [int]$GpuCacheMiB = 4096,
    [int]$GpuKvMiB = 1024,
    [string]$Adapter = '',
    [int]$Runs = 3,
    [int]$Warmup = 1,
    [string]$OutFile = 'benchmark-result.json'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Push-Location $PSScriptRoot
try {
    . (Join-Path $PSScriptRoot 'scripts\Common.ps1')
    $Model = Resolve-QwenModel -Model $Model
    $exe = Build-QwenEngine
    $runArgs = @('bench','--model',$Model,'--tokens',$Tokens,'--ctx',"$Context",'--gpu-cache-mib',"$GpuCacheMiB",'--gpu-kv-mib',"$GpuKvMiB",'--runs',"$Runs",'--warmup',"$Warmup",'--compare','--device','vulkan')
    if ($Adapter) { $runArgs += @('--adapter',$Adapter) }
    $jsonLines = & $exe @runArgs
    $status = $LASTEXITCODE
    $jsonText = $jsonLines -join [Environment]::NewLine
    if ($jsonText) {
        $report = $jsonText | ConvertFrom-Json
        $destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutFile)
        [System.IO.File]::WriteAllText($destination, $jsonText, (New-Object System.Text.UTF8Encoding($false)))
        Write-Host "Relatorio: $destination"
        Write-Host ('Referencia: {0:N3} tokens/s | Residente: {1:N3} tokens/s | Razao: {2:N3}x | Paridade: {3}' -f $report.reference.forward_tokens_per_second,$report.resident.forward_tokens_per_second,$report.reference_over_resident_ratio,$report.parity.passed)
    }
    if ($status -ne 0) { throw 'Benchmark/paridade falhou. Veja o erro e o relatorio, se gerado.' }
} finally { Pop-Location }

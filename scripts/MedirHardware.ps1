param(
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [Parameter(Mandatory = $true)][string]$OutFile,
    [ValidateRange(0, 1000000)][int]$Samples = 0
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (Test-Path -LiteralPath $OutFile) { throw "CSV ja existe: $OutFile" }
$root = [IO.Path]::GetPathRoot($ModelPath)
$volume = if ($root -match '^[A-Za-z]:') { $root.Substring(0, 2) } else { '' }
for ($sample = 0; $Samples -eq 0 -or $sample -lt $Samples; $sample++) {
    $now = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $memory = Get-CimInstance -ClassName Win32_PerfFormattedData_PerfOS_Memory
    $disks = @(Get-CimInstance -ClassName Win32_PerfFormattedData_PerfDisk_LogicalDisk)
    $modelDisk = $disks | Where-Object { $_.Name -eq $volume } | Select-Object -First 1
    $totalDisk = $disks | Where-Object { $_.Name -eq '_Total' } | Select-Object -First 1
    $modelReadBytes = if ($modelDisk) { $modelDisk.DiskReadBytesPersec }
    $totalReadBytes = if ($totalDisk) { $totalDisk.DiskReadBytesPersec }
    [pscustomobject]@{
        UnixMsUtc = $now
        ModelVolume = $volume
        AvailableRamMiB = $memory.AvailableMBytes
        CommittedBytes = $memory.CommittedBytes
        CommitLimitBytes = $memory.CommitLimit
        PageReadsPerSec = $memory.PageReadsPersec
        ModelVolumeReadBytesPerSec = $modelReadBytes
        AllVolumesReadBytesPerSec = $totalReadBytes
    } | Export-Csv -LiteralPath $OutFile -NoTypeInformation -Append -Encoding UTF8
    if ($Samples -eq 0 -or $sample + 1 -lt $Samples) { Start-Sleep -Seconds 1 }
}

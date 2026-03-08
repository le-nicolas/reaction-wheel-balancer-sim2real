param(
    [ValidateSet("wheel_only", "base_assist")]
    [string]$Profile = "wheel_only",

    [ValidateSet("smooth", "robust")]
    [string]$Mode = "robust",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$outPath = Join-Path $repoRoot "esp32_rw_base_platformio\include\controller_params.h"

$cmd = @(
    "final/export_firmware_params.py",
    "--mode", $Mode,
    "--hardware-safe",
    "--control-hz", "250",
    "--bus-voltage-v", "11.1",
    "--wheel-current-limit-a", "2.5",
    "--wheel-torque-limit", "0.03",
    "--max-wheel-speed", "45",
    "--max-tilt-rate", "1.5",
    "--crash-angle-deg", "6",
    "--out", $outPath
)

if ($Profile -eq "wheel_only") {
    $cmd += @("--wheel-only", "--no-allow-base-motion")
}

if ($ExtraArgs) {
    $cmd += $ExtraArgs
}

Write-Host ("python " + ($cmd -join " "))
Push-Location $repoRoot
try {
    python @cmd
}
finally {
    Pop-Location
}

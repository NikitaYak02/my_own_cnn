param(
  [string]$Configuration = "Release",
  [string]$Target = "conv_bench"
)

$ErrorActionPreference = "Stop"

function Require-Command {
  param([string]$Name)
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Required command not found: $Name"
  }
}

Require-Command cmake
Require-Command nvidia-smi

$gpuInfo = nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>$null
if (-not $gpuInfo) {
  throw "Failed to detect NVIDIA GPU via nvidia-smi."
}

$firstGpu = ($gpuInfo | Select-Object -First 1).Trim()
Write-Host "Detected GPU: $firstGpu"
Write-Host "Configuring preset: auto"

cmake --preset auto
if ($LASTEXITCODE -ne 0) {
  throw "cmake --preset auto failed"
}

Write-Host "Building target '$Target' ($Configuration)"
cmake --build build-auto --config $Configuration --target $Target -- /m:4
if ($LASTEXITCODE -ne 0) {
  throw "cmake build failed"
}

Write-Host "Build completed: build-auto"

<#
.SYNOPSIS
  Bring up the full GroundCheck stack (API + static app) in mock mode — no API key.

.DESCRIPTION
  Installs the engine + API editable, then serves groundcheck_api on 127.0.0.1:<Port>
  with GROUNDCHECK_LLM=mock so the money demo runs end-to-end with no key. The page
  is served same-origin under /app, so it fetches /check with no CORS.

.EXAMPLE
  ./scripts/dev.ps1            # serve on http://127.0.0.1:8000/
  ./scripts/dev.ps1 -Port 8137 # serve on a different port
#>
param(
  [int]$Port = 8000,
  [switch]$NoInstall
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
  if (-not $NoInstall) {
    Write-Host "Installing core + api (editable)..."
    python -m pip install -e "./core[dev]" -e "./api[dev]"
  }
  $env:GROUNDCHECK_LLM = "mock"
  Write-Host ""
  Write-Host "GroundCheck is up at  http://127.0.0.1:$Port/   (mock mode, no key)"
  Write-Host "  POST /check   GET /examples   GET /health   GET /app/GroundCheck.dc.html"
  Write-Host "Press Ctrl+C to stop."
  Write-Host ""
  python -m uvicorn groundcheck_api.main:app --host 127.0.0.1 --port $Port
} finally {
  Pop-Location
}

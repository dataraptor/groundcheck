<#
.SYNOPSIS
  Run the full NO-KEY test suite: core + api + eval + browser e2e + axe audit.

.DESCRIPTION
  Runs each layer's pytest suite with `-m "not api"` (so the live-key smokes skip),
  then the Playwright e2e + axe-core audit under e2e/. Exits non-zero if any suite
  fails. The e2e tests skip cleanly (not fail) when chromium or network is absent.

.EXAMPLE
  ./scripts/test.ps1
  ./scripts/test.ps1 -NoInstall      # skip the editable install
  ./scripts/test.ps1 -SkipE2E        # core + api + eval only (no browser)
#>
param(
  [switch]$NoInstall,
  [switch]$SkipE2E
)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
$failed = $false
try {
  if (-not $NoInstall) {
    python -m pip install -e "./core[dev]" -e "./api[dev]" | Out-Null
  }
  $env:GROUNDCHECK_LLM = "mock"

  $suites = @(
    @{ Path = "core/tests"; Name = "core" },
    @{ Path = "api/tests";  Name = "api (incl. contract)" },
    @{ Path = "eval/tests"; Name = "eval" }
  )
  if (-not $SkipE2E) { $suites += @{ Path = "e2e"; Name = "e2e + axe audit" } }

  foreach ($s in $suites) {
    Write-Host ""
    Write-Host "=== pytest $($s.Path)  ($($s.Name)) ===" -ForegroundColor Cyan
    python -m pytest $s.Path -m "not api"
    if ($LASTEXITCODE -ne 0) { $failed = $true }
  }
} finally {
  Pop-Location
}
Write-Host ""
if ($failed) {
  Write-Host "FAIL — at least one no-key suite failed." -ForegroundColor Red
  exit 1
} else {
  Write-Host "PASS — all no-key suites green." -ForegroundColor Green
  exit 0
}

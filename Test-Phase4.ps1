param([string]$Path = 'C:\Francis\agent\phase4.ps1')
$all = Get-Content -LiteralPath $Path

# A) no path mistakenly on the left of '='
$lhs = Select-String -Path $Path -Pattern '^\s*\\?([A-Za-z]:\\|/).+?=' -AllMatches

# B) no stray "\" before variable names
$slashVars = Select-String -Path $Path -Pattern '(^|\s)\\[A-Za-z_]\w*' -AllMatches

# C) every $psi has FileName assignment soon after
$psiHits = Select-String -Path $Path -Pattern '^\s*\$psi\s*=\s*New-Object\s+System\.Diagnostics\.ProcessStartInfo' -AllMatches
$missing = @()
foreach ($hit in $psiHits) {
  $ln = $hit.LineNumber
  $ok = $false
  $start = $ln + 1
  $end   = [Math]::Min($ln + 8, $all.Count)
  foreach ($i in $start..$end) {
    if ($all[$i-1] -match '^\s*\$psi\.FileName\s*=\s*\$python\b') { $ok = $true; break }
    if ($all[$i-1] -match '^\s*\$psi\s*=\s*New-Object\s+System\.Diagnostics\.ProcessStartInfo') { break }
  }
  if (-not $ok) { $missing += $ln }
}

[pscustomobject]@{
  Path                     = $Path
  BadLHSCount              = ($lhs | Measure-Object).Count
  BackslashVarsCount       = ($slashVars | Measure-Object).Count
  MissingFileNameNearLines = ($missing -join ', ')
}

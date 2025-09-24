Describe "phase4.ps1 structure" {
  $text = Get-Content 'C:\Francis\agent\phase4.ps1' -Raw

  It "has a python resolver outcome" {
    $text | Should Match '\$psi\.FileName\s*=\s*\$python'
  }

  It "does not contain bad LHS paths" {
    (($text -split "`r?`n") -join "`n") | Should Not Match '^\s*\\?([A-Za-z]:\\|/).+?='
  }
}

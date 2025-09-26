function Release-Patch {
  param([string]$notes = "")
  poetry version patch
  $ver = poetry version -s
  if (Test-Path CHANGELOG.md -PathType Leaf) { }
  else { New-Item -Type File CHANGELOG.md | Out-Null }
  if ($notes) { Add-Content CHANGELOG.md "`n# v$ver`n- $notes" }
  git add pyproject.toml CHANGELOG.md
  poetry run pre-commit run --all-files
  git commit -m "chore: release v$ver"
  git tag "v$ver"
  git push && git push --tags
}
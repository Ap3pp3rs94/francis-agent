function Release-Patch {
  param([string]$notes = "")
  poetry version patch
  $ver = poetry version -s

  if (-not (Test-Path CHANGELOG.md)) { New-Item -Type File CHANGELOG.md | Out-Null }
  if ($notes) { Add-Content CHANGELOG.md "`n# v$ver`n- $notes" }

  git add -A
  pre-commit run -a

  git commit -m "chore: release v$ver"

  # If tag exists locally or remotely, remove it first
  if (git tag | Select-String -Quiet "^v$ver$") {
    git tag -d "v$ver" | Out-Null
    git push --delete origin "v$ver" 2>$null | Out-Null
  }

  git tag "v$ver"
  git push
  git push --tags
}

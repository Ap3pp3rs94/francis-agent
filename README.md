# Francis Agent — Phase 4

## Run
powershell -NoProfile -ExecutionPolicy Bypass -File .\phase4.ps1

## Dry-run
powershell -NoProfile -ExecutionPolicy Bypass -File .\phase4.ps1 -DryRun

## Tests
.\Test-Phase4.ps1 | Format-List
Invoke-Pester .\Tests -OutputFormat NUnitXml -OutputFile .\test-results.xml

## Latest secure backup
Get-ChildItem .\backups\*_secure.zip | Sort-Object LastWriteTime -Desc | Select-Object -First 1

## Restore (example)
Expand-Archive .\backups\full_notes_YYYYMMDD_HHMMSS.zip -DestinationPath .\restore\full_notes_YYYYMMDD_HHMMSS

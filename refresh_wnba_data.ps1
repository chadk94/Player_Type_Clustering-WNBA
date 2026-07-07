$ErrorActionPreference = 'Stop'

$repoPath = "C:\Users\chad_\PycharmProjects\Player_Type_Clustering - WNBA"
$python = "C:\Users\chad_\PycharmProjects\WNBAApp\venv\Scripts\python.exe"
$logFile = Join-Path $repoPath "refresh_wnba_data.log"

Start-Transcript -Path $logFile -Append
try {
    Set-Location $repoPath

    & $python fetch_live_data.py
    $fetchExit = $LASTEXITCODE

    if ($fetchExit -ne 0) {
        Write-Output "fetch_live_data.py failed (exit $fetchExit); leaving existing CSVs in place, skipping commit."
        exit $fetchExit
    }

    git add current_season_box.csv todays_matchups.csv
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m "Refresh live WNBA data"
        git push
        Write-Output "Committed and pushed refreshed data."
    } else {
        Write-Output "No changes to commit."
    }
}
finally {
    Stop-Transcript
}

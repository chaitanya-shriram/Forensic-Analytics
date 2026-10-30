# Builds the UI (first run) and starts the API + console on http://127.0.0.1:8000
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path "$root\frontend\dist")) {
    Push-Location "$root\frontend"; npm install; npm run build; Pop-Location
}
Push-Location "$root\backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Pop-Location

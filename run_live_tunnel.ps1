# ==============================================================================
# 🚀 1-CLICK INSTANT PUBLIC LIVE ACCESSIBILITY SCRIPT (CLOUDFLARE TUNNEL)
# ==============================================================================
# WHAT: Starts the web application server and creates a secure public HTTPS URL.
# ACCESSIBLE BY EVERYONE 24/7 as long as this script is running.
# ==============================================================================

param (
    [string]$AppType = "fastapi"  # Options: "fastapi" or "streamlit"
)

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "   🤖 ML Master Suite - Starting Live Public Tunnel" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

if ($AppType -eq "streamlit") {
    $port = 8501
    Write-Host "Starting Streamlit App on port $port..." -ForegroundColor Yellow
    $serverProcess = Start-Process python -ArgumentList "-m streamlit run app.py --server.port $port --server.headless true" -PassThru -NoNewWindow
} else {
    $port = 8000
    Write-Host "Starting FastAPI Web App on port $port..." -ForegroundColor Yellow
    $serverProcess = Start-Process python -ArgumentList "-m uvicorn api.index:app --host 0.0.0.0 --port $port" -PassThru -NoNewWindow
}

Start-Sleep -Seconds 3

Write-Host "`nLaunching Cloudflare Tunnel via cloudflared.exe..." -ForegroundColor Green
Write-Host "Generating live HTTPS public URL accessible anywhere worldwide..." -ForegroundColor Green
Write-Host "=========================================================`n" -ForegroundColor Cyan

& ".\cloudflared.exe" tunnel --url "http://localhost:$port"

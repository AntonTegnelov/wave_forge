Write-Host "Running tests for core packages..." -ForegroundColor Green
Write-Host "Testing wfc-core..." -ForegroundColor Cyan
cargo test -p wfc-core
Write-Host "Testing wfc-rules..." -ForegroundColor Cyan
cargo test -p wfc-rules 
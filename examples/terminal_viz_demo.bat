@echo off
REM Terminal Visualization Demo for Wave Forge
REM This script demonstrates terminal visualization mode

echo Running Wave Forge with terminal visualization...
echo.
echo Press T to toggle visualization on/off
echo.

REM Change to project root directory (adjust if needed)
cd %~dp0\..

REM Run Wave Forge with terminal visualization
cargo run --release --bin wave-forge-app -- ^
  --rule-file examples/simple-pattern.ron ^
  --width 20 ^
  --height 15 ^
  --depth 5 ^
  --visualization-mode terminal ^
  --report-progress-interval 250ms ^
  --progress-log-level info

echo.
echo Demo completed
pause 
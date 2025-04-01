@echo off
REM Simple2D Visualization Demo for Wave Forge
REM This script demonstrates Simple2D visualization mode

echo Running Wave Forge with Simple2D visualization...
echo.
echo Controls:
echo - Press T to toggle visualization on/off
echo - Press ESC to close the visualization window
echo - Use arrow keys to navigate through Z-layers
echo.

REM Change to project root directory (adjust if needed)
cd %~dp0\..

REM Run Wave Forge with Simple2D visualization
cargo run --release --bin wave-forge-app -- ^
  --rule-file examples/simple-pattern.ron ^
  --width 40 ^
  --height 40 ^
  --depth 10 ^
  --visualization-mode simple2d ^
  --report-progress-interval 100ms ^
  --progress-log-level info

echo.
echo Demo completed
pause 
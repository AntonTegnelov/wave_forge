@echo off
REM Visualization Benchmarking Comparison for Wave Forge
REM This script demonstrates the performance impact of different visualization modes

echo ===============================================
echo    Wave Forge Visualization Benchmark Test
echo ===============================================
echo.
echo This script will run three tests with identical parameters:
echo 1. No visualization (baseline)
echo 2. Terminal visualization
echo 3. Simple2D visualization
echo.
echo Results will be saved as CSV files for comparison
echo.

REM Change to project root directory (adjust if needed)
cd %~dp0\..

REM Create output directory if it doesn't exist
if not exist "benchmark_results" mkdir benchmark_results

REM Define common parameters
set COMMON_PARAMS=--rule-file examples/simple-pattern.ron --width 50 --height 50 --depth 15 --benchmark-mode --report-progress-interval 100ms

echo Running baseline test (no visualization)...
cargo run --release --bin wave-forge-app -- %COMMON_PARAMS% ^
  --visualization-mode none ^
  --benchmark-csv-output benchmark_results/no_viz_benchmark.csv
echo.

echo Running terminal visualization test...
echo Note: The visualization will slow down the algorithm
cargo run --release --bin wave-forge-app -- %COMMON_PARAMS% ^
  --visualization-mode terminal ^
  --benchmark-csv-output benchmark_results/terminal_viz_benchmark.csv
echo.

echo Running Simple2D visualization test...
echo Note: The visualization will slow down the algorithm significantly
cargo run --release --bin wave-forge-app -- %COMMON_PARAMS% ^
  --visualization-mode simple2d ^
  --benchmark-csv-output benchmark_results/simple2d_viz_benchmark.csv
echo.

echo ===============================================
echo All tests completed!
echo.
echo Benchmark results saved to:
echo - benchmark_results/no_viz_benchmark.csv
echo - benchmark_results/terminal_viz_benchmark.csv
echo - benchmark_results/simple2d_viz_benchmark.csv
echo.
echo You can compare these files to see the performance impact
echo of different visualization modes.
echo ===============================================

pause 
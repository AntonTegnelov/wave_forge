# Analyzing Benchmark Results

This guide provides instructions on how to run and analyze benchmark results using Wave Forge's benchmarking tools.

## Running Benchmarks

Wave Forge includes comprehensive benchmarking capabilities that can be accessed through command-line arguments:

```bash
wave-forge --rule-file <path_to_rule_file> --benchmark-mode
```

Additional benchmark options:

- `--benchmark-csv-output <path>`: Save benchmark results to a CSV file
- `--width <n> --height <n> --depth <n>`: Set custom grid dimensions
- `--global-log-level debug`: Get more detailed output during benchmarking

## Interpreting Console Output

When running in benchmark mode, Wave Forge outputs detailed performance information to the console:

```
=== BENCHMARK COMPARISON ===
Grid Size: 16x16x16
Number of Tiles: 4

Metric               | GPU
-------------------------------------------------------------------
Total Time           | 0.48s
Iterations           | 4096
Collapsed Cells      | 4096
Memory Usage         | 18.2 MB
Status               | Success

=== Performance Hotspots ===
Section                   | GPU
-------------------------------------------------------------------
Entropy Calculation       | 0.08s
Constraint Propagation    | 0.25s
Collapse Cell             | 0.10s
Find Lowest Entropy       | 0.05s
```

### Key Metrics to Analyze

1. **Total Time**: The wall-clock time for the entire operation. Lower is better.
2. **Relative Speed**: How much faster/slower one implementation is compared to the other.
3. **Iterations**: The number of iterations taken by the algorithm to complete.
4. **Memory Usage**: The amount of memory consumed during execution.
5. **Performance Hotspots**: The most time-consuming operations, helping identify bottlenecks.

## Working with CSV Output

When using the `--benchmark-csv-output` option, results are saved in CSV format for further analysis. The CSV file contains the following columns:

- Width, Height, Depth: Grid dimensions
- Num Tiles: Number of unique tiles used
- Total Time (ms): Execution time in milliseconds
- Iterations: Number of WFC iterations
- Collapsed Cells: Number of cells collapsed
- Result: "Ok" for success, "Fail" for failure

### Analyzing CSV Results with Python

You can use Python with pandas and matplotlib to analyze the CSV data:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('benchmark_results.csv')

# Calculate grid volume
df['Volume'] = df['Width'] * df['Height'] * df['Depth']


## Identifying Bottlenecks

When analyzing benchmark results, pay attention to:

1. **Sections with Highest Total Time**: These are the primary bottlenecks
2. **Scaling with Grid Size**: How execution time grows with problem size
3. **Memory Usage Patterns**: How memory consumption scales with grid size

### Common Bottlenecks

- **Constraint Propagation**: Can be expensive for complex rule sets
- **GPU Data Transfer**: Moving data between CPU and GPU can be costly
- **Find Lowest Entropy**: This operation doesn't parallelize as efficiently

## Optimizing Based on Benchmark Results

After identifying bottlenecks:

1. **For CPU bottlenecks**:
   - Improve cache locality by restructuring data
   - Consider using SIMD instructions for numerical operations
   - Adjust threading granularity for better parallelism

2. **For GPU bottlenecks**:
   - Optimize shader code in hotspot areas
   - Adjust workgroup sizes for better occupancy
   - Minimize CPU-GPU data transfers
   - Use shared memory for frequently accessed data

## Advanced Analysis

For more advanced analysis, consider:

- **Profiling with external tools**: Use platform-specific profilers like NVIDIA NSight for GPU or VTune for CPU
- **Memory profiling**: Track memory allocations to find leaks or excessive usage
- **Custom instrumentation**: Add custom timing points for more granular analysis

## Getting Help

If you need assistance interpreting benchmark results, please:

1. File an issue on GitHub with your benchmark results attached
2. Include details about your hardware and operating system
3. Specify which optimization you're trying to achieve
```

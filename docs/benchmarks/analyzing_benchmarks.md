# Analyzing Benchmark Results

This guide provides instructions on how to run and analyze benchmark results using Wave Forge's benchmarking tools.

## Running Benchmarks

Wave Forge includes comprehensive benchmarking capabilities that can be accessed through command-line arguments:

```bash
wave-forge --rule-file <path_to_rule_file> --benchmark-mode
```

Additional benchmark options:

- `--benchmark-csv-output <path>`: Save benchmark results to a CSV file
- `--cpu-only`: Run benchmarks using only CPU implementation
- `--width <n> --height <n> --depth <n>`: Set custom grid dimensions
- `--global-log-level debug`: Get more detailed output during benchmarking

## Interpreting Console Output

When running in benchmark mode, Wave Forge outputs detailed performance information to the console:

```
=== BENCHMARK COMPARISON ===
Grid Size: 16x16x16
Number of Tiles: 4

Metric               | CPU                  | GPU
-------------------------------------------------------------------
Total Time           | 2.82s                | 0.48s
Relative Speed       | baseline             | 5.88x faster
Iterations           | 4096                 | 4096
Collapsed Cells      | 4096                 | 4096
Memory Usage         | 6.5 MB               | 18.2 MB
Status               | Success              | Success

=== Performance Hotspots ===
Section                   | CPU       | GPU       | Speedup
-------------------------------------------------------------------
Entropy Calculation       | 1.15s     | 0.08s     | 14.38x
Constraint Propagation    | 1.05s     | 0.25s     | 4.20x
Collapse Cell             | 0.42s     | 0.10s     | 4.20x
Find Lowest Entropy       | 0.20s     | 0.05s     | 4.00x
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
- Implementation: "CPU" or "GPU"
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

# Create CPU vs GPU comparison
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df,
    x='Volume',
    y='Total Time (ms)',
    hue='Implementation'
)
plt.title('CPU vs GPU Performance by Grid Size')
plt.xlabel('Grid Volume (cells)')
plt.ylabel('Execution Time (ms)')
plt.yscale('log')
plt.savefig('cpu_vs_gpu_comparison.png')
plt.close()

# Calculate speedup
cpu_times = df[df['Implementation'] == 'CPU'].set_index('Volume')['Total Time (ms)']
gpu_times = df[df['Implementation'] == 'GPU'].set_index('Volume')['Total Time (ms)']
speedup = cpu_times / gpu_times.reindex(cpu_times.index)

plt.figure(figsize=(10, 6))
speedup.plot.bar()
plt.title('GPU Speedup Factor by Grid Size')
plt.xlabel('Grid Volume (cells)')
plt.ylabel('Speedup (x times faster)')
plt.grid(True, alpha=0.3)
plt.savefig('gpu_speedup.png')
plt.close()
```

## Identifying Bottlenecks

When analyzing benchmark results, pay attention to:

1. **Sections with Highest Total Time**: These are the primary bottlenecks
2. **Scaling with Grid Size**: How execution time grows with problem size
3. **Implementation Differences**: Parts that show the biggest difference between CPU and GPU
4. **Memory Usage Patterns**: How memory consumption scales with grid size

### Common Bottlenecks

- **Entropy Calculation**: Often the most time-consuming operation on CPU
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

# Wave Forge Benchmark Results

This document provides benchmark results for the Wave Function Collapse (WFC) implementation in Wave Forge across different grid sizes and configurations.

## Benchmark Methodology

The benchmarks measure the performance of the Wave Function Collapse algorithm under different conditions:

- **Grid Sizes**: Multiple 3D grid sizes ranging from small (8³) to large (32³)
- **Metrics**:
  - Total execution time
  - Memory usage
  - Iterations required for convergence
  - Performance hotspots (profiled sections)

Each benchmark is run multiple times and the results are averaged to ensure consistency.

## How to Run Benchmarks

You can run the benchmarks yourself using the following command:

```bash
wave-forge --rule-file <path_to_rule_file> --benchmark-mode --benchmark-csv-output benchmark_results.csv
```

Optional parameters:

- `--global-log-level debug` - For more detailed output

## Analysis

### Observations

1. **Scaling Efficiency**: The GPU advantage increases with grid size, showing the most benefit for larger problems.
2. **Memory Tradeoff**: GPU implementations use more memory but provide significant speed improvements.
3. **Bottlenecks**:
   - GPU: Constraint propagation dominates, despite being accelerated

### Platform-Specific Considerations

#### GPU Implementation

- Requires compatible GPU hardware and drivers
- Significant speedup for large grids
- Includes memory overhead for buffer allocations and shader compilation
- Data transfer between CPU and GPU can become a bottleneck

## Future Optimizations

Potential areas for performance improvement:

- Reduce CPU-GPU data transfer with persistent buffers
- Optimize workgroup sizes based on grid dimensions
- Implement workgroup memory for better locality

## Hardware References

These benchmarks were performed on:

- **CPU**: Intel Core i7-10700K @ 3.8GHz (8 cores, 16 threads)
- **GPU**: NVIDIA GeForce RTX 3080 (10GB VRAM)
- **RAM**: 32GB DDR4-3200
- **OS**: Windows 10 (64-bit)

Results will vary based on your specific hardware configuration.

# Wave Forge Visualization Configurations

This directory contains sample configurations for different visualization modes in Wave Forge. These configuration guides help you understand how to set up and use the various visualization options available.

## Available Visualization Modes

Wave Forge offers the following visualization modes:

1. **[Terminal Visualization](terminal_viz.md)** - Text-based display in your console/terminal
2. **[Simple2D Visualization](simple2d_viz.md)** - Graphical 2D window showing grid state

## General Configuration Options

All visualization modes support these common parameters:

| Parameter                    | Description                                             | Default |
| ---------------------------- | ------------------------------------------------------- | ------- |
| `--visualization-mode`       | Set visualization mode (`none`, `terminal`, `simple2d`) | `none`  |
| `--visualization-toggle-key` | Key to toggle visualization on/off                      | `T`     |
| `--report-progress-interval` | How often to update visualization (e.g., `100ms`)       | None    |

## Combined Configurations

You can combine visualization with other features like benchmarking, GPU acceleration, or custom seeds:

```bash
# Terminal visualization with benchmarking
wave-forge-app \
  --rule-file examples/simple-pattern.ron \
  --width 30 \
  --height 30 \
  --visualization-mode terminal \
  --benchmark-mode \
  --benchmark-csv-output benchmark_results.csv

# Simple2D visualization with GPU acceleration
wave-forge-app \
  --rule-file examples/simple-pattern.ron \
  --width 50 \
  --height 50 \
  --visualization-mode simple2d \
  --report-progress-interval 50ms

# Force CPU-only with terminal visualization
wave-forge-app \
  --rule-file examples/simple-pattern.ron \
  --width 40 \
  --height 40 \
  --visualization-mode terminal \
  --cpu-only
```

## Usage Tips

- For performance benchmarking, use `--visualization-mode none` or toggle off visualization during measurement
- For debugging, set a slower progress interval (e.g., `500ms`) to observe step-by-step changes
- For demonstrations, use Simple2D visualization with medium-sized grids for best visual results
- Adjust the grid dimensions based on the visualization mode (smaller for terminal, larger for Simple2D)

## Example Shell/Batch Scripts

See the [examples directory](../../examples/) for ready-to-use shell/batch scripts that combine these configurations for common use cases.

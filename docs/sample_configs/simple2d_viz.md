# Simple2D Visualization Mode Configuration

This configuration file shows how to set up and use the Simple2D visualization mode in Wave Forge.

## Command Line Usage

To run Wave Forge with Simple2D visualization:

```bash
wave-forge-app --rule-file examples/simple-pattern.ron --width 30 --height 30 --depth 10 --visualization-mode simple2d
```

## Key Features

- Real-time graphical 2D window showing grid state
- Color-coded visualization:
  - Collapsed cells: Grayscale based on tile ID
  - Uncollapsed cells: Blue with intensity based on entropy
  - Contradictions: Red
- Window resizes automatically based on grid dimensions
- Supports toggle key to enable/disable visualization during runtime

## Configuration Options

| Parameter                       | Description                        | Default |
| ------------------------------- | ---------------------------------- | ------- |
| `--visualization-mode simple2d` | Activates Simple2D visualization   | None    |
| `--visualization-toggle-key T`  | Key to toggle visualization on/off | T       |

## Example Configuration

```bash
# Basic Simple2D visualization
wave-forge-app \
  --rule-file examples/simple-pattern.ron \
  --width 50 \
  --height 50 \
  --depth 20 \
  --visualization-mode simple2d

# With custom toggle key
wave-forge-app \
  --rule-file examples/simple-pattern.ron \
  --width 50 \
  --height 50 \
  --depth 20 \
  --visualization-mode simple2d \
  --visualization-toggle-key V

# With benchmarking mode
wave-forge-app \
  --rule-file examples/simple-pattern.ron \
  --width 50 \
  --height 50 \
  --depth 20 \
  --visualization-mode simple2d \
  --benchmark-mode \
  --report-progress-interval 100ms
```

## Window Controls

During execution with Simple2D visualization:

- Press the toggle key (default: 'T') to turn visualization on/off
- Press 'ESC' to close the visualization window
- Use arrow keys to navigate through Z-layers (up/down)

## System Requirements

- Simple2D visualization requires a graphical environment
- Uses the minifb library for window creation and buffer management
- Works on Windows, macOS, and Linux with graphical interfaces
- Falls back to terminal visualization if window creation fails

## Tips

- Simple2D visualization is ideal for:
  - Watching the algorithm progress in real-time
  - Debugging pattern generation issues
  - Demonstrations and presentations
- For large grids, visualization may slow down the algorithm
- Toggle visualization off (with toggle key) during benchmarking for accurate measurements

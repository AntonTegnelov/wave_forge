# Terminal Visualization Mode Configuration

This configuration file shows how to set up and use the terminal visualization mode in Wave Forge.

## Command Line Usage

To run Wave Forge with terminal visualization:

```bash
wave-forge-app --rule-file examples/simple-pattern.ron --width 20 --height 20 --depth 5 --visualization-mode terminal
```

## Key Features

- Text-based rendering in your terminal
- Lightweight with minimal dependencies
- Shows a 2D slice of the 3D grid
- Displays the current Z-layer being viewed
- Color coding of cell states:
  - Collapsed cells: Shown with specific characters based on tile ID
  - Uncollapsed cells: Indicated by the number of remaining possibilities
  - Contradictions: Highlighted in red

## Configuration Options

| Parameter                       | Description                        | Default |
| ------------------------------- | ---------------------------------- | ------- |
| `--visualization-mode terminal` | Activates terminal visualization   | None    |
| `--visualization-toggle-key T`  | Key to toggle visualization on/off | T       |

## Example Configuration

```bash
# Basic terminal visualization with customized key
wave-forge-app \
  --rule-file examples/simple-pattern.ron \
  --width 30 \
  --height 20 \
  --depth 10 \
  --visualization-mode terminal \
  --visualization-toggle-key V

# With progress reporting for more verbose output
wave-forge-app \
  --rule-file examples/simple-pattern.ron \
  --width 30 \
  --height 20 \
  --depth 10 \
  --visualization-mode terminal \
  --report-progress-interval 200ms \
  --progress-log-level debug
```

## Tips

- Terminal visualization is ideal for:
  - Headless environments
  - SSH sessions
  - Quick debugging
  - Systems with minimal graphics support
- The visualization update rate is tied to the progress reporting interval if specified
- Use a smaller grid size for clearer visualization in the terminal

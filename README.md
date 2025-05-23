# Wave Forge

**Note:** This project is currently under heavy development and is not yet ready for use.

GPU-accelerated 3D terrain generation using Wave Function Collapse (WFC).

## Building

_Instructions on how to build the project will go here._

## Usage

_Instructions on how to run or use the project will go here._

### Command-line Options

Wave Forge supports various command-line options to configure its behavior:

```
USAGE:
    wave-forge-app [OPTIONS] --rule-file <FILE>

OPTIONS:
    -r, --rule-file <FILE>                    Path to the RON rule file defining tiles and adjacencies
        --width <WIDTH>                       Width of the output grid [default: 10]
        --height <HEIGHT>                     Height of the output grid [default: 10]
        --depth <DEPTH>                       Depth of the output grid [default: 10]
        --seed <SEED>                         Optional seed for the random number generator
    -o, --output-path <FILE>                  Path to save the generated output grid [default: output.txt]
        --benchmark-mode                      Run in benchmark mode
        --report-progress-interval <DURATION> Report progress updates every specified interval (e.g., "1s", "500ms")
        --progress-log-level <LEVEL>          Log level to use for progress reporting [default: info]
                                              [possible values: trace, debug, info, warn]
        --visualization-mode <MODE>           Choose the visualization mode [default: none]
                                              [possible values: none, terminal, simple2d]
        --visualization-toggle-key <KEY>      Key to toggle visualization on/off during runtime [default: T]
        --benchmark-csv-output <CSV_FILE>     Optional: Path to save benchmark results as a CSV file
```

### Progress Reporting

The project supports configurable progress reporting with two main settings:

1. **Report Interval** (`--report-progress-interval`):

   - Controls how frequently progress updates are displayed
   - Example: `--report-progress-interval 500ms` for updates every half second
   - If not specified, progress reporting is disabled

2. **Log Level** (`--progress-log-level`):
   - Controls at which log level progress messages are emitted
   - Options:
     - `trace` - Very detailed logging, typically only visible when trace logging is enabled
     - `debug` - Detailed logging visible when debug logging is enabled
     - `info` - Standard logging level (default)
     - `warn` - Less frequent, higher priority logging
       You can also configure the progress log level in the project's configuration file.

These settings allow you to control both the frequency and visibility of progress updates. For example:

```
# Show progress every second at INFO level (default)
wave-forge-app --rule-file rules.ron --report-progress-interval 1s

# Show progress every 100ms at DEBUG level (more verbose but requires debug logging enabled)
wave-forge-app --rule-file rules.ron --report-progress-interval 100ms --progress-log-level debug
```

To control the overall log level at runtime, use the `RUST_LOG` environment variable:

```
# Windows PowerShell
$env:RUST_LOG="debug"; wave-forge-app --rule-file rules.ron

# Windows Command Prompt
set RUST_LOG=debug
wave-forge-app --rule-file rules.ron

# Linux/macOS
RUST_LOG=debug wave-forge-app --rule-file rules.ron
```

### Feature Flags

- `winapi`: Enables platform-specific code, primarily for attempting to measure memory usage during benchmarks on Windows using the `winapi` crate. This feature is optional and the application will build and run without it, although memory usage reporting might be less accurate or unavailable on Windows.

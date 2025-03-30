# Wave Forge

**Note:** This project is currently under heavy development and is not yet ready for use.

Multithreaded 3D GPU Terrain generation using Wave Function Collapse (WFC).

## Building

_Instructions on how to build the project will go here._

## Usage

_Instructions on how to run or use the project will go here._

## Running Tests

There are several ways to run tests in this project:

1. Run tests for a specific crate:

   ```
   cargo test -p wfc-core
   cargo test -p wfc-rules
   cargo test -p wfc-gpu
   cargo test -p wave-forge-app
   ```

2. Run tests for all workspace members:

   ```
   cargo test --workspace
   ```

3. Use the convenience scripts:
   - PowerShell: `.\test-core.ps1` (tests core packages excluding GPU)
   - Batch: `test-core.bat` (same as above, for Command Prompt)
   - PowerShell: `.\test-all.ps1` (tests all packages including GPU)

Note: GPU tests may take longer to run as they need to initialize the GPU.

## Contributing

_Information about contributing to the project._

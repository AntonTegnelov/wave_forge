# Wave Forge Application - TODO List

## General Improvements

- [x] Standardize error handling approach across modules for consistency (using `thiserror` for library crates, `anyhow` for the application binary)
- [ ] Improve resource management with graceful cleanup for all resources
- [x] Better document feature flags and how they interact

## main.rs

- [x] Split main.rs into smaller files for better maintainability
- [x] Improve visualization thread management to ensure threads are properly joined
- [x] Add graceful shutdown logic for all components

## config.rs

- [x] Add support for configuration files in addition to command-line arguments
- [x] Add validation for interdependent configuration options

## output.rs

- [ ] Support for incremental/streaming output

## progress.rs

- [x] Implement more detailed progress statistics
- [x] Add option to save progress logs to a file

## Performance Optimizations

- [ ] optimize critical code paths for large grid sizes
- [ ] Implement more efficient data structures
- [ ] Explore additional parallelization opportunities
- [ ] Optimize memory usage

# Wave Forge Application - TODO List

## GPU Migration

- [x] Figure out everywhere that need to be adjusted. Mark them with comments and create a markdown doc that keeps all the places and what needs to be done
- [x] Make all wave collapse operations GPU-based
- [x] Implement the changes in the doc
- [x] Remove CPU implementation entirely
- [x] Refactor `main.rs` to remove CPU-specific code paths
- [x] Update benchmarking to focus on GPU performance only
- [x] Update documentation to reflect GPU-only approach

## General Improvements

- [x] Standardize error handling approach across modules for consistency (using `thiserror` for library crates, `anyhow` for the application binary)
- [ ] Improve resource management with graceful cleanup for all resources
- [x] Better document feature flags and how they interact

## main.rs

- [x] Split main.rs into smaller files for better maintainability
- [x] Improve visualization thread management to ensure threads are properly joined
- [x] Add graceful shutdown logic for all components

## config.rs

- [ ] Add support for configuration files in addition to command-line arguments
- [ ] Add validation for interdependent configuration options

## output.rs

- [ ] Support for incremental/streaming output

## progress.rs

- [ ] Implement more detailed progress statistics
- [ ] Add option to save progress logs to a file

## Performance Optimizations

- [ ] optimize critical code paths for large grid sizes
- [ ] Implement more efficient data structures
- [ ] Explore additional parallelization opportunities
- [ ] Optimize memory usage

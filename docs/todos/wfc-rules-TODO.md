# WFC Rules Module TODOs

## Format Handling Improvements

- [x] Create a proper trait/interface for format parsers to implement
- [ ] Implement full support for bitcode serialization format as default (https://crates.io/crates/bitcode)
- [ ] Separate format-specific parsing from validation logic

## Error Handling

- [ ] Make error variants more specific with additional context
- [ ] Improve "Internal error" messages with more detailed diagnostics
- [ ] Add validation for invalid or contradictory axis combinations
- [ ] Add validation for bidirectional consistency in adjacency rules

## API Design

- [ ] Separate parsing logic from deserialization for better testability
- [ ] Add support for loading directly from embedded resources

## Documentation and Testing

- [ ] Add comprehensive module-level documentation
- [ ] Create test cases for all parsing logic branches
- [ ] Add tests for error handling conditions
- [ ] Verify and validate the examples in documentation

## Performance Optimizations

- [ ] Avoid allocating entire adjacency matrix; use sparse representation
- [ ] Reduce string allocations and HashMaps during parsing

## New Features

- [ ] Add support for tile symmetry and rotation rules
- [ ] Add adjacency rule visualization/verification tools
- [ ] Create helper functions for common rule patterns

## Implementation Refinements

- [ ] Document and formalize 3D axes assumption or make it configurable
- [ ] Optimize memory usage of rule representation
- [ ] Consider custom iterators for more efficient rule traversal

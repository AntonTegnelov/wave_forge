# GPU Migration Plan

This document outlines the steps and locations required to migrate the core Wave Function Collapse logic from CPU to GPU using the `wfc-gpu` crate.

## Files and Sections Requiring Modification

- **File:** `path/to/file1.rs`
  - **Section/Function:** `function_name_1`
  - **Change Needed:** Description of the change.
- **File:** `path/to/file2.rs`
  - **Section/Function:** `struct_name::method_name`
  - **Change Needed:** Description of the change.

* **File:** `wfc-core/src/runner.rs`

  - **Section/Function:** `run<P: ConstraintPropagator, E: EntropyCalculator>(...)`
  - **Change Needed:**
    - Modify generic bounds or use dynamic dispatch to select the GPU implementations of `ConstraintPropagator` and `EntropyCalculator` (provided by `wfc-gpu`).
    - Ensure the initial propagation call (`propagator.propagate(grid, all_coords, rules)`) uses the GPU implementation.
    - Ensure the entropy calculation call (`entropy_calculator.calculate_entropy(grid)`) uses the GPU implementation.
    - Ensure the lowest entropy finding call (`entropy_calculator.find_lowest_entropy(&entropy_grid)`) uses the GPU implementation.
    - Ensure the main propagation call (`propagator.propagate(grid, vec![(x, y, z)], rules)`) uses the GPU implementation.

* **File:** `wfc-gpu/src/lib.rs` (or relevant modules within)

  - **Section/Function:** TBD (Needs implementation)
  - **Change Needed:** Implement `wfc_core::entropy::EntropyCalculator` trait using wgpu.

* **File:** `wfc-gpu/src/lib.rs` (or relevant modules within)

  - **Section/Function:** TBD (Needs implementation)
  - **Change Needed:** Implement `wfc_core::propagator::ConstraintPropagator` trait using wgpu.

* **File:** `wave-forge-app/src/main.rs` (and potentially `benchmark.rs`)
  - **Section/Function:** WFC execution setup code (where `wfc_core::run` is called).
  - **Change Needed:** Instantiate and pass the GPU versions of the propagator and entropy calculator instead of the CPU versions.

## Integration Strategy

1.  **Trait Implementation:** The `wfc-gpu` crate must implement the `EntropyCalculator` and `ConstraintPropagator` traits from `wfc-core`.
2.  **Data Transfer:** `wfc-gpu` implementations will need to manage copying `PossibilityGrid` data to GPU buffers (likely using `wgpu::Buffer::write_buffer`) before computation and reading results back (e.g., using `wgpu::CommandEncoder::copy_buffer_to_buffer` and mapping the staging buffer).
3.  **State Management:** The GPU implementations will need access to the `wgpu` Device and Queue. This might be managed within a `GpuContext` struct passed during initialization or held within the GPU propagator/calculator structs themselves.
4.  **Instantiation:** The application (`wave-forge-app`) will need to initialize the wgpu backend and then create instances of the GPU propagator and calculator, passing them to `wfc_core::run`.
5.  **Error Handling:** GPU-specific errors (`PropagationError::Gpu*`) need to be handled appropriately in the application.

## Open Questions / Considerations

- How will the `wgpu::Device` and `wgpu::Queue` be managed and passed to the `wfc-gpu` implementations? (Global context? Passed into constructors?)
- What is the most efficient way to transfer the `PossibilityGrid` (likely `BitVec` data) to and from the GPU?
- How should tile weights be handled in the GPU entropy calculation (if implementing Shannon entropy)? (Passed as uniform? Storage buffer?)
- How should adjacency rules be represented and accessed on the GPU for propagation? (Storage buffer?)
- Synchronization between entropy calculation, lowest entropy finding (reduction), and propagation steps.

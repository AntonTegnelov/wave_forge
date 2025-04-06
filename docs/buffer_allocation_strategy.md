# Buffer Allocation Strategy in WFC-GPU

This document explains the buffer allocation strategy used in the GPU-accelerated Wave Function Collapse implementation. Understanding how buffers are created, sized, and managed is crucial for optimizing performance and memory usage in the algorithm.

## Buffer Types and Purposes

The `GpuBuffers` struct manages several types of buffers, each serving a specific role in the WFC algorithm:

### Primary State Buffers

- **Grid Possibilities Buffer**: Stores the current state of all cell possibilities (the core WFC grid state). Each cell's possibilities are represented as a bit mask packed into one or more 32-bit integers.
- **Entropy Buffer**: Stores the calculated entropy value for each cell, used to determine which cell to collapse next.
- **Min Entropy Info Buffer**: Stores information about the cell with minimum entropy (the value and its index).

### Propagation Buffers

- **Worklist Buffers (A/B)**: Double-buffered design for storing the list of cells that need propagation updates. One buffer serves as input, the other as output, and they swap roles between propagation passes.
- **Worklist Count Buffer**: Tracks the number of cells in the current worklist.

### Status and Control Buffers

- **Contradiction Flag Buffer**: A single-value buffer that is set when a contradiction is detected during propagation.
- **Contradiction Location Buffer**: Stores the location (cell index) where a contradiction was found.
- **Pass Statistics Buffer**: Records metrics about each propagation pass, such as cells processed and possibilities eliminated.

### Rule-Related Buffers

- **Adjacency Rules Buffer**: Stores tile adjacency rules in a GPU-optimized format.
- **Rule Weights Buffer**: Stores weights for adjacency rules when using weighted constraints.

### Parameter Buffers

- **Params Uniform Buffer**: Holds algorithm parameters like grid dimensions, tile counts, and configuration settings.
- **Entropy Params Buffer**: Holds parameters specific to entropy calculation.

### Staging Buffers

- Each primary buffer has a corresponding staging buffer used for CPU-GPU data transfer.

## Buffer Sizing Strategy

### Initial Sizing

When buffers are first created in `GpuBuffers::new()`, their sizes are determined based on:

1. **Grid Dimensions**: Width × Height × Depth determines the number of cells
2. **Tile Count**: Affects how many bits are needed to represent each cell's state
3. **Usage Pattern**: Whether the buffer needs to store data for all cells or just a subset

```rust
// Example: Grid Possibilities buffer size calculation
let num_cells = width * height * depth;
let u32s_per_cell = (num_tiles + 31) / 32; // Ceiling division to get number of u32s needed
let buffer_size = (num_cells * u32s_per_cell * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
```

### Bit Packing

For the grid possibilities buffer, a bit-packing strategy is used:

- Each tile possibility is represented by one bit (1 = possible, 0 = impossible)
- These bits are packed into 32-bit words (`u32`)
- The number of `u32` words needed per cell is calculated as `(num_tiles + 31) / 32`

This approach minimizes memory usage while providing efficient bit operations on the GPU.

## Dynamic Buffer Management

The `DynamicBufferConfig` struct controls how buffers adapt to changing requirements:

```rust
pub struct DynamicBufferConfig {
    pub growth_factor: f32,        // How much to grow buffers when resizing (e.g., 1.5 = 50% growth)
    pub min_buffer_size: u64,      // Minimum size for any buffer
    pub max_buffer_size: u64,      // Maximum size allowed for any buffer
    pub auto_resize: bool,         // Whether to automatically resize buffers
}
```

### Buffer Resizing Policies

When a buffer needs to grow (e.g., for a larger grid or more tiles):

1. **Growth Factor**: Buffers grow by a factor (default 1.5×) to reduce frequent resizing
2. **Minimum Size**: All buffers have a minimum size to handle small grids efficiently
3. **Maximum Size**: A cap prevents excessive memory allocation
4. **Auto-Resize**: Buffers can resize automatically when operations would exceed their capacity

The resize operation in `resize_buffer()` creates a new, larger buffer rather than attempting to resize in place, as this is more compatible with the WGPU API's design.

## Buffer Usage Optimization

Several strategies are employed to optimize buffer usage:

### 1. Double Buffering for Propagation

Propagation uses a double-buffered approach with `worklist_buf_a` and `worklist_buf_b`:

- One serves as the input (cells to process in this pass)
- The other as output (cells to process in the next pass)
- They swap roles between propagation passes
- This eliminates the need for barrier synchronization within a pass

### 2. Structure of Arrays (SoA) Layout

Instead of storing all data for each cell together (Array of Structures), the implementation uses a Structure of Arrays approach:

- All possibilities bits for the first slot of each cell are stored together
- This improves memory access patterns on the GPU
- Enables coalesced memory operations for better throughput

### 3. Staging Buffers for Efficient Data Transfer

Data transfer between CPU and GPU uses staging buffers:

- CPU-side changes are written to staging buffers
- These are then copied to device-local buffers
- Results are read from device-local to staging, then to CPU
- This approach aligns with WGPU's memory model and improves performance

### 4. Smart Buffer Reuse

The implementation avoids creating new buffers when existing ones can be reused:

- Buffers are checked for sufficient size before operations
- Only resized when necessary
- Buffer usage flags are carefully chosen to enable necessary operations while avoiding excessive capabilities

## Error Recovery and Resiliency

The buffer allocation strategy includes features for error recovery:

1. **Original Grid Dimensions Storage**: Kept for potential recovery operations
2. **Adaptive Timeouts**: Timeout durations for buffer operations scale with grid size and complexity
3. **Recoverable Operations**: Buffer operations are wrapped in `RecoverableGpuOp` for automatic retry on transient failures

## Memory Management

### Cleanup

The `GpuBuffers` struct implements `Drop` to ensure proper cleanup of GPU resources.

### Arc Wrapping

All buffers are wrapped in `Arc` (Atomic Reference Counting):

- Enables safe sharing between threads
- Allows asynchronous operations without complex lifetime management
- Ensures buffers are only freed when all references are dropped

## Buffer Creation Functions

The main buffer creation functions include:

- `create_buffer()`: Creates a standard WGPU buffer with specified usage flags
- `resize_buffer()`: Creates a new larger buffer according to dynamic buffer config
- `ensure_grid_possibilities_buffer()`: Ensures the grid buffer is large enough for current dimensions
- `ensure_entropy_buffer()`: Same for entropy calculations
- `ensure_worklist_buffers()`: Same for propagation worklists

## Optimizing for Different Hardware

The strategy includes adaptations for different GPU hardware:

1. **Feature Detection**: Buffer usage patterns adapt based on available GPU features
2. **Fallback Implementations**: Alternative implementations are provided for hardware lacking specific capabilities
3. **Workgroup Size Adaptation**: Computation patterns adjust to match the GPU's preferred workgroup sizes

## Performance Considerations

When working with WFC-GPU buffers, consider:

1. **Memory Overhead**: The implementation balances between memory usage and performance
2. **Transfer Costs**: Minimize CPU-GPU data transfers, especially for large grids
3. **Growth Factor**: The default 1.5× growth factor balances allocation frequency against wasted space
4. **Alignment Requirements**: All buffers respect GPU-specific alignment requirements, particularly for uniform buffers

## Examples

### Creating a New Buffer Set

```rust
let buffers = GpuBuffers::new(
    &device,
    &queue,
    &initial_grid,
    &rules,
    wfc_core::BoundaryCondition::Periodic,
)?;
```

### Enabling Dynamic Resizing

```rust
let buffers = GpuBuffers::new(...)?
    .with_dynamic_buffer_config(DynamicBufferConfig {
        growth_factor: 2.0,      // More aggressive growth
        min_buffer_size: 4096,    // Larger minimum size
        max_buffer_size: 1 << 30, // 1GB maximum
        auto_resize: true,
    });
```

### Manually Resizing for a Larger Grid

```rust
buffers.resize_for_grid(
    &device,
    new_width,
    new_height,
    new_depth,
    num_tiles,
    &DynamicBufferConfig::default(),
)?;
```

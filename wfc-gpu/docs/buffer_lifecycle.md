# WFC-GPU Buffer Lifecycle and Synchronization

This document explains the management of GPU buffers within the `wfc-gpu` crate, focusing on their lifecycle (creation, potential resizing) and the synchronization mechanisms used for data transfer between the CPU and GPU.

## Core Components

- **`GpuBuffers` (`buffers.rs`)**: Primarily responsible for the **creation** and **storage** of all necessary WGPU buffers used by the WFC algorithm. It holds `Arc<wgpu::Buffer>` references.
  - Owns the buffer definitions and initial creation logic.
  - Handles buffer resizing logic (via `ensure_*_buffer` methods) if dynamic resizing is enabled.
  - Provides accessors to the buffers.
  - **Retains** the `download_results` method, as this orchestrates the download of _multiple_ potential buffers based on a request, rather than being a simple, single-buffer transfer.
- **`GpuSynchronizer` (`sync.rs`)**: Responsible for **data transfer** (upload/download) between CPU and GPU for specific buffers, and for **resetting** GPU buffer states.
  - Uses the `wgpu::Queue` for `write_buffer` operations (uploads, resets).
  - Uses `copy_buffer_to_buffer` and buffer mapping for downloads.
  - Holds references to `Arc<wgpu::Device>`, `Arc<wgpu::Queue>`, and `Arc<GpuBuffers>`.
  - Provides methods like `upload_grid`, `download_grid`, `upload_initial_updates`, `reset_min_entropy_buffer`, `download_contradiction_status`, etc.

## Buffer Lifecycle

1.  **Creation**: Buffers are created within `GpuBuffers::new` based on the initial grid dimensions and rules. Initial data (like packed grid possibilities and rules) is uploaded using `device.create_buffer_init`.
2.  **Usage**: Compute shaders (managed by `ComputePipelines`) access these buffers via bind groups. `GpuSynchronizer` methods are called to upload new data (e.g., initial worklist) or download results.
3.  **Resizing (Optional)**: If `DynamicBufferConfig` is enabled, methods like `ensure_grid_possibilities_buffer` within `GpuBuffers` might be called (potentially by `GpuAccelerator` or a coordinator layer) before certain operations to ensure buffers are large enough. Resizing typically involves creating a _new_ larger buffer and replacing the `Arc` in `GpuBuffers`. The old buffer's memory is reclaimed when its `Arc` reference count drops to zero.
4.  **Download**: `GpuSynchronizer` methods handle copying data from GPU buffers to staging buffers and mapping them for CPU access.
5.  **Cleanup**: Buffer memory is managed by `wgpu` and `Arc`. When the `Arc<wgpu::Buffer>` references held by `GpuBuffers` (and potentially temporarily by other components) are dropped, `wgpu` eventually reclaims the GPU memory.

## Synchronization Strategy

- **Uploads/Resets**: Primarily use `queue.write_buffer`, which is generally asynchronous from the CPU's perspective but ordered on the GPU timeline.
- **Downloads**: Use a two-step process:
  1.  `encoder.copy_buffer_to_buffer` (GPU -> Staging Buffer): A GPU-side copy scheduled via a command encoder.
  2.  `staging_buffer.slice(..).map_async()`: Asynchronously requests the CPU to map the staging buffer memory once the GPU copy is complete. Polling (`device.poll`) or async runtimes (`tokio`) are used to wait for the mapping result.
- **Coordination**: Higher-level components like `GpuAccelerator` or `GpuPropagator` coordinate calls to `GpuSynchronizer` methods and compute shader dispatches to ensure correct data flow.
- **Error Handling**: `GpuSynchronizer` methods return `Result<_, GpuError>`. The `download_results` method within `GpuBuffers` uses `RecoverableGpuOp` for more robust handling of potential mapping timeouts or errors during bulk downloads.

## Ownership

- `GpuBuffers` logically _owns_ the buffer definitions and holds the primary `Arc` references.
- `GpuSynchronizer` _borrows_ access to these buffers via its `Arc<GpuBuffers>` field to perform data transfers.
- Other components (`GpuPropagator`, `GpuEntropyCalculator`, `ComputePipelines`) receive `Arc<GpuBuffers>` or references to specific buffers as needed for their operations (e.g., creating bind groups). `Arc` ensures shared ownership without unnecessary cloning of the underlying GPU resources.

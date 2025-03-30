use crate::GpuError;
use bytemuck::{Pod, Zeroable};
use wfc_core::{grid::PossibilityGrid, rules::AdjacencyRules};
use wgpu;
use wgpu::util::DeviceExt; // Import for create_buffer_init

// Uniform buffer structure - MUST match shader layout
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuParamsUniform {
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    num_tiles: u32,
    num_tiles_u32: u32, // Number of u32s needed per cell for possibilities
    num_axes: u32,
    // Add padding if necessary for alignment (std140 layout)
    _padding1: u32,
    _padding2: u32,
}

// Placeholder struct for managing GPU buffers
pub struct GpuBuffers {
    // Grid state (possibilities) - likely atomic u32 for bitvec representation
    pub grid_possibilities_buf: wgpu::Buffer,
    // Adjacency rules (flattened)
    pub rules_buf: wgpu::Buffer,
    // Entropy output buffer
    pub entropy_buf: wgpu::Buffer,
    // Buffer for updated coordinates (input to propagation)
    pub updates_buf: wgpu::Buffer,
    // Staging buffers for reading results back to CPU (e.g., entropy, contradiction)
    pub entropy_staging_buf: wgpu::Buffer,
    pub params_uniform_buf: wgpu::Buffer,
    pub contradiction_flag_buf: wgpu::Buffer,
    pub contradiction_staging_buf: wgpu::Buffer,
}

impl GpuBuffers {
    pub fn new(
        device: &wgpu::Device,
        initial_grid: &PossibilityGrid,
        rules: &AdjacencyRules,
    ) -> Result<Self, GpuError> {
        let width = initial_grid.width;
        let height = initial_grid.height;
        let depth = initial_grid.depth;
        let num_cells = width * height * depth;
        let num_tiles = rules.num_tiles();
        let num_axes = rules.num_axes();

        // --- Pack Possibilities (Manual Bit Packing) ---
        let bits_per_cell = num_tiles;
        let u32s_per_cell = (bits_per_cell + 31) / 32; // Ceiling division
        let mut packed_possibilities: Vec<u32> = Vec::with_capacity(num_cells * u32s_per_cell);

        for cell_bitvec in initial_grid.get_cell_data() {
            let mut cell_data_u32 = vec![0u32; u32s_per_cell];
            for (i, bit) in cell_bitvec.iter().by_vals().enumerate() {
                if bit {
                    let u32_idx = i / 32;
                    let bit_idx = i % 32;
                    if u32_idx < cell_data_u32.len() {
                        // Ensure index is in bounds
                        cell_data_u32[u32_idx] |= 1 << bit_idx;
                    }
                }
            }
            packed_possibilities.extend_from_slice(&cell_data_u32);
        }
        let _grid_buffer_size = (packed_possibilities.len() * std::mem::size_of::<u32>()) as u64;

        // --- Pack Rules ---
        let num_rules = num_axes * num_tiles * num_tiles;
        let u32s_for_rules = (num_rules + 31) / 32;
        let mut packed_rules = vec![0u32; u32s_for_rules];
        for (i, &allowed) in rules.get_allowed_rules().iter().enumerate() {
            if allowed {
                let u32_idx = i / 32;
                let bit_idx = i % 32;
                packed_rules[u32_idx] |= 1 << bit_idx;
            }
        }
        let _rules_buffer_size = (packed_rules.len() * std::mem::size_of::<u32>()) as u64;

        // --- Create Uniform Buffer Data ---
        let params = GpuParamsUniform {
            grid_width: width as u32,
            grid_height: height as u32,
            grid_depth: depth as u32,
            num_tiles: num_tiles as u32,
            num_tiles_u32: u32s_per_cell as u32,
            num_axes: num_axes as u32,
            _padding1: 0, // Explicit padding
            _padding2: 0,
        };
        let _params_buffer_size = std::mem::size_of::<GpuParamsUniform>() as u64;

        // --- Calculate Other Buffer Sizes ---
        let entropy_buffer_size = (num_cells * std::mem::size_of::<f32>()) as u64;
        // Max updates could be num_cells, passing index (u32)
        let updates_buffer_size = (num_cells * std::mem::size_of::<u32>()) as u64;
        let contradiction_buffer_size = std::mem::size_of::<u32>() as u64;

        // --- Create Buffers --- (Use calculated sizes)
        let grid_possibilities_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Possibilities"),
            contents: bytemuck::cast_slice(&packed_possibilities),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let rules_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Rules"),
            contents: bytemuck::cast_slice(&packed_rules),
            usage: wgpu::BufferUsages::STORAGE, // Read-only in shader
        });

        let params_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Uniform"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let entropy_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Entropy"),
            size: entropy_buffer_size, // Use calculated size
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let updates_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Updates"),
            size: updates_buffer_size, // Use calculated size
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let contradiction_flag_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Contradiction Flag"),
            size: contradiction_buffer_size, // Use calculated size
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let entropy_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Entropy Staging"),
            size: entropy_buffer_size, // Use calculated size
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let contradiction_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Contradiction Staging"),
            size: contradiction_buffer_size, // Use calculated size
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            grid_possibilities_buf,
            rules_buf,
            params_uniform_buf,
            entropy_buf,
            updates_buf,
            contradiction_flag_buf,
            entropy_staging_buf,
            contradiction_staging_buf,
        })
    }

    // TODO: Add methods for uploading updates (updated_coords) to updates_buf
    // pub fn upload_updates(&self, queue: &wgpu::Queue, updates: &[(usize, usize, usize)]) { ... }

    // TODO: Add methods for downloading results (entropy, contradiction flag) from staging buffers
    // pub async fn download_entropy(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<EntropyGrid, GpuError> { ... }
    // pub async fn check_contradiction(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<bool, GpuError> { ... }
}

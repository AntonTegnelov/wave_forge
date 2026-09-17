//! Can one workgroup solve one whole chunk?
//!
//! The run loop in `GpuAccelerator` blocks on the device several times per collapse, so the GPU
//! spends its time waiting rather than computing (docs/solver-redesign.md, "The round-trip audit").
//! This benchmark tests the formulation that keeps the entire solve inside a single dispatch: each
//! workgroup holds one chunk's domains in workgroup memory and loops over sweeps without ever
//! returning to the CPU. Parallelism is spent across chunks, where WFC is embarrassingly parallel,
//! rather than inside one propagation, where it is not.
//!
//! ```text
//! cargo test -p wfc-gpu --release --test block_solver_bench -- --ignored --nocapture
//! ```
//!
//! Correctness is asserted against the CPU reference solver (`wfc_devtools::reference`); timings
//! are printed. A timing describes one build on one machine and driver stack.

use wfc_devtools::city;
use wfc_devtools::reference::{self, ReferenceSolver};

/// Chunk dimensions. Workgroup memory holds the chunk, so these bound what fits (see `KERNEL`).
const CX: u32 = 8;
const CY: u32 = 8;
const CZ: u32 = 8;
const CELLS: u32 = CX * CY * CZ;
/// Words per cell in the kernel, sized for the city's 81 variants.
const WORDS: u32 = 3;
/// Entries of the per-chunk statistics record the kernel writes.
const STATS: u32 = 8;

/// The kernel, as a state machine that advances once per workgroup-wide step.
///
/// Every step runs the same shape: per-lane work chosen by the shared state, a barrier, then lane 0
/// alone reads the shared flags and decides the next state, which every lane picks up through
/// `workgroupUniformLoad`. WGSL only allows barriers under control flow that is identical for all
/// invocations, and a value loaded from workgroup memory is not identical by construction; routing
/// every decision through one uniform load is what makes the loop legal.
///
/// Propagation is a gather sweep: each cell intersects, for every neighbour changed since the
/// previous sweep, the union of that neighbour's rule rows. Only a cell's own lane writes it, and
/// domains only shrink, so a concurrent read sees a superset of the neighbour's final state and can
/// never remove a supported tile; the fixpoint does not depend on thread order.
///
/// Workgroup memory at 8x8x8 and 81 tiles: domains 6144 B, rules 5832 B, epochs 2048 B, reduction
/// keys 1024 B, control 40 B, which is 15088 B against the 16384 B WebGPU default.
const KERNEL: &str = r#"
const CX: u32 = {CX}u;
const CY: u32 = {CY}u;
const CELLS: u32 = {CELLS}u;
const NT: u32 = {NT}u;
const WG: u32 = 256u;
const RULE_WORDS: u32 = {RULE_WORDS}u;
const STATS: u32 = {STATS}u;
const NONE: u32 = 0xFFFFFFFFu;

const LOAD: u32 = 0u;
const PROPAGATE: u32 = 1u;
const WRITE: u32 = 2u;
const DONE: u32 = 3u;

const STATUS_OK: u32 = 0u;
const STATUS_FAILED: u32 = 1u;
const STATUS_CAP: u32 = 2u;
const STATUS_BOUNDARY: u32 = 3u;

struct Params { mode: u32, max_steps: u32, seed: u32, max_attempts: u32 };

struct Ctrl {
    phase: u32,
    sweep: u32,
    status: u32,
    steps: u32,
    contra_cell: u32,
    sweeps: u32,
    collapses: u32,
    restarts: u32,
    // Collapses in the current attempt; zero means a contradiction came from the chunk's own borders.
    step: u32,
};

@group(0) @binding(0) var<storage, read> rules: array<u32>;
@group(0) @binding(1) var<storage, read> init: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<storage, read_write> stats: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> weights: array<f32>;

var<workgroup> dom: array<atomic<u32>, {DOM_WORDS}>;
var<workgroup> rules_s: array<u32, {RULE_WORDS}>;
var<workgroup> epoch: array<atomic<u32>, {CELLS}>;
var<workgroup> keys: array<u32, 256>;
var<workgroup> changed: atomic<u32>;
// CELLS - c for the lowest emptied cell c, so zero means none and atomicMax keeps the lowest.
var<workgroup> contra: atomic<u32>;
var<workgroup> ctrl: Ctrl;

fn load_dom(c: u32) -> vec3<u32> {
    return vec3<u32>(atomicLoad(&dom[c * 3u]), atomicLoad(&dom[c * 3u + 1u]), atomicLoad(&dom[c * 3u + 2u]));
}

fn store_dom(c: u32, d: vec3<u32>) {
    atomicStore(&dom[c * 3u], d.x);
    atomicStore(&dom[c * 3u + 1u], d.y);
    atomicStore(&dom[c * 3u + 2u], d.z);
}

fn tile_count(d: vec3<u32>) -> u32 {
    let n = countOneBits(d);
    return n.x + n.y + n.z;
}

fn neighbour(c: u32, axis: u32) -> u32 {
    let x = c % CX;
    let y = (c / CX) % CY;
    let z = c / (CX * CY);
    switch axis {
        case 0u: { if (x + 1u < CX) { return c + 1u; } }
        case 1u: { if (x > 0u) { return c - 1u; } }
        case 2u: { if (y + 1u < CY) { return c + CX; } }
        case 3u: { if (y > 0u) { return c - CX; } }
        case 4u: { if (c + CX * CY < CELLS) { return c + CX * CY; } }
        case 5u: { if (z > 0u) { return c - CX * CY; } }
        default: {}
    }
    return NONE;
}

// Tiles allowed along `axis` of a cell holding any tile of `d`.
fn allowed_by(d: vec3<u32>, axis: u32) -> vec3<u32> {
    var acc = vec3<u32>(0u);
    for (var w = 0u; w < 3u; w++) {
        var bits = d[w];
        while (bits != 0u) {
            let row = (axis * NT + w * 32u + countTrailingZeros(bits)) * 3u;
            bits &= bits - 1u;
            acc |= vec3<u32>(rules_s[row], rules_s[row + 1u], rules_s[row + 2u]);
        }
    }
    return acc;
}

// Jarzynski and Olano's pcg3d: a stateless hash, so a choice depends only on its inputs.
fn pcg3d(v0: vec3<u32>) -> vec3<u32> {
    var v = v0 * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3<u32>(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// A tile of `d` chosen in proportion to its weight, `u` in [0, 1).
fn weighted_tile(d: vec3<u32>, u: f32) -> u32 {
    var total = 0.0;
    for (var w = 0u; w < 3u; w++) {
        var bits = d[w];
        while (bits != 0u) {
            total += weights[w * 32u + countTrailingZeros(bits)];
            bits &= bits - 1u;
        }
    }
    var pick = u * total;
    var chosen = NONE;
    for (var w = 0u; w < 3u; w++) {
        var bits = d[w];
        while (bits != 0u) {
            let tile = w * 32u + countTrailingZeros(bits);
            bits &= bits - 1u;
            if (pick > 0.0 || chosen == NONE) {
                chosen = tile;
                pick -= weights[tile];
            }
        }
    }
    return chosen;
}

// One gather sweep over this lane's cells; also leaves each lane's best selection key.
fn sweep_lane(lane: u32, sweep: u32) {
    var best = NONE;
    for (var c = lane; c < CELLS; c += WG) {
        let d = load_dom(c);
        var nd = d;
        for (var axis = 0u; axis < 6u; axis++) {
            let n = neighbour(c, axis);
            if (n == NONE || atomicLoad(&epoch[n]) + 1u < sweep) {
                continue;
            }
            // The neighbour lies along `axis` from c, so c lies along the opposite axis from it.
            nd &= allowed_by(load_dom(n), axis ^ 1u);
        }
        if (any(nd != d)) {
            store_dom(c, nd);
            atomicStore(&epoch[c], sweep);
            atomicStore(&changed, 1u);
            if (all(nd == vec3<u32>(0u))) {
                atomicMax(&contra, CELLS - c);
            }
        }
        let n = tile_count(nd);
        if (n > 1u) {
            best = min(best, (n << 16u) | c);
        }
    }
    keys[lane] = best;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let chunk = wid.x;
    let base = chunk * CELLS * 3u;
    var st = Ctrl(LOAD, 1u, STATUS_OK, 0u, NONE, 0u, 0u, 0u, 0u);
    loop {
        // Per-lane work for the current state. No barriers in here.
        if (st.phase == LOAD) {
            for (var i = lane * 6u; i < min(lane * 6u + 6u, RULE_WORDS); i++) {
                rules_s[i] = rules[i];
            }
            for (var c = lane; c < CELLS; c += WG) {
                for (var w = 0u; w < 3u; w++) {
                    atomicStore(&dom[c * 3u + w], init[base + c * 3u + w]);
                }
                atomicStore(&epoch[c], st.sweep);
            }
        } else if (st.phase == PROPAGATE) {
            sweep_lane(lane, st.sweep);
        } else if (st.phase == WRITE) {
            for (var c = lane; c < CELLS; c += WG) {
                for (var w = 0u; w < 3u; w++) {
                    out[base + c * 3u + w] = atomicLoad(&dom[c * 3u + w]);
                }
            }
        }
        workgroupBarrier();

        // Lane 0 decides the next state; every other lane is past the barrier and idle.
        if (lane == 0u) {
            var next = st;
            next.steps += 1u;
            if (st.phase == LOAD) {
                next.phase = PROPAGATE;
                next.sweep = st.sweep + 1u;
            } else if (st.phase == PROPAGATE) {
                next.sweeps += 1u;
                let emptied = atomicLoad(&contra);
                let moved = atomicLoad(&changed);
                atomicStore(&contra, 0u);
                atomicStore(&changed, 0u);
                if (emptied != 0u) {
                    next.contra_cell = CELLS - emptied;
                    if (st.step == 0u) {
                        // No choice has been made, so another attempt would fail the same way.
                        next.status = STATUS_BOUNDARY;
                        next.phase = WRITE;
                    } else if (st.restarts + 1u >= params.max_attempts) {
                        next.status = STATUS_FAILED;
                        next.phase = WRITE;
                    } else {
                        next.restarts += 1u;
                        next.step = 0u;
                        next.phase = LOAD;
                        next.sweep = st.sweep + 1u;
                    }
                } else if (moved != 0u) {
                    next.sweep = st.sweep + 1u;
                } else if (params.mode == 0u) {
                    next.phase = WRITE;
                } else {
                    // At a fixpoint the keys from this last sweep describe the current domains.
                    var best = NONE;
                    for (var i = 0u; i < WG; i++) {
                        best = min(best, keys[i]);
                    }
                    if (best == NONE) {
                        next.phase = WRITE;
                    } else {
                        let cell = best & 0xFFFFu;
                        let h = pcg3d(vec3<u32>(params.seed ^ (chunk * 0x9E3779B9u), st.restarts, st.step)).x;
                        let tile = weighted_tile(load_dom(cell), f32(h >> 8u) / 16777216.0);
                        var one = vec3<u32>(0u);
                        one[tile / 32u] = 1u << (tile % 32u);
                        store_dom(cell, one);
                        atomicStore(&epoch[cell], st.sweep);
                        next.sweep = st.sweep + 1u;
                        next.step = st.step + 1u;
                        next.collapses += 1u;
                    }
                }
            } else if (st.phase == WRITE) {
                next.phase = DONE;
                stats[chunk * STATS] = next.status;
                stats[chunk * STATS + 1u] = next.sweeps;
                stats[chunk * STATS + 2u] = next.collapses;
                stats[chunk * STATS + 3u] = next.restarts;
                stats[chunk * STATS + 4u] = next.contra_cell;
                stats[chunk * STATS + 5u] = next.steps;
            }
            // A hung shader resets the host's graphics driver, so every run has a hard step budget.
            if (next.steps >= params.max_steps && next.phase < WRITE) {
                next.status = STATUS_CAP;
                next.phase = WRITE;
            }
            ctrl = next;
        }
        st = workgroupUniformLoad(&ctrl);
        if (st.phase == DONE) {
            break;
        }
    }
}
"#;

struct BenchDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// A device with the workgroup memory the kernel needs.
///
/// `WgpuBackend::new` requests default limits; this mirrors its instance setup (so `WGPU_*`
/// environment variables, which the dozen adapter depends on, still apply) and raises the workgroup
/// storage limit to what the adapter offers.
fn bench_device() -> BenchDevice {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("a GPU adapter");
    let offered = adapter.limits();
    let required_limits = wgpu::Limits {
        max_compute_workgroup_storage_size: offered.max_compute_workgroup_storage_size.min(32_768),
        ..wgpu::Limits::default()
    };
    eprintln!(
        "block_solver: adapter {:?}, workgroup storage {} B",
        adapter.get_info().name,
        required_limits.max_compute_workgroup_storage_size
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("block solver bench"),
        required_features: wgpu::Features::empty(),
        required_limits,
        ..Default::default()
    }))
    .expect("a device");
    BenchDevice { device, queue }
}

fn kernel_source(num_tiles: usize) -> String {
    let rule_words = 6 * num_tiles as u32 * WORDS;
    KERNEL
        .replace("{CX}", &CX.to_string())
        .replace("{CY}", &CY.to_string())
        .replace("{CELLS}", &CELLS.to_string())
        .replace("{DOM_WORDS}", &(CELLS * WORDS).to_string())
        .replace("{NT}", &num_tiles.to_string())
        .replace("{RULE_WORDS}", &rule_words.to_string())
        .replace("{STATS}", &STATS.to_string())
}

/// The rule table in the layout the solver's own shaders use: one `WORDS`-word mask per
/// `(axis, tile)` of the tiles allowed in the neighbour along `axis`.
fn pack_rules(rules: &wfc_rules::AdjacencyRules) -> Vec<u32> {
    let n = rules.num_tiles();
    let words = WORDS as usize;
    let mut table = vec![0u32; 6 * n * words];
    for axis in 0..6 {
        for a in 0..n {
            for b in 0..n {
                if rules.check(a, b, axis) {
                    table[(axis * n + a) * words + b / 32] |= 1 << (b % 32);
                }
            }
        }
    }
    table
}

fn to_words(cells: &[reference::Cell]) -> Vec<u32> {
    cells
        .iter()
        .flat_map(|cell| {
            let low = cell[0];
            let high = cell[1];
            [low as u32, (low >> 32) as u32, high as u32]
        })
        .collect()
}

fn from_words(words: &[u32]) -> Vec<reference::Cell> {
    words
        .as_chunks::<3>()
        .0
        .iter()
        .map(|w| [u64::from(w[0]) | (u64::from(w[1]) << 32), u64::from(w[2])])
        .collect()
}

struct Kernel {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    init: wgpu::Buffer,
    out: wgpu::Buffer,
    stats: wgpu::Buffer,
    params: wgpu::Buffer,
    chunks: u32,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Params {
    /// 0 propagates to a fixpoint and stops; 1 solves the chunk.
    mode: u32,
    max_steps: u32,
    seed: u32,
    max_attempts: u32,
}

impl Kernel {
    fn new(
        gpu: &BenchDevice,
        rules: &wfc_rules::AdjacencyRules,
        weights: &[f32],
        chunks: u32,
    ) -> Self {
        let device = &gpu.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("block solver"),
            source: wgpu::ShaderSource::Wgsl(kernel_source(rules.num_tiles()).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("block solver"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let storage = |label: &str, size: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
                mapped_at_creation: false,
            })
        };
        let rule_table = pack_rules(rules);
        let rules_buf = storage(
            "rules",
            (rule_table.len() * 4) as u64,
            wgpu::BufferUsages::empty(),
        );
        gpu.queue
            .write_buffer(&rules_buf, 0, bytemuck::cast_slice(&rule_table));
        let weights_buf = storage(
            "weights",
            std::mem::size_of_val(weights) as u64,
            wgpu::BufferUsages::empty(),
        );
        gpu.queue
            .write_buffer(&weights_buf, 0, bytemuck::cast_slice(weights));
        let domain_bytes = u64::from(chunks * CELLS * WORDS * 4);
        let init = storage("init", domain_bytes, wgpu::BufferUsages::empty());
        let out = storage("out", domain_bytes, wgpu::BufferUsages::COPY_SRC);
        let stats = storage(
            "stats",
            u64::from(chunks * STATS * 4),
            wgpu::BufferUsages::COPY_SRC,
        );
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("block solver"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: rules_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: init.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: stats.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: weights_buf.as_entire_binding(),
                },
            ],
        });
        Self {
            pipeline,
            bind_group,
            init,
            out,
            stats,
            params,
            chunks,
        }
    }

    /// Runs one dispatch over every chunk and waits for it.
    fn run(&self, gpu: &BenchDevice, params: Params) {
        gpu.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.chunks, 1, 1);
        }
        gpu.queue.submit(Some(encoder.finish()));
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the dispatch completes");
    }
}

/// Copies `buffer` to the CPU.
fn read_u32s(gpu: &BenchDevice, buffer: &wgpu::Buffer) -> Vec<u32> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| result.expect("readback maps"));
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the readback completes");
    let words = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
    staging.unmap();
    words
}

/// Propagation alone, on one chunk, must reach exactly the CPU reference fixpoint.
#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn one_chunk_propagates_to_the_reference_fixpoint() {
    let city = city::city();
    let m = &city.modules;
    let (w, h, d) = (CX as usize, CY as usize, CZ as usize);
    let initial = reference::city_initial_cells(&city, w, h, d);
    let solver = ReferenceSolver::new(&m.rules, &m.tileset.weights, w, h, d);
    let mut expected = initial.clone();
    let mut stack: Vec<usize> = (0..expected.len()).collect();
    solver
        .propagate(&mut expected, &mut stack)
        .expect("the city chunk is consistent");
    // A kernel that did nothing would pass if propagation had nothing to do.
    let narrowed = initial
        .iter()
        .zip(&expected)
        .filter(|(i, e)| i != e)
        .count();
    assert!(
        narrowed > 0,
        "the reference fixpoint differs from the initial domains"
    );
    eprintln!(
        "block_solver: propagation narrows {narrowed} of {} cells",
        initial.len()
    );
    let gpu = bench_device();
    let kernel = Kernel::new(&gpu, &m.rules, &m.tileset.weights, 1);
    gpu.queue
        .write_buffer(&kernel.init, 0, bytemuck::cast_slice(&to_words(&initial)));

    kernel.run(
        &gpu,
        Params {
            mode: 0,
            max_steps: 10_000,
            seed: 1,
            max_attempts: 1,
        },
    );

    let stats = read_u32s(&gpu, &kernel.stats);
    eprintln!("block_solver: propagate-only stats {stats:?}");
    assert_eq!(stats[0], 0, "status OK");
    let actual = from_words(&read_u32s(&gpu, &kernel.out));
    let differing = actual.iter().zip(&expected).filter(|(a, e)| a != e).count();
    assert_eq!(
        differing, 0,
        "cells whose domain differs from the reference fixpoint"
    );
}

/// A chunk's tiles, after checking every cell was decided within its initial domain.
fn decided_tiles(domains: &[reference::Cell], initial: &[reference::Cell]) -> Vec<usize> {
    domains
        .iter()
        .zip(initial)
        .map(|(&cell, &start)| {
            assert_eq!(reference::count(cell), 1, "every cell is decided");
            assert_eq!(
                [cell[0] & !start[0], cell[1] & !start[1]],
                [0, 0],
                "a tile outside the initial domain"
            );
            reference::set_bits(cell).next().expect("one tile")
        })
        .collect()
}

/// Solving one chunk inside one dispatch yields a valid, reproducible chunk.
#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn one_chunk_solves_validly_and_reproducibly() {
    let city = city::city();
    let m = &city.modules;
    let (w, h, d) = (CX as usize, CY as usize, CZ as usize);
    let initial = reference::city_initial_cells(&city, w, h, d);
    let gpu = bench_device();
    let kernel = Kernel::new(&gpu, &m.rules, &m.tileset.weights, 1);
    gpu.queue
        .write_buffer(&kernel.init, 0, bytemuck::cast_slice(&to_words(&initial)));
    let solve = |seed: u32| {
        kernel.run(
            &gpu,
            Params {
                mode: 1,
                max_steps: 50_000,
                seed,
                max_attempts: 64,
            },
        );
        let stats = read_u32s(&gpu, &kernel.stats);
        eprintln!(
            "block_solver: seed={seed} status={} sweeps={} collapses={} restarts={} last_contradiction={} steps={}",
            stats[0], stats[1], stats[2], stats[3], stats[4] as i32, stats[5]
        );
        assert_eq!(stats[0], 0, "status OK");
        from_words(&read_u32s(&gpu, &kernel.out))
    };

    let first = solve(1);
    let again = solve(1);
    let other = solve(2);

    let tiles = decided_tiles(&first, &initial);
    let grid = wfc_devtools::TileGrid::new(w, h, d, tiles).expect("dimensions match");
    let violations =
        wfc_devtools::adjacency_violations(&grid, &m.rules, wfc_core::BoundaryCondition::Finite);
    assert!(
        violations.is_empty(),
        "{} adjacency violations, first {:?}",
        violations.len(),
        violations.first()
    );
    assert!(first == again, "the same seed reproduces the chunk");
    assert!(first != other, "another seed gives another chunk");
}

/// Median of `samples`, which must not be empty.
fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// How chunk throughput scales with the number of chunks in one dispatch, next to one CPU thread
/// running the reference solver on the same chunk in the same build.
#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn chunk_throughput_against_the_cpu_reference() {
    let city = city::city();
    let m = &city.modules;
    let (w, h, d) = (CX as usize, CY as usize, CZ as usize);
    let initial = reference::city_initial_cells(&city, w, h, d);

    // The CPU yardstick: sixteen seeds of the same chunk, one thread, after one warm-up solve.
    let solver = ReferenceSolver::new(&m.rules, &m.tileset.weights, w, h, d);
    solver.solve(initial.clone(), 0);
    let cpu_ms = median(
        (1..=16)
            .map(|seed| solver.solve(initial.clone(), seed).seconds * 1000.0)
            .collect(),
    );
    eprintln!(
        "block_solver: cpu reference {cpu_ms:.3} ms per {w}x{h}x{d} chunk (median of 16 seeds, one thread)"
    );

    let gpu = bench_device();
    let params = Params {
        mode: 1,
        max_steps: 50_000,
        seed: 7,
        max_attempts: 64,
    };
    for chunks in [1u32, 16, 64, 256] {
        let kernel = Kernel::new(&gpu, &m.rules, &m.tileset.weights, chunks);
        let words: Vec<u32> = (0..chunks).flat_map(|_| to_words(&initial)).collect();
        gpu.queue
            .write_buffer(&kernel.init, 0, bytemuck::cast_slice(&words));
        // Pipeline creation and the first dispatches run cold; the GPU also raises its clocks under
        // load, so only warm samples are compared.
        for _ in 0..3 {
            kernel.run(&gpu, params);
        }
        let mut samples = Vec::new();
        for _ in 0..5 {
            let started = std::time::Instant::now();
            kernel.run(&gpu, params);
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        let wall_ms = median(samples);

        let stats = read_u32s(&gpu, &kernel.stats);
        let records: Vec<&[u32]> = stats.chunks(STATS as usize).collect();
        let domains = from_words(&read_u32s(&gpu, &kernel.out));
        let mut failed = 0;
        for (chunk, record) in records.iter().enumerate() {
            if record[0] != 0 {
                failed += 1;
                continue;
            }
            let cells = &domains[chunk * CELLS as usize..(chunk + 1) * CELLS as usize];
            let grid = wfc_devtools::TileGrid::new(w, h, d, decided_tiles(cells, &initial))
                .expect("dimensions match");
            let violations = wfc_devtools::adjacency_violations(
                &grid,
                &m.rules,
                wfc_core::BoundaryCondition::Finite,
            );
            assert!(
                violations.is_empty(),
                "chunk {chunk}: {} violations",
                violations.len()
            );
        }
        let collapses: u32 = records.iter().map(|r| r[2]).sum();
        let sweeps: u32 = records.iter().map(|r| r[1]).sum();
        let mut restarts: Vec<u32> = records.iter().map(|r| r[3]).collect();
        restarts.sort_unstable();
        let cells = f64::from(chunks * CELLS);
        eprintln!(
            "block_solver: chunks={chunks} wall_ms={wall_ms:.2} ms_per_chunk={:.3} cells_per_s={:.0} \
             vs_cpu_thread={:.2}x failed={failed} collapses={collapses} sweeps_per_collapse={:.2} \
             us_per_collapse={:.1} restarts[min,median,max]=[{},{},{}]",
            wall_ms / f64::from(chunks),
            cells / (wall_ms / 1000.0),
            cpu_ms * f64::from(chunks) / wall_ms,
            f64::from(sweeps) / f64::from(collapses.max(1)),
            wall_ms * 1000.0 / f64::from(collapses.max(1)),
            restarts[0],
            restarts[restarts.len() / 2],
            restarts[restarts.len() - 1],
        );
    }
}

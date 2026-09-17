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
/// The chunk shape of the single-chunk and throughput tests.
const CHUNK: [u32; 3] = [CX, CY, CZ];
/// Checkpoints kept per chunk for undo.
const RING: u32 = 32;
/// Invocations per workgroup in the correctness tests. Each lane owns every `INVOCATIONS`-th cell.
const INVOCATIONS: u32 = 256;

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
/// keys 1024 B, per-cell keys 2048 B, control 44 B: 17140 B, over the 16384 B WebGPU default, so
/// the bench requests the adapter's limit (32768 B through dozen).
const KERNEL: &str = r#"
const CX: u32 = {CX}u;
const CY: u32 = {CY}u;
const CELLS: u32 = {CELLS}u;
const CZ: u32 = CELLS / (CX * CY);
const NT: u32 = {NT}u;
const WG: u32 = {WG}u;
const RULE_WORDS: u32 = {RULE_WORDS}u;
const STATS: u32 = {STATS}u;
const NONE: u32 = 0xFFFFFFFFu;

const LOAD: u32 = 0u;
const PROPAGATE: u32 = 1u;
const WRITE: u32 = 2u;
const DONE: u32 = 3u;
const SELECT: u32 = 4u;
const RESTORE: u32 = 5u;
const RING: u32 = {RING}u;

const STATUS_OK: u32 = 0u;
const STATUS_FAILED: u32 = 1u;
const STATUS_CAP: u32 = 2u;
const STATUS_BOUNDARY: u32 = 3u;
// A checkpoint held an empty cell: a bug in the checkpoint ring, never a property of the rules.
const STATUS_BAD_CHECKPOINT: u32 = 4u;

struct Params {
    mode: u32,
    max_steps: u32,
    seed: u32,
    max_attempts: u32,
    // 0 collapses the single global minimum per round; r > 0 collapses every local minimum within
    // Chebyshev radius r at once.
    radius: u32,
    // 0 restarts the chunk on a contradiction; 1 restores the checkpoint before the failing round.
    undo: u32,
    pad0: u32,
    pad1: u32,
};

struct Ctrl {
    phase: u32,
    sweep: u32,
    status: u32,
    steps: u32,
    contra_cell: u32,
    sweeps: u32,
    collapses: u32,
    restarts: u32,
    // Rounds completed in the current attempt; zero means a contradiction came from the borders.
    step: u32,
    // Contradictions so far. It salts the choice hash, so a retried round chooses differently.
    tries: u32,
    // Rounds undone by the last backtrack, and the round count at that failure: failing again at
    // or before that round doubles the undo.
    undo: u32,
    fail_step: u32,
    backtracks: u32,
    // The most rounds this attempt has reached. Slot k is rewritten by round k + RING, so it is only
    // trustworthy while k + RING exceeds this.
    max_step: u32,
};

@group(0) @binding(0) var<storage, read> rules: array<u32>;
@group(0) @binding(1) var<storage, read> init: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<storage, read_write> stats: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> weights: array<f32>;
// A ring of RING checkpoints per chunk: slot k holds the fixpoint reached before round k.
@group(0) @binding(6) var<storage, read_write> snaps: array<u32>;

var<workgroup> dom: array<atomic<u32>, {DOM_WORDS}>;
var<workgroup> rules_s: array<u32, {RULE_WORDS}>;
var<workgroup> epoch: array<atomic<u32>, {CELLS}>;
var<workgroup> keys: array<u32, {WG}>;
// Each cell's (count << 16 | index) key from the latest sweep, NONE once decided.
var<workgroup> cell_keys: array<u32, {CELLS}>;
var<workgroup> chosen: atomic<u32>;
var<workgroup> restored_empty: atomic<u32>;
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

fn snap_index(chunk: u32, slot: u32, c: u32) -> u32 {
    return ((chunk * RING + slot) * CELLS + c) * 3u;
}

fn cell_key(c: u32, d: vec3<u32>) -> u32 {
    let n = tile_count(d);
    if (n > 1u) {
        return (n << 16u) | c;
    }
    return NONE;
}

// One gather sweep over this lane's cells; also leaves each lane's best selection key and, when
// undo is on, this lane's part of the checkpoint for the coming round. Only the last sweep before a
// round changes nothing, so the copy that survives is the fixpoint.
fn sweep_lane(lane: u32, chunk: u32, st: Ctrl) {
    let sweep = st.sweep;
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
        let key = cell_key(c, nd);
        best = min(best, key);
        cell_keys[c] = key;
        if (params.undo != 0u) {
            let i = snap_index(chunk, st.step % RING, c);
            snaps[i] = nd.x;
            snaps[i + 1u] = nd.y;
            snaps[i + 2u] = nd.z;
        }
    }
    keys[lane] = best;
}

// Puts back this lane's cells from the checkpoint before round `st.step`. It is a fixpoint, so no
// cell needs another sweep.
fn restore_lane(lane: u32, chunk: u32, st: Ctrl) {
    var best = NONE;
    for (var c = lane; c < CELLS; c += WG) {
        let i = snap_index(chunk, st.step % RING, c);
        let d = vec3<u32>(snaps[i], snaps[i + 1u], snaps[i + 2u]);
        store_dom(c, d);
        atomicStore(&epoch[c], 0u);
        if (all(d == vec3<u32>(0u))) {
            atomicStore(&restored_empty, 1u);
        }
        let key = cell_key(c, d);
        best = min(best, key);
        cell_keys[c] = key;
    }
    keys[lane] = best;
}

// Lane 0 only: collapses the fewest-possibilities cell. False when every cell is decided.
fn collapse_global_minimum(chunk: u32, st: Ctrl) -> bool {
    var best = NONE;
    for (var i = 0u; i < WG; i++) {
        best = min(best, keys[i]);
    }
    if (best == NONE) {
        return false;
    }
    let cell = best & 0xFFFFu;
    let h = pcg3d(vec3<u32>(params.seed ^ (chunk * 0x9E3779B9u), st.tries, st.step)).x;
    let tile = weighted_tile(load_dom(cell), f32(h >> 8u) / 16777216.0);
    var one = vec3<u32>(0u);
    one[tile / 32u] = 1u << (tile % 32u);
    store_dom(cell, one);
    atomicStore(&epoch[cell], st.sweep);
    return true;
}

// Whether no undecided cell within Chebyshev `radius` of `c` has a smaller key. Two such local
// minima are always more than `radius` apart, and the global minimum is always one.
fn is_local_minimum(c: u32, radius: i32) -> bool {
    let key = cell_keys[c];
    let x = i32(c % CX);
    let y = i32((c / CX) % CY);
    let z = i32(c / (CX * CY));
    for (var dz = max(-radius, -z); dz <= min(radius, i32(CZ) - 1 - z); dz++) {
        for (var dy = max(-radius, -y); dy <= min(radius, i32(CY) - 1 - y); dy++) {
            for (var dx = max(-radius, -x); dx <= min(radius, i32(CX) - 1 - x); dx++) {
                let n = u32((z + dz) * i32(CX * CY) + (y + dy) * i32(CX) + x + dx);
                if (cell_keys[n] < key) {
                    return false;
                }
            }
        }
    }
    return true;
}

// Collapses every local minimum this lane owns, each with its own hashed choice.
fn select_lane(lane: u32, chunk: u32, st: Ctrl) {
    for (var c = lane; c < CELLS; c += WG) {
        if (cell_keys[c] == NONE || !is_local_minimum(c, i32(params.radius))) {
            continue;
        }
        let h = pcg3d(vec3<u32>(params.seed ^ (chunk * 0x9E3779B9u), st.tries, st.step * CELLS + c)).x;
        let tile = weighted_tile(load_dom(c), f32(h >> 8u) / 16777216.0);
        var one = vec3<u32>(0u);
        one[tile / 32u] = 1u << (tile % 32u);
        store_dom(c, one);
        atomicStore(&epoch[c], st.sweep);
        atomicAdd(&chosen, 1u);
    }
}

@compute @workgroup_size({WG})
fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let chunk = wid.x;
    let base = chunk * CELLS * 3u;
    var st = Ctrl(LOAD, 1u, STATUS_OK, 0u, NONE, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    loop {
        // Per-lane work for the current state. No barriers in here.
        if (st.phase == LOAD) {
            let per_lane = (RULE_WORDS + WG - 1u) / WG;
            for (var i = lane * per_lane; i < min((lane + 1u) * per_lane, RULE_WORDS); i++) {
                rules_s[i] = rules[i];
            }
            for (var c = lane; c < CELLS; c += WG) {
                for (var w = 0u; w < 3u; w++) {
                    atomicStore(&dom[c * 3u + w], init[base + c * 3u + w]);
                }
                atomicStore(&epoch[c], st.sweep);
            }
        } else if (st.phase == PROPAGATE) {
            sweep_lane(lane, chunk, st);
        } else if (st.phase == SELECT) {
            select_lane(lane, chunk, st);
        } else if (st.phase == RESTORE) {
            restore_lane(lane, chunk, st);
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
                    next.tries = st.tries + 1u;
                    // Failing again without getting past the last failure means the cause lies
                    // further back, so undo twice as far; getting past it starts again from one.
                    let again = st.step <= st.fail_step;
                    var undo = 1u;
                    if (again) {
                        undo = min(st.undo * 2u, RING - 1u);
                    }
                    if (st.step == 0u) {
                        // No choice has been made, so another attempt would fail the same way.
                        next.status = STATUS_BOUNDARY;
                        next.phase = WRITE;
                    } else if (params.undo != 0u && undo <= st.step && st.step - undo + RING > st.max_step
                        && !(again && st.undo >= RING - 1u)) {
                        next.backtracks += 1u;
                        next.undo = undo;
                        next.fail_step = st.step;
                        next.step = st.step - undo;
                        next.phase = RESTORE;
                        next.sweep = st.sweep + 1u;
                    } else if (st.restarts + 1u >= params.max_attempts) {
                        next.status = STATUS_FAILED;
                        next.phase = WRITE;
                    } else {
                        next.restarts += 1u;
                        next.step = 0u;
                        next.max_step = 0u;
                        next.undo = 0u;
                        next.fail_step = 0u;
                        next.phase = LOAD;
                        next.sweep = st.sweep + 1u;
                    }
                } else if (moved != 0u) {
                    next.sweep = st.sweep + 1u;
                } else if (params.mode == 0u) {
                    next.phase = WRITE;
                } else if (params.radius > 0u) {
                    next.phase = SELECT;
                } else if (collapse_global_minimum(chunk, st)) {
                    // At a fixpoint the keys from this last sweep describe the current domains.
                    next.sweep = st.sweep + 1u;
                    next.step = st.step + 1u;
                    next.max_step = max(st.max_step, st.step + 1u);
                    next.collapses += 1u;
                } else {
                    next.phase = WRITE;
                }
            } else if (st.phase == RESTORE) {
                if (atomicLoad(&restored_empty) != 0u) {
                    next.status = STATUS_BAD_CHECKPOINT;
                    next.phase = WRITE;
                } else if (params.radius > 0u) {
                    next.phase = SELECT;
                } else if (collapse_global_minimum(chunk, st)) {
                    next.phase = PROPAGATE;
                    next.sweep = st.sweep + 1u;
                    next.step = st.step + 1u;
                    next.max_step = max(st.max_step, st.step + 1u);
                    next.collapses += 1u;
                } else {
                    next.phase = WRITE;
                }
            } else if (st.phase == SELECT) {
                let n = atomicLoad(&chosen);
                atomicStore(&chosen, 0u);
                if (n == 0u) {
                    // No undecided cell was left to be a minimum.
                    next.phase = WRITE;
                } else {
                    next.phase = PROPAGATE;
                    next.sweep = st.sweep + 1u;
                    next.step = st.step + 1u;
                    next.max_step = max(st.max_step, st.step + 1u);
                    next.collapses += n;
                }
            } else if (st.phase == WRITE) {
                next.phase = DONE;
                stats[chunk * STATS] = next.status;
                stats[chunk * STATS + 1u] = next.sweeps;
                stats[chunk * STATS + 2u] = next.collapses;
                stats[chunk * STATS + 3u] = next.restarts;
                stats[chunk * STATS + 4u] = next.contra_cell;
                stats[chunk * STATS + 5u] = next.steps;
                stats[chunk * STATS + 6u] = next.backtracks;
                stats[chunk * STATS + 7u] = next.tries;
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

fn kernel_source(num_tiles: usize, invocations: u32, shape: [u32; 3]) -> String {
    let cells = shape[0] * shape[1] * shape[2];
    let rule_words = 6 * num_tiles as u32 * WORDS;
    KERNEL
        .replace("{CX}", &shape[0].to_string())
        .replace("{CY}", &shape[1].to_string())
        .replace("{CELLS}", &cells.to_string())
        .replace("{DOM_WORDS}", &(cells * WORDS).to_string())
        .replace("{NT}", &num_tiles.to_string())
        .replace("{RULE_WORDS}", &rule_words.to_string())
        .replace("{STATS}", &STATS.to_string())
        .replace("{WG}", &invocations.to_string())
        .replace("{RING}", &RING.to_string())
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
    /// 0 collapses one cell per round; `r` collapses every local minimum within radius `r`.
    radius: u32,
    /// 0 restarts a chunk on a contradiction; 1 restores the checkpoint before the failing round.
    undo: u32,
    padding: [u32; 2],
}

impl Kernel {
    fn new(
        gpu: &BenchDevice,
        rules: &wfc_rules::AdjacencyRules,
        weights: &[f32],
        chunks: u32,
        invocations: u32,
        shape: [u32; 3],
    ) -> Self {
        let cells = shape[0] * shape[1] * shape[2];
        let device = &gpu.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("block solver"),
            source: wgpu::ShaderSource::Wgsl(
                kernel_source(rules.num_tiles(), invocations, shape).into(),
            ),
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
        let domain_bytes = u64::from(chunks * cells * WORDS * 4);
        let init = storage("init", domain_bytes, wgpu::BufferUsages::empty());
        let out = storage("out", domain_bytes, wgpu::BufferUsages::COPY_SRC);
        let snaps = storage(
            "snaps",
            u64::from(chunks * RING * cells * WORDS * 4),
            wgpu::BufferUsages::empty(),
        );
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
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: snaps.as_entire_binding(),
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
    let kernel = Kernel::new(&gpu, &m.rules, &m.tileset.weights, 1, INVOCATIONS, CHUNK);
    gpu.queue
        .write_buffer(&kernel.init, 0, bytemuck::cast_slice(&to_words(&initial)));

    kernel.run(
        &gpu,
        Params {
            mode: 0,
            max_steps: 10_000,
            seed: 1,
            max_attempts: 1,
            radius: 0,
            undo: 0,
            padding: [0; 2],
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
    let kernel = Kernel::new(&gpu, &m.rules, &m.tileset.weights, 1, INVOCATIONS, CHUNK);
    gpu.queue
        .write_buffer(&kernel.init, 0, bytemuck::cast_slice(&to_words(&initial)));
    let solve = |seed: u32, radius: u32, undo: u32| {
        kernel.run(
            &gpu,
            Params {
                mode: 1,
                max_steps: 50_000,
                seed,
                max_attempts: 64,
                radius,
                undo,
                padding: [0; 2],
            },
        );
        let stats = read_u32s(&gpu, &kernel.stats);
        eprintln!(
            "block_solver: radius={radius} undo={undo} seed={seed} status={} sweeps={} collapses={} restarts={} last_contradiction={} steps={} backtracks={} tries={}",
            stats[0], stats[1], stats[2], stats[3], stats[4] as i32, stats[5], stats[6], stats[7]
        );
        assert_eq!(stats[0], 0, "status OK");
        from_words(&read_u32s(&gpu, &kernel.out))
    };

    // One cell per round or every local minimum within radius 2, recovering by restart or by undo.
    for (radius, undo) in [(0, 0), (2, 0), (0, 1), (2, 1)] {
        let first = solve(1, radius, undo);
        let again = solve(1, radius, undo);
        let other = solve(2, radius, undo);

        let tiles = decided_tiles(&first, &initial);
        let grid = wfc_devtools::TileGrid::new(w, h, d, tiles).expect("dimensions match");
        let violations = wfc_devtools::adjacency_violations(
            &grid,
            &m.rules,
            wfc_core::BoundaryCondition::Finite,
        );
        assert!(
            violations.is_empty(),
            "radius {radius}, undo {undo}: {} adjacency violations, first {:?}",
            violations.len(),
            violations.first()
        );
        assert!(
            first == again,
            "radius {radius}, undo {undo}: the same seed reproduces the chunk"
        );
        assert!(
            first != other,
            "radius {radius}, undo {undo}: another seed gives another chunk"
        );
    }
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
    // The same 256 chunks spread over every hardware thread, which is what a GPU dispatch of 256
    // chunks actually competes with. The reference's naive undo occasionally thrashes to its
    // backtrack cap; those runs are counted and their time is included, as a GPU chunk's would be.
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let mut thrashed = 0;
    let all_cores_ms = median(
        (0..3)
            .map(|_| {
                let started = std::time::Instant::now();
                thrashed = std::thread::scope(|scope| {
                    let workers: Vec<_> = (0..threads)
                        .map(|thread| {
                            let (solver, initial) = (&solver, &initial);
                            scope.spawn(move || {
                                (0..256u64)
                                    .filter(|seed| *seed as usize % threads == thread)
                                    .filter(|&seed| solver.solve(initial.clone(), seed).thrashed)
                                    .count()
                            })
                        })
                        .collect();
                    workers
                        .into_iter()
                        .map(|worker| worker.join().expect("a solver thread"))
                        .sum::<usize>()
                });
                started.elapsed().as_secs_f64() * 1000.0
            })
            .collect(),
    );
    eprintln!(
        "block_solver: cpu reference on {threads} threads: 256 chunks in {all_cores_ms:.1} ms, {:.3} ms per chunk, {thrashed} thrashed (median of 3)",
        all_cores_ms / 256.0
    );

    let gpu = bench_device();
    // Invocations per workgroup trade parallelism inside a sweep against the cost of synchronising
    // every step; the radius trades collapses per round against choices made blind to each other.
    for (invocations, radius, undo) in [
        (1u32, 0u32, 0u32),
        (4, 0, 0),
        (16, 0, 0),
        (64, 0, 0),
        (256, 0, 0),
        (256, 1, 0),
        (256, 2, 0),
        (256, 3, 0),
        (256, 0, 1),
        (256, 1, 1),
        (256, 2, 1),
        (256, 3, 1),
    ] {
        let params = Params {
            mode: 1,
            max_steps: 50_000,
            seed: 7,
            max_attempts: 64,
            radius,
            undo,
            padding: [0; 2],
        };
        // Windows resets the device when one dispatch runs for about two seconds, and a reset here
        // takes the host's display driver with it. Chunk counts grow by 4, so a dispatch is only
        // attempted while four times the previous one stays well inside that limit.
        let mut previous_ms = 0.0;
        for chunks in [1u32, 4, 16, 64, 256] {
            if previous_ms * 4.0 > 600.0 {
                eprintln!(
                    "block_solver: invocations={invocations} radius={radius} undo={undo} chunks={chunks} skipped: {previous_ms:.0} ms at a quarter of the chunks risks the device timeout"
                );
                break;
            }
            let kernel = Kernel::new(
                &gpu,
                &m.rules,
                &m.tileset.weights,
                chunks,
                invocations,
                CHUNK,
            );
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
            previous_ms = wall_ms;

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
            let mut backtracks: Vec<u32> = records.iter().map(|r| r[6]).collect();
            backtracks.sort_unstable();
            // A dispatch finishes when its slowest workgroup does, so if chunks really run in
            // parallel the slowest chunk's step count, not the total, predicts wall time.
            let max_steps = records.iter().map(|r| r[5]).max().expect("a chunk");
            let mean_steps =
                records.iter().map(|r| f64::from(r[5])).sum::<f64>() / records.len() as f64;
            let cells = f64::from(chunks * CELLS);
            eprintln!(
                "block_solver: invocations={invocations} radius={radius} undo={undo} chunks={chunks} wall_ms={wall_ms:.2} ms_per_chunk={:.3} cells_per_s={:.0} \
             vs_cpu_thread={:.2}x failed={failed} collapses={collapses} sweeps_per_collapse={:.2} \
             us_per_collapse={:.1} restarts[min,median,max]=[{},{},{}] \
             backtracks[median,max]=[{},{}] max_steps={max_steps} mean_steps={mean_steps:.0} us_per_step_of_slowest={:.1}",
                wall_ms / f64::from(chunks),
                cells / (wall_ms / 1000.0),
                cpu_ms * f64::from(chunks) / wall_ms,
                f64::from(sweeps) / f64::from(collapses.max(1)),
                wall_ms * 1000.0 / f64::from(collapses.max(1)),
                restarts[0],
                restarts[restarts.len() / 2],
                restarts[restarts.len() - 1],
                backtracks[backtracks.len() / 2],
                backtracks[backtracks.len() - 1],
                wall_ms * 1000.0 / f64::from(max_steps),
            );
        }
    }
}

/// Every chunk of a many-chunk dispatch that reports success is valid, in every selection and
/// recovery mode. One chunk can hide a rare failure; 64 different random streams rarely do.
#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn every_reported_success_is_a_valid_chunk() {
    let city = city::city();
    let m = &city.modules;
    let (w, h, d) = (CX as usize, CY as usize, CZ as usize);
    let initial = reference::city_initial_cells(&city, w, h, d);
    let chunks = 64u32;
    let gpu = bench_device();
    let kernel = Kernel::new(
        &gpu,
        &m.rules,
        &m.tileset.weights,
        chunks,
        INVOCATIONS,
        CHUNK,
    );
    let words: Vec<u32> = (0..chunks).flat_map(|_| to_words(&initial)).collect();
    gpu.queue
        .write_buffer(&kernel.init, 0, bytemuck::cast_slice(&words));

    for (seed, radius, undo) in [7, 11, 13]
        .into_iter()
        .flat_map(|seed| [(0, 0), (2, 0), (0, 1), (2, 1)].map(|(r, u)| (seed, r, u)))
    {
        kernel.run(
            &gpu,
            Params {
                mode: 1,
                max_steps: 50_000,
                seed,
                max_attempts: 64,
                radius,
                undo,
                padding: [0; 2],
            },
        );

        let stats = read_u32s(&gpu, &kernel.stats);
        let domains = from_words(&read_u32s(&gpu, &kernel.out));
        let mut invalid = Vec::new();
        for chunk in 0..chunks as usize {
            let record = &stats[chunk * STATS as usize..(chunk + 1) * STATS as usize];
            assert_ne!(
                record[0], 4,
                "chunk {chunk} restored an empty checkpoint: {record:?}"
            );
            if record[0] != 0 {
                continue;
            }
            let cells = &domains[chunk * CELLS as usize..(chunk + 1) * CELLS as usize];
            let empty = cells.iter().filter(|&&c| reference::count(c) == 0).count();
            let open = cells.iter().filter(|&&c| reference::count(c) > 1).count();
            if empty + open > 0 {
                invalid.push((chunk, empty, open, record.to_vec()));
            }
        }
        eprintln!(
            "block_solver: seed={seed} radius={radius} undo={undo} chunks reporting success but invalid: {} \
             (chunk, empty cells, undecided cells, [status, sweeps, collapses, restarts, \
             contradiction, steps, backtracks, tries]) first: {:?}",
            invalid.len(),
            invalid.first()
        );
        assert!(
            invalid.is_empty(),
            "seed {seed}, radius {radius}, undo {undo}"
        );
    }
}

/// The tiles a cell may hold when its neighbour along `axis` holds `tile`, per the rules.
fn allowed_next_to(rules: &wfc_rules::AdjacencyRules, tile: usize, axis: usize) -> reference::Cell {
    let mut cell = [0u64; reference::WORDS];
    for other in 0..rules.num_tiles() {
        if rules.check(tile, other, axis) {
            cell[other / 64] |= 1 << (other % 64);
        }
    }
    cell
}

/// What stitching a world from chunks produced.
struct Stitched {
    undecided_cells: usize,
    violations: usize,
    border_contradictions: usize,
    repaired: usize,
}

/// An 8x8-chunk city world assembled from block-kernel solves.
struct World<'a> {
    city: &'a city::City,
    gpu: &'a BenchDevice,
    chunks_x: usize,
    width: usize,
    height: usize,
    /// Every cell's domain before anything is decided: street level at the bottom, air on top, and
    /// no path leaving the world's outer faces. Faces between chunks are left to the solver.
    initial: Vec<reference::Cell>,
    /// The layering alone, for halo cells beyond the world's edge: the middle column of a grid too
    /// small to have an interior rim.
    open_column: Vec<reference::Cell>,
    tiles: Vec<Option<usize>>,
}

impl<'a> World<'a> {
    fn new(city: &'a city::City, gpu: &'a BenchDevice) -> Self {
        let (chunks_x, chunks_y) = (8, 8);
        let (width, height) = (chunks_x * CX as usize, chunks_y * CY as usize);
        let grid = reference::city_initial_cells(city, 3, 3, CZ as usize);
        Self {
            city,
            gpu,
            chunks_x,
            width,
            height,
            initial: reference::city_initial_cells(city, width, height, CZ as usize),
            open_column: (0..CZ as usize)
                .map(|z| grid[(z * 3 + 1) * 3 + 1])
                .collect(),
            tiles: vec![None; width * height * CZ as usize],
        }
    }

    fn index(&self, (x, y, z): (isize, isize, isize)) -> Option<usize> {
        let inside = x >= 0
            && y >= 0
            && z >= 0
            && (x as usize) < self.width
            && (y as usize) < self.height
            && (z as usize) < CZ as usize;
        inside.then(|| (z as usize * self.height + y as usize) * self.width + x as usize)
    }

    /// World coordinates of chunk `k` widened by `halo`, in row-major order, and whether each cell
    /// belongs to the chunk itself.
    fn region(&self, k: usize, halo: usize) -> Vec<((isize, isize, isize), bool)> {
        let (cx, cy) = (CX as usize, CY as usize);
        let (ox, oy) = (
            ((k % self.chunks_x) * cx) as isize,
            ((k / self.chunks_x) * cy) as isize,
        );
        let h = halo as isize;
        (0..CZ as isize)
            .flat_map(|z| {
                (0..(cy + 2 * halo) as isize).flat_map(move |j| {
                    (0..(cx + 2 * halo) as isize).map(move |i| {
                        let inner = i >= h && i < h + cx as isize && j >= h && j < h + cy as isize;
                        ((ox + i - h, oy + j - h, z), inner)
                    })
                })
            })
            .collect()
    }

    /// Starting domains for a region. Decided cells inside it are pinned to their tiles, unless
    /// `release` frees them to be solved again; decided cells just outside restrict their
    /// neighbours inside.
    fn region_init(&self, k: usize, halo: usize, release: bool) -> Vec<reference::Cell> {
        let region = self.region(k, halo);
        let members: std::collections::HashSet<(isize, isize, isize)> =
            region.iter().map(|(at, _)| *at).collect();
        region
            .iter()
            .map(|&(at, _)| {
                let Some(i) = self.index(at) else {
                    return self.open_column[at.2 as usize];
                };
                if let (Some(tile), false) = (self.tiles[i], release) {
                    let mut pinned = [0u64; reference::WORDS];
                    pinned[tile / 64] = 1 << (tile % 64);
                    return pinned;
                }
                let mut cell = self.initial[i];
                for (axis, (dx, dy, dz)) in
                    wfc_devtools::invariants::AXIS_OFFSETS.iter().enumerate()
                {
                    let next = (at.0 + dx, at.1 + dy, at.2 + dz);
                    if members.contains(&next) {
                        continue;
                    }
                    if let Some(tile) = self.index(next).and_then(|n| self.tiles[n]) {
                        // The neighbour lies along `axis` from this cell, so this cell lies along
                        // the opposite axis from the neighbour.
                        let allowed = allowed_next_to(&self.city.modules.rules, tile, axis ^ 1);
                        cell = [cell[0] & allowed[0], cell[1] & allowed[1]];
                    }
                }
                cell
            })
            .collect()
    }

    /// Solves the regions of `chunks` in one dispatch and commits the successes: every chunk cell,
    /// plus, when `release` is set, the halo cells that were decided before. Returns each chunk's
    /// status.
    fn solve(
        &mut self,
        label: &str,
        chunks: &[usize],
        halo: usize,
        release: bool,
        seed: u32,
    ) -> Vec<u32> {
        let m = &self.city.modules;
        let (rx, ry, cz) = (CX as usize + 2 * halo, CY as usize + 2 * halo, CZ as usize);
        let region_cells = rx * ry * cz;
        let init: Vec<reference::Cell> = chunks
            .iter()
            .flat_map(|&k| self.region_init(k, halo, release))
            .collect();
        let kernel = Kernel::new(
            self.gpu,
            &m.rules,
            &m.tileset.weights,
            chunks.len() as u32,
            INVOCATIONS,
            [rx as u32, ry as u32, CZ],
        );
        self.gpu
            .queue
            .write_buffer(&kernel.init, 0, bytemuck::cast_slice(&to_words(&init)));
        let started = std::time::Instant::now();
        kernel.run(
            self.gpu,
            Params {
                mode: 1,
                max_steps: 50_000,
                seed,
                max_attempts: 64,
                radius: 1,
                undo: 1,
                padding: [0; 2],
            },
        );
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;

        let stats = read_u32s(self.gpu, &kernel.stats);
        let domains = from_words(&read_u32s(self.gpu, &kernel.out));
        let mut statuses = Vec::with_capacity(chunks.len());
        let mut counts = [0usize; 5];
        for (r, &k) in chunks.iter().enumerate() {
            let status = stats[r * STATS as usize];
            statuses.push(status);
            counts[status as usize] += 1;
            let span = r * region_cells..(r + 1) * region_cells;
            if status == 3 {
                // A border contradiction must be real: the CPU reference, propagating the same
                // initial domains, has to empty a cell too.
                let mut cells = init[span.clone()].to_vec();
                let mut stack: Vec<usize> = (0..cells.len()).collect();
                let emptied = ReferenceSolver::new(&m.rules, &m.tileset.weights, rx, ry, cz)
                    .propagate(&mut cells, &mut stack)
                    .expect_err("the kernel reports a border contradiction the CPU does not find");
                if counts[3] == 1 {
                    let (at, inner) = self.region(k, halo)[emptied];
                    eprintln!(
                        "block_solver: {label}: chunk {k} first border contradiction at world {at:?}, {}",
                        if inner {
                            "inside the chunk"
                        } else {
                            "in the halo"
                        }
                    );
                }
            }
            if status != 0 {
                continue;
            }
            for ((at, inner), cell) in self.region(k, halo).into_iter().zip(&domains[span]) {
                let Some(i) = self.index(at) else { continue };
                if inner || (release && self.tiles[i].is_some()) {
                    assert_eq!(
                        reference::count(*cell),
                        1,
                        "chunk {k} left {at:?} undecided"
                    );
                    self.tiles[i] = reference::set_bits(*cell).next();
                }
            }
        }
        eprintln!(
            "block_solver: {label}: {} chunks in {wall_ms:.1} ms (cold), \
             [ok, failed, step cap, border contradiction, bad checkpoint] = {counts:?}",
            chunks.len()
        );
        statuses
    }

    /// Checks and renders the world; only pairs of decided cells can violate a rule.
    fn report(&self, name: &str, border_contradictions: usize, repaired: usize) -> Stitched {
        let undecided_cells = self.tiles.iter().filter(|t| t.is_none()).count();
        let air = self.city.air;
        let tiles: Vec<usize> = self.tiles.iter().map(|t| t.unwrap_or(air)).collect();
        let grid = wfc_devtools::TileGrid::new(self.width, self.height, CZ as usize, tiles)
            .expect("dimensions match");
        let decided = |(x, y, z): (usize, usize, usize)| {
            self.tiles[(z * self.height + y) * self.width + x].is_some()
        };
        let violations = wfc_devtools::adjacency_violations(
            &grid,
            &self.city.modules.rules,
            wfc_core::BoundaryCondition::Finite,
        )
        .into_iter()
        .filter(|v| decided(v.cell) && decided(v.neighbor))
        .count();
        eprintln!(
            "block_solver: {name} world {}x{}x{CZ}: undecided cells {undecided_cells}, \
             violations between decided cells {violations}, border contradictions {border_contradictions}, \
             repaired chunks {repaired}",
            self.width, self.height
        );
        let path =
            std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}_city.png"));
        wfc_devtools::render::render_voxel_isometric(&grid, &self.city.voxels, 2)
            .save(&path)
            .expect("write PNG");
        eprintln!("block_solver: rendered {}", path.display());
        Stitched {
            undecided_cells,
            violations,
            border_contradictions,
            repaired,
        }
    }
}

/// Solves an 8x8-chunk world pass by pass, each pass one dispatch over its chunks.
///
/// Each chunk is solved as a region `halo` cells wider on every side, with decided cells pinned
/// and the rest of the halo discarded afterwards: with no halo a chunk's free faces constrain
/// nothing, so it can leave border tiles no row of neighbours can complete. With `repair`, a chunk
/// that still fails is solved again on its own with its halo released, which may rewrite
/// neighbouring cells (modifying in blocks), widening the halo up to three cells.
fn stitch_world(
    name: &str,
    passes: usize,
    halo: usize,
    repair: bool,
    pass_of: impl Fn(usize, usize) -> usize,
) -> Stitched {
    let city = city::city();
    let gpu = bench_device();
    let mut world = World::new(&city, &gpu);
    let (mut border_contradictions, mut repaired) = (0, 0);
    for pass in 0..passes {
        let members: Vec<usize> = (0..64).filter(|k| pass_of(k % 8, k / 8) == pass).collect();
        // Chunks are indexed per dispatch in the hash, so each dispatch takes its own seed.
        let statuses = world.solve(
            &format!("{name} pass {pass}"),
            &members,
            halo,
            false,
            7 + pass as u32,
        );
        border_contradictions += statuses.iter().filter(|&&s| s == 3).count();
        if !repair {
            continue;
        }
        for (&k, _) in members.iter().zip(&statuses).filter(|(_, s)| **s != 0) {
            // One chunk per dispatch: released halos of chunks in the same wave can overlap.
            for widened in 1..=3 {
                let seed = 1000 + (pass * 64 + k) as u32 * 4 + widened as u32;
                let label = format!("{name} pass {pass} repair chunk {k} halo {widened}");
                if world.solve(&label, &[k], widened, true, seed)[0] == 0 {
                    repaired += 1;
                    break;
                }
            }
        }
    }
    world.report(name, border_contradictions, repaired)
}

/// Checkerboard: even chunks first with free faces, then odd chunks with all four side faces fixed.
/// Decided cells never violate the rules, even where a chunk could not be solved.
#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn checkerboard_schedule_never_breaks_a_seam() {
    let stitched = stitch_world("checkerboard", 2, 0, false, |x, y| (x + y) % 2);

    eprintln!(
        "block_solver: checkerboard border contradictions {}",
        stitched.border_contradictions
    );
    assert_eq!(stitched.violations, 0);
}

/// Diagonal waves, as in N-WFC: chunks with the same x + y share no face, so each wave is one
/// dispatch, and every chunk has at most two fixed faces (towards -x and -y).
#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn diagonal_schedule_never_breaks_a_seam() {
    let stitched = stitch_world("diagonal", 15, 0, false, |x, y| x + y);

    eprintln!(
        "block_solver: diagonal border contradictions {}, undecided cells {}",
        stitched.border_contradictions, stitched.undecided_cells
    );
    assert_eq!(stitched.violations, 0);
}

/// Diagonal waves again, each chunk solved with a one-cell halo that is discarded afterwards.
#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn diagonal_schedule_with_halo_never_breaks_a_seam() {
    let stitched = stitch_world("diagonal_halo1", 15, 1, false, |x, y| x + y);

    eprintln!(
        "block_solver: diagonal with halo 1: border contradictions {}, undecided cells {}",
        stitched.border_contradictions, stitched.undecided_cells
    );
    assert_eq!(stitched.violations, 0);
}

/// Checkerboard with a one-cell halo.
#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn checkerboard_schedule_with_halo_never_breaks_a_seam() {
    let stitched = stitch_world("checkerboard_halo1", 2, 1, false, |x, y| (x + y) % 2);

    eprintln!(
        "block_solver: checkerboard with halo 1: border contradictions {}, undecided cells {}",
        stitched.border_contradictions, stitched.undecided_cells
    );
    assert_eq!(stitched.violations, 0);
}

/// Diagonal waves with a one-cell halo, repairing every chunk that still fails by releasing its
/// halo. The world must come out complete as well as seamless.
#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn diagonal_schedule_with_repair_completes_the_world() {
    let stitched = stitch_world("diagonal_repair", 15, 1, true, |x, y| x + y);

    assert_eq!(stitched.violations, 0);
    assert_eq!(
        stitched.undecided_cells, 0,
        "{} chunks repaired",
        stitched.repaired
    );
}

/// Checkerboard with a one-cell halo and repair.
#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn checkerboard_schedule_with_repair_completes_the_world() {
    let stitched = stitch_world("checkerboard_repair", 2, 1, true, |x, y| (x + y) % 2);

    assert_eq!(stitched.violations, 0);
    assert_eq!(
        stitched.undecided_cells, 0,
        "{} chunks repaired",
        stitched.repaired
    );
}

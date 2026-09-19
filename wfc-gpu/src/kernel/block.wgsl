// One workgroup solves one whole region, start to finish, inside a single dispatch.
//
// The region's domains live in workgroup memory, so a collapse and the propagation it causes never
// leave the device: the host hears from a region twice, when it is handed over and when it is done.
// Parallelism is spent across regions, where wave function collapse is embarrassingly parallel,
// rather than inside one propagation, where arc consistency is inherently sequential.
//
// The loop is a state machine that advances once per workgroup-wide step: per-lane work chosen by
// the shared state, a barrier, then lane 0 alone reads the shared flags and decides the next state,
// which every lane picks up through workgroupUniformLoad. WGSL only allows barriers under control
// flow identical for all invocations, and a value loaded from workgroup memory is not identical by
// construction; routing every decision through one uniform load is what makes the loop legal.
//
// Propagation is a gather sweep: each cell intersects, for every neighbour changed since the
// previous sweep, the union of that neighbour's rule rows. Only a cell's own lane writes it, and
// domains only shrink, so a concurrent read sees a superset of the neighbour's final state and can
// never remove a supported tile; the fixpoint does not depend on thread order.
//
// Every decision is integer arithmetic, and every choice is a hash of (seed, chunk id, attempt,
// step), so two backends and two machines agree on the result. Keep this in step with
// wfc_core::hash, which is the same specification in Rust.
//
// The braced names are substituted by KernelSpec::wgsl before compilation.

const CX: u32 = {CX}u;
const CY: u32 = {CY}u;
const CELLS: u32 = {CELLS}u;
const CZ: u32 = CELLS / (CX * CY);
const NT: u32 = {NT}u;
// Words a cell's mask needs, 32 tiles each.
const W: u32 = {W}u;
const WG: u32 = {WG}u;
const RULE_WORDS: u32 = {RULE_WORDS}u;
const STATS: u32 = {STATS}u;
const RING: u32 = {RING}u;
const NONE: u32 = 0xFFFFFFFFu;

const LOAD: u32 = 0u;
const PROPAGATE: u32 = 1u;
const WRITE: u32 = 2u;
const DONE: u32 = 3u;
const SELECT: u32 = 4u;
const RESTORE: u32 = 5u;

const STATUS_OK: u32 = 0u;
const STATUS_FAILED: u32 = 1u;
const STATUS_CAP: u32 = 2u;
const STATUS_BOUNDARY: u32 = 3u;
// A checkpoint held an empty cell: a bug in the checkpoint ring, never a property of the rules.
const STATUS_BAD_CHECKPOINT: u32 = 4u;

// One cell's possible tiles, and everything that reads or writes one. KernelSpec::wgsl generates
// these with the words written out; see the note on mask_prelude for why nothing loops over them.
{MASK_PRELUDE}
struct Params {
    // 0 propagates to a fixpoint and stops; 1 solves the region.
    mode: u32,
    max_steps: u32,
    max_attempts: u32,
    // 0 collapses the single global minimum per round; r > 0 collapses every local minimum within
    // Chebyshev radius r at once.
    radius: u32,
    // 0 restarts the region on a contradiction; 1 restores the checkpoint before the failing round.
    undo: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
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
    // The most rounds this attempt has reached. Slot k is rewritten by round k + RING, so it is
    // only trustworthy while k + RING exceeds this.
    max_step: u32,
};

@group(0) @binding(0) var<storage, read> rules: array<u32>;
@group(0) @binding(1) var<storage, read> init: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<storage, read_write> stats: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
// One integer weight per tile. Integers, because a float sum may be contracted into a fused
// multiply-add by one driver and not another, and the same seed would then pick differently.
@group(0) @binding(5) var<storage, read> weights: array<u32>;
// A ring of RING checkpoints per region: slot k holds the fixpoint reached before round k.
@group(0) @binding(6) var<storage, read_write> snaps: array<u32>;
// The world identity of each region's chunk, so its choices depend on where it is rather than on
// its place in the batch: an evicted chunk regenerates identically.
@group(0) @binding(7) var<storage, read> ids: array<u32>;
// Each region's seed. A repair of one chunk shares a dispatch with first attempts at others.
@group(0) @binding(8) var<storage, read> seeds: array<u32>;

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

fn neighbour(c: u32, axis: u32) -> u32 {
    let x = c % CX;
    let y = (c / CX) % CY;
    let z = c / (CX * CY);
    switch axis {
        case 0u: { if (x + 1u < CX) { return c + 1u; } }
        case 1u: { if (x > 0u) { return c - 1u; } }
        case 2u: { if (y + 1u < CY) { return c + CX; } }
        case 3u: { if (y > 0u) { return c - CX; } }
        case 4u: { if (z + 1u < CZ) { return c + CX * CY; } }
        case 5u: { if (z > 0u) { return c - CX * CY; } }
        default: {}
    }
    return NONE;
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

fn choice_hash(seed: u32, id: u32, tries: u32, step: u32) -> u32 {
    return pcg3d(vec3<u32>(seed ^ (id * 0x9E3779B9u), tries, step)).x;
}

fn snap_index(region: u32, slot: u32, c: u32) -> u32 {
    return ((region * RING + slot) * CELLS + c) * W;
}

fn cell_key(c: u32, mask: Mask) -> u32 {
    let count = mask_count(mask);
    if (count > 1u) {
        return (count << 16u) | c;
    }
    return NONE;
}

// One gather sweep over this lane's cells; also leaves each lane's best selection key and, when
// undo is on, this lane's part of the checkpoint for the coming round. Only the last sweep before a
// round changes nothing, so the copy that survives is the fixpoint.
fn sweep_lane(lane: u32, region: u32, st: Ctrl) {
    let sweep = st.sweep;
    var best = NONE;
    for (var c = lane; c < CELLS; c += WG) {
        let before = dom_load(c);
        var mask = before;
        for (var axis = 0u; axis < 6u; axis++) {
            let n = neighbour(c, axis);
            if (n == NONE || atomicLoad(&epoch[n]) + 1u < sweep) {
                continue;
            }
            // The neighbour lies along `axis` from c, so c lies along the opposite axis from it.
            mask = mask_and(mask, allowed_by(dom_load(n), axis ^ 1u));
        }
        if (mask_differs(mask, before)) {
            dom_store(c, mask);
            atomicStore(&epoch[c], sweep);
            atomicStore(&changed, 1u);
            if (mask_empty(mask)) {
                atomicMax(&contra, CELLS - c);
            }
        }
        let key = cell_key(c, mask);
        best = min(best, key);
        cell_keys[c] = key;
        if (params.undo != 0u) {
            snap_store(snap_index(region, st.step % RING, c), mask);
        }
    }
    keys[lane] = best;
}

// Puts back this lane's cells from the checkpoint before round `st.step`. It is a fixpoint, so no
// cell needs another sweep.
fn restore_lane(lane: u32, region: u32, st: Ctrl) {
    var best = NONE;
    for (var c = lane; c < CELLS; c += WG) {
        let mask = snap_load(snap_index(region, st.step % RING, c));
        dom_store(c, mask);
        atomicStore(&epoch[c], 0u);
        if (mask_empty(mask)) {
            atomicStore(&restored_empty, 1u);
        }
        let key = cell_key(c, mask);
        best = min(best, key);
        cell_keys[c] = key;
    }
    keys[lane] = best;
}

// Lane 0 only: collapses the fewest-possibilities cell. False when every cell is decided.
fn collapse_global_minimum(id: u32, seed: u32, st: Ctrl) -> bool {
    var best = NONE;
    for (var i = 0u; i < WG; i++) {
        best = min(best, keys[i]);
    }
    if (best == NONE) {
        return false;
    }
    let cell = best & 0xFFFFu;
    let tile = weighted_tile(dom_load(cell), choice_hash(seed, id, st.tries, st.step));
    dom_store(cell, mask_one(tile));
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
fn select_lane(lane: u32, id: u32, seed: u32, st: Ctrl) {
    for (var c = lane; c < CELLS; c += WG) {
        if (cell_keys[c] == NONE || !is_local_minimum(c, i32(params.radius))) {
            continue;
        }
        let hash = choice_hash(seed, id, st.tries, st.step * CELLS + c);
        let tile = weighted_tile(dom_load(c), hash);
        dom_store(c, mask_one(tile));
        atomicStore(&epoch[c], st.sweep);
        atomicAdd(&chosen, 1u);
    }
}

@compute @workgroup_size({WG})
fn solve_region(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) workgroup: vec3<u32>,
) {
    let region = workgroup.x;
    let id = ids[region];
    let seed = seeds[region];
    let base = region * CELLS * W;
    var st = Ctrl(LOAD, 1u, STATUS_OK, 0u, NONE, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    loop {
        // Per-lane work for the current state. No barriers in here.
        if (st.phase == LOAD) {
            let per_lane = (RULE_WORDS + WG - 1u) / WG;
            for (var i = lane * per_lane; i < min((lane + 1u) * per_lane, RULE_WORDS); i++) {
                rules_s[i] = rules[i];
            }
            for (var c = lane; c < CELLS; c += WG) {
                dom_store(c, init_load(base + c * W));
                atomicStore(&epoch[c], st.sweep);
            }
        } else if (st.phase == PROPAGATE) {
            sweep_lane(lane, region, st);
        } else if (st.phase == SELECT) {
            select_lane(lane, id, seed, st);
        } else if (st.phase == RESTORE) {
            restore_lane(lane, region, st);
        } else if (st.phase == WRITE) {
            for (var c = lane; c < CELLS; c += WG) {
                out_store(base + c * W, dom_load(c));
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
                    } else if (params.undo != 0u && undo <= st.step
                        && st.step - undo + RING > st.max_step
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
                } else if (collapse_global_minimum(id, seed, st)) {
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
                } else if (collapse_global_minimum(id, seed, st)) {
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
                stats[region * STATS] = next.status;
                stats[region * STATS + 1u] = next.sweeps;
                stats[region * STATS + 2u] = next.collapses;
                stats[region * STATS + 3u] = next.restarts;
                stats[region * STATS + 4u] = next.contra_cell;
                stats[region * STATS + 5u] = next.steps;
                stats[region * STATS + 6u] = next.backtracks;
                stats[region * STATS + 7u] = next.tries;
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

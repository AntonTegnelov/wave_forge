//! A sequential CPU solver on the city rules, kept as the yardstick every GPU solver is measured
//! against (docs/solver-redesign.md, "The leads").
//!
//! It is deliberately plain: two `u64` words per cell, a stack of changed cells for propagation, a
//! full scan for the fewest-possibilities cell, and marian42's undo-doubling on contradiction. It is
//! not a product and not a CPU fallback. It answers one question: what does a single CPU thread
//! achieve on this workload, so that a GPU number can be read as a win or a loss.
//!
//! ```text
//! cargo test -p wfc-devtools --release --test cpu_reference -- --ignored --nocapture
//! ```
//!
//! Each run prints one `cpu_reference:` line. Numbers describe one build on one machine; the full
//! grid scan makes selection quadratic in grid size, so large grids understate what a CPU can do.

use std::collections::VecDeque;
use std::time::Instant;
use wfc_core::BoundaryCondition;
use wfc_core::grid::PossibilityGrid;
use wfc_devtools::city::{self, City};
use wfc_devtools::{TileGrid, adjacency_violations};

/// Words per cell. The city has 81 variants, which fit in two.
const WORDS: usize = 2;
/// Snapshots kept for undo, as in the GPU solver.
const MAX_HISTORY: usize = 2048;
/// A run that backtracks this often is reported as thrashing rather than left to spin.
const MAX_BACKTRACKS: usize = 20_000;

type Cell = [u64; WORDS];

/// SplitMix64: a small, seedable generator, so a run is reproducible without a dependency.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A float in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn set_bits(cell: Cell) -> impl Iterator<Item = usize> {
    cell.into_iter().enumerate().flat_map(|(word, mut bits)| {
        std::iter::from_fn(move || {
            (bits != 0).then(|| {
                let tile = word * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                tile
            })
        })
    })
}

fn count(cell: Cell) -> u32 {
    cell.iter().map(|word| word.count_ones()).sum()
}

struct Solver<'a> {
    width: usize,
    height: usize,
    depth: usize,
    num_tiles: usize,
    /// One mask per `(axis, tile)`: the tiles allowed in the neighbour along `axis`.
    masks: Vec<Cell>,
    weights: &'a [f32],
}

/// What a finished run reports.
struct Outcome {
    grid: Vec<Cell>,
    collapses: usize,
    backtracks: usize,
    seconds: f64,
    thrashed: bool,
}

impl Solver<'_> {
    /// Propagates from every cell on `stack` to a fixpoint, or returns the cell that emptied.
    fn propagate(&self, grid: &mut [Cell], stack: &mut Vec<usize>) -> Result<(), usize> {
        let (w, h, d) = (self.width, self.height, self.depth);
        while let Some(i) = stack.pop() {
            let (x, y, z) = (i % w, (i / w) % h, i / (w * h));
            let current = grid[i];
            for axis in 0..6 {
                let j = match axis {
                    0 if x + 1 < w => i + 1,
                    1 if x > 0 => i - 1,
                    2 if y + 1 < h => i + w,
                    3 if y > 0 => i - w,
                    4 if z + 1 < d => i + w * h,
                    5 if z > 0 => i - w * h,
                    _ => continue,
                };
                let mut allowed = [0u64; WORDS];
                for tile in set_bits(current) {
                    let row = self.masks[axis * self.num_tiles + tile];
                    for k in 0..WORDS {
                        allowed[k] |= row[k];
                    }
                }
                let old = grid[j];
                let mut new = old;
                for k in 0..WORDS {
                    new[k] &= allowed[k];
                }
                if new != old {
                    if count(new) == 0 {
                        return Err(j);
                    }
                    grid[j] = new;
                    stack.push(j);
                }
            }
        }
        Ok(())
    }

    fn solve(&self, mut grid: Vec<Cell>, seed: u64) -> Outcome {
        let started = Instant::now();
        let mut stack: Vec<usize> = (0..grid.len()).collect();
        self.propagate(&mut grid, &mut stack)
            .expect("the constrained city is arc consistent");
        let mut rng = SplitMix64(seed);
        let mut history: VecDeque<(Vec<Cell>, usize, usize)> = VecDeque::new();
        let (mut collapses, mut backtracks) = (0usize, 0usize);
        // Fewest possibilities first, lowest index on ties.
        while let Some((_, cell)) = grid
            .iter()
            .enumerate()
            .map(|(i, &c)| (count(c), i))
            .filter(|&(n, _)| n > 1)
            .min()
        {
            let total: f32 = set_bits(grid[cell]).map(|t| self.weights[t]).sum();
            let mut pick = rng.unit() * total;
            let mut chosen = 0;
            for tile in set_bits(grid[cell]) {
                chosen = tile;
                pick -= self.weights[tile];
                if pick <= 0.0 {
                    break;
                }
            }
            if history.len() == MAX_HISTORY {
                history.pop_front();
            }
            history.push_back((grid.clone(), cell, chosen));
            grid[cell] = [0; WORDS];
            grid[cell][chosen / 64] = 1 << (chosen % 64);
            collapses += 1;
            stack.push(cell);
            let mut undo = 1usize;
            while self.propagate(&mut grid, &mut stack).is_err() {
                stack.clear();
                backtracks += 1;
                if backtracks >= MAX_BACKTRACKS {
                    return Outcome {
                        grid,
                        collapses,
                        backtracks,
                        seconds: 0.0,
                        thrashed: true,
                    };
                }
                let mut restored = None;
                for _ in 0..undo {
                    restored = history.pop_back().or(restored);
                }
                undo = (undo * 2).min(64);
                let (snapshot, banned_cell, banned_tile) = restored.expect("history is not empty");
                grid = snapshot;
                grid[banned_cell][banned_tile / 64] &= !(1 << (banned_tile % 64));
                // An emptied cell fails the loop condition again and undoes further.
                stack.push(banned_cell);
            }
        }
        Outcome {
            grid,
            collapses,
            backtracks,
            seconds: started.elapsed().as_secs_f64(),
            thrashed: false,
        }
    }
}

fn initial_cells(city: &City, width: usize, height: usize, depth: usize) -> Vec<Cell> {
    let mut initial = PossibilityGrid::new(width, height, depth, city.modules.variants.len());
    city::constrain_city(&mut initial, city);
    (0..width * height * depth)
        .map(|i| {
            let possible = initial
                .get(i % width, (i / width) % height, i / (width * height))
                .expect("in bounds");
            let mut cell = [0u64; WORDS];
            for tile in possible.iter_ones() {
                cell[tile / 64] |= 1 << (tile % 64);
            }
            cell
        })
        .collect()
}

#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn cpu_reference_city() {
    let city = city::city();
    let m = &city.modules;
    let num_tiles = m.variants.len();
    assert!(
        num_tiles <= WORDS * 64,
        "{num_tiles} variants do not fit {WORDS} words"
    );
    let mut masks = vec![[0u64; WORDS]; 6 * num_tiles];
    for axis in 0..6 {
        for a in 0..num_tiles {
            for b in 0..num_tiles {
                if m.rules.check(a, b, axis) {
                    masks[axis * num_tiles + a][b / 64] |= 1 << (b % 64);
                }
            }
        }
    }

    for (width, height, depth) in [(8, 8, 8), (12, 12, 6), (24, 24, 8), (48, 48, 10)] {
        let solver = Solver {
            width,
            height,
            depth,
            num_tiles,
            masks: masks.clone(),
            weights: &m.tileset.weights,
        };
        let initial = initial_cells(&city, width, height, depth);
        for seed in 1..=8 {
            let outcome = solver.solve(initial.clone(), seed);
            let cells = width * height * depth;
            if outcome.thrashed {
                eprintln!(
                    "cpu_reference: {width}x{height}x{depth} seed={seed} thrashed backtracks={}",
                    outcome.backtracks
                );
                continue;
            }
            let tiles = outcome
                .grid
                .iter()
                .map(|&cell| {
                    assert_eq!(count(cell), 1, "every cell collapses to one tile");
                    set_bits(cell).next().expect("one tile")
                })
                .collect();
            let grid = TileGrid::new(width, height, depth, tiles).expect("dimensions match");
            let violations = adjacency_violations(&grid, &m.rules, BoundaryCondition::Finite);
            assert!(
                violations.is_empty(),
                "{} adjacency violations",
                violations.len()
            );
            eprintln!(
                "cpu_reference: {width}x{height}x{depth} seed={seed} cells={cells} collapses={} backtracks={} run_s={:.4} cells_per_s={:.0}",
                outcome.collapses,
                outcome.backtracks,
                outcome.seconds,
                cells as f64 / outcome.seconds
            );
        }
    }
}

//! What the block solver must get right, checked against the CPU reference and the rules.
//!
//! These run on any adapter, including a software one, so they stay in the ordinary test run. The
//! timings live in `block_solver_bench.rs`.

use std::sync::Arc;
use std::time::{Duration, Instant};
use wfc_core::reference::ReferenceSolver;
use wfc_core::rules::AXES;
use wfc_core::{
    ChunkCoord, ChunkShape, ChunkStore, Domains, Prior, Region, RegionBatch, RegionShape,
    RegionStatus, Ruleset, SolveBudget, Solver, SolverError, TileMask, WorldExtent, region_init,
};
use wfc_devtools::city::{self, city_prior};
use wfc_gpu::block_solver::BlockSolver;
use wfc_gpu::error::GpuError;
use wfc_gpu::kernel::{Params, SolverConfig};
use wfc_gpu::wgpu_backend::WgpuBackend;
use wfc_rules::AdjacencyRules;

/// A world of one chunk, which is what these tests solve.
struct OneChunk {
    region: Region,
    init: Domains,
    ruleset: Arc<Ruleset>,
}

impl OneChunk {
    /// A chunk of `shape` under `ruleset`, with `prior` deciding its starting domains.
    fn new(shape: ChunkShape, ruleset: Ruleset, prior: &Prior) -> Self {
        let extent = WorldExtent::new(shape)
            .with_x(0..1)
            .with_y(0..1)
            .with_z(0..1);
        let region = Region::new(ChunkCoord::new(0, 0, 0), shape.region([0, 0, 0]));
        let store = ChunkStore::new(extent);
        let init = region_init(&store, prior, &ruleset, &region, false);
        Self {
            region,
            init,
            ruleset: Arc::new(ruleset),
        }
    }

    /// The city as one chunk.
    fn city(shape: ChunkShape) -> Self {
        let city = city::city();
        let ruleset = Ruleset::from_modules(&city.modules).expect("the city rule set compiles");
        let prior = city_prior(&city, shape.z);
        Self::new(shape, ruleset, &prior)
    }

    fn shape(&self) -> RegionShape {
        self.region.shape()
    }

    /// A batch that solves this chunk once per `(id, seed)` pair.
    fn batch(&self, runs: &[(u32, u32)]) -> RegionBatch {
        let mut init = self.init.clone();
        for _ in 1..runs.len() {
            let more = self.init.clone();
            init.append(&more);
        }
        RegionBatch {
            region: self.shape(),
            ids: runs.iter().map(|(id, _)| *id).collect(),
            seeds: runs.iter().map(|(_, seed)| *seed).collect(),
            init,
            budget: None,
            portfolio: false,
        }
    }

    fn solver(&self, config: SolverConfig) -> BlockSolver<WgpuBackend> {
        let backend = WgpuBackend::from_env().expect("a compute device");
        BlockSolver::new(backend, Arc::clone(&self.ruleset), config).expect("a solver")
    }
}

/// Tiles that may only touch themselves, so one choice decides a whole region. With more tiles than
/// fit a word, a solve has to cross words to stay correct.
fn stripes(tiles: usize) -> Ruleset {
    let tuples: Vec<(usize, usize, usize)> = (0..AXES)
        .flat_map(|axis| (0..tiles).map(move |tile| (axis, tile, tile)))
        .collect();
    Ruleset::new(
        &AdjacencyRules::from_allowed_tuples(tiles, AXES, tuples),
        &vec![1.0; tiles],
    )
    .expect("stripe rules")
}

fn tiles_of(domains: &Domains) -> Vec<u32> {
    (0..domains.cells())
        .map(|cell| domains.decided(cell).expect("every cell is decided"))
        .collect()
}

#[test]
fn propagation_reaches_the_reference_fixpoint() {
    let chunk = OneChunk::city(ChunkShape::cube(8));
    let config = SolverConfig::default();
    let mut expected = chunk.init.clone();
    let mut stack: Vec<u32> = (0..expected.cells()).collect();
    ReferenceSolver::new(Arc::clone(&chunk.ruleset))
        .propagate(chunk.shape(), &mut expected, &mut stack)
        .expect("the city chunk is consistent");
    let narrowed = (0..expected.cells())
        .filter(|cell| expected.mask(*cell) != chunk.init.mask(*cell))
        .count();
    assert!(
        narrowed > 0,
        "a kernel that did nothing would pass if propagation had nothing to do"
    );
    let mut solver = chunk.solver(config);

    let job = solver
        .start_with(chunk.batch(&[(0, 1)]), Params::propagate_only(&config))
        .expect("a well-formed batch");
    let result = solver.wait(job).expect("the dispatch finishes");

    assert_eq!(result.statuses, vec![RegionStatus::Solved]);
    assert_eq!(
        result.domains, expected,
        "{narrowed} cells were narrowed by propagation"
    );
}

#[test]
fn a_region_solves_validly_and_the_same_seed_repeats_it() {
    let chunk = OneChunk::city(ChunkShape::cube(8));
    let mut solver = chunk.solver(SolverConfig::default());

    let job = solver
        .start(chunk.batch(&[(7, 1), (7, 1), (8, 1)]))
        .expect("a well-formed batch");
    let result = solver.wait(job).expect("the dispatch finishes");

    let cells = chunk.shape().cells();
    assert_eq!(result.statuses, vec![RegionStatus::Solved; 3]);
    let first = result.region(0, cells);
    assert_eq!(
        first,
        result.region(1, cells),
        "the same chunk and seed repeat"
    );
    assert_ne!(first, result.region(2, cells), "another chunk differs");
    for (cell, tile) in tiles_of(&first).into_iter().enumerate() {
        assert!(
            chunk.init.mask(cell as u32).contains(tile),
            "cell {cell} holds tile {tile}, which its starting domain forbade"
        );
    }
    let adjacency = wfc_devtools::adjacency_violations(
        &wfc_devtools::TileGrid::from_domains(&first, 8, 8, 8).expect("decided"),
        &city::city().modules.rules,
        wfc_devtools::BoundaryCondition::Finite,
    );
    assert!(
        adjacency.is_empty(),
        "{} adjacency violations",
        adjacency.len()
    );
}

#[test]
fn the_result_does_not_depend_on_invocations_per_workgroup() {
    let chunk = OneChunk::city(ChunkShape::cube(8));
    let runs = [(3, 11)];

    let solved = [64, 256].map(|invocations| {
        let mut solver = chunk.solver(SolverConfig {
            invocations,
            ..SolverConfig::default()
        });
        let job = solver
            .start(chunk.batch(&runs))
            .expect("a well-formed batch");
        let result = solver.wait(job).expect("the dispatch finishes");
        assert_eq!(result.statuses, vec![RegionStatus::Solved]);
        result.domains
    });

    assert_eq!(
        solved[0], solved[1],
        "how the work is spread must not change the result"
    );
}

#[test]
fn a_portfolio_stops_seeds_above_the_winner_without_changing_the_winner() {
    // One attempt each, so some seeds fail and the winner is not always the first.
    let chunk = OneChunk::city(ChunkShape::cube(8));
    let runs: Vec<(u32, u32)> = (0..32).map(|seed| (5, seed * 7919 + 3)).collect();
    let budget = SolveBudget {
        max_attempts: 1,
        max_steps: 50_000,
    };
    let solve = |portfolio: bool| {
        let mut solver = chunk.solver(SolverConfig::default());
        let job = solver
            .start(RegionBatch {
                budget: Some(budget),
                portfolio,
                ..chunk.batch(&runs)
            })
            .expect("a well-formed batch");
        solver.wait(job).expect("the dispatch finishes")
    };

    let all = solve(false);
    let early = solve(true);

    let winner = all
        .statuses
        .iter()
        .position(|status| status.is_solved())
        .expect("some seed solves");
    assert_eq!(
        early.statuses.iter().position(|status| status.is_solved()),
        Some(winner),
        "the same seed wins: {:?} against {:?}",
        early.statuses,
        all.statuses
    );
    let cells = chunk.shape().cells();
    assert_eq!(early.region(winner, cells), all.region(winner, cells));
    assert_eq!(
        early.statuses[..winner],
        all.statuses[..winner],
        "every seed below the winner runs to its end"
    );
    for (index, (&stopped, &ran)) in early.statuses.iter().zip(&all.statuses).enumerate() {
        assert!(
            stopped == ran || (stopped == RegionStatus::Superseded && index > winner),
            "seed {index}: {stopped:?} with the portfolio, {ran:?} without"
        );
    }
    assert!(!all.statuses.contains(&RegionStatus::Superseded));
    eprintln!(
        "block_solver: portfolio winner {winner}, {} of 32 superseded",
        early
            .statuses
            .iter()
            .filter(|status| **status == RegionStatus::Superseded)
            .count()
    );
}

#[test]
fn weights_bias_the_choice() {
    let shape = ChunkShape { x: 4, y: 4, z: 1 };
    let rules = AdjacencyRules::from_allowed_tuples(
        2,
        AXES,
        (0..AXES).flat_map(|axis| (0..2).flat_map(move |a| (0..2).map(move |b| (axis, a, b)))),
    );
    let heavy = Ruleset::new(&rules, &[1.0, 1000.0]).expect("valid weights");
    let chunk = OneChunk::new(shape, heavy, &Prior::open(2));
    let mut solver = chunk.solver(SolverConfig::default());
    let runs: Vec<(u32, u32)> = (0..8).map(|seed| (seed, 1)).collect();

    let job = solver
        .start(chunk.batch(&runs))
        .expect("a well-formed batch");
    let result = solver.wait(job).expect("the dispatch finishes");

    let cells = chunk.shape().cells();
    let tiles: Vec<u32> = (0..runs.len())
        .flat_map(|region| tiles_of(&result.region(region, cells)))
        .collect();
    let heavy = tiles.iter().filter(|tile| **tile == 1).count();
    assert!(
        heavy * 10 > tiles.len() * 9,
        "the tile weighing a thousand times more took only {heavy} of {} cells",
        tiles.len()
    );
}

#[test]
fn rule_sets_wider_than_one_word_decide_tiles_in_every_word() {
    let chunk = OneChunk::new(
        ChunkShape { x: 4, y: 4, z: 2 },
        stripes(40),
        &Prior::open(40),
    );
    let mut solver = chunk.solver(SolverConfig::default());
    let runs: Vec<(u32, u32)> = (0..16).map(|id| (id, 5)).collect();

    let job = solver
        .start(chunk.batch(&runs))
        .expect("a well-formed batch");
    let result = solver.wait(job).expect("the dispatch finishes");

    let cells = chunk.shape().cells();
    let mut chosen = Vec::new();
    for region in 0..runs.len() {
        let tiles = tiles_of(&result.region(region, cells));
        assert!(
            tiles.iter().all(|tile| *tile == tiles[0]),
            "these rules allow one tile per region"
        );
        chosen.push(tiles[0]);
    }
    assert!(
        chosen.iter().any(|tile| *tile >= 32),
        "no region chose a tile beyond the first word"
    );
    assert!(
        chosen.iter().any(|tile| *tile < 32),
        "no region chose a tile in the first word"
    );
}

#[test]
fn a_rule_set_wider_than_four_words_solves_too() {
    // Five words per cell, so a mask needs two vectors and the kernel's generated helpers take
    // their second path. The region is small because the rule table also lives in workgroup memory.
    let chunk = OneChunk::new(
        ChunkShape { x: 4, y: 4, z: 1 },
        stripes(130),
        &Prior::open(130),
    );
    let mut solver = chunk.solver(SolverConfig::default());
    let runs: Vec<(u32, u32)> = (0..16).map(|id| (id, 9)).collect();

    let job = solver
        .start(chunk.batch(&runs))
        .expect("a well-formed batch");
    let result = solver.wait(job).expect("the dispatch finishes");

    assert_eq!(chunk.ruleset.words_per_cell(), 5);
    let cells = chunk.shape().cells();
    let chosen: Vec<u32> = (0..runs.len())
        .map(|region| {
            let tiles = tiles_of(&result.region(region, cells));
            assert!(
                tiles.iter().all(|tile| *tile == tiles[0]),
                "one tile per region"
            );
            tiles[0]
        })
        .collect();
    assert!(
        chosen.iter().any(|tile| *tile >= 128),
        "no region chose a tile in the fifth word"
    );
    assert!(
        chosen.iter().any(|tile| *tile < 32),
        "no region chose a tile in the first word"
    );
}

#[test]
fn an_impossible_border_is_reported_rather_than_searched() {
    let chunk = OneChunk::new(ChunkShape { x: 4, y: 4, z: 1 }, stripes(8), &Prior::open(8));
    let mut init = chunk.init.clone();
    init.set(0, TileMask::single(0));
    init.set(1, TileMask::single(1));
    let batch = RegionBatch {
        region: chunk.shape(),
        ids: vec![1],
        seeds: vec![1],
        init,
        budget: None,
        portfolio: false,
    };
    let mut solver = chunk.solver(SolverConfig::default());

    let job = solver.start(batch).expect("a well-formed batch");
    let result = solver.wait(job).expect("the dispatch finishes");

    assert_eq!(result.statuses, vec![RegionStatus::BorderContradiction]);
    assert_eq!(
        result.stats[0].collapses, 0,
        "it did not search a region it cannot solve"
    );
    assert!(
        result.stats[0].contradiction_cell.is_some(),
        "it says where it emptied"
    );
}

#[test]
fn a_region_too_large_for_the_device_says_so_in_numbers() {
    // At 81 tiles a 16x16x8 region needs about 40 KiB of workgroup memory; no device offers that.
    let chunk = OneChunk::city(ChunkShape { x: 16, y: 16, z: 8 });
    let mut solver = chunk.solver(SolverConfig::default());

    let refused = solver.start(chunk.batch(&[(0, 1)]));

    match refused {
        Err(SolverError::WorkgroupStorage { needed, available }) => {
            assert!(needed > available, "{needed} B against {available} B");
        }
        other => panic!("expected a workgroup storage error, got {other:?}"),
    }
}

#[test]
fn asking_for_more_invocations_than_the_device_allows_is_refused() {
    let chunk = OneChunk::city(ChunkShape::cube(8));
    let backend = WgpuBackend::from_env().expect("a compute device");
    let available = wfc_gpu::ComputeBackend::limits(&backend).invocations_per_workgroup;

    let refused = BlockSolver::new(
        backend,
        Arc::clone(&chunk.ruleset),
        SolverConfig {
            invocations: available * 2,
            ..SolverConfig::default()
        },
    );

    assert!(
        matches!(refused, Err(GpuError::Invocations { .. })),
        "a device limit is not a panic"
    );
}

#[test]
fn a_batch_is_polled_rather_than_waited_on() {
    let chunk = OneChunk::city(ChunkShape::cube(8));
    let mut solver = chunk.solver(SolverConfig::default());
    assert!(solver.can_poll(), "wgpu answers without blocking");
    let runs: Vec<(u32, u32)> = (0..16).map(|id| (id, 3)).collect();

    let job = solver
        .start(chunk.batch(&runs))
        .expect("a well-formed batch");
    // A poll never blocks, so how many it takes depends on the device's speed; what is bounded is
    // how long the device may take, and a software device is slow.
    let started = Instant::now();
    let mut polls = 0;
    let result = loop {
        polls += 1;
        if let Some(result) = solver.poll(job).expect("the job is the running one") {
            break result;
        }
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "the dispatch never finished"
        );
    };

    eprintln!(
        "block_solver: on {}, the batch was polled {polls} times before it finished",
        solver.backend().describe()
    );
    assert_eq!(result.statuses.len(), runs.len());
    assert!(
        matches!(solver.poll(job), Err(SolverError::UnknownJob(_))),
        "a job is taken once"
    );
}

#[test]
fn a_malformed_batch_is_refused_before_the_device_sees_it() {
    let chunk = OneChunk::city(ChunkShape::cube(8));
    let mut solver = chunk.solver(SolverConfig::default());
    let mut batch = chunk.batch(&[(0, 1)]);
    batch.seeds.clear();

    assert!(matches!(
        solver.start(batch),
        Err(SolverError::Malformed(_))
    ));
    let empty = RegionBatch {
        region: chunk.shape(),
        ids: Vec::new(),
        seeds: Vec::new(),
        init: Domains::from_words(0, chunk.ruleset.words_per_cell(), Vec::new()).expect("empty"),
        budget: None,
        portfolio: false,
    };
    assert!(matches!(
        solver.start(empty),
        Err(SolverError::Malformed(_))
    ));
}

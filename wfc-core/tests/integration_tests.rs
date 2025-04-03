use bitvec::prelude::{bitvec, Lsb0};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use wfc_core::{
    entropy::cpu::CpuEntropyCalculator,
    grid::PossibilityGrid,
    propagator::cpu::CpuConstraintPropagator,
    runner::{run, WfcConfig},
    BoundaryMode, WfcError,
};
use wfc_rules::{AdjacencyRules, TileSet, Transformation};

// --- Test Setup Helpers (similar to runner unit tests) ---
const TEST_GRID_DIM_INT: usize = 4; // Slightly larger grid for integration
const TEST_NUM_TILES_INT: usize = 3; // More tiles

fn setup_grid_int() -> PossibilityGrid {
    PossibilityGrid::new(TEST_GRID_DIM_INT, TEST_GRID_DIM_INT, 1, TEST_NUM_TILES_INT)
    // 2D for simplicity
}

fn create_tileset_int(num_base_tiles: usize) -> TileSet {
    let weights = vec![1.0; num_base_tiles];
    let allowed_transforms = vec![vec![Transformation::Identity]; num_base_tiles];
    TileSet::new(weights, allowed_transforms).expect("Failed to create test tileset")
}

fn create_uniform_rules_int(num_tiles: usize) -> AdjacencyRules {
    let num_axes = 6; // Still 3D rules even if grid is 2D (depth 1)
    let mut allowed_tuples = Vec::new();
    for axis in 0..num_axes {
        for ttid1 in 0..num_tiles {
            for ttid2 in 0..num_tiles {
                allowed_tuples.push((axis, ttid1, ttid2));
            }
        }
    }
    AdjacencyRules::from_allowed_tuples(num_tiles, num_axes, allowed_tuples)
}

fn create_simple_pattern_rules_int() -> AdjacencyRules {
    // Rules for a simple 2-tile checkerboard pattern
    // T0 must have T1 neighbors on X/Y
    // T1 must have T0 neighbors on X/Y
    // Anything allowed on Z
    let num_tiles = 2;
    let num_axes = 6;
    let mut allowed = Vec::new();
    // Axis 0 (+X), 1 (-X), 2 (+Y), 3 (-Y)
    allowed.push((0, 0, 1)); // +X: T0 -> T1
    allowed.push((1, 1, 0)); // -X: T1 -> T0
    allowed.push((0, 1, 0)); // +X: T1 -> T0
    allowed.push((1, 0, 1)); // -X: T0 -> T1
    allowed.push((2, 0, 1)); // +Y: T0 -> T1
    allowed.push((3, 1, 0)); // -Y: T1 -> T0
    allowed.push((2, 1, 0)); // +Y: T1 -> T0
    allowed.push((3, 0, 1)); // -Y: T0 -> T1

    // Z axis (4, 5) - allow anything
    for axis in 4..6 {
        for t1 in 0..num_tiles {
            for t2 in 0..num_tiles {
                allowed.push((axis, t1, t2));
            }
        }
    }
    AdjacencyRules::from_allowed_tuples(num_tiles, num_axes, allowed)
}

// --- Integration Tests ---

#[test]
fn test_run_success_uniform_rules() {
    let mut grid = setup_grid_int();
    let tileset = Arc::new(create_tileset_int(TEST_NUM_TILES_INT));
    let rules = create_uniform_rules_int(TEST_NUM_TILES_INT);
    let propagator = Box::new(CpuConstraintPropagator::new(BoundaryMode::Clamped));
    let entropy_calculator = Box::new(CpuEntropyCalculator::new(
        tileset.clone(),
        wfc_core::entropy::SelectionStrategy::FirstMinimum,
    ));
    let config = WfcConfig::default();

    let result = run(
        &mut grid,
        &tileset,
        &rules,
        propagator,
        entropy_calculator,
        &config,
    );
    assert!(result.is_ok(), "WFC run failed: {:?}", result.err());

    // Check if fully collapsed
    let is_fully_collapsed = (0..grid.depth).all(|z| {
        (0..grid.height).all(|y| {
            (0..grid.width).all(|x| grid.get(x, y, z).map_or(false, |c| c.count_ones() == 1))
        })
    });
    assert!(
        is_fully_collapsed,
        "Grid was not fully collapsed with uniform rules"
    );
}

#[test]
fn test_run_contradiction_simple_pattern() {
    // Use 2 tiles for checkerboard
    let mut grid = PossibilityGrid::new(2, 1, 1, 2); // Small grid 2x1
    let tileset = Arc::new(create_tileset_int(2));
    let rules = create_simple_pattern_rules_int(); // Checkerboard rules

    // Force initial state that violates rules
    *grid.get_mut(0, 0, 0).unwrap() = bitvec![usize, Lsb0; 1, 0]; // T0
    *grid.get_mut(1, 0, 0).unwrap() = bitvec![usize, Lsb0; 1, 0]; // Also T0 (violates +X T0->T1)

    let propagator = Box::new(CpuConstraintPropagator::new(BoundaryMode::Clamped));
    let entropy_calculator = Box::new(CpuEntropyCalculator::new(
        tileset.clone(),
        wfc_core::entropy::SelectionStrategy::FirstMinimum,
    ));
    let config = WfcConfig::default();

    let result = run(
        &mut grid,
        &tileset,
        &rules,
        propagator,
        entropy_calculator,
        &config,
    );

    // Expect a contradiction error from initial propagation or first iteration
    assert!(
        matches!(
            result,
            Err(WfcError::Contradiction(_, _, _)) | Err(WfcError::PropagationError(_))
        ),
        "Expected Contradiction, got {:?}",
        result
    );
}

#[test]
fn test_run_timeout() {
    // Difficult setup required - potentially unsatisfiable rules on a larger grid
    // Or just set a very low iteration limit
    let mut grid = PossibilityGrid::new(5, 5, 1, 2); // 5x5 grid, 2 tiles
    let tileset = Arc::new(create_tileset_int(2));
    let rules = create_simple_pattern_rules_int(); // Checkerboard should be satisfiable

    let propagator = Box::new(CpuConstraintPropagator::new(BoundaryMode::Clamped));
    let entropy_calculator = Box::new(CpuEntropyCalculator::new(
        tileset.clone(),
        wfc_core::entropy::SelectionStrategy::RandomLowest, // Use randomness
    ));
    // Set a very low iteration limit
    let config = WfcConfig::builder().max_iterations(3).build();

    let result = run(
        &mut grid,
        &tileset,
        &rules,
        propagator,
        entropy_calculator,
        &config,
    );

    assert!(
        matches!(result, Err(WfcError::TimeoutOrInfiniteLoop)),
        "Expected Timeout, got {:?}",
        result
    );
}

#[test]
fn test_run_interrupted() {
    let mut grid = setup_grid_int();
    let tileset = Arc::new(create_tileset_int(TEST_NUM_TILES_INT));
    let rules = create_uniform_rules_int(TEST_NUM_TILES_INT); // Uniform should run longer
    let propagator = Box::new(CpuConstraintPropagator::new(BoundaryMode::Clamped));
    let entropy_calculator = Box::new(CpuEntropyCalculator::new(
        tileset.clone(),
        wfc_core::entropy::SelectionStrategy::FirstMinimum,
    ));

    let shutdown_signal = Arc::new(AtomicBool::new(false));
    let config = WfcConfig::builder()
        .shutdown_signal(shutdown_signal.clone())
        // Maybe add a progress callback to trigger the signal after a few iterations
        .progress_callback(Box::new(move |info| {
            if info.iterations > 2 {
                shutdown_signal.store(true, Ordering::Relaxed);
                log::info!(">>> Shutdown signal sent via callback <<<");
            }
            Ok(())
        }))
        .build();

    // Initialize logger for callback message (optional)
    // let _ = env_logger::builder().is_test(true).try_init();

    let result = run(
        &mut grid,
        &tileset,
        &rules,
        propagator,
        entropy_calculator,
        &config,
    );

    assert!(
        matches!(result, Err(WfcError::Interrupted)),
        "Expected Interrupted, got {:?}",
        result
    );
}

// TODO: Add tests for different BoundaryModes (Periodic)
// TODO: Add tests for checkpointing/resuming within integration context
// TODO: Add tests using different EntropySelection strategies

// Remove trailing quote if present

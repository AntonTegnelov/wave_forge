//! Kernel compilation for the suites that time generation on a device.

use std::time::Instant;
use wave_forge::{BlockSolver, RegionShape, WgpuBackend, WorldGenerator};

/// Compiles every kernel a run will need, which is what a game would do when it loads: the first
/// dispatch of a new specialisation otherwise pays for translating it to the device's own language,
/// and that is seconds, not milliseconds.
///
/// A kernel is specialised per region shape and per batch capacity, so this compiles each of
/// `capacities` for every repair halo the device fits.
pub fn warm(world: &mut WorldGenerator<BlockSolver<WgpuBackend>>, capacities: &[u32]) {
    let config = world.config().clone();
    let solver = world.solver_mut();
    let shapes: Vec<(u32, RegionShape)> = (1..=config.repair.max_halo)
        .map(|halo| config.chunk.region(config.extent.halo(halo)))
        .filter(|region| solver.fits(*region))
        .flat_map(|region| capacities.iter().map(move |&capacity| (capacity, region)))
        .collect();
    let started = Instant::now();
    solver.warm(&shapes).expect("the kernels compile");
    eprintln!(
        "kernels: compiled {} in {:.1} s",
        shapes.len(),
        started.elapsed().as_secs_f64()
    );
}

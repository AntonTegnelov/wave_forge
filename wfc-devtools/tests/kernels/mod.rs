//! Kernel compilation for the suites that time generation on a device.

use std::time::Instant;
use wave_forge::{BlockSolver, WgpuBackend, WorldGenerator};

/// Compiles every kernel a run with focus points of up to `radius` can dispatch, which is what a
/// game would do when it loads: the first dispatch of a new specialisation otherwise pays for
/// translating it to the device's own language, and that is seconds, not milliseconds.
pub fn warm(world: &mut WorldGenerator<BlockSolver<WgpuBackend>>, radius: u32) {
    let shapes = world.kernel_shapes(radius);
    let started = Instant::now();
    world
        .solver_mut()
        .warm(&shapes)
        .expect("the kernels compile");
    eprintln!(
        "kernels: compiled {} in {:.1} s",
        shapes.len(),
        started.elapsed().as_secs_f64()
    );
}

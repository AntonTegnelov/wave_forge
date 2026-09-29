//! Volume surfaces built on a thread of their own: the same surface `volume_mesh` builds, and only
//! the answer to a chunk's newest request.

use std::sync::Arc;
use std::time::{Duration, Instant};
use wave_forge::stages::{Pack, Product, Runtime};
use wave_forge::{ChunkCoord, FocusPoint, SurfaceWorker, VolumeMesh, volume_mesh};

const VOXEL: [f32; 3] = [2.0, 1.5, 2.0];

const PACK: &str = r#"(
    version: 1,
    noises: {"caves": (noise_type: SimplexSmooth, seed: 11, frequency: 0.12, fractal_octaves: 2)},
    stages: [
        (name: "caves", kind: Volume(
            density: Max(Min(FastNoise("caves"), Sub(Constant(5.0), Z)), Sub(Constant(-3.0), Z)),
            bottom: -4,
            top: 6,
        )),
    ],
)"#;

fn runtime() -> Runtime {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        3,
        [8, 8],
    );
    let focus: Vec<FocusPoint> = (-2..=2)
        .flat_map(|y| (-2..=2).map(move |x| FocusPoint::new(ChunkCoord::new(x, y, 0), 0)))
        .collect();
    runtime.request(&focus, &["caves"]).expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    runtime
}

/// The volumes around `chunk`, row by row from its lower corner, shared.
fn around(runtime: &Runtime, chunk: ChunkCoord) -> [Arc<Product>; 9] {
    std::array::from_fn(|i| {
        let at = ChunkCoord::new(chunk.x + i as i32 % 3 - 1, chunk.y + i as i32 / 3 - 1, 0);
        Arc::new(Product::Volume(
            runtime.volume("caves", at).expect("generated").clone(),
        ))
    })
}

/// What `worker` hands back within a few seconds, until it is waiting for nothing.
fn finished(worker: &mut SurfaceWorker) -> Vec<VolumeMesh> {
    let started = Instant::now();
    let mut out = Vec::new();
    while worker.building() > 0 {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "no surface in 10 s"
        );
        out.extend(worker.drain());
        std::thread::yield_now();
    }
    out
}

#[test]
fn a_surface_from_the_worker_is_the_one_volume_mesh_builds() {
    let runtime = runtime();
    let chunks = [ChunkCoord::new(0, 0, 0), ChunkCoord::new(-1, 1, 0)];
    let mut worker = SurfaceWorker::spawn();

    for chunk in chunks {
        worker.build(chunk, around(&runtime, chunk), VOXEL);
    }
    let built = finished(&mut worker);

    assert_eq!(built.len(), 2);
    for mesh in built {
        let expected = volume_mesh(mesh.chunk, |at| runtime.volume("caves", at), VOXEL);
        assert_eq!(Some(mesh), expected);
    }
}

#[test]
fn only_a_chunks_newest_request_is_answered_and_a_cancelled_one_never_is() {
    let runtime = runtime();
    let (kept, cancelled) = (ChunkCoord::new(0, 0, 0), ChunkCoord::new(1, 0, 0));
    let mut worker = SurfaceWorker::spawn();

    worker.build(kept, around(&runtime, kept), [1.0; 3]);
    worker.build(kept, around(&runtime, kept), VOXEL);
    worker.build(cancelled, around(&runtime, cancelled), VOXEL);
    worker.cancel(cancelled);
    let built = finished(&mut worker);

    assert_eq!(built.len(), 1);
    assert_eq!(
        Some(&built[0]),
        volume_mesh(kept, |at| runtime.volume("caves", at), VOXEL).as_ref()
    );
    assert!(!worker.is_building(cancelled));
}

//! Aquifer stages: fluid in a volume's empty space, pooled per jittered cell at the cell's own
//! level, as Minecraft's aquifers decide it, so caves below one pool's level do not all flood.

use std::sync::Arc;
use wave_forge::stages::{Pack, PackError, Runtime, Volume};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    noises: {"caves": (noise_type: SimplexSmooth, seed: 4, frequency: 0.1, fractal_octaves: 2)},
    stages: [
        (name: "rock", kind: Volume(
            density: Min(Sub(Constant(12.0), Z), Mul(FastNoise("caves"), Constant(6.0))),
            bottom: -24,
            top: 16,
        )),
        (name: "fluid", kind: Aquifer(volume: "rock", cell: (8, 6), level: (-24.0, 4.0),
            materials: Some((rules: [(category: "lava", when: [Less(Z, Constant(-14.0))])], otherwise: "water")))),
    ],
)"#;

fn runtime(chunks: &[ChunkCoord]) -> Runtime {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        6,
        [8, 8],
    );
    let focus: Vec<FocusPoint> = chunks.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime.request(&focus, &["fluid"]).expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    runtime
}

fn area() -> Vec<ChunkCoord> {
    (-2..2)
        .flat_map(|y| (-2..2).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// Every voxel of `volume` with its column, level and height in cells.
fn voxels(volume: &Volume) -> impl Iterator<Item = (u32, u32, u32, f32)> + '_ {
    (0..volume.size[2]).flat_map(move |level| {
        (0..volume.size[1]).flat_map(move |y| {
            (0..volume.size[0])
                .map(move |x| (x, y, level, (volume.bottom + level as i32) as f32 + 0.5))
        })
    })
}

#[test]
fn fluid_fills_only_empty_rock() {
    let chunks = area();
    let runtime = runtime(&chunks);

    let mut wet = 0;
    for &chunk in &chunks {
        let rock = runtime.volume("rock", chunk).expect("an input");
        let fluid = runtime.volume("fluid", chunk).expect("generated");
        assert_eq!((fluid.size, fluid.bottom), (rock.size, rock.bottom));
        for (x, y, level, z) in voxels(fluid) {
            if fluid.get(x, y, level) > 0.0 {
                assert!(rock.get(x, y, level) < 0.0, "fluid in rock at {z}");
                wet += 1;
            }
        }
    }
    assert!(wet > 100, "only {wet} voxels of fluid");
}

#[test]
fn caves_below_one_pools_level_are_dry_in_another_pool() {
    let chunks = area();
    let runtime = runtime(&chunks);

    // Empty voxels below the highest level any pool can have, wet or dry.
    let (mut wet, mut dry) = (0, 0);
    for &chunk in &chunks {
        let rock = runtime.volume("rock", chunk).expect("an input");
        let fluid = runtime.volume("fluid", chunk).expect("generated");
        for (x, y, level, z) in voxels(fluid) {
            if rock.get(x, y, level) < 0.0 && z < 4.0 {
                if fluid.get(x, y, level) > 0.0 {
                    wet += 1;
                } else {
                    dry += 1;
                }
            }
        }
    }
    assert!(
        wet > 100 && dry > 100,
        "{wet} wet and {dry} dry cave voxels"
    );
}

#[test]
fn fluid_lies_under_its_pools_level_and_pools_under_the_lava_line_are_lava() {
    let chunks = area();
    let runtime = runtime(&chunks);

    // Fluid lies under its pool's level, and a pool is lava when its level is under -14, so no
    // fluid lies above the highest level and no lava above the lava line.
    let (mut water, mut lava) = (0, 0);
    for &chunk in &chunks {
        let fluid = runtime.volume("fluid", chunk).expect("generated");
        for (x, y, level, z) in voxels(fluid) {
            if fluid.get(x, y, level) <= 0.0 {
                continue;
            }
            assert!(z < 4.0, "fluid at {z}, above every pool");
            match fluid.material(x, y, level) {
                0 => {
                    assert!(z < -14.0, "lava at {z}, above the lava line");
                    lava += 1;
                }
                1 => water += 1,
                other => panic!("material {other}"),
            }
        }
    }
    assert!(water > 50 && lava > 50, "{water} water and {lava} lava");
}

#[test]
fn fluid_comes_back_the_same_and_chunks_agree_whatever_their_order() {
    let chunks = area();
    let forward = runtime(&chunks);
    let reversed: Vec<ChunkCoord> = chunks.iter().rev().copied().collect();
    let backward = runtime(&reversed);

    for &chunk in &chunks {
        assert_eq!(
            forward.volume("fluid", chunk),
            backward.volume("fluid", chunk)
        );
    }
}

#[test]
fn cells_of_nothing_levels_the_wrong_way_or_a_coarse_volume_are_refused() {
    let parse = |aquifer: &str, scale: u32| {
        Pack::parse(&format!(
            r#"(version: 1, stages: [
                (name: "rock", scale: {scale}, kind: Volume(density: Z, bottom: 0, top: 4)),
                (name: "fluid", kind: Aquifer(volume: "rock", {aquifer})),
            ])"#
        ))
    };

    let results = [
        parse("cell: (0, 4), level: (0.0, 1.0)", 1),
        parse("cell: (4, 0), level: (0.0, 1.0)", 1),
        parse("cell: (4, 4), level: (2.0, 1.0)", 1),
        parse("cell: (4, 4), level: (0.0, 1.0)", 2),
    ];

    for result in results {
        assert!(
            matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "fluid"),
            "{result:?}"
        );
    }
}

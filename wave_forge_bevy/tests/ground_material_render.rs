//! The ground material drawn by Bevy's own renderer: a chunk whose west half is one material and
//! east half another comes out in their palette colours, blended only around the border.
//!
//! ```text
//! cargo test -p wave_forge_bevy --release --test ground_material_render -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`d: it needs a device. The app is headless and renders into an image, which it reads
//! back from the GPU; the material is unlit and the camera does no tonemapping, so what comes back
//! is the palette's colours themselves.

use bevy::app::PluginsState;
use bevy::asset::RenderAssetUsages;
use bevy::camera::{ImageRenderTarget, RenderTarget, ScalingMode};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::view::Msaa;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wave_forge::stages::{Pack, Runtime};
use wave_forge::{ChunkCoord, FocusPoint, ground, ground_materials};
use wave_forge_bevy::materials::{WaveForgeMaterialsPlugin, ground_material_of, palette_image};
use wave_forge_bevy::stages::ground_mesh;

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Constant(0.0))),
        (name: "side", kind: Rules(rules: [(category: "west", when: [Less(X, Constant(4.0))])], otherwise: "east")),
    ],
)"#;

const SIZE: u32 = 64;
const CELL: f32 = 1.0;
const WEST: [u8; 3] = [230, 40, 30];
const EAST: [u8; 3] = [20, 60, 220];

#[derive(Resource, Default)]
struct Pixels(Option<Vec<u8>>);

#[test]
#[ignore = "needs a device; run with --ignored in release mode"]
fn a_chunks_materials_come_out_in_their_palette_colours() {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        1,
        [8, 8],
    );
    let chunk = ChunkCoord::new(0, 0, 0);
    runtime
        .request(&[FocusPoint::new(chunk, 1)], &["height", "side"])
        .expect("stages");
    runtime.run_until_idle().expect("the stages run");
    let mesh = ground(chunk, |at| runtime.field("height", at), [CELL; 3]).expect("a ground");
    let ids = ground_materials(chunk, |at| runtime.categories("side", at)).expect("materials");

    let mut app = App::new();
    app.add_plugins((DefaultPlugins, WaveForgeMaterialsPlugin))
        .init_resource::<Pixels>();
    // What `App::run` does before the first frame: let the renderer's device future resolve.
    let started = Instant::now();
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "the renderer did not become ready"
        );
    }
    app.finish();
    app.cleanup();
    let world = app.world_mut();
    let mut target = Image::new_fill(
        Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    target.texture_descriptor.usage |=
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING;
    let target = world.resource_mut::<Assets<Image>>().add(target);
    let palette = palette_image(&[
        Color::srgb_u8(WEST[0], WEST[1], WEST[2]),
        Color::srgb_u8(EAST[0], EAST[1], EAST[2]),
    ]);
    let palette = world.resource_mut::<Assets<Image>>().add(palette);
    let material = {
        let mut images = world.resource_mut::<Assets<Image>>();
        ground_material_of(
            &mesh,
            &ids,
            Vec3::ZERO,
            Vec3::splat(CELL),
            StandardMaterial {
                unlit: true,
                ..default()
            },
            palette,
            &mut images,
        )
    };
    let material = world
        .resource_mut::<Assets<wave_forge_bevy::materials::GroundMaterial>>()
        .add(material);
    let mesh = world.resource_mut::<Assets<Mesh>>().add(ground_mesh(&mesh));
    world.spawn((Mesh3d(mesh), MeshMaterial3d(material)));
    // The mesh spans the columns' centres, 0.5 to 8.5 cells; the camera looks straight down on it.
    world.spawn((
        Camera3d::default(),
        RenderTarget::Image(ImageRenderTarget::from(target.clone())),
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed {
                width: 8.0,
                height: 8.0,
            },
            ..OrthographicProjection::default_3d()
        }),
        Tonemapping::None,
        Msaa::Off,
        Transform::from_xyz(4.5, 20.0, 4.5).looking_at(Vec3::new(4.5, 0.0, 4.5), Vec3::NEG_Z),
    ));
    world.spawn(Readback::texture(target)).observe(
        |readback: On<ReadbackComplete>, mut pixels: ResMut<Pixels>| {
            pixels.0 = Some(readback.data.clone());
        },
    );

    let started = Instant::now();
    let mut frames = 0;
    // Pipelines compile over the first frames; keep reading until the picture has settled.
    while frames < 60 || app.world().resource::<Pixels>().0.is_none() {
        app.update();
        frames += 1;
        assert!(started.elapsed() < Duration::from_secs(60), "no picture");
    }

    let pixels = app
        .world()
        .resource::<Pixels>()
        .0
        .clone()
        .expect("read back");
    let pixel = |x: u32, y: u32| {
        let at = ((y * SIZE + x) * 4) as usize;
        [pixels[at], pixels[at + 1], pixels[at + 2]]
    };
    let near = |a: [u8; 3], b: [u8; 3]| a.iter().zip(b).all(|(a, b)| a.abs_diff(b) <= 2);
    let (left, right) = (pixel(4, SIZE / 2), pixel(SIZE - 5, SIZE / 2));
    let (west, east) = if near(left, WEST) {
        (4, SIZE - 5)
    } else {
        (SIZE - 5, 4)
    };
    println!("left {left:?}, right {right:?}");
    assert!(
        (near(left, WEST) && near(right, EAST)) || (near(left, EAST) && near(right, WEST)),
        "left {left:?}, right {right:?}"
    );
    // The border lies at x = 4 cells, 3.5 cells into the 8 the camera sees: only a band a cell
    // wide around it blends.
    for y in (0..SIZE).step_by(8) {
        for x in 0..SIZE {
            let from_west = if west < east { x } else { SIZE - 1 - x };
            let colour = pixel(x, y);
            if from_west < SIZE * 3 / 8 - SIZE / 16 {
                assert!(near(colour, WEST), "({x}, {y}): {colour:?}");
            } else if from_west > SIZE * 3 / 8 + SIZE / 8 + SIZE / 16 {
                assert!(near(colour, EAST), "({x}, {y}): {colour:?}");
            }
        }
    }
}

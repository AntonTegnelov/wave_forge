//! The ground material drawn by Bevy's own renderer: a chunk whose west half is one material and
//! east half another. With the flat look it comes out in their palette colours, blended only around
//! the border; with the default look the colours vary around the palette's.
//!
//! ```text
//! cargo test -p wave_forge_bevy --release --test ground_material_render -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`d: it needs a device. The app is headless and renders into an image, which it reads
//! back from the GPU; the material is unlit and the camera does no tonemapping, so what comes back
//! is the material's colours themselves.

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
use wave_forge_bevy::materials::{
    GroundLook, WaveForgeMaterialsPlugin, ground_channels_image, ground_material_of, palette_image,
};
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

/// The chunk drawn with `look`, seen from straight above: `SIZE` by `SIZE` pixels, RGBA. With
/// `channels`, the west half's vertices are fully hollow and the east half's fully wet.
fn render_halves(look: GroundLook, channels: bool) -> Vec<u8> {
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
    let mut material = {
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
    material.extension.look = look;
    if channels {
        let values: Vec<f32> = ids
            .iter()
            .flat_map(|&id| if id == 0 { [1.0, 0.0] } else { [0.0, 1.0] })
            .collect();
        let image = ground_channels_image(mesh.size, &values);
        material.extension.channels = world.resource_mut::<Assets<Image>>().add(image);
    }
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
    app.world()
        .resource::<Pixels>()
        .0
        .clone()
        .expect("read back")
}

fn pixel_of(pixels: &[u8], x: u32, y: u32) -> [u8; 3] {
    let at = ((y * SIZE + x) * 4) as usize;
    [pixels[at], pixels[at + 1], pixels[at + 2]]
}

#[test]
#[ignore = "needs a device; run with --ignored in release mode"]
fn with_the_flat_look_a_chunks_materials_come_out_in_their_palette_colours() {
    let pixels = render_halves(GroundLook::flat(), false);

    let pixel = |x: u32, y: u32| pixel_of(&pixels, x, y);
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

#[test]
#[ignore = "needs a device; run with --ignored in release mode"]
fn with_the_default_look_the_colours_vary_around_the_palettes() {
    let pixels = render_halves(GroundLook::default(), false);

    // The two quarters furthest from the border, which lies 3.5 cells into the 8 the camera sees.
    let (left, right): (Vec<[u8; 3]>, Vec<[u8; 3]>) = (0..SIZE)
        .flat_map(|y| (0..SIZE / 4).map(move |x| (x, y)))
        .map(|(x, y)| (pixel_of(&pixels, x, y), pixel_of(&pixels, SIZE - 1 - x, y)))
        .unzip();
    for (side, colours) in [("left", left), ("right", right)] {
        let mean = |channel: usize| {
            colours.iter().map(|c| f32::from(c[channel])).sum::<f32>() / colours.len() as f32
        };
        let palette = if mean(0) > mean(2) { WEST } else { EAST };
        let strongest = if palette == WEST { 0 } else { 2 };
        let values: Vec<u8> = colours.iter().map(|c| c[strongest]).collect();
        let spread = values.iter().max().expect("pixels") - values.iter().min().expect("pixels");
        let off =
            (mean(strongest) - f32::from(palette[strongest])).abs() / f32::from(palette[strongest]);
        println!(
            "{side}: spread {spread}, mean {:.1} against {}",
            mean(strongest),
            palette[strongest]
        );
        assert!(
            spread >= 8,
            "{side}: the colour does not vary, spread {spread}"
        );
        assert!(
            off < 0.2,
            "{side}: the mean is {:.0}% off the palette",
            off * 100.0
        );
    }
}

#[test]
#[ignore = "needs a device; run with --ignored in release mode"]
fn a_hollow_darkens_the_ground_and_wetness_darkens_it_less() {
    let look = GroundLook::flat();
    let pixels = render_halves(look, true);

    // The factors darken the linear colour, which the target stores as sRGB.
    let darkened = |colour: [u8; 3], by: f32| {
        let linear = Color::srgb_u8(colour[0], colour[1], colour[2]).to_linear();
        let dark = LinearRgba::rgb(linear.red * by, linear.green * by, linear.blue * by);
        let srgb = Srgba::from(dark).to_u8_array();
        [srgb[0], srgb[1], srgb[2]]
    };
    let hollow = darkened(WEST, 1.0 - look.cavity_darkening);
    let wet = darkened(EAST, 1.0 - look.wet_darkening);
    let near = |a: [u8; 3], b: [u8; 3]| a.iter().zip(b).all(|(a, b)| a.abs_diff(b) <= 2);
    let (left, right) = (
        pixel_of(&pixels, 4, SIZE / 2),
        pixel_of(&pixels, SIZE - 5, SIZE / 2),
    );
    println!("left {left:?}, right {right:?}; hollow {hollow:?}, wet {wet:?}");

    assert!(
        (near(left, hollow) && near(right, wet)) || (near(left, wet) && near(right, hollow)),
        "left {left:?}, right {right:?}; hollow {hollow:?}, wet {wet:?}"
    );
}

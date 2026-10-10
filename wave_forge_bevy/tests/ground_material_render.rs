//! The ground material drawn by Bevy's own renderer: a chunk whose west half is one material and
//! east half another. With the flat look it comes out in their palette colours, blended only around
//! the border; with the default look the colours vary around the palette's. The ground's channels
//! darken it where it is hollow or wet and tint it where it is covered, and a material's ramp
//! carries its colour toward another by the ground's cavity or height.
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
    GrassFade, GroundLook, GroundRamp, WaveForgeMaterialsPlugin, ground_channels_image,
    ground_material_of, palette_image, ramp_image,
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
/// `channels`, the vertices of the half of material id 0 take the first channels and the other
/// half's the second.
fn render_halves(look: GroundLook, channels: Option<[[f32; 3]; 2]>) -> Vec<u8> {
    render_halves_on("Constant(0.0)", look, channels)
}

/// What [`render_halves`] draws, on the ground of height `height`, an expression.
fn render_halves_on(height: &str, look: GroundLook, channels: Option<[[f32; 3]; 2]>) -> Vec<u8> {
    render_halves_with(height, look, channels, &[])
}

/// What [`render_halves_on`] draws, with the materials' `ramps`.
fn render_halves_with(
    height: &str,
    look: GroundLook,
    channels: Option<[[f32; 3]; 2]>,
    ramps: &[GroundRamp],
) -> Vec<u8> {
    let text = PACK.replace("Constant(0.0)", height);
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(&text).expect("a valid pack")),
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
    // The grass's fade reaches every ground material through its resource, as an app gives it.
    app.add_plugins((DefaultPlugins, WaveForgeMaterialsPlugin))
        .init_resource::<Pixels>()
        .insert_resource(look.cover_fade);
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
    material.extension.ramps = world.resource_mut::<Assets<Image>>().add(ramp_image(ramps));
    if let Some([first, second]) = channels {
        let values: Vec<f32> = ids
            .iter()
            .flat_map(|&id| if id == 0 { first } else { second })
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
    let pixels = render_halves(GroundLook::flat(), None);

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
    let pixels = render_halves(GroundLook::default(), None);

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
    // Fully hollow on one half, fully wet on the other.
    let pixels = render_halves(look, Some([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]));

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

/// The tint the cover gives: fully covered ground takes it whole beyond the grass.
const COVER: [u8; 3] = [40, 200, 60];

/// The halves' colours, left and right, with one half fully covered and the other bare, and the
/// grass fading out over `cover_fade`.
fn covered_halves(band: Vec2) -> ([u8; 3], [u8; 3]) {
    let look = GroundLook {
        cover_tint: 1.0,
        cover_colour: Color::srgb_u8(COVER[0], COVER[1], COVER[2])
            .to_linear()
            .to_vec4(),
        cover_fade: GrassFade {
            band,
            ..GrassFade::default()
        },
        ..GroundLook::flat()
    };
    let pixels = render_halves(look, Some([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]));
    (
        pixel_of(&pixels, 4, SIZE / 2),
        pixel_of(&pixels, SIZE - 5, SIZE / 2),
    )
}

fn near(a: [u8; 3], b: [u8; 3]) -> bool {
    a.iter().zip(b).all(|(a, b)| a.abs_diff(b) <= 2)
}

fn bare(colour: [u8; 3]) -> bool {
    near(colour, WEST) || near(colour, EAST)
}

#[test]
#[ignore = "needs a device; run with --ignored in release mode"]
fn covered_ground_beyond_the_grass_takes_the_cover_colour() {
    // The camera stands over the chunk, so all of it is beyond a fade that ends behind it.
    let (left, right) = covered_halves(Vec2::new(-2.0, -1.0));

    println!("left {left:?}, right {right:?}; cover {COVER:?}");
    assert!(
        (near(left, COVER) && bare(right)) || (bare(left) && near(right, COVER)),
        "left {left:?}, right {right:?}; cover {COVER:?}"
    );
}

#[test]
#[ignore = "needs a device; run with --ignored in release mode"]
fn covered_ground_where_the_blades_stand_keeps_its_colour() {
    let (left, right) = covered_halves(Vec2::new(100.0, 200.0));

    println!("left {left:?}, right {right:?}");
    assert!(bare(left) && bare(right), "left {left:?}, right {right:?}");
}

#[test]
#[ignore = "needs a device; run with --ignored in release mode"]
fn gullies_stripe_a_steep_slope_across_its_fall_line() {
    // Ground rising a cell a cell toward +x, so its fall line runs along x.
    let slope = "X";
    let spread = |gullies: f32| {
        let look = GroundLook {
            gullies,
            ..GroundLook::flat()
        };
        let pixels = render_halves_on(slope, look, None);
        // Down a line of pixels across the fall line, inside the west half.
        let reds: Vec<u8> = (4..SIZE - 4)
            .map(|y| pixel_of(&pixels, SIZE / 4, y)[0])
            .collect();
        reds.iter().max().expect("pixels") - reds.iter().min().expect("pixels")
    };

    let (without, with) = (spread(0.0), spread(1.0));

    println!("across the fall line the red spans {without} without gullies, {with} with them");
    assert!(without <= 2, "the plain slope varies by {without}");
    assert!(with >= 20, "the gullies vary the slope by only {with}");
}

/// What the ramps carry the west material toward.
const TOWARD: [u8; 3] = [40, 200, 60];

/// `colour` a `share` of the way to `toward`, mixed in linear light as the shader mixes it.
fn mixed(colour: [u8; 3], toward: [u8; 3], share: f32) -> [u8; 3] {
    let from = Color::srgb_u8(colour[0], colour[1], colour[2]).to_linear();
    let to = Color::srgb_u8(toward[0], toward[1], toward[2]).to_linear();
    let srgb = Srgba::from(from.mix(&to, share)).to_u8_array();
    [srgb[0], srgb[1], srgb[2]]
}

/// The west material's ramp, to [`TOWARD`] by `by`, and none for the east.
fn west_ramp(by: Vec4) -> [GroundRamp; 1] {
    [GroundRamp {
        toward: Color::srgb_u8(TOWARD[0], TOWARD[1], TOWARD[2]),
        by,
    }]
}

#[test]
#[ignore = "needs a device; run with --ignored in release mode"]
fn a_ramp_by_cavity_carries_a_hollow_material_to_its_far_colour() {
    // Both halves fully hollow, darkening nothing, so only the ramps change them.
    let look = GroundLook {
        cavity_darkening: 0.0,
        ..GroundLook::flat()
    };
    let pixels = render_halves_with(
        "Constant(0.0)",
        look,
        Some([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        &west_ramp(Vec4::new(0.0, 0.0, 1.0, 0.0)),
    );

    let (left, right) = (
        pixel_of(&pixels, 4, SIZE / 2),
        pixel_of(&pixels, SIZE - 5, SIZE / 2),
    );
    println!("left {left:?}, right {right:?}; toward {TOWARD:?}");
    assert!(
        (near(left, TOWARD) && near(right, EAST)) || (near(left, EAST) && near(right, TOWARD)),
        "left {left:?}, right {right:?}; toward {TOWARD:?}"
    );
}

#[test]
#[ignore = "needs a device; run with --ignored in release mode"]
fn a_ramp_by_height_goes_as_far_as_the_ground_stands_within_its_heights() {
    // The ground stands at 5, halfway up heights of 0 to 10.
    let look = GroundLook {
        ramp_heights: Vec2::new(0.0, 10.0),
        ..GroundLook::flat()
    };
    let pixels = render_halves_with(
        "Constant(5.0)",
        look,
        None,
        &west_ramp(Vec4::new(1.0, 0.0, 0.0, 0.0)),
    );

    let halfway = mixed(WEST, TOWARD, 0.5);
    let (left, right) = (
        pixel_of(&pixels, 4, SIZE / 2),
        pixel_of(&pixels, SIZE - 5, SIZE / 2),
    );
    println!("left {left:?}, right {right:?}; halfway {halfway:?}");
    assert!(
        (near(left, halfway) && near(right, EAST)) || (near(left, EAST) && near(right, halfway)),
        "left {left:?}, right {right:?}; halfway {halfway:?}"
    );
}

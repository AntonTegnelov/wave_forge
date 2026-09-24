//! Reference materials for what the stages build (docs/reference/bevy.md, "Ground materials").
//!
//! [`WaveForgeMaterialsPlugin`] registers them and embeds their shaders; a game that renders adds
//! it after Bevy's own plugins. [`GroundMaterial`] draws a chunk's ground with a colour per
//! material, blended between the ground's vertices, from [`WaveForgeStages::ground_materials`].

use crate::stages::WaveForgeStages;
use bevy_app::{App, Plugin};
use bevy_asset::{Asset, Assets, Handle, RenderAssetUsages, embedded_asset};
use bevy_color::{Color, ColorToPacked};
use bevy_image::Image;
use bevy_math::{Vec3, Vec4};
use bevy_pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial};
use bevy_reflect::TypePath;
use bevy_render::render_resource::{AsBindGroup, Extent3d, TextureDimension, TextureFormat};
use bevy_shader::ShaderRef;
use wave_forge::{ChunkCoord, GroundMesh};

/// How many colours a palette holds: one per category a Rules stage can have.
const PALETTE: usize = 256;

/// What [`GroundMaterial`] adds to a `StandardMaterial`: the chunk's grid, its material id per
/// ground vertex, and the palette.
#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct GroundMaterials {
    /// The chunk's corner on the ground plane (x, z) and a cell's width along x and z.
    #[uniform(100)]
    pub grid: Vec4,
    /// One texel per ground vertex, its material id / 255 in red.
    #[texture(101, sample_type = "float", filterable = false)]
    pub materials: Handle<Image>,
    /// A colour per material id, 256 by 1.
    #[texture(102, sample_type = "float", filterable = false)]
    pub palette: Handle<Image>,
}

impl MaterialExtension for GroundMaterials {
    fn fragment_shader() -> ShaderRef {
        "embedded://wave_forge_bevy/shaders/ground.wesl".into()
    }
}

/// A chunk's ground with a colour per material: Bevy's `StandardMaterial` for everything but the
/// base colour, which the ground's materials give.
pub type GroundMaterial = ExtendedMaterial<StandardMaterial, GroundMaterials>;

/// Registers the reference materials and their shaders. Add it after Bevy's rendering plugins.
pub struct WaveForgeMaterialsPlugin;

impl Plugin for WaveForgeMaterialsPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/ground.wesl");
        app.add_plugins(MaterialPlugin::<GroundMaterial>::default());
    }
}

/// A palette for [`GroundMaterial`]: `colours` by category index, the categories past its end
/// taking colours of their own from their index.
#[must_use]
pub fn palette_image(colours: &[Color]) -> Image {
    let texels: Vec<u8> = (0..PALETTE)
        .flat_map(|index| {
            colours
                .get(index)
                .copied()
                .unwrap_or_else(|| Color::hsl(index as f32 * 222.5 % 360.0, 0.35, 0.45))
                .to_srgba()
                .to_u8_array()
        })
        .collect();
    Image::new(
        Extent3d {
            width: PALETTE as u32,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        texels,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

/// The ground material of `chunk`, once its ground is built with materials: `base` for all but
/// the colour, and a colour per material from `palette`. Its id image goes into `images`.
#[must_use]
pub fn ground_material(
    stages: &WaveForgeStages,
    chunk: ChunkCoord,
    base: StandardMaterial,
    palette: Handle<Image>,
    images: &mut Assets<Image>,
) -> Option<GroundMaterial> {
    let (mesh, ids) = (stages.ground(chunk)?, stages.ground_materials(chunk)?);
    let settings = stages.settings();
    Some(ground_material_of(
        mesh,
        ids,
        stages.chunk_corner(chunk),
        settings.cell_size,
        base,
        palette,
        images,
    ))
}

/// The ground material of a chunk's ground `mesh` whose vertices' material ids are `ids`
/// ([`wave_forge::ground_materials`]), with its corner at `corner` and cells of `cell`: what
/// [`ground_material`] makes from the plugin's resource, for a ground built some other way.
#[must_use]
pub fn ground_material_of(
    mesh: &GroundMesh,
    ids: &[u8],
    corner: Vec3,
    cell: Vec3,
    base: StandardMaterial,
    palette: Handle<Image>,
    images: &mut Assets<Image>,
) -> GroundMaterial {
    let materials = Image::new(
        Extent3d {
            width: mesh.size[0],
            height: mesh.size[1],
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        ids.to_vec(),
        TextureFormat::R8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    GroundMaterial {
        base,
        extension: GroundMaterials {
            grid: Vec4::new(corner.x, corner.z, cell.x, cell.z),
            materials: images.add(materials),
            palette,
        },
    }
}

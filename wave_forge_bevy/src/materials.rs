//! Reference materials for what the stages build (docs/reference/bevy.md, "Ground materials" and
//! "Grass").
//!
//! [`WaveForgeMaterialsPlugin`] registers them and embeds their shaders; a game that renders adds
//! it after Bevy's own plugins. [`GroundMaterial`] draws a chunk's ground with a colour per
//! material, blended between the ground's vertices, from [`WaveForgeStages::ground_materials`].
//! [`GrassMaterial`] draws grass from a cover field over one mesh of blades every chunk shares,
//! swaying in the [`Wind`].

use crate::stages::WaveForgeStages;
use bevy_app::{App, Plugin, Update};
use bevy_asset::{Asset, Assets, Handle, RenderAssetUsages, embedded_asset};
use bevy_camera::primitives::Aabb;
use bevy_camera::visibility::NoAutoAabb;
use bevy_color::{Color, ColorToPacked};
use bevy_ecs::change_detection::DetectChanges;
use bevy_ecs::prelude::{Res, ResMut, Resource};
use bevy_image::Image;
use bevy_math::{IVec4, Vec3, Vec4};
use bevy_mesh::{Mesh, PrimitiveTopology};
use bevy_pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial};
use bevy_reflect::TypePath;
use bevy_render::render_resource::{
    AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat,
};
use bevy_shader::ShaderRef;
use wave_forge::stages::Field;
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
        embedded_asset!(app, "shaders/grass.wesl");
        app.add_plugins((
            MaterialPlugin::<GroundMaterial>::default(),
            MaterialPlugin::<GrassMaterial>::default(),
        ))
        .init_resource::<Wind>()
        .add_systems(Update, blow);
    }
}

/// The wind grass sways in: a direction along x and z, a strength in world units at a blade's tip,
/// and a speed. Changing it reaches every grass material in the next frame.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Wind(pub Vec4);

impl Default for Wind {
    /// Blowing gently along +x.
    fn default() -> Self {
        Self(Vec4::new(1.0, 0.0, 0.15, 1.5))
    }
}

/// Gives every grass material the wind, when it changes.
fn blow(wind: Res<Wind>, mut grass: ResMut<Assets<GrassMaterial>>) {
    if !wind.is_changed() {
        return;
    }
    for (_, material) in grass.iter_mut() {
        material.extension.settings.wind = wind.0;
    }
}

/// What the grass shader reads about a chunk's grass.
#[derive(ShaderType, Clone, Copy, Debug)]
pub struct GrassSettings {
    /// A cell's width along x and z, then a blade's height and width.
    pub cell_and_blade: Vec4,
    /// The wind, as [`Wind`] gives it.
    pub wind: Vec4,
    /// The chunk along x and z, then blades per column.
    pub chunk_and_count: IVec4,
    /// A blade's colour at its root, linear RGBA.
    pub base_colour: Vec4,
    /// A blade's colour at its tip, linear RGBA.
    pub tip_colour: Vec4,
}

/// What [`GrassMaterial`] adds to a `StandardMaterial`: the chunk's settings, the ground's height
/// per vertex and the cover per column.
#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct GrassMaterials {
    /// The chunk's grass: its grid, blades, wind and colours.
    #[uniform(100)]
    pub settings: GrassSettings,
    /// One texel per ground vertex, its height in world units.
    #[texture(101, sample_type = "float", filterable = false)]
    pub heights: Handle<Image>,
    /// One texel per column, its cover from 0 to 1.
    #[texture(102, sample_type = "float", filterable = false)]
    pub cover: Handle<Image>,
}

impl MaterialExtension for GrassMaterials {
    fn vertex_shader() -> ShaderRef {
        "embedded://wave_forge_bevy/shaders/grass.wesl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://wave_forge_bevy/shaders/grass.wesl".into()
    }

    /// The blades are placed in the vertex shader, which the prepass would not run.
    fn enable_prepass() -> bool {
        false
    }

    /// Grass casts no shadow: thousands of blades would cost the shadow pass more than their
    /// shadows add.
    fn enable_shadows() -> bool {
        false
    }
}

/// A chunk's grass: Bevy's `StandardMaterial` for lighting, the blades placed and coloured by the
/// grass shader.
pub type GrassMaterial = ExtendedMaterial<StandardMaterial, GrassMaterials>;

/// The mesh of blades every chunk's grass shares: `per_column` blades for each of `columns`, each a
/// triangle carrying its index in its second UV channel, from which the grass shader places it.
#[must_use]
pub fn grass_mesh(columns: [u32; 2], per_column: u32) -> Mesh {
    let blades = (columns[0] * columns[1] * per_column) as usize;
    let shape = [[-0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.0]];
    let positions: Vec<[f32; 3]> = (0..blades).flat_map(|_| shape).collect();
    let normals: Vec<[f32; 3]> = vec![[0.0, 1.0, 0.0]; positions.len()];
    let uvs: Vec<[f32; 2]> = positions.iter().map(|p| [p[0], p[1]]).collect();
    let blade: Vec<[f32; 2]> = (0..blades)
        .flat_map(|index| [[index as f32, 0.0]; 3])
        .collect();
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, blade)
}

/// A chunk's grass material, once its ground is built and `cover`, a field stage's product for the
/// chunk, has arrived: `per_column` blades a column, as its [`grass_mesh`] has, `base` for
/// lighting. Its images go into `images`.
#[must_use]
pub fn grass_material(
    stages: &WaveForgeStages,
    chunk: ChunkCoord,
    cover: &Field,
    per_column: u32,
    base: StandardMaterial,
    images: &mut Assets<Image>,
) -> Option<GrassMaterial> {
    let mesh = stages.ground(chunk)?;
    Some(grass_material_of(
        mesh,
        cover,
        stages.settings().cell_size,
        per_column,
        base,
        images,
    ))
}

/// The grass material of a chunk whose ground is `mesh` and cover `cover`, with cells of `cell`:
/// what [`grass_material`] makes from the plugin's resource, for a ground built some other way.
#[must_use]
pub fn grass_material_of(
    mesh: &GroundMesh,
    cover: &Field,
    cell: Vec3,
    per_column: u32,
    base: StandardMaterial,
    images: &mut Assets<Image>,
) -> GrassMaterial {
    let heights: Vec<u8> = mesh.heights.iter().flat_map(|h| h.to_le_bytes()).collect();
    let heights = Image::new(
        Extent3d {
            width: mesh.size[0],
            height: mesh.size[1],
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        heights,
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    let covered: Vec<u8> = cover
        .values
        .iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let covered = Image::new(
        Extent3d {
            width: cover.size[0],
            height: cover.size[1],
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        covered,
        TextureFormat::R8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    GrassMaterial {
        // A blade is one triangle, seen from both sides.
        base: StandardMaterial {
            cull_mode: None,
            ..base
        },
        extension: GrassMaterials {
            settings: GrassSettings {
                cell_and_blade: Vec4::new(cell.x, cell.z, 0.6, 0.1),
                wind: Wind::default().0,
                chunk_and_count: IVec4::new(mesh.chunk.x, mesh.chunk.y, per_column as i32, 0),
                base_colour: Vec4::new(0.03, 0.08, 0.01, 1.0),
                tip_colour: Vec4::new(0.26, 0.5, 0.07, 1.0),
            },
            heights: images.add(heights),
            cover: images.add(covered),
        },
    }
}

/// A chunk's grass's bounds, relative to its corner: the chunk and its heights, since the shader
/// moves the blades far from where the mesh has them. Insert both components on the grass's
/// entity: without [`NoAutoAabb`], Bevy replaces the bounds with the mesh's own whenever the
/// entity's mesh changes, and culls the chunk's grass wherever the mesh's corner is out of view.
#[must_use]
pub fn grass_bounds(stages: &WaveForgeStages, chunk: ChunkCoord) -> Option<(Aabb, NoAutoAabb)> {
    Some(grass_bounds_of(
        stages.ground(chunk)?,
        stages.settings().cell_size,
    ))
}

/// The bounds of grass over the ground `mesh` with cells of `cell`, relative to the chunk's corner.
#[must_use]
pub fn grass_bounds_of(mesh: &GroundMesh, cell: Vec3) -> (Aabb, NoAutoAabb) {
    let (low, high) = mesh
        .heights
        .iter()
        .fold((f32::MAX, f32::MIN), |(low, high), &h| {
            (low.min(h), high.max(h))
        });
    let max = Vec3::new(
        mesh.size[0] as f32 * cell.x,
        high + 2.0 * cell.y,
        mesh.size[1] as f32 * cell.z,
    );
    (
        Aabb::from_min_max(Vec3::new(0.0, low - cell.y, 0.0), max),
        NoAutoAabb,
    )
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

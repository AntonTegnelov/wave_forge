//! Grass over the ground (docs/reference/godot.md, "Grass").
//!
//! One MultiMesh of blades, every blade with the identity transform, is shared by all chunks: its
//! buffer is uploaded once. Each chunk near the followed position draws it as an instance of its
//! own with a copy of the grass material holding the chunk's cover and ground heights, and the
//! grass shader places each blade from its `INSTANCE_ID`. Grass casts no shadow and takes no GI.

use crate::gi::Gi;
use godot::classes::image::Format as ImageFormat;
use godot::classes::rendering_server::{
    ArrayType, MultimeshTransformFormat, PrimitiveType, ShadowCastingSetting,
};
use godot::classes::{Image, ImageTexture, Material, RenderingServer, Shader, ShaderMaterial};
use godot::obj::EngineEnum;
use godot::prelude::*;
use std::collections::HashMap;
use wave_forge::stages::Field;
use wave_forge::{ChunkCoord, GroundMesh};

/// The reference grass shader.
pub(crate) const GRASS_SHADER: &str = include_str!("shaders/grass.gdshader");

/// The shared blades, the material each chunk's is copied from, and each chunk's instance.
pub(crate) struct Grass {
    blade: Rid,
    multimesh: Rid,
    per_column: u32,
    template: Gd<ShaderMaterial>,
    chunks: HashMap<ChunkCoord, (Rid, Gd<ShaderMaterial>)>,
}

impl Grass {
    /// Blades for chunks of `columns`, `per_column` to a column, drawn with `material`, or with
    /// the reference grass shader when it is `None`.
    ///
    /// # Errors
    /// If `material` is not a `ShaderMaterial`.
    pub(crate) fn new(
        material: Option<&Gd<Material>>,
        per_column: u32,
        columns: [u32; 2],
    ) -> Result<Self, String> {
        let template = match material {
            None => {
                let mut shader = Shader::new_gd();
                shader.set_code(GRASS_SHADER);
                let mut material = ShaderMaterial::new_gd();
                material.set_shader(&shader);
                material
            }
            Some(material) => material
                .clone()
                .try_cast::<ShaderMaterial>()
                .map_err(|_| "grass_material must be a ShaderMaterial".to_owned())?,
        };
        let mut rendering = RenderingServer::singleton();
        let blade = rendering.mesh_create();
        let mut arrays = VarArray::new();
        arrays.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
        let vertices = PackedVector3Array::from(
            [
                Vector3::new(-0.5, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(0.5, 0.0, 0.0),
            ]
            .as_slice(),
        );
        arrays.set(ArrayType::VERTEX.ord() as usize, &vertices.to_variant());
        rendering.mesh_add_surface_from_arrays(blade, PrimitiveType::TRIANGLES, &arrays);
        let count = columns[0] * columns[1] * per_column;
        let multimesh = rendering.multimesh_create();
        rendering.multimesh_set_mesh(multimesh, blade);
        rendering
            .multimesh_allocate_data_ex(
                multimesh,
                count as i32,
                MultimeshTransformFormat::TRANSFORM_3D,
            )
            .done();
        let identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let buffer: Vec<f32> = (0..count).flat_map(|_| identity).collect();
        rendering.multimesh_set_buffer(multimesh, &PackedFloat32Array::from(buffer.as_slice()));
        Ok(Self {
            blade,
            multimesh,
            per_column,
            template,
            chunks: HashMap::new(),
        })
    }

    /// Frees the grass of chunks for which `keep` fails, then grows grass on at most `budget` of
    /// `candidates` that have none, in their order, when `ground` and `cover` give both. Returns
    /// how many chunks got grass.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update<'a>(
        &mut self,
        scenario: Rid,
        cell: Vector3,
        corner: impl Fn(ChunkCoord) -> Vector3,
        keep: impl Fn(ChunkCoord) -> bool,
        candidates: &[ChunkCoord],
        budget: usize,
        ground: impl Fn(ChunkCoord) -> Option<&'a GroundMesh>,
        cover: impl Fn(ChunkCoord) -> Option<&'a Field>,
    ) -> usize {
        let mut rendering = RenderingServer::singleton();
        let gone: Vec<ChunkCoord> = self
            .chunks
            .keys()
            .copied()
            .filter(|&chunk| !keep(chunk))
            .collect();
        for chunk in gone {
            if let Some((instance, _)) = self.chunks.remove(&chunk) {
                rendering.free_rid(instance);
            }
        }
        let mut grown = 0;
        for &chunk in candidates {
            if grown == budget {
                break;
            }
            if self.chunks.contains_key(&chunk) {
                continue;
            }
            let (Some(mesh), Some(cover)) = (ground(chunk), cover(chunk)) else {
                continue;
            };
            let material = self.material(chunk, mesh, cover, cell);
            let instance = rendering.instance_create2(self.multimesh, scenario);
            rendering
                .instance_set_transform(instance, Transform3D::new(Basis::IDENTITY, corner(chunk)));
            rendering.instance_geometry_set_material_override(instance, material.get_rid());
            rendering
                .instance_geometry_set_cast_shadows_setting(instance, ShadowCastingSetting::OFF);
            Gi::Off.apply(instance);
            // The blades are moved in the shader, so the bounds are the chunk's, not the blades'.
            let (low, high) = mesh
                .heights
                .iter()
                .fold((f32::MAX, f32::MIN), |(low, high), &h| {
                    (low.min(h), high.max(h))
                });
            let extent = Vector3::new(
                mesh.size[0] as f32 * cell.x,
                high - low + 2.0 * cell.y,
                mesh.size[1] as f32 * cell.z,
            );
            rendering.instance_set_custom_aabb(
                instance,
                Aabb::new(Vector3::new(0.0, low - cell.y, 0.0), extent),
            );
            self.chunks.insert(chunk, (instance, material));
            grown += 1;
        }
        grown
    }

    /// Frees every chunk's grass, keeping the blades.
    pub(crate) fn clear(&mut self) {
        let mut rendering = RenderingServer::singleton();
        for (_, (instance, _)) in self.chunks.drain() {
            rendering.free_rid(instance);
        }
    }

    /// The chunks that have grass.
    pub(crate) fn chunks(&self) -> impl Iterator<Item = ChunkCoord> + '_ {
        self.chunks.keys().copied()
    }

    /// A chunk's copy of the grass material.
    pub(crate) fn material_of(&self, chunk: ChunkCoord) -> Option<Gd<ShaderMaterial>> {
        self.chunks
            .get(&chunk)
            .map(|(_, material)| material.clone())
    }

    /// A copy of the template holding the chunk's ground heights and cover.
    fn material(
        &self,
        chunk: ChunkCoord,
        mesh: &GroundMesh,
        cover: &Field,
        cell: Vector3,
    ) -> Gd<ShaderMaterial> {
        let heights: Vec<u8> = mesh.heights.iter().flat_map(|h| h.to_le_bytes()).collect();
        let heights = Image::create_from_data(
            mesh.size[0] as i32,
            mesh.size[1] as i32,
            false,
            ImageFormat::RF,
            &PackedByteArray::from(heights.as_slice()),
        )
        .expect("an image of one float per vertex");
        let covered: Vec<u8> = cover
            .values
            .iter()
            .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        let covered = Image::create_from_data(
            cover.size[0] as i32,
            cover.size[1] as i32,
            false,
            ImageFormat::R8,
            &PackedByteArray::from(covered.as_slice()),
        )
        .expect("an image of one byte per column");
        let mut material = self.template.duplicate_resource();
        for (name, value) in [
            (
                "wave_forge_heights",
                ImageTexture::create_from_image(&heights).to_variant(),
            ),
            (
                "wave_forge_cover",
                ImageTexture::create_from_image(&covered).to_variant(),
            ),
            ("wave_forge_cell", Vector2::new(cell.x, cell.z).to_variant()),
            (
                "wave_forge_per_column",
                (self.per_column as i32).to_variant(),
            ),
            (
                "wave_forge_chunk",
                Vector2i::new(chunk.x, chunk.y).to_variant(),
            ),
        ] {
            material.set_shader_parameter(name, &value);
        }
        material
    }
}

impl Drop for Grass {
    fn drop(&mut self) {
        self.clear();
        let mut rendering = RenderingServer::singleton();
        rendering.free_rid(self.multimesh);
        rendering.free_rid(self.blade);
    }
}

/// Registers the global wind parameter the grass and vegetation shaders read, blowing along +x
/// gently, unless the project's settings declare it. The extension calls it once, as it loads.
pub(crate) fn ensure_wind() {
    if godot::classes::ProjectSettings::singleton().has_setting("shader_globals/wave_forge_wind") {
        return;
    }
    RenderingServer::singleton().global_shader_parameter_add(
        "wave_forge_wind",
        godot::classes::rendering_server::GlobalShaderParameterType::VEC4,
        &Vector4::new(1.0, 0.0, 0.15, 1.5).to_variant(),
    );
}

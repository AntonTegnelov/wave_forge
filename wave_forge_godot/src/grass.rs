//! Grass over the ground (docs/reference/godot.md, "Grass").
//!
//! A few MultiMeshes of blades, every blade with the identity transform, are shared by all chunks,
//! their buffers uploaded once: one of every blade a column has, and smaller ones of a column's
//! first blades by rank. A chunk draws the smallest that holds every blade the grass shader shows
//! at its nearest point to the followed position, where blades thin out with distance, so grass
//! far away costs less. Each chunk near the followed position draws it as an instance of its
//! own with a copy of the grass material holding the chunk's cover and ground heights, and the
//! ground's materials and palette when its ground has them, and the grass shader places each blade
//! from its `INSTANCE_ID`. Grass casts no shadow and takes no GI.

use crate::gi::Gi;
use godot::classes::image::Format as ImageFormat;
use godot::classes::rendering_server::{
    ArrayType, MultimeshTransformFormat, PrimitiveType, ShadowCastingSetting,
};
use godot::classes::{Image, ImageTexture, Material, RenderingServer, ShaderMaterial};
use godot::obj::EngineEnum;
use godot::prelude::*;
use std::collections::HashMap;
use wave_forge::stages::Field;
use wave_forge::{ChunkCoord, GrassThinning, GroundMesh, grass_blades_needed};

/// Where the grass fades out, as the reference grass and ground shaders read it: by how many chunks
/// a point lies from the followed chunk's centre along the farther axis, so the band follows the
/// square of chunks that have grass wherever the camera is.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GrassFade {
    /// The chunks from the followed chunk's centre over which blades shrink away.
    pub(crate) band: Vector2,
    /// A chunk's width along x and z, in world units.
    pub(crate) span: Vector2,
    /// The followed chunk's centre on the ground plane, in world units.
    pub(crate) focus: Vector2,
}

impl GrassFade {
    /// The fade of grass on the chunks within `radius` of the followed one, chunks `span` wide: over
    /// the outer chunk of the square they make, or the outer half of it for the followed chunk alone.
    pub(crate) fn of_radius(radius: u32, span: Vector2) -> Self {
        let end = radius as f32 + 0.5;
        Self {
            band: Vector2::new(end - (end / 2.0).min(1.0), end),
            span,
            focus: Vector2::ZERO,
        }
    }

    /// Hands the fade to a copy of the reference grass or ground shader.
    pub(crate) fn apply(&self, material: &mut Gd<ShaderMaterial>) {
        material.set_shader_parameter("wave_forge_fade", &self.band.to_variant());
        material.set_shader_parameter("wave_forge_span", &self.span.to_variant());
        material.set_shader_parameter("wave_forge_focus", &self.focus.to_variant());
    }
}

/// The shares of a column's blades the smaller MultiMeshes hold.
const TIERS: [f32; 4] = [1.0, 0.5, 0.25, 0.125];

/// How far, in world units along the ground, the camera moves before the chunks' tiers are chosen
/// again: each holds every blade the shader shows while the camera is within it.
const ROOM: f32 = 1.0;

/// The shared blades, the material each chunk's is copied from, and each chunk's instance and the
/// tier of blades it draws.
pub(crate) struct Grass {
    blade: Rid,
    /// Each tier's MultiMesh and its blades per column, the first of every blade.
    tiers: Vec<(Rid, u32)>,
    per_column: u32,
    /// How the shader thins the blades with distance; none for a grass material that does not.
    thinning: Option<GrassThinning>,
    /// Where blades shrink away.
    fade: GrassFade,
    /// Where the camera stood, along the ground, when the tiers were last chosen; none before it is
    /// known, when every chunk draws every blade.
    tiered_at: Option<Vector2>,
    template: Gd<ShaderMaterial>,
    chunks: HashMap<ChunkCoord, (Rid, Gd<ShaderMaterial>, usize)>,
}

impl Grass {
    /// Blades for chunks of `columns`, `per_column` to a column, drawn with `material`, or with
    /// the reference grass shader when it is `None`, shrinking away where `fade` says.
    ///
    /// # Errors
    /// If `material` is not a `ShaderMaterial`.
    pub(crate) fn new(
        material: Option<&Gd<Material>>,
        per_column: u32,
        columns: [u32; 2],
        fade: GrassFade,
    ) -> Result<Self, String> {
        let template = match material {
            None => {
                let mut material = ShaderMaterial::new_gd();
                material.set_shader(&crate::shaders::reference(
                    "grass.gdshader",
                    crate::shaders::GRASS,
                ));
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
        let thinning = match (
            template
                .get_shader_parameter("thin_distance")
                .try_to::<Vector2>(),
            template.get_shader_parameter("thin_keep").try_to::<f32>(),
        ) {
            (Ok(distance), Ok(keep)) => Some(GrassThinning {
                from: distance.x,
                to: distance.y,
                keep,
            }),
            // The reference shader's defaults, which its material holds no value for.
            _ if material.is_none() => Some(GrassThinning::REFERENCE),
            _ => None,
        };
        let mut tiers: Vec<(Rid, u32)> = Vec::new();
        for share in TIERS {
            let blades = ((per_column as f32 * share).ceil() as u32).max(1);
            if tiers.last().is_some_and(|&(_, last)| last == blades) {
                continue;
            }
            let count = columns[0] * columns[1] * blades;
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
            tiers.push((multimesh, blades));
        }
        Ok(Self {
            blade,
            tiers,
            per_column,
            thinning,
            fade,
            tiered_at: None,
            template,
            chunks: HashMap::new(),
        })
    }

    /// Where blades shrink away.
    pub(crate) fn fade(&self) -> GrassFade {
        self.fade
    }

    /// Follows the followed chunk's centre, `focus`, with every chunk's fade.
    pub(crate) fn set_focus(&mut self, focus: Vector2) {
        self.fade.focus = focus;
        for (_, material, _) in self.chunks.values_mut() {
            self.fade.apply(material);
        }
    }

    /// Follows the camera, at `camera` along the ground: once it has moved farther than `ROOM`
    /// since the tiers were chosen, every chunk's grass draws the tier of blades its new distance
    /// needs.
    pub(crate) fn set_camera(&mut self, camera: Vector2) {
        if self
            .tiered_at
            .is_some_and(|at| at.distance_to(camera) <= ROOM)
        {
            return;
        }
        self.tiered_at = Some(camera);
        let mut rendering = RenderingServer::singleton();
        let chunks: Vec<ChunkCoord> = self.chunks.keys().copied().collect();
        for chunk in chunks {
            let tier = self.tier_of(chunk);
            let (instance, material, drawn) = self.chunks.get_mut(&chunk).expect("a chunk listed");
            if *drawn != tier {
                let (multimesh, blades) = self.tiers[tier];
                rendering.instance_set_base(*instance, multimesh);
                material
                    .set_shader_parameter("wave_forge_per_column", &(blades as i32).to_variant());
                *drawn = tier;
            }
        }
    }

    /// The smallest tier of blades that holds every blade the shader shows on `chunk` while the
    /// camera stays within `ROOM` of where it stood when the tiers were chosen; every blade before
    /// the camera is known.
    fn tier_of(&self, chunk: ChunkCoord) -> usize {
        let Some(camera) = self.tiered_at else {
            return 0;
        };
        let blades: Vec<u32> = self.tiers.iter().map(|&(_, blades)| blades).collect();
        tier_for(
            chunk,
            camera,
            self.fade.span,
            self.thinning,
            self.per_column,
            &blades,
        )
    }

    /// Frees the grass of chunks for which `keep` fails, then grows grass on at most `budget` of
    /// `candidates` that have none, in their order, when `ground` and `cover` give both, its
    /// blades' bases taking the colours of the ground material `looks` gives, if any. Returns how
    /// many chunks got grass.
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
        looks: impl Fn(ChunkCoord) -> Option<&'a Gd<ShaderMaterial>>,
    ) -> usize {
        let mut rendering = RenderingServer::singleton();
        let gone: Vec<ChunkCoord> = self
            .chunks
            .keys()
            .copied()
            .filter(|&chunk| !keep(chunk))
            .collect();
        for chunk in gone {
            if let Some((instance, _, _)) = self.chunks.remove(&chunk) {
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
            let mut material = self.material(chunk, mesh, cover, cell);
            // The ground's materials and palette, from the copy of the ground shader its ground
            // is drawn with.
            if let Some(look) = looks(chunk) {
                for name in ["wave_forge_materials", "wave_forge_palette"] {
                    material.set_shader_parameter(name, &look.get_shader_parameter(name));
                }
                material.set_shader_parameter("wave_forge_ground_colours", &true.to_variant());
            }
            let tier = self.tier_of(chunk);
            let (multimesh, blades) = self.tiers[tier];
            material.set_shader_parameter("wave_forge_per_column", &(blades as i32).to_variant());
            let instance = rendering.instance_create2(multimesh, scenario);
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
            self.chunks.insert(chunk, (instance, material, tier));
            grown += 1;
        }
        grown
    }

    /// Frees a chunk's grass, if it has any, to be grown again on its ground built again.
    pub(crate) fn remove(&mut self, chunk: ChunkCoord) {
        if let Some((instance, _, _)) = self.chunks.remove(&chunk) {
            RenderingServer::singleton().free_rid(instance);
        }
    }

    /// Frees every chunk's grass, keeping the blades.
    pub(crate) fn clear(&mut self) {
        let mut rendering = RenderingServer::singleton();
        for (_, (instance, _, _)) in self.chunks.drain() {
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
            .map(|(_, material, _)| material.clone())
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
            ("wave_forge_blades", (self.per_column as i32).to_variant()),
            (
                "wave_forge_chunk",
                Vector2i::new(chunk.x, chunk.y).to_variant(),
            ),
        ] {
            material.set_shader_parameter(name, &value);
        }
        self.fade.apply(&mut material);
        material
    }
}

/// The smallest of the tiers of `blades` per column, from the most to the fewest, that holds every
/// blade the grass shader shows on `chunk` (`wave_forge::grass_blades_needed`) while the camera
/// stays within `ROOM` of `camera`, chunks `span` wide; the first, every blade, without thinning.
fn tier_for(
    chunk: ChunkCoord,
    camera: Vector2,
    span: Vector2,
    thinning: Option<GrassThinning>,
    per_column: u32,
    blades: &[u32],
) -> usize {
    let Some(thinning) = thinning else {
        return 0;
    };
    let needed = grass_blades_needed(
        chunk,
        [camera.x, camera.y],
        ROOM,
        [span.x, span.y],
        thinning,
        per_column,
    );
    blades
        .iter()
        .rposition(|&count| count >= needed)
        .unwrap_or(0)
}

impl Drop for Grass {
    fn drop(&mut self) {
        self.clear();
        let mut rendering = RenderingServer::singleton();
        for &(multimesh, _) in &self.tiers {
            rendering.free_rid(multimesh);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    const SPAN: Vector2 = Vector2::new(8.0, 8.0);
    const CAMERA: Vector2 = Vector2::new(4.0, 4.0);
    const TIERS: [u32; 3] = [8, 4, 2];

    fn tier(x: i32, thinning: Option<GrassThinning>) -> usize {
        tier_for(ChunkCoord::new(x, 0, 0), CAMERA, SPAN, thinning, 8, &TIERS)
    }

    #[test]
    fn a_chunk_draws_the_fewest_blades_that_hold_every_one_it_shows() {
        let thinning = Some(GrassThinning {
            from: 8.0,
            to: 30.0,
            keep: 0.3,
        });

        // Near, every blade; three out, about 0.65 of 8, 6, which only the full set holds; five
        // out, 0.3 of 8, 3, which 4 hold and 2 do not.
        assert_eq!(tier(0, thinning), 0);
        assert_eq!(tier(3, thinning), 0);
        assert_eq!(tier(5, thinning), 1);
    }

    #[test]
    fn the_fewest_blades_serve_once_the_thinning_keeps_few_enough() {
        let thinning = Some(GrassThinning {
            from: 8.0,
            to: 30.0,
            keep: 0.2,
        });

        assert_eq!(tier(20, thinning), 2);
    }

    #[test]
    fn a_grass_that_does_not_thin_draws_every_blade() {
        assert_eq!(tier(20, None), 0);
    }
}

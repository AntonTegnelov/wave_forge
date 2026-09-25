//! Far stand-ins for the world's chunks (docs/reference/godot.md, "Far proxies").
//!
//! Each generated chunk gets one mesh of coloured boxes ([`wave_forge::proxy_mesh`]) with its
//! coarser levels as its `lods`, drawn from `proxy_distance` on, so a far chunk costs one draw call
//! where its modules would cost one per module. A game hides its own near drawing of the chunk
//! where the proxy shows by making the proxy its `visibility_parent`.

use crate::gi::Gi;
use crate::lods::add_levelled_surface;
use godot::classes::base_material_3d::Flags as MaterialFlags;
use godot::classes::rendering_server::{ArrayType, ShadowCastingSetting, VisibilityRangeFadeMode};
use godot::classes::{RenderingServer, StandardMaterial3D};
use godot::obj::EngineEnum;
use godot::prelude::*;
use std::collections::HashMap;
use wave_forge::{ChunkCoord, ProxyMesh};

/// Each chunk's proxy, its mesh and its instance; `None` for a chunk with nothing to stand in for.
pub(crate) struct Proxies {
    material: Gd<StandardMaterial3D>,
    chunks: HashMap<ChunkCoord, Option<(Rid, Rid)>>,
}

impl Default for Proxies {
    fn default() -> Self {
        let mut material = StandardMaterial3D::new_gd();
        material.set_flag(MaterialFlags::ALBEDO_FROM_VERTEX_COLOR, true);
        // A game gives its colours as it sees them, in sRGB.
        material.set_flag(MaterialFlags::SRGB_VERTEX_COLOR, true);
        Self {
            material,
            chunks: HashMap::new(),
        }
    }
}

impl Proxies {
    /// The chunks that have been given their proxy.
    pub(crate) fn chunks(&self) -> impl Iterator<Item = ChunkCoord> + '_ {
        self.chunks.keys().copied()
    }

    /// The instance of `chunk`'s proxy, if it has one.
    pub(crate) fn instance(&self, chunk: ChunkCoord) -> Option<Rid> {
        self.chunks
            .get(&chunk)
            .copied()
            .flatten()
            .map(|(_, instance)| instance)
    }

    /// Gives `proxy`'s chunk its proxy at `corner` in `scenario`, drawn from `begin` on,
    /// replacing the one it had.
    pub(crate) fn build(&mut self, scenario: Rid, corner: Vector3, proxy: &ProxyMesh, begin: f32) {
        self.drop_chunk(proxy.chunk);
        let [finest, coarser @ ..] = proxy.levels.as_slice() else {
            unreachable!("a proxy has its finest level at least");
        };
        if finest.indices.is_empty() {
            self.chunks.insert(proxy.chunk, None);
            return;
        }
        let mut rendering = RenderingServer::singleton();
        let mesh = rendering.mesh_create();
        let mut arrays = VarArray::new();
        arrays.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
        let vector = |&[x, y, z]: &[f32; 3]| Vector3::new(x, y, z);
        let vertices: PackedVector3Array = proxy.positions.iter().map(vector).collect();
        let normals: PackedVector3Array = proxy.normals.iter().map(vector).collect();
        let colours: PackedColorArray = proxy
            .colours
            .iter()
            .map(|&[r, g, b, a]| Color::from_rgba(r, g, b, a))
            .collect();
        arrays.set(ArrayType::VERTEX.ord() as usize, &vertices.to_variant());
        arrays.set(ArrayType::NORMAL.ord() as usize, &normals.to_variant());
        arrays.set(ArrayType::COLOR.ord() as usize, &colours.to_variant());
        // Godot refuses a level with as many triangles as the surface, which half-filled boxes can
        // make of a chunk full of holes, and an empty one draws nothing.
        let coarser: Vec<(&[u32], f32)> = coarser
            .iter()
            .filter(|level| !level.indices.is_empty() && level.indices.len() < finest.indices.len())
            .map(|level| (level.indices.as_slice(), level.error))
            .collect();
        add_levelled_surface(mesh, &mut arrays, &finest.indices, &coarser);
        let instance = rendering.instance_create2(mesh, scenario);
        rendering.instance_set_transform(instance, Transform3D::new(Basis::IDENTITY, corner));
        rendering.instance_geometry_set_material_override(instance, self.material.get_rid());
        rendering.instance_geometry_set_cast_shadows_setting(instance, ShadowCastingSetting::OFF);
        Gi::Static.apply(instance);
        set_begin(instance, begin);
        self.chunks.insert(proxy.chunk, Some((mesh, instance)));
    }

    /// Draws every proxy from `begin` on.
    pub(crate) fn set_begin(&self, begin: f32) {
        for &(_, instance) in self.chunks.values().flatten() {
            set_begin(instance, begin);
        }
    }

    /// Frees `chunk`'s proxy.
    pub(crate) fn drop_chunk(&mut self, chunk: ChunkCoord) {
        if let Some(Some((mesh, instance))) = self.chunks.remove(&chunk) {
            let mut rendering = RenderingServer::singleton();
            rendering.free_rid(instance);
            rendering.free_rid(mesh);
        }
    }

    /// Frees every proxy, which belong to the rendering server rather than to the tree.
    pub(crate) fn clear(&mut self) {
        let chunks: Vec<ChunkCoord> = self.chunks().collect();
        for chunk in chunks {
            self.drop_chunk(chunk);
        }
    }
}

/// Draws `instance` from `begin` on, with no end.
fn set_begin(instance: Rid, begin: f32) {
    RenderingServer::singleton().instance_geometry_set_visibility_range(
        instance,
        begin,
        0.0,
        0.0,
        0.0,
        VisibilityRangeFadeMode::DISABLED,
    );
}

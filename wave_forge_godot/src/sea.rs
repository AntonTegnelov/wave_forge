//! The sea: a plane at the pack's water level under the followed chunk, drawn with the node's
//! `sea_material` through the rendering server (docs/reference/godot.md, "Ground and colliders").
//! It covers the pack's level only; a lake above it is not drawn.

use godot::classes::{Material, PlaneMesh, RenderingServer};
use godot::prelude::*;

/// A plane drawn at the sea's level, centred under the followed chunk.
pub(crate) struct Sea {
    /// Kept so the mesh the instance draws lives as long as it.
    _mesh: Gd<PlaneMesh>,
    instance: Rid,
    height: f32,
}

impl Sea {
    /// A plane `side` units square at `height` in Godot's world, drawn with `material` in
    /// `scenario`.
    pub(crate) fn new(material: &Gd<Material>, side: f32, height: f32, scenario: Rid) -> Self {
        let mut mesh = PlaneMesh::new_gd();
        mesh.set_size(Vector2::new(side, side));
        mesh.set_material(material);
        let instance = RenderingServer::singleton().instance_create2(mesh.get_rid(), scenario);
        Self {
            _mesh: mesh,
            instance,
            height,
        }
    }

    /// Centres the plane under `centre`, in Godot's world, at the sea's level.
    pub(crate) fn place(&self, centre: Vector3) {
        let at = Transform3D::IDENTITY.translated(Vector3::new(centre.x, self.height, centre.z));
        RenderingServer::singleton().instance_set_transform(self.instance, at);
    }
}

impl Drop for Sea {
    fn drop(&mut self) {
        RenderingServer::singleton().free_rid(self.instance);
    }
}

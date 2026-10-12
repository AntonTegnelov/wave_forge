//! The sea: a plane at the pack's water level under the followed chunk, drawn with the node's
//! `sea_material` through the rendering server (docs/reference/godot.md, "Ground and colliders").
//! It covers the pack's level only; a lake above it is not drawn. Where the chunks around the
//! followed one draw their own water, the sea's among it, the plane leaves a hole over them, so the
//! coasts there are the chunks' water, tinted by its depth, and the plane is the sea beyond.

use godot::classes::mesh::{ArrayType, PrimitiveType};
use godot::classes::{ArrayMesh, Material, RenderingServer};
use godot::obj::EngineEnum;
use godot::prelude::*;

/// The depth, in world units, every vertex of the plane carries in its second UV's x, as a water
/// mesh carries its own: deep enough that the reference water shader draws it as open sea.
const DEEP: f32 = 1000.0;

/// A plane drawn at the sea's level, centred under the followed chunk.
pub(crate) struct Sea {
    /// Kept so the mesh the instance draws lives as long as it.
    _mesh: Gd<ArrayMesh>,
    instance: Rid,
    height: f32,
    /// The hole left for the chunks' own water, along Godot's x and z; zero for none.
    pub(crate) hole: [f32; 2],
}

impl Sea {
    /// A plane `side` units square at `height` in Godot's world, with a hole `hole` units along
    /// Godot's x and z (none where either is 0) whose centre lies `shift` from the plane's, drawn
    /// with `material` in `scenario`.
    pub(crate) fn new(
        material: &Gd<Material>,
        side: f32,
        hole: [f32; 2],
        shift: [f32; 2],
        height: f32,
        scenario: Rid,
    ) -> Self {
        let (outer, inner) = (side / 2.0, [hole[0] / 2.0, hole[1] / 2.0]);
        let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        let mut vertices = PackedVector3Array::new();
        for (x, z) in corners {
            vertices.push(Vector3::new(x * outer, 0.0, z * outer));
        }
        let indices: Vec<i32> = if hole[0] > 0.0 && hole[1] > 0.0 {
            for (x, z) in corners {
                vertices.push(Vector3::new(
                    x * inner[0] + shift[0],
                    0.0,
                    z * inner[1] + shift[1],
                ));
            }
            // Each side of the ring, from the outer edge to the hole's, clockwise seen from above
            // as Godot's front faces are.
            (0..4)
                .flat_map(|side| {
                    let next = (side + 1) % 4;
                    [side, next, side + 4, next, next + 4, side + 4]
                })
                .collect()
        } else {
            vec![0, 1, 3, 1, 2, 3]
        };
        let count = vertices.len();
        let normals = PackedVector3Array::from(vec![Vector3::UP; count].as_slice());
        let depths = PackedVector2Array::from(vec![Vector2::new(DEEP, 0.0); count].as_slice());
        let mut arrays = VarArray::new();
        arrays.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
        arrays.set(ArrayType::VERTEX.ord() as usize, &vertices.to_variant());
        arrays.set(ArrayType::NORMAL.ord() as usize, &normals.to_variant());
        arrays.set(ArrayType::TEX_UV2.ord() as usize, &depths.to_variant());
        arrays.set(
            ArrayType::INDEX.ord() as usize,
            &PackedInt32Array::from(indices.as_slice()).to_variant(),
        );
        let mut mesh = ArrayMesh::new_gd();
        mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, &arrays);
        mesh.surface_set_material(0, material);
        let instance = RenderingServer::singleton().instance_create2(mesh.get_rid(), scenario);
        Self {
            _mesh: mesh,
            instance,
            height,
            hole,
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

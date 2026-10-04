//! A kit of modules read from Godot for the import that proposes its module set (N6): the meshes of
//! a `MeshLibrary`'s items, and a `MeshLibrary` made from a folder of scenes.

use godot::classes::mesh::{ArrayType as MeshArray, PrimitiveType};
use godot::classes::{
    CollisionShape3D, ConcavePolygonShape3D, DirAccess, Mesh, MeshLibrary, Node, Node3D,
    PackedScene, Shape3D, StaticBody3D, SurfaceTool,
};
use godot::prelude::*;

/// Two points of a kit's face count as one within this fraction of a cell.
pub(crate) const TOLERANCE: f32 = 1.0 / 16.0;

/// Each item of `library` by name, with its mesh's vertices in the lattice's frame: taken as a
/// `GridMap` centres it in a cell of `cell_size`, with the item's mesh transform, from 0 to 1 across
/// the cell; no vertices for an item without a mesh.
pub(crate) fn items(library: &Gd<MeshLibrary>, cell_size: Vector3) -> Vec<(String, Vec<[f32; 3]>)> {
    library
        .get_item_list()
        .as_slice()
        .iter()
        .map(|&item| {
            let name = library.get_item_name(item).to_string();
            let transform = library.get_item_mesh_transform(item);
            let mut positions = Vec::new();
            if let Some(mesh) = library.get_item_mesh(item) {
                for surface in 0..mesh.get_surface_count() {
                    let arrays = mesh.surface_get_arrays(surface);
                    let vertices = arrays
                        .get(MeshArray::VERTEX.ord() as usize)
                        .and_then(|vertices| vertices.try_to::<PackedVector3Array>().ok())
                        .unwrap_or_default();
                    for &vertex in vertices.as_slice() {
                        let at = transform * vertex;
                        // The lattice's x, y and up from Godot's x, z and y, 0 to 1 across the
                        // cell.
                        positions.push([
                            at.x / cell_size.x + 0.5,
                            at.z / cell_size.z + 0.5,
                            at.y / cell_size.y + 0.5,
                        ]);
                    }
                }
            }
            (name, positions)
        })
        .collect()
}

/// The collision shapes of `library`'s items, by item name, each as one shape centred on the cell as
/// a `GridMap` places its item, which a node's `set_collision_shape` takes: an item's only shape if
/// it has one at no offset, so a box stays a box, or else the faces of all of them where they stand,
/// as one concave shape. An item without shapes is left out.
pub(crate) fn item_shapes(library: &Gd<MeshLibrary>) -> Vec<(String, Gd<Shape3D>)> {
    let mut found = Vec::new();
    for &item in library.get_item_list().as_slice() {
        // Godot gives an item's shapes as a flat list: a shape, then its transform, for each.
        let listed: Vec<Variant> = library.get_item_shapes(item).iter_shared().collect();
        let shapes: Vec<(Gd<Shape3D>, Transform3D)> = listed
            .chunks(2)
            .filter_map(|pair| {
                Some((
                    pair[0].try_to::<Gd<Shape3D>>().ok()?,
                    pair.get(1)?.try_to::<Transform3D>().ok()?,
                ))
            })
            .collect();
        let name = library.get_item_name(item).to_string();
        match shapes.as_slice() {
            [] => {}
            [(shape, at)] if *at == Transform3D::IDENTITY => found.push((name, shape.clone())),
            _ => {
                let faces: PackedVector3Array = shapes
                    .iter()
                    .filter_map(|(shape, at)| Some((shape.get_debug_mesh()?, *at)))
                    .flat_map(|(mesh, at)| {
                        mesh.get_faces()
                            .as_slice()
                            .iter()
                            .map(|&corner| at * corner)
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let mut merged = ConcavePolygonShape3D::new_gd();
                merged.set_faces(&faces);
                found.push((name, merged.upcast()));
            }
        }
    }
    found
}

/// Every `CollisionShape3D` under a `StaticBody3D` in `node`, with where it stands relative to the
/// scene's root, `at` being where `node` stands: the shapes a scene's item carries, as Godot 4.8's
/// scene-to-MeshLibrary import takes them.
fn body_shapes(node: &Gd<Node>, at: Transform3D, in_body: bool) -> Vec<(Gd<Shape3D>, Transform3D)> {
    let mut found = Vec::new();
    for child in node.get_children().iter_shared() {
        let placed = child
            .clone()
            .try_cast::<Node3D>()
            .map_or(at, |child| at * child.get_transform());
        if in_body
            && let Ok(collision) = child.clone().try_cast::<CollisionShape3D>()
            && let Some(shape) = collision.get_shape()
        {
            found.push((shape, placed));
        }
        let body = in_body || child.clone().try_cast::<StaticBody3D>().is_ok();
        found.extend(body_shapes(&child, placed, body));
    }
    found
}

/// `items` as the kit the import reads.
pub(crate) fn modules(items: &[(String, Vec<[f32; 3]>)]) -> Vec<wave_forge::import::KitModule<'_>> {
    items
        .iter()
        .map(|(name, positions)| wave_forge::import::KitModule { name, positions })
        .collect()
}

/// A `MeshLibrary` of the scenes in `directory`, as `WaveForgeWorld.mesh_library_from_scenes`
/// describes.
pub(crate) fn library_from_scenes(directory: &str) -> Result<Gd<MeshLibrary>, String> {
    if !DirAccess::dir_exists_absolute(directory) {
        return Err(format!("no directory {directory}"));
    }
    let mut files: Vec<String> = DirAccess::get_files_at(directory)
        .as_slice()
        .iter()
        .map(ToString::to_string)
        .filter(|file| {
            [".tscn", ".scn", ".glb", ".gltf"]
                .iter()
                .any(|ext| file.ends_with(ext))
        })
        .collect();
    files.sort();
    let mut library = MeshLibrary::new_gd();
    for (id, file) in files.iter().enumerate() {
        let id = i32::try_from(id).expect("fewer scenes than an i32 counts");
        let path = format!("{}/{file}", directory.trim_end_matches('/'));
        let scene = try_load::<PackedScene>(&path).map_err(|error| format!("{path}: {error}"))?;
        let root = scene
            .instantiate()
            .ok_or_else(|| format!("{path} does not instantiate"))?;
        let meshes = crate::placements::meshes(&root, Transform3D::IDENTITY);
        let shapes = body_shapes(&root, Transform3D::IDENTITY, false);
        root.free();
        let name = file
            .rsplit_once('.')
            .map_or(file.as_str(), |(stem, _)| stem);
        library.create_item(id);
        library.set_item_name(id, name);
        let mut listed = VarArray::new();
        for (shape, at) in &shapes {
            listed.push(&shape.to_variant());
            listed.push(&at.to_variant());
        }
        library.set_item_shapes(id, &listed);
        if meshes.is_empty() {
            continue;
        }
        let mut tool = SurfaceTool::new_gd();
        tool.begin(PrimitiveType::TRIANGLES);
        for (mesh, at) in &meshes {
            for surface in 0..mesh.get_surface_count() {
                tool.append_from(mesh, surface, *at);
            }
        }
        let merged = tool
            .commit()
            .ok_or_else(|| format!("{path}: its meshes do not merge"))?;
        library.set_item_mesh(id, &merged.upcast::<Mesh>());
    }
    Ok(library)
}

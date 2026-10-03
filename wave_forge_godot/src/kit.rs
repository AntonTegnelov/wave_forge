//! A kit of modules read from Godot for the import that proposes its module set (N6): the meshes of
//! a `MeshLibrary`'s items, and a `MeshLibrary` made from a folder of scenes.

use godot::classes::mesh::{ArrayType as MeshArray, PrimitiveType};
use godot::classes::{
    DirAccess, Mesh, MeshInstance3D, MeshLibrary, Node, Node3D, PackedScene, SurfaceTool,
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
        let mut meshes = Vec::new();
        collect_meshes(&root, Transform3D::IDENTITY, &mut meshes);
        root.free();
        let name = file
            .rsplit_once('.')
            .map_or(file.as_str(), |(stem, _)| stem);
        library.create_item(id);
        library.set_item_name(id, name);
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

/// Every mesh under `node`, depth first, with where it sits relative to the scene's root, given
/// that `node` sits at `at`.
fn collect_meshes(node: &Gd<Node>, at: Transform3D, into: &mut Vec<(Gd<Mesh>, Transform3D)>) {
    if let Ok(instance) = node.clone().try_cast::<MeshInstance3D>()
        && let Some(mesh) = instance.get_mesh()
    {
        into.push((mesh, at));
    }
    for child in node.get_children().iter_shared() {
        let placed = child
            .clone()
            .try_cast::<Node3D>()
            .map_or(at, |child| at * child.get_transform());
        collect_meshes(&child, placed, into);
    }
}

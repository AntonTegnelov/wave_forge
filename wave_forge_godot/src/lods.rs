//! Meshes with levels of detail as Godot draws them: a surface of the finest triangles, with the
//! coarser levels as its `lods`, which Godot picks from by distance (docs/reference/godot.md,
//! "Ground and colliders").

use godot::classes::RenderingServer;
use godot::classes::rendering_server::{ArrayType, PrimitiveType};
use godot::obj::EngineEnum;
use godot::prelude::*;

/// Adds to mesh `rid` a surface of `arrays`, whose triangles are `finest`, with the `coarser`
/// levels, each its triangles and how far in world units it strays from the finest, as its `lods`.
/// Triangles are the library's, counter-clockwise seen from their front.
pub(crate) fn add_levelled_surface(
    rid: Rid,
    arrays: &mut VarArray,
    finest: &[u32],
    coarser: &[(&[u32], f32)],
) {
    arrays.set(
        ArrayType::INDEX.ord() as usize,
        &godot_triangles(finest).to_variant(),
    );
    let errors: Vec<f32> = coarser.iter().map(|&(_, error)| error).collect();
    let mut lods = VarDictionary::new();
    // A coarser level overwrites a finer one of the same key, so of levels that stray alike the
    // coarsest is drawn.
    for (key, &(indices, _)) in lod_keys(&errors).into_iter().zip(coarser) {
        lods.set(key, &godot_triangles(indices).to_variant());
    }
    RenderingServer::singleton()
        .mesh_add_surface_from_arrays_ex(rid, PrimitiveType::TRIANGLES, &*arrays)
        .lods(&lods)
        .done();
}

/// Triangles for Godot: the library's are counter-clockwise seen from their front, Godot's
/// clockwise.
fn godot_triangles(indices: &[u32]) -> PackedInt32Array {
    indices
        .chunks(3)
        .flat_map(|triangle| [triangle[0], triangle[2], triangle[1]])
        .map(|index| index as i32)
        .collect()
}

/// The keys of the coarser levels in a surface's `lods`, from their errors in world units. Godot
/// draws a level while its key, projected to the screen, stays under the viewport's
/// `mesh_lod_threshold` in pixels, and stops at the first level that does not, in order of key; so
/// a key is at least every finer level's. It skips a key that is not positive, and a level that
/// strays nowhere is drawn at every distance, so such a key is the smallest positive float.
fn lod_keys(errors: &[f32]) -> Vec<f32> {
    errors
        .iter()
        .scan(f32::MIN_POSITIVE, |key, &error| {
            *key = key.max(error);
            Some(*key)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lod_keys_never_fall_below_a_finer_levels() {
        let keys = lod_keys(&[0.5, 0.25, 2.0]);

        assert_eq!(keys, [0.5, 0.5, 2.0]);
    }

    #[test]
    fn a_level_that_strays_nowhere_gets_a_positive_key() {
        let keys = lod_keys(&[0.0, 0.0]);

        assert_eq!(keys, [f32::MIN_POSITIVE, f32::MIN_POSITIVE]);
    }
}

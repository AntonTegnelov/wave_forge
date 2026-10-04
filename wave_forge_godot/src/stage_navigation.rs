//! Navigation for a world of stages: one region per chunk, baked on the navigation server's threads
//! from the triangles agents walk on in the chunk and, out to a border, in its neighbours
//! (`wave_forge::surface_nav_source`), so neighbouring regions meet on edges built from the same
//! geometry, as `WaveForgeWorld`'s do.

use godot::classes::{NavigationMesh, NavigationMeshSourceGeometryData3D, NavigationServer3D};
use godot::prelude::*;
use std::collections::HashMap;
use wave_forge::{ChunkCoord, NavSource, NavSourceError};

/// A region's newest mesh on its way into the navigation map. Since Godot 4.4 a region makes its
/// polygons of a new mesh on a later sync, and the map takes them in on a later iteration of its
/// own, built on a thread of its own, so a path asked for as soon as the mesh is set, on
/// `navigation_ready` say, misses it: in `verify.gd` no chunk had a path then, nor after six forced
/// updates of the map in the same frame, nor once the region's iteration had moved on. So the map is
/// asked directly: the mesh is in once the map says the region owns the centre of one of the mesh's
/// polygons. A mesh without polygons has nothing to wait for. A chunk baked again keeps its old
/// mesh, of the same region, until the new one is in; a path over it is found either way.
#[derive(Clone, Copy)]
pub(crate) struct Arrival {
    /// The centre of the mesh's first polygon, if it has one.
    probe: Option<Vector3>,
}

impl Arrival {
    /// The arrival of `mesh`, just set on its region.
    pub(crate) fn of(mesh: &Gd<NavigationMesh>) -> Self {
        let vertices = mesh.get_vertices();
        let probe = (mesh.get_polygon_count() > 0).then(|| {
            let corners = mesh.get_polygon(0);
            let sum = corners
                .as_slice()
                .iter()
                .fold(Vector3::ZERO, |sum, &corner| {
                    sum + vertices[corner as usize]
                });
            sum / corners.len() as f32
        });
        Self { probe }
    }

    /// Whether the mesh is in `map` now, a path over it found from now on. A map not yet
    /// synchronised once holds nothing, and asking it would be reported as an error.
    pub(crate) fn arrived(self, server: &Gd<NavigationServer3D>, map: Rid, region: Rid) -> bool {
        self.probe.is_none_or(|probe| {
            server.map_get_iteration_id(map) > 0
                && server.map_get_closest_point_owner(map, probe) == region
        })
    }
}

/// Each chunk's navigation region and the bake that fills it.
pub(crate) struct StageNavigation<S> {
    chunks: HashMap<ChunkCoord, Region<S>>,
    /// How many bakes have finished and gone into their regions.
    pub(crate) baked: i64,
}

/// A chunk's region, and the mesh baked, or being baked, for it from what `from` describes.
struct Region<S> {
    region: Rid,
    mesh: Gd<NavigationMesh>,
    baking: bool,
    /// Whether a mesh baked for the chunk is in the map, this one or an earlier one.
    in_map: bool,
    /// The newest mesh on its way into the map, once it is set on the region.
    arriving: Option<Arrival>,
    from: S,
}

impl<S: PartialEq> StageNavigation<S> {
    pub(crate) fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            baked: 0,
        }
    }

    /// The chunks whose latest mesh is in the map; a chunk being baked again keeps its last one
    /// until the new one is done.
    pub(crate) fn chunks(&self) -> impl Iterator<Item = ChunkCoord> + '_ {
        self.chunks
            .iter()
            .filter(|(_, chunk)| chunk.in_map)
            .map(|(&coord, _)| coord)
    }

    /// Frees every region, which belongs to the navigation server rather than to the node.
    pub(crate) fn clear(&mut self) {
        let mut server = NavigationServer3D::singleton();
        for (_, chunk) in self.chunks.drain() {
            server.free_rid(chunk.region);
        }
    }

    /// Frees the regions of chunks no longer `wanted`, and puts every finished bake in its region.
    ///
    /// Returns the chunks whose newest mesh `map` has taken in since the last call ([`Arrival`]), a
    /// path over them found from now on.
    pub(crate) fn settle(&mut self, wanted: &[ChunkCoord], map: Rid) -> Vec<ChunkCoord> {
        let mut server = NavigationServer3D::singleton();
        let gone: Vec<ChunkCoord> = self
            .chunks
            .keys()
            .copied()
            .filter(|chunk| !wanted.contains(chunk))
            .collect();
        for coord in gone {
            if let Some(chunk) = self.chunks.remove(&coord) {
                server.free_rid(chunk.region);
            }
        }
        let mut ready = Vec::new();
        for (&coord, chunk) in &mut self.chunks {
            if chunk.baking && !server.is_baking_navigation_mesh(&chunk.mesh) {
                server.region_set_navigation_mesh(chunk.region, &chunk.mesh);
                chunk.arriving = Some(Arrival::of(&chunk.mesh));
                chunk.baking = false;
                self.baked += 1;
            }
            if let Some(arrival) = chunk.arriving
                && arrival.arrived(&server, map, chunk.region)
            {
                chunk.arriving = None;
                chunk.in_map = true;
                ready.push(coord);
            }
        }
        ready
    }

    /// Starts one bake, of the chunk of `wanted` nearest `focus` that is not being baked and has
    /// no mesh, or one baked from something other than what `from` says the chunk holds now: from
    /// the source `source` gathers with a border as wide as the template needs. A chunk whose
    /// source is missing a neighbour waits for it. One bake at a time, since preparing one costs
    /// Godot's thread up to a millisecond.
    ///
    /// # Errors
    /// What `source` gives other than a missing neighbour.
    pub(crate) fn start_bake(
        &mut self,
        map: Rid,
        template: Option<&Gd<NavigationMesh>>,
        mut wanted: Vec<ChunkCoord>,
        focus: ChunkCoord,
        from: impl Fn(ChunkCoord) -> S,
        source: impl Fn(ChunkCoord, f32) -> Result<NavSource, NavSourceError>,
    ) -> Result<(), NavSourceError> {
        let mut server = NavigationServer3D::singleton();
        server.map_set_use_edge_connections(map, false);
        let distance =
            |chunk: &ChunkCoord| (chunk.x - focus.x).abs().max((chunk.y - focus.y).abs());
        wanted.sort_by_key(|chunk| (distance(chunk), *chunk));
        let cell_size = server.map_get_cell_size(map);
        let template = template.cloned().unwrap_or_else(NavigationMesh::new_gd);
        // Recast's own padding for tiles: the agent's radius in whole cells, and three more.
        let border = ((template.get_agent_radius() / cell_size).ceil() + 3.0) * cell_size;
        for coord in wanted {
            let built_from = from(coord);
            if self
                .chunks
                .get(&coord)
                .is_some_and(|chunk| chunk.baking || chunk.from == built_from)
            {
                continue;
            }
            let nav = match source(coord, border) {
                Ok(nav) => nav,
                Err(NavSourceError::Missing(_)) => continue,
                Err(error) => return Err(error),
            };
            let corners = i32::try_from(nav.triangles.len() / 3).expect("a source fits i32");
            let indices: Vec<i32> = (0..corners).collect();
            let mut geometry = NavigationMeshSourceGeometryData3D::new_gd();
            geometry.set_vertices(&PackedFloat32Array::from(nav.triangles.as_slice()));
            geometry.set_indices(&PackedInt32Array::from(indices.as_slice()));
            let mut mesh = template.duplicate_resource();
            mesh.set_cell_size(cell_size);
            mesh.set_cell_height(server.map_get_cell_height(map));
            mesh.set_filter_baking_aabb(Aabb::new(
                Vector3::from_array(nav.bounds_origin),
                Vector3::from_array(nav.bounds_size),
            ));
            mesh.set_border_size(nav.border);
            server.bake_from_source_geometry_data_async(&mesh, &geometry);
            let (region, in_map) = match self.chunks.remove(&coord) {
                Some(existing) => (existing.region, existing.in_map),
                None => {
                    let region = server.region_create();
                    server.region_set_map(region, map);
                    server.region_set_enabled(region, true);
                    // Given new meshes while an asynchronous iteration of it was under way, a region
                    // stopped the map from synchronising at all in Godot 4.7.2 (verify_world.gd
                    // found no path across chunks in 4 runs of 4 with it, and one in each of 3
                    // without). A chunk's region is small, so the map's own iteration takes it in.
                    server.region_set_use_async_iterations(region, false);
                    (region, false)
                }
            };
            self.chunks.insert(
                coord,
                Region {
                    region,
                    mesh,
                    baking: true,
                    in_map,
                    arriving: None,
                    from: built_from,
                },
            );
            break;
        }
        Ok(())
    }
}

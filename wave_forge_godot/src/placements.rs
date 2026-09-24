//! Scenes bound to what Scatter and Assemble stages place (docs/reference/godot.md).
//!
//! A kind (a Scatter point's kind, or an Assemble piece's name) is bound to a scene. A scene whose
//! root is a lone `MeshInstance3D` without a script is drawn as one `RenderingServer` MultiMesh per
//! chunk and kind, never as nodes; any other scene is instantiated as nodes, off the tree, and
//! attached under a per-frame time budget nearest the followed position first. With promotion, a
//! scene placed as nodes is so only near the followed position; farther out it is drawn as a
//! MultiMesh of its first mesh, and a chunk crossing the radius is placed again. A chunk that is
//! dropped takes its MultiMeshes and nodes with it.

use godot::classes::rendering_server::MultimeshTransformFormat;
use godot::classes::{
    Mesh, MeshInstance3D, Node, Node3D, PackedScene, RenderingServer, ResourceLoader,
};
use godot::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use wave_forge::ChunkCoord;

/// `ResourceLoader.ThreadLoadStatus` values.
const IN_PROGRESS: i64 = 1;
const LOADED: i64 = 3;

/// What a kind is drawn with.
enum Binding {
    /// A scene still loading on Godot's loader threads, by its path.
    Loading(GString),
    /// A lone mesh, drawn as a MultiMesh per chunk.
    Mesh(Gd<Mesh>),
    /// A scene instantiated as nodes, and its first mesh with where it sits in the scene, which
    /// stands in for it beyond the promotion radius.
    Nodes(Gd<PackedScene>, Option<(Gd<Mesh>, Transform3D)>),
}

/// One thing to place: its kind, where it stands in Godot's world, and its id within its chunk.
pub(crate) struct Item {
    pub(crate) kind: String,
    pub(crate) transform: Transform3D,
    pub(crate) id: i64,
}

/// What a stage's chunk placed: its MultiMeshes with their instances and how many each draws,
/// and its nodes.
#[derive(Default)]
struct Placed {
    multimeshes: Vec<(Rid, Rid, usize)>,
    nodes: Vec<Gd<Node3D>>,
    /// Whether it was placed within the promotion radius.
    near: bool,
}

/// Every kind's binding, the chunks waiting to be placed, and what each placed.
#[derive(Default)]
pub(crate) struct Placements {
    bindings: BTreeMap<String, Binding>,
    due: BTreeSet<(String, ChunkCoord)>,
    placed: HashMap<(String, ChunkCoord), Placed>,
}

/// A node placed: the node, its chunk and its id, for `instance_spawned`.
pub(crate) type Spawned = (Gd<Node3D>, ChunkCoord, i64);

impl Placements {
    /// The bindings of `scenes`, kind to a `PackedScene` or a path to one, whose loading starts
    /// on Godot's loader threads.
    ///
    /// # Errors
    /// Naming the kind whose value is neither, or whose path cannot be loaded.
    pub(crate) fn new(scenes: &VarDictionary) -> Result<Self, String> {
        let mut bindings = BTreeMap::new();
        for (kind, value) in scenes.iter_shared() {
            let kind = kind.to::<GString>().to_string();
            let binding = if let Ok(scene) = value.try_to::<Gd<PackedScene>>() {
                classify(&scene)
            } else if let Ok(path) = value.try_to::<GString>() {
                // godot-rust leaves the threaded loader out of builds with its thread safeguards,
                // because a Rust resource loaded on a loader thread aborts the process
                // (docs/architecture/engine-integration.md); a plain scene holds none.
                let started = ResourceLoader::singleton().call(
                    "load_threaded_request",
                    &[path.to_variant(), "PackedScene".to_variant()],
                );
                if started.try_to::<i64>().ok() != Some(0) {
                    return Err(format!("scene of {kind:?}: {path} could not be loaded"));
                }
                Binding::Loading(path)
            } else {
                return Err(format!(
                    "scene of {kind:?}: a PackedScene or a path to one, not {value}"
                ));
            };
            bindings.insert(kind, binding);
        }
        Ok(Self {
            bindings,
            ..Self::default()
        })
    }

    /// Whether any kind is bound, so the node looks for things to place at all.
    pub(crate) fn any(&self) -> bool {
        !self.bindings.is_empty()
    }

    /// The kinds bound to a scene.
    pub(crate) fn kinds(&self) -> Vec<String> {
        self.bindings.keys().cloned().collect()
    }

    /// A stage's chunk has arrived: it is placed in a later frame.
    pub(crate) fn arrived(&mut self, stage: &str, chunk: ChunkCoord) {
        self.due.insert((stage.to_owned(), chunk));
    }

    /// A stage's chunk was dropped: what it placed is freed.
    pub(crate) fn dropped(&mut self, stage: &str, chunk: ChunkCoord) {
        let key = (stage.to_owned(), chunk);
        self.due.remove(&key);
        if let Some(placed) = self.placed.remove(&key) {
            free(placed);
        }
    }

    /// How many nodes and how many MultiMesh instances are placed.
    pub(crate) fn counts(&self) -> (usize, usize) {
        self.placed
            .values()
            .fold((0, 0), |(nodes, instances), placed| {
                (
                    nodes + placed.nodes.len(),
                    instances
                        + placed
                            .multimeshes
                            .iter()
                            .map(|&(_, _, count)| count)
                            .sum::<usize>(),
                )
            })
    }

    /// Chunks waiting to be placed.
    pub(crate) fn pending(&self) -> usize {
        self.due.len()
    }

    /// Places due chunks, and chunks that crossed the promotion radius, nearest `focus` first,
    /// until `budget_ms` is spent: each chunk's items as `items` gives them, a node scene's as nodes
    /// only where `near` holds. Waits while any scene is still loading. Returns the nodes placed.
    ///
    /// # Errors
    /// Naming a scene that failed to load or whose root is not a `Node3D`.
    pub(crate) fn place(
        &mut self,
        holder: &mut Gd<Node>,
        scenario: Rid,
        focus: ChunkCoord,
        budget_ms: f64,
        near: impl Fn(ChunkCoord) -> bool,
        items: impl Fn(&str, ChunkCoord) -> Option<Vec<Item>>,
    ) -> Result<Vec<Spawned>, String> {
        let started = std::time::Instant::now();
        if !self.loaded()? {
            return Ok(Vec::new());
        }
        let crossed: Vec<(String, ChunkCoord)> = self
            .placed
            .iter()
            .filter(|((_, chunk), placed)| placed.near != near(*chunk))
            .map(|(key, _)| key.clone())
            .collect();
        self.due.extend(crossed);
        let mut due: Vec<(String, ChunkCoord)> = self.due.iter().cloned().collect();
        due.sort_by_key(|(stage, chunk)| {
            let distance = (chunk.x - focus.x).abs().max((chunk.y - focus.y).abs());
            (distance, *chunk, stage.clone())
        });
        let mut spawned = Vec::new();
        for (stage, chunk) in due {
            if started.elapsed().as_secs_f64() * 1000.0 >= budget_ms {
                break;
            }
            let key = (stage, chunk);
            self.due.remove(&key);
            // Dropped since it arrived: nothing to place.
            let Some(items) = items(&key.0, chunk) else {
                continue;
            };
            let mut by_kind: BTreeMap<&str, Vec<&Item>> = BTreeMap::new();
            for item in &items {
                by_kind.entry(&item.kind).or_default().push(item);
            }
            let mut placed = Placed {
                near: near(chunk),
                ..Placed::default()
            };
            for (kind, items) in by_kind {
                match &self.bindings[kind] {
                    Binding::Mesh(mesh) => placed.multimeshes.push(multimesh(
                        mesh,
                        Transform3D::IDENTITY,
                        &items,
                        scenario,
                    )),
                    Binding::Nodes(_, proxy) if !placed.near => {
                        if let Some((mesh, offset)) = proxy {
                            placed
                                .multimeshes
                                .push(multimesh(mesh, *offset, &items, scenario));
                        }
                    }
                    Binding::Nodes(scene, _) => {
                        for item in items {
                            let mut node = scene
                                .try_instantiate_as::<Node3D>()
                                .ok_or_else(|| format!("the scene of {kind:?} is not a Node3D"))?;
                            node.set_transform(item.transform);
                            holder.add_child(&node);
                            spawned.push((node.clone(), chunk, item.id));
                            placed.nodes.push(node);
                        }
                    }
                    Binding::Loading(_) => unreachable!("placing waits for every scene to load"),
                }
            }
            if let Some(old) = self.placed.insert(key, placed) {
                free(old);
            }
        }
        Ok(spawned)
    }

    /// Frees everything placed, and forgets what was due.
    pub(crate) fn clear(&mut self) {
        self.due.clear();
        for (_, placed) in self.placed.drain() {
            free(placed);
        }
    }

    /// Whether every scene has loaded, classifying those that just did.
    fn loaded(&mut self) -> Result<bool, String> {
        let mut loader = ResourceLoader::singleton();
        let mut all = true;
        for (kind, binding) in &mut self.bindings {
            let Binding::Loading(path) = binding else {
                continue;
            };
            let status = loader.call("load_threaded_get_status", &[path.to_variant()]);
            match status.try_to::<i64>() {
                Ok(IN_PROGRESS) => all = false,
                Ok(LOADED) => {
                    let scene = loader
                        .call("load_threaded_get", &[path.to_variant()])
                        .try_to::<Gd<PackedScene>>()
                        .map_err(|_| format!("scene of {kind:?}: {path} is not a PackedScene"))?;
                    *binding = classify(&scene);
                }
                _ => return Err(format!("scene of {kind:?}: {path} failed to load")),
            }
        }
        Ok(all)
    }
}

/// How a scene is drawn: as a MultiMesh when its root is a lone `MeshInstance3D` without a script
/// holding a mesh, otherwise as nodes.
fn classify(scene: &Gd<PackedScene>) -> Binding {
    let Some(root) = scene.instantiate() else {
        return Binding::Nodes(scene.clone(), None);
    };
    let lone = root.get_child_count() == 0 && root.get_script().is_none();
    let mesh = root
        .clone()
        .try_cast::<MeshInstance3D>()
        .ok()
        .and_then(|instance| instance.get_mesh());
    let binding = match mesh {
        Some(mesh) if lone => Binding::Mesh(mesh),
        _ => Binding::Nodes(scene.clone(), first_mesh(&root, Transform3D::IDENTITY)),
    };
    root.free();
    binding
}

/// The first mesh under `node`, depth first, and where it sits relative to the scene's root, given
/// that `node` sits at `at`.
fn first_mesh(node: &Gd<Node>, at: Transform3D) -> Option<(Gd<Mesh>, Transform3D)> {
    if let Ok(instance) = node.clone().try_cast::<MeshInstance3D>()
        && let Some(mesh) = instance.get_mesh()
    {
        return Some((mesh, at));
    }
    node.get_children().iter_shared().find_map(|child| {
        let placed = child
            .clone()
            .try_cast::<Node3D>()
            .map_or(at, |child| at * child.get_transform());
        first_mesh(&child, placed)
    })
}

/// A MultiMesh of `mesh` at every item's transform, placed at `offset` within it, drawn in
/// `scenario`: the MultiMesh, its instance and how many it draws.
fn multimesh(
    mesh: &Gd<Mesh>,
    offset: Transform3D,
    items: &[&Item],
    scenario: Rid,
) -> (Rid, Rid, usize) {
    let mut rendering = RenderingServer::singleton();
    let multimesh = rendering.multimesh_create();
    rendering.multimesh_set_mesh(multimesh, mesh.get_rid());
    rendering
        .multimesh_allocate_data_ex(
            multimesh,
            items.len() as i32,
            MultimeshTransformFormat::TRANSFORM_3D,
        )
        .done();
    let buffer: Vec<f32> = items
        .iter()
        .flat_map(|item| {
            let Transform3D { basis, origin } = item.transform * offset;
            let [a, b, c] = basis.rows;
            [
                a.x, a.y, a.z, origin.x, b.x, b.y, b.z, origin.y, c.x, c.y, c.z, origin.z,
            ]
        })
        .collect();
    rendering.multimesh_set_buffer(multimesh, &PackedFloat32Array::from(buffer.as_slice()));
    let instance = rendering.instance_create2(multimesh, scenario);
    (multimesh, instance, items.len())
}

fn free(placed: Placed) {
    let mut rendering = RenderingServer::singleton();
    for (multimesh, instance, _) in placed.multimeshes {
        rendering.free_rid(instance);
        rendering.free_rid(multimesh);
    }
    for mut node in placed.nodes {
        node.queue_free();
    }
}

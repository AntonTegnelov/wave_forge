//! A good first look without the addon owning the lighting (docs/reference/godot.md, "First look"):
//! whether a scene has a sun and a sky as Godot's editor preview judges it, the inspector's "Add sun
//! and sky", which copies the addon's `sun_and_sky.tscn` into the scene, and the warning a running
//! game gives once when it draws Wave Forge content with neither.

use godot::classes::resource::DeepDuplicateMode;
use godot::classes::{EditorInterface, Engine, Node, PackedScene, WorldEnvironment};
use godot::prelude::*;

/// The scene the inspector's "Add sun and sky" copies its nodes from.
const SUN_AND_SKY: &str = "res://addons/wave_forge/sun_and_sky.tscn";

/// Whether `scene`, or a node under it in its own or an instanced scene, is a `class`: how Godot's
/// editor counts the suns and environments that turn its preview off.
fn holds(scene: &Gd<Node>, class: &str) -> bool {
    scene.is_class(class)
        || !scene
            .find_children_ex("*")
            .type_(class)
            .owned(false)
            .done()
            .is_empty()
}

/// Whether `scene` lacks a sun, a `DirectionalLight3D`, and whether it lacks a sky, a
/// `WorldEnvironment` with no project default to stand in for it.
pub(crate) fn missing(scene: &Gd<Node>) -> (bool, bool) {
    let default_environment = godot::classes::ProjectSettings::singleton()
        .get_setting("rendering/environment/defaults/default_environment")
        .to::<GString>();
    (
        !holds(scene, "DirectionalLight3D"),
        !holds(scene, "WorldEnvironment") && default_environment.is_empty(),
    )
}

/// Adds the addon's sun and sky to `scene`, each only if it lacks one, at the top of the scene and
/// owned by it, as Godot's own "Add Sun to Scene" does: plain nodes the user then owns. In the
/// editor it is one undo action. Returns the names of the nodes added.
pub(crate) fn add_sun_and_sky(scene: &Gd<Node>) -> Vec<String> {
    let (no_sun, no_sky) = missing(scene);
    let Some(packed) = try_load::<PackedScene>(SUN_AND_SKY).ok() else {
        godot_error!("wave forge: {SUN_AND_SKY} is missing, so there is no sun and sky to add");
        return Vec::new();
    };
    let Some(template) = packed.instantiate() else {
        godot_error!("wave forge: {SUN_AND_SKY} could not be instanced");
        return Vec::new();
    };
    let mut template = template;
    let mut added: Vec<Gd<Node>> = Vec::new();
    if no_sun && let Some(sun) = template.find_child("Sun") {
        template.remove_child(&sun);
        added.push(sun);
    }
    if no_sky && let Some(sky) = template.find_child("WorldEnvironment") {
        template.remove_child(&sky);
        let mut sky = sky.cast::<WorldEnvironment>();
        // The scene keeps an environment of its own, not one held by the addon's file.
        if let Some(environment) = sky.get_environment() {
            let own = environment
                .duplicate_resource_ex()
                .deep(DeepDuplicateMode::ALL)
                .done();
            sky.set_environment(&own);
        }
        added.push(sky.upcast());
    }
    template.free();
    let names = added
        .iter()
        .map(|node| node.get_name().to_string())
        .collect();
    if added.is_empty() {
        godot_print!("wave forge: the scene already has a sun and a sky");
        return names;
    }
    if Engine::singleton().is_editor_hint() {
        let mut undo = EditorInterface::singleton()
            .get_editor_undo_redo()
            .expect("the editor's undo history");
        undo.create_action("Add sun and sky");
        for node in &added {
            undo.add_do_method(scene, "add_child", &[node.to_variant()]);
            undo.add_do_method(scene, "move_child", &[node.to_variant(), 0.to_variant()]);
            undo.add_do_property(node, "owner", &scene.to_variant());
            undo.add_do_reference(node);
            undo.add_undo_method(scene, "remove_child", &[node.to_variant()]);
        }
        undo.commit_action();
    } else {
        let mut scene = scene.clone();
        for node in &mut added {
            scene.add_child(&*node);
            scene.move_child(&*node, 0);
            node.set_owner(&scene);
        }
    }
    names
}

/// Warns once, in a running game, that `node` draws Wave Forge content in a world with no sun or
/// no sky: the editor's preview sun and sky are not the game's. Returns whether it warned.
pub(crate) fn warn_if_unlit(node: &Gd<Node>) -> bool {
    if Engine::singleton().is_editor_hint() {
        return false;
    }
    if !node.is_inside_tree() {
        return false;
    }
    // Everything the game runs, its autoloads included, which may hold the sun and sky.
    let Some(root) = node.get_tree().get_root() else {
        return false;
    };
    let root = root.upcast::<Node>();
    let (no_sun, no_sky) = missing(&root);
    if !no_sun && !no_sky {
        return false;
    }
    let what = match (no_sun, no_sky) {
        (true, true) => "no sun and no sky",
        (true, false) => "no sun",
        _ => "no sky",
    };
    godot_warn!(
        "wave forge: the running scene has {what}, so the terrain is drawn dark and flat; the \
         editor's preview sun and sky are not the game's. Add a DirectionalLight3D and a \
         WorldEnvironment, or press \"Add sun and sky\" on the Wave Forge node."
    );
    true
}

//! What the nodes show as configuration warnings in the editor: settings that leave a world dark,
//! without bodies or without sound, said where a user sees them (docs/reference/godot.md, "Editor").

use godot::classes::{AudioServer, Engine, Node, ProjectSettings};
use godot::prelude::*;
use std::time::{Duration, Instant};

/// How often the editor's warnings are looked at again: a light added anywhere in the scene changes
/// them, and nothing tells the node so.
const REFRESH_EVERY: Duration = Duration::from_millis(500);

/// That the scene the node is in has no light and no environment, which leaves the world dark: the
/// editor lights its viewport with a preview sun and sky of its own, which a running game has not.
pub(crate) fn lighting(node: &Gd<Node>) -> Vec<String> {
    let Some(scene) = scene_of(node) else {
        return Vec::new();
    };
    let has = |class: &str| {
        scene.is_class(class)
            || !scene
                .find_children_ex("*")
                .type_(class)
                .owned(false)
                .done()
                .is_empty()
    };
    let mut warnings = Vec::new();
    if !has("Light3D") {
        warnings.push(
            "The scene has no light, so the world is drawn dark when the game runs: add a \
             DirectionalLight3D as its sun."
                .to_owned(),
        );
    }
    let default_environment = ProjectSettings::singleton()
        .get_setting("rendering/environment/defaults/default_environment")
        .to::<GString>();
    if !has("WorldEnvironment") && default_environment.is_empty() {
        warnings.push(
            "The scene has no WorldEnvironment, so the world has no sky and no ambient light \
             when the game runs: add one."
                .to_owned(),
        );
    }
    warnings
}

/// That colliders are asked for while the project's 3D physics is not Jolt, which the node's
/// bodies and their measured costs assume.
pub(crate) fn physics(collider_radius: i32) -> Option<String> {
    let engine = ProjectSettings::singleton()
        .get_setting("physics/3d/physics_engine")
        .to::<GString>();
    (collider_radius >= 0 && engine != "Jolt Physics").then(|| {
        format!(
            "collider_radius builds bodies, but the project's 3D physics engine is {engine}, not \
             Jolt Physics: set it in Project Settings, Physics, 3D."
        )
    })
}

/// That occluders are asked for while occlusion culling is off, so they hide nothing.
pub(crate) fn occlusion(occluder_radius: i32) -> Option<String> {
    let culling = ProjectSettings::singleton()
        .get_setting("rendering/occlusion_culling/use_occlusion_culling")
        .to::<bool>();
    (occluder_radius >= 0 && !culling).then(|| {
        "occluder_radius builds occluders, but occlusion culling is off, so they hide nothing: \
         turn on Project Settings, Rendering, Occlusion Culling."
            .to_owned()
    })
}

/// That an audio bus a setting names does not exist, so its sounds play on the master bus.
pub(crate) fn buses(named: &[(&str, &StringName)]) -> Vec<String> {
    named
        .iter()
        .filter(|(_, bus)| !bus.is_empty() && AudioServer::singleton().get_bus_index(*bus) < 0)
        .map(|(setting, bus)| {
            format!("{setting} names the audio bus {bus}, which the project's bus layout lacks.")
        })
        .collect()
}

/// The scene the node is in: the one being edited in the editor, otherwise the topmost node above
/// it below the tree's root.
fn scene_of(node: &Gd<Node>) -> Option<Gd<Node>> {
    if !node.is_inside_tree() {
        return None;
    }
    let tree = node.get_tree();
    if Engine::singleton().is_editor_hint() {
        return tree.get_edited_scene_root();
    }
    let mut at = node.clone();
    while let Some(parent) = at.get_parent() {
        if parent.get_parent().is_none() {
            break;
        }
        at = parent;
    }
    Some(at)
}

/// When a node's warnings were last looked at, and what they were.
#[derive(Default)]
pub(crate) struct Refresh {
    at: Option<Instant>,
    shown: Vec<String>,
}

impl Refresh {
    /// Whether the warnings are due to be looked at again, in the editor only.
    pub(crate) fn due(&self) -> bool {
        Engine::singleton().is_editor_hint()
            && self.at.is_none_or(|at| at.elapsed() >= REFRESH_EVERY)
    }

    /// Records `warnings` as looked at now; whether they differ from those shown.
    pub(crate) fn changed(&mut self, warnings: Vec<String>) -> bool {
        self.at = Some(Instant::now());
        let changed = warnings != self.shown;
        self.shown = warnings;
        changed
    }
}

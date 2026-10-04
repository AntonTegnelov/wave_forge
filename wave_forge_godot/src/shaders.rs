//! The reference shaders the nodes draw with when a game gives no material of its own: the ground's,
//! the grass's, the fluid's and the plants' in the wind. They ship as `.gdshader` files in the
//! addon, which the nodes load by path, so Godot's shader baker finds them when a game is exported
//! and an export compiles them ahead of time; the extension holds the same code for the
//! `*_shader_code()` functions.

use godot::classes::{Material, Shader, ShaderMaterial};
use godot::prelude::*;

/// The reference ground shader.
pub(crate) const GROUND: &str = include_str!("../godot/addons/wave_forge/shaders/ground.gdshader");
/// The reference grass shader.
pub(crate) const GRASS: &str = include_str!("../godot/addons/wave_forge/shaders/grass.gdshader");
/// The reference fluid shader.
pub(crate) const FLUID: &str = include_str!("../godot/addons/wave_forge/shaders/fluid.gdshader");
/// The reference vegetation shader: plants bending in the global wind.
pub(crate) const VEGETATION: &str =
    include_str!("../godot/addons/wave_forge/shaders/vegetation.gdshader");

/// The reference shader in the addon's `shaders/<file>`, whose code is `code`. A project that has
/// moved the addon's files gets a shader of the same code, with a warning, since an export's shader
/// baker cannot find that one.
pub(crate) fn reference(file: &str, code: &str) -> Gd<Shader> {
    let path = format!("res://addons/wave_forge/shaders/{file}");
    try_load::<Shader>(&path).unwrap_or_else(|_| {
        godot_warn!("wave forge: no shader at {path}; the reference shader is built from the extension's own code, which an export cannot bake");
        let mut shader = Shader::new_gd();
        shader.set_code(code);
        shader
    })
}

/// `material` as a baked scene holds it: a copy holding a copy of its shader if its shader is one
/// of the addon's reference shaders, which the scene then carries itself, so it opens in a project
/// without the extension; any other material as it is.
pub(crate) fn self_contained(material: Gd<Material>) -> Gd<Material> {
    let Ok(shading) = material.clone().try_cast::<ShaderMaterial>() else {
        return material;
    };
    let Some(shader) = shading.get_shader() else {
        return material;
    };
    if !shader
        .get_path()
        .to_string()
        .starts_with("res://addons/wave_forge/shaders/")
    {
        return material;
    }
    let mut copy = shading.duplicate_resource();
    let mut own = Shader::new_gd();
    own.set_code(&shader.get_code());
    copy.set_shader(&own);
    copy.upcast()
}

//! The directories a node reads and writes chunks in, as the file-system paths the library's stores
//! open on their own threads, where Godot's file API cannot be called.

use godot::classes::{Os, ProjectSettings};
use godot::prelude::*;
use std::path::{Path, PathBuf};

/// `path`, a `res://`, `user://` or file-system path, as a file-system path. In the editor and in
/// a project run from its folder, Godot resolves `res://` to the project's directory. In an
/// exported game it resolves it to a path relative to the working directory, which depends on how
/// the game was started (an exported pack gives `res://wave_forge_world` as `wave_forge_world`), so
/// a relative path is taken from the executable's directory instead: a directory shipped beside the
/// executable is found however the game was started.
pub(crate) fn directory_path(path: &GString) -> String {
    let globalized = ProjectSettings::singleton()
        .globalize_path(path)
        .to_string();
    let executable = Os::singleton().get_executable_path().to_string();
    beside_executable(&globalized, Path::new(&executable))
        .to_string_lossy()
        .into_owned()
}

/// `globalized` itself if it is absolute, and otherwise the same path in `executable`'s directory.
fn beside_executable(globalized: &str, executable: &Path) -> PathBuf {
    let path = Path::new(globalized);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        executable
            .parent()
            .expect("an executable lies in a directory")
            .join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_absolute_path_is_kept() {
        let path = beside_executable("/games/save/world", Path::new("/opt/game/game.x86_64"));

        assert_eq!(path, PathBuf::from("/games/save/world"));
    }

    #[test]
    fn a_relative_path_lies_beside_the_executable() {
        let path = beside_executable("wave_forge_world", Path::new("/opt/game/game.x86_64"));

        assert_eq!(path, PathBuf::from("/opt/game/wave_forge_world"));
    }
}

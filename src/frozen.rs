//! Where chunks kept as they were first generated wait while no request needs them, so they can
//! leave memory without being lost (docs/reference/packs.md, "Persistence and saves").

use std::path::PathBuf;
use wfc_core::ChunkCoord;

/// Where the chunks of a frozen stage wait while no request needs them: a game's save directory, a
/// database, or memory. Wave Forge hands a chunk over when it leaves memory and asks for it when it
/// is needed again. What a chunk's bytes hold is Wave Forge's business; a store only keeps them,
/// by layer (a stage's name) and chunk.
pub trait FrozenStore: Send {
    /// Keeps `bytes` for `chunk` of `layer`, replacing anything kept for it.
    ///
    /// # Errors
    /// If the store could not keep them.
    fn keep(&mut self, layer: &str, chunk: ChunkCoord, bytes: Vec<u8>) -> Result<(), StoreError>;

    /// What was kept for `chunk` of `layer`, or `None` if nothing was.
    ///
    /// # Errors
    /// If the store could not be read.
    fn fetch(&mut self, layer: &str, chunk: ChunkCoord) -> Result<Option<Vec<u8>>, StoreError>;
}

/// A store that could not keep or give back a chunk, with its reason.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the frozen store failed: {0}")]
pub struct StoreError(pub String);

/// A store of files under a directory: one per chunk, at `<directory>/<layer>/<x>_<y>_<z>`.
#[derive(Clone, Debug)]
pub struct DirectoryStore {
    directory: PathBuf,
}

impl DirectoryStore {
    /// A store under `directory`, which it creates as it needs to.
    #[must_use]
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    fn path(&self, layer: &str, chunk: ChunkCoord) -> PathBuf {
        self.directory
            .join(layer)
            .join(format!("{}_{}_{}", chunk.x, chunk.y, chunk.z))
    }
}

impl FrozenStore for DirectoryStore {
    fn keep(&mut self, layer: &str, chunk: ChunkCoord, bytes: Vec<u8>) -> Result<(), StoreError> {
        let path = self.path(layer, chunk);
        let parent = path
            .parent()
            .expect("a chunk's file is inside its layer's directory");
        std::fs::create_dir_all(parent)
            .and_then(|()| std::fs::write(&path, bytes))
            .map_err(|error| StoreError(format!("{}: {error}", path.display())))
    }

    fn fetch(&mut self, layer: &str, chunk: ChunkCoord) -> Result<Option<Vec<u8>>, StoreError> {
        let path = self.path(layer, chunk);
        match std::fs::read(&path) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(StoreError(format!("{}: {error}", path.display()))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_directory_store_gives_back_what_it_kept_and_nothing_for_the_rest() {
        let directory =
            std::env::temp_dir().join(format!("wave_forge_store_{}", std::process::id()));
        let mut store = DirectoryStore::new(&directory);
        let chunk = ChunkCoord::new(3, -2, 0);

        store.keep("places", chunk, vec![1, 2, 3]).expect("kept");
        let kept = store.fetch("places", chunk).expect("read");
        let other_layer = store.fetch("trees", chunk).expect("read");
        let other_chunk = store
            .fetch("places", ChunkCoord::new(0, 0, 0))
            .expect("read");
        std::fs::remove_dir_all(&directory).expect("cleaned up");

        assert_eq!(kept, Some(vec![1, 2, 3]));
        assert_eq!(other_layer, None);
        assert_eq!(other_chunk, None);
    }
}

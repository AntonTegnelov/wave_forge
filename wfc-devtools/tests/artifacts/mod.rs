//! Where the end-to-end tests put the images they render, so a passing or failing run can be
//! looked at rather than guessed about.

use std::path::PathBuf;

/// Cargo's per-target temporary directory by default; override with `WFC_ARTIFACT_DIR`.
pub fn dir() -> PathBuf {
    let dir = std::env::var_os("WFC_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("e2e-artifacts"));
    std::fs::create_dir_all(&dir).expect("create artifact directory");
    dir
}

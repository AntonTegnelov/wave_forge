use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use wfc_core::grid::PossibilityGrid;
use wfc_core::tile::TileId;

// TODO: Add different output format options (binary, RON)

/// Saves the final collapsed grid to a file in a simple text format.
///
/// Assumes the grid is fully collapsed, containing only one possibility (TileId) per cell.
/// Format: Space-separated TileIds along X, newline per row (Y), blank line per layer (Z).
pub fn save_grid_to_file(grid: &PossibilityGrid, output_path: &Path) -> Result<()> {
    log::info!("Attempting to save grid to {:?}...", output_path);

    let file = File::create(output_path)
        .with_context(|| format!("Failed to create output file: {:?}", output_path))?;
    let mut writer = io::BufWriter::new(file);

    for z in 0..grid.depth {
        if z > 0 {
            // Separator between Z slices
            writeln!(writer)?;
        }
        for y in 0..grid.height {
            if y > 0 {
                // Separator between Y layers within a Z slice
                // Removed extra blank line for simplicity; rows just follow each other.
            }
            let mut line = String::new();
            for x in 0..grid.width {
                let possibilities = grid.get(x, y, z).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Internal error: Failed to access grid cell ({},{},{})",
                        x,
                        y,
                        z
                    )
                })?;

                let tile_id = match possibilities.iter_ones().next() {
                    Some(id) => {
                        // Check if *only* one bit is set
                        if possibilities.count_ones() == 1 {
                            TileId(id)
                        } else {
                            bail!("Grid cell ({},{},{}) is not fully collapsed ({} possibilities), cannot save.", x, y, z, possibilities.count_ones());
                        }
                    }
                    None => {
                        bail!("Grid cell ({},{},{}) has a contradiction (0 possibilities), cannot save.", x, y, z);
                    }
                };

                if x > 0 {
                    line.push(' '); // Space separator for X
                }
                line.push_str(&tile_id.0.to_string());
            }
            writeln!(writer, "{}", line)
                .with_context(|| format!("Failed to write line for coords (:, {}, {})", y, z))?;
        }
    }

    writer
        .flush()
        .context("Failed to flush writer for output file")?;
    log::info!("Successfully saved grid to {:?}", output_path);

    Ok(())
}

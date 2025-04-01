//! Provides traits and implementations for visualizing the WFC grid state.

use anyhow::Result;
use colored::*;
use wfc_core::grid::PossibilityGrid; // Import colored crate

// TODO: Define specific error type for visualization?

/// Trait for types that can visualize the state of the WFC `PossibilityGrid`.
///
/// Implementors of this trait define how the grid's state (e.g., possibilities,
/// entropy, final collapsed state) is presented to the user, such as via
/// terminal output or a graphical window.
pub trait Visualizer {
    /// Displays or updates the visualization based on the current state of the grid.
    ///
    /// This method is called to render the grid's state. The specific behavior
    /// depends on the implementing visualizer (e.g., print to console, update GUI).
    ///
    /// # Arguments
    ///
    /// * `grid` - A reference to the `PossibilityGrid` whose state should be displayed.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the display was successful.
    /// * `Err(anyhow::Error)` if an error occurred during visualization.
    fn display_state(&mut self, grid: &PossibilityGrid) -> Result<(), anyhow::Error>; // Using anyhow for app errors

    // TODO: Add methods for initialization, updates during run, final display?
    // TODO: Add methods for toggling, focusing layers?
}

// --- Implementations ---

// TODO: Implement TerminalVisualizer
/// A visualizer that renders a 2D slice of the grid state in the terminal.
pub struct TerminalVisualizer {
    /// The Z-layer index to display.
    z_layer: usize,
}

impl TerminalVisualizer {
    /// Creates a new TerminalVisualizer focused on layer 0.
    pub fn new() -> Self {
        Self { z_layer: 0 }
    }

    /// Sets the Z-layer index to be displayed.
    pub fn set_layer(&mut self, z_layer: usize) {
        self.z_layer = z_layer;
    }
}

impl Visualizer for TerminalVisualizer {
    fn display_state(&mut self, grid: &PossibilityGrid) -> Result<(), anyhow::Error> {
        println!("--- Visualization Frame (Z={} Slice) ---", self.z_layer);
        if self.z_layer >= grid.depth || grid.height == 0 || grid.width == 0 {
            println!(
                "(Grid is empty or Z-layer {} is out of bounds [0..{}])",
                self.z_layer,
                grid.depth.saturating_sub(1)
            );
            println!("-------------------------------------");
            return Ok(());
        }

        let z = self.z_layer; // Use the stored layer index
        for y in 0..grid.height {
            for x in 0..grid.width {
                if let Some(cell) = grid.get(x, y, z) {
                    let possibilities = cell.count_ones();
                    match possibilities {
                        0 => print!("{}", "X".red().bold()), // Contradiction
                        1 => {
                            // Find the single possible TileId
                            let tile_id_index = cell.iter_ones().next().unwrap_or(0);
                            // Simple mapping: Use different colors/styles for different tiles (example)
                            let tile_char = if tile_id_index < 10 {
                                std::char::from_digit(tile_id_index as u32, 10).unwrap_or('#')
                            } else {
                                '+'
                            };
                            // Example coloring based on tile index parity
                            if tile_id_index % 2 == 0 {
                                print!("{}", tile_char.to_string().green());
                            } else {
                                print!("{}", tile_char.to_string().blue());
                            }
                        }
                        _ => print!("{}", "?".yellow()), // Multiple possibilities
                    };
                } else {
                    print!("{}", "E".magenta().bold()); // Error getting cell
                }
            }
            println!(); // Newline after each row
        }
        println!("-------------------------------------");
        Ok(())
    }
}

// TODO: Implement Simple2DVisualizer

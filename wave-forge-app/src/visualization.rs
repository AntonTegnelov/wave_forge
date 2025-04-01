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
/// A placeholder visualizer intended to render the grid state in the terminal.
// Placeholder implementation
pub struct TerminalVisualizer;

impl Visualizer for TerminalVisualizer {
    fn display_state(&mut self, grid: &PossibilityGrid) -> Result<(), anyhow::Error> {
        println!("--- Visualization Frame (Z=0 Slice) ---");
        if grid.depth == 0 || grid.height == 0 || grid.width == 0 {
            println!("(Grid is empty or flat)");
            println!("-------------------------------------");
            return Ok(());
        }

        let z = 0; // Display only the first layer for simplicity
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

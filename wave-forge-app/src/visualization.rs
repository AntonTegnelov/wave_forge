use wfc_core::grid::PossibilityGrid;

// TODO: Define specific error type for visualization?

/// Trait for visualizing the state of the WFC grid.
pub trait Visualizer {
    /// Displays the current state of the grid.
    ///
    /// This might involve printing to console, updating a window, etc.
    /// It might take the grid directly, or potentially other info like entropy.
    fn display_state(&mut self, grid: &PossibilityGrid) -> Result<(), anyhow::Error>; // Using anyhow for app errors

    // TODO: Add methods for initialization, updates during run, final display?
    // TODO: Add methods for toggling, focusing layers?
}

// --- Implementations ---

// TODO: Implement TerminalVisualizer
// Placeholder implementation
pub struct TerminalVisualizer;

impl Visualizer for TerminalVisualizer {
    fn display_state(&mut self, grid: &PossibilityGrid) -> Result<(), anyhow::Error> {
        // Placeholder: Print basic grid info
        println!("--- Visualization Frame ---");
        println!("Grid: {}x{}x{}", grid.width, grid.height, grid.depth);
        // TODO: Implement actual terminal rendering
        println!("---------------------------");
        Ok(())
    }
}

// TODO: Implement Simple2DVisualizer

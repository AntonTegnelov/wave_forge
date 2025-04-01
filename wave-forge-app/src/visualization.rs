use wfc_core::grid::PossibilityGrid;

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
        // Placeholder: Print basic grid info
        println!("--- Visualization Frame ---");
        println!("Grid: {}x{}x{}", grid.width, grid.height, grid.depth);
        // TODO: Implement actual terminal rendering
        println!("---------------------------");
        Ok(())
    }
}

// TODO: Implement Simple2DVisualizer

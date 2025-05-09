//! Provides traits and implementations for visualizing the WFC grid state.

use anyhow::Result;
use bitvec::prelude::{BitSlice, Lsb0};
use colored::*;
use minifb::{Key, Window, WindowOptions};
use wfc_core::grid::PossibilityGrid; // Import colored crate // Add missing imports

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

    /// Checks if the visualizer is currently enabled (active).
    ///
    /// # Returns
    ///
    /// * `true` if visualization is currently enabled
    /// * `false` if visualization is disabled/paused
    fn is_enabled(&self) -> bool;

    /// Toggles the enabled state of the visualizer.
    ///
    /// # Returns
    ///
    /// * The new enabled state after toggling
    fn toggle_enabled(&mut self) -> bool;

    /// Processes any pending user input to check for toggle commands, etc.
    ///
    /// This method should be called regularly to handle user input events.
    /// It provides a way for users to interact with the visualization,
    /// such as toggling on/off, changing layers, etc.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` if visualization should continue (not permanently closed)
    /// * `Ok(false)` if visualization was permanently closed (e.g., window closed)
    /// * `Err(anyhow::Error)` if an error occurred while processing input
    fn process_input(&mut self) -> Result<bool, anyhow::Error>;

    // TODO: Add methods for initialization, final display?
}

// --- Implementations ---

// TODO: Implement TerminalVisualizer
/// A visualizer that renders a 2D slice of the grid state in the terminal.
pub struct TerminalVisualizer {
    /// The Z-layer index to display.
    z_layer: usize,
    /// Whether visualization is currently enabled
    enabled: bool,
}

impl Default for TerminalVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalVisualizer {
    /// Creates a new TerminalVisualizer focused on layer 0.
    pub fn new() -> Self {
        Self {
            z_layer: 0,
            enabled: true,
        }
    }

    /// Creates a new TerminalVisualizer with specified toggle key.
    /// The toggle key is provided for API compatibility but not used internally.
    pub fn with_toggle_key(_toggle_key: char) -> Self {
        Self {
            z_layer: 0,
            enabled: true,
        }
    }

    /// Sets the Z-layer index to be displayed.
    #[allow(dead_code)]
    pub fn set_layer(&mut self, z_layer: usize) {
        self.z_layer = z_layer;
    }
}

impl Visualizer for TerminalVisualizer {
    fn display_state(&mut self, grid: &PossibilityGrid) -> Result<(), anyhow::Error> {
        // Only display if enabled
        if !self.enabled {
            return Ok(());
        }

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

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn toggle_enabled(&mut self) -> bool {
        self.enabled = !self.enabled;
        if self.enabled {
            println!("Terminal visualization enabled");
        } else {
            println!("Terminal visualization disabled");
        }
        self.enabled
    }

    fn process_input(&mut self) -> Result<bool, anyhow::Error> {
        // Terminal visualizer doesn't process input directly
        // Toggling will be handled via explicit toggle_enabled calls
        Ok(true) // Always continue
    }
}

// --- Simple 2D Visualizer (using minifb) ---

const DEFAULT_WINDOW_WIDTH: usize = 640;
const DEFAULT_WINDOW_HEIGHT: usize = 480;

/// Determines the color for a cell based on its state.
fn get_color(possibilities: &BitSlice<usize, Lsb0>, num_tiles: usize) -> u32 {
    let count = possibilities.count_ones();
    match count {
        0 => 0xFF_FF_00_00, // Red for contradiction
        1 => {
            // Collapsed - use a grayscale value based on TileId
            let tile_id_index = possibilities.iter_ones().next().unwrap_or(0);
            // Simple mapping: scale index to grayscale (ensure division by zero is handled)
            let gray = if num_tiles > 1 {
                ((tile_id_index * 255) / (num_tiles - 1)).min(255) as u32
            } else {
                255 // Only one tile, make it white
            };
            (gray << 16) | (gray << 8) | gray // R=G=B
        }
        _ => {
            // Not collapsed - use blue, intensity based on entropy (possibility count proxy)
            // More possibilities = darker blue
            (255 - (count * 200 / num_tiles).min(200)) as u32
        }
    }
}

/// A visualizer that displays a 2D slice of the grid in a simple window using `minifb`.
pub struct Simple2DVisualizer {
    window: Window,
    buffer: Vec<u32>,
    width: usize,
    height: usize,
    z_layer: usize,  // Layer to display
    enabled: bool,   // Whether visualization is currently enabled
    toggle_key: Key, // Key used to toggle visualization on/off (default is 'T')
}

impl Simple2DVisualizer {
    /// Creates a new Simple2DVisualizer window.
    ///
    /// # Parameters
    ///
    /// * `title` - The title of the visualization window
    /// * `grid_width` - The width of the grid (currently using fixed window size instead)
    /// * `grid_height` - The height of the grid (currently using fixed window size instead)
    /// * `toggle_key` - The key used to toggle visualization on/off (default is 'T')
    ///
    /// # Errors
    ///
    /// Returns an error if the window cannot be created.
    pub fn new(
        title: &str,
        _grid_width: usize,
        _grid_height: usize,
        toggle_key: char,
    ) -> Result<Self, anyhow::Error> {
        // Using fixed window size rather than scaling based on grid dimensions
        // This allows consistent window size regardless of grid size
        // TODO: Consider implementing proper scaling based on grid dimensions
        let window_width = DEFAULT_WINDOW_WIDTH;
        let window_height = DEFAULT_WINDOW_HEIGHT;

        let window = Window::new(title, window_width, window_height, WindowOptions::default())
            .map_err(|e| anyhow::anyhow!("Failed to create minifb window: {}", e))?;

        let buffer = vec![0; window_width * window_height];

        // Convert the toggle_key character to a minifb Key
        // This is a simple mapping for common keys
        let key = match toggle_key.to_ascii_uppercase() {
            'A' => Key::A,
            'B' => Key::B,
            'C' => Key::C,
            'D' => Key::D,
            'E' => Key::E,
            'F' => Key::F,
            'G' => Key::G,
            'H' => Key::H,
            'I' => Key::I,
            'J' => Key::J,
            'K' => Key::K,
            'L' => Key::L,
            'M' => Key::M,
            'N' => Key::N,
            'O' => Key::O,
            'P' => Key::P,
            'Q' => Key::Q,
            'R' => Key::R,
            'S' => Key::S,
            'T' => Key::T,
            'U' => Key::U,
            'V' => Key::V,
            'W' => Key::W,
            'X' => Key::X,
            'Y' => Key::Y,
            'Z' => Key::Z,
            '0' => Key::Key0,
            '1' => Key::Key1,
            '2' => Key::Key2,
            '3' => Key::Key3,
            '4' => Key::Key4,
            '5' => Key::Key5,
            '6' => Key::Key6,
            '7' => Key::Key7,
            '8' => Key::Key8,
            '9' => Key::Key9,
            _ => {
                log::warn!("Unsupported toggle key '{}', using 'T' instead", toggle_key);
                Key::T // Default to T for unsupported keys
            }
        };

        Ok(Self {
            window,
            buffer,
            width: window_width,
            height: window_height,
            z_layer: 0,
            enabled: true,
            toggle_key: key,
        })
    }

    /// Sets the Z-layer index to be displayed.
    #[allow(dead_code)]
    pub fn set_layer(&mut self, z_layer: usize) {
        self.z_layer = z_layer;
    }
}

impl Visualizer for Simple2DVisualizer {
    fn display_state(&mut self, grid: &PossibilityGrid) -> Result<(), anyhow::Error> {
        // Skip rendering if disabled
        if !self.enabled {
            // Still need to update window and check for input even when not rendering
            // This ensures the window remains responsive
            if !self.window.is_open() {
                return Ok(());
            }
            // Update window with current buffer (no changes)
            self.window
                .update_with_buffer(&self.buffer, self.width, self.height)
                .map_err(|e| anyhow::anyhow!("Failed to update minifb window buffer: {}", e))?;
            return Ok(());
        }

        if !self.window.is_open() || self.window.is_key_down(Key::Escape) {
            // Allow closing window gracefully - maybe return a specific error?
            // For now, just stop displaying but don't error out the whole WFC.
            log::info!("Visualization window closed or Escape pressed.");
            // To signal closure upwards, we might need a different return type or mechanism.
            return Ok(()); // Pretend success for now
        }

        let grid_w = grid.width;
        let grid_h = grid.height;
        let num_tiles = grid.num_tiles();

        if grid_w == 0 || grid_h == 0 || self.z_layer >= grid.depth {
            // Handle empty grid or invalid layer - maybe display black?
            self.buffer.fill(0);
        } else {
            // Render the grid slice to the buffer
            // Scale grid cells to fit window buffer
            let scale_x = self.width as f32 / grid_w as f32;
            let scale_y = self.height as f32 / grid_h as f32;

            for wy in 0..self.height {
                for wx in 0..self.width {
                    // Map window pixel back to grid cell
                    let gx = (wx as f32 / scale_x).floor() as usize;
                    let gy = (wy as f32 / scale_y).floor() as usize;

                    let color = match grid.get(gx, gy, self.z_layer) {
                        Some(cell_possibilities) => get_color(cell_possibilities, num_tiles),
                        None => 0xFF_80_00_80, // Magenta for error/out-of-bounds (shouldn't happen with checks)
                    };
                    self.buffer[wy * self.width + wx] = color;
                }
            }
        }

        // Update the window with the buffer content
        self.window
            .update_with_buffer(&self.buffer, self.width, self.height)
            .map_err(|e| anyhow::anyhow!("Failed to update minifb window buffer: {}", e))?;

        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn toggle_enabled(&mut self) -> bool {
        self.enabled = !self.enabled;
        if self.enabled {
            log::info!("2D visualization enabled");
        } else {
            log::info!("2D visualization disabled");
        }
        self.enabled
    }

    fn process_input(&mut self) -> Result<bool, anyhow::Error> {
        // Check if window is still open
        if !self.window.is_open() {
            return Ok(false); // Signal that visualization was closed
        }

        // Check for visualization toggle key (using the configured key)
        if self.window.is_key_released(self.toggle_key) {
            self.toggle_enabled();
        }

        // Check for layer navigation keys (UP/DOWN arrows)
        if self.window.is_key_released(Key::Up) {
            self.z_layer = self.z_layer.saturating_add(1);
            log::info!("Visualization layer changed to z={}", self.z_layer);
        }
        if self.window.is_key_released(Key::Down) {
            self.z_layer = self.z_layer.saturating_sub(1);
            log::info!("Visualization layer changed to z={}", self.z_layer);
        }

        // Window is still open and can continue
        Ok(true)
    }
}

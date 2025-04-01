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
            let blue = (255 - (count * 200 / num_tiles).min(200)) as u32; // More possibilities = darker blue
            blue
        }
    }
}

/// A visualizer that displays a 2D slice of the grid in a simple window using `minifb`.
pub struct Simple2DVisualizer {
    window: Window,
    buffer: Vec<u32>,
    width: usize,
    height: usize,
    z_layer: usize, // Layer to display
}

impl Simple2DVisualizer {
    /// Creates a new Simple2DVisualizer window.
    ///
    /// # Errors
    ///
    /// Returns an error if the window cannot be created.
    pub fn new(
        title: &str,
        _grid_width: usize,
        _grid_height: usize,
    ) -> Result<Self, anyhow::Error> {
        // Prefix unused vars
        // Determine window size - maybe scale based on grid size?
        // For now, use defaults, but ensure buffer matches window.
        let window_width = DEFAULT_WINDOW_WIDTH;
        let window_height = DEFAULT_WINDOW_HEIGHT;

        let window = Window::new(title, window_width, window_height, WindowOptions::default())
            .map_err(|e| anyhow::anyhow!("Failed to create minifb window: {}", e))?;

        let buffer = vec![0; window_width * window_height];

        Ok(Self {
            window,
            buffer,
            width: window_width,
            height: window_height,
            z_layer: 0,
        })
    }

    /// Sets the Z-layer index to be displayed.
    pub fn set_layer(&mut self, z_layer: usize) {
        self.z_layer = z_layer;
    }
}

impl Visualizer for Simple2DVisualizer {
    fn display_state(&mut self, grid: &PossibilityGrid) -> Result<(), anyhow::Error> {
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
}

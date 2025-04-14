//! Handles the setup of the visualization thread based on configuration.

use crate::config::{AppConfig, VisualizationMode};
use crate::visualization::{self, TerminalVisualizer, Visualizer};
use log; // Add log crate
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;
use wfc_core::grid::PossibilityGrid;

// Move VizMessage enum here
#[derive(Debug)] // Add Debug derive for potential logging
pub enum VizMessage {
    UpdateGrid(Box<PossibilityGrid>),
}

/// Sets up the visualization thread based on the application configuration.
///
/// Returns a tuple containing an optional Sender channel for sending grid updates
/// and an optional JoinHandle for the spawned visualization thread.
pub fn setup_visualization(
    config: &AppConfig,
) -> (Option<Sender<VizMessage>>, Option<thread::JoinHandle<()>>) {
    match config.visualization_mode {
        VisualizationMode::None => (None, None),
        _ => {
            // Logic moved from main.rs
            let (tx, rx): (Sender<VizMessage>, Receiver<VizMessage>) = mpsc::channel();

            let viz_mode = config.visualization_mode.clone();
            let grid_width = config.width;
            let grid_height = config.height;
            let toggle_key = config.visualization_toggle_key;

            log::info!("Starting visualization thread with mode: {:?}", viz_mode);
            log::info!("Visualization toggle key: '{}'", toggle_key);

            let handle = thread::spawn(move || {
                let mut visualizer: Box<dyn Visualizer> = match viz_mode {
                    VisualizationMode::Terminal => {
                        Box::new(TerminalVisualizer::with_toggle_key(toggle_key))
                    }
                    VisualizationMode::Simple2D => {
                        match visualization::Simple2DVisualizer::new(
                            &format!("Wave Forge - {}x{}", grid_width, grid_height),
                            grid_width,
                            grid_height,
                            toggle_key,
                        ) {
                            Ok(viz) => Box::new(viz),
                            Err(e) => {
                                log::error!("Failed to create Simple2DVisualizer: {}", e);
                                Box::new(TerminalVisualizer::with_toggle_key(toggle_key))
                            }
                        }
                    }
                    VisualizationMode::None => unreachable!(), // Should not happen if outer match is not None
                };

                log::info!("Visualization thread started");

                let mut running = true;
                while running {
                    match rx.recv() {
                        Ok(VizMessage::UpdateGrid(grid)) => {
                            if let Ok(continue_viz) = visualizer.process_input() {
                                if !continue_viz {
                                    log::info!("Visualization stopped by user input");
                                    running = false;
                                    continue;
                                }
                                if visualizer.is_enabled() {
                                    if let Err(e) = visualizer.display_state(&grid) {
                                        log::error!("Failed to display grid: {}", e);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            log::error!("Error receiving visualization update: {}", e);
                            running = false;
                        }
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                log::info!("Visualization thread terminated");
            });

            (Some(tx), Some(handle))
        }
    }
}

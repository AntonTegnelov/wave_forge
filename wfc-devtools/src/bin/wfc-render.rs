//! Renders a grid written by the `wave_forge` CLI to a PNG (developer tool).

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use wfc_devtools::TileGrid;
use wfc_devtools::render::{self, Style};

#[derive(Clone, Copy, ValueEnum)]
enum View {
    /// One z layer seen from above.
    Layer,
    /// Top, isometric, front and side views of the whole grid on one sheet.
    FourView,
}

#[derive(Parser)]
#[command(about = "Render a Wave Forge output grid to PNG (developer tool)")]
struct Args {
    /// Grid text file written by `wave-forge --output`.
    grid: PathBuf,
    /// PNG file to write.
    #[arg(short, long, default_value = "grid.png")]
    out: PathBuf,
    /// What to draw.
    #[arg(long, value_enum, default_value_t = View::FourView)]
    view: View,
    /// Layer to draw with `--view layer`.
    #[arg(long, default_value_t = 0)]
    z: usize,
    /// Edge length of one cell in pixels.
    #[arg(long, default_value_t = 12)]
    cell_px: u32,
    /// Tile index that represents empty space (for example air) in 3D views. Repeatable.
    #[arg(long = "empty-tile")]
    empty_tiles: Vec<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let text = std::fs::read_to_string(&args.grid)
        .with_context(|| format!("reading {}", args.grid.display()))?;
    let grid = TileGrid::parse_text(&text).map_err(anyhow::Error::msg)?;
    let style = Style {
        palette: &[],
        empty_tiles: &args.empty_tiles,
        cell_px: args.cell_px,
    };

    let image = match args.view {
        View::Layer => {
            ensure!(
                args.z < grid.depth,
                "layer {} is out of range for depth {}",
                args.z,
                grid.depth
            );
            render::render_layer(&grid, args.z, &style)
        }
        View::FourView => render::render_four_view(&grid, &style),
    };
    image
        .save(&args.out)
        .with_context(|| format!("writing {}", args.out.display()))?;
    println!("wrote {}", args.out.display());
    Ok(())
}

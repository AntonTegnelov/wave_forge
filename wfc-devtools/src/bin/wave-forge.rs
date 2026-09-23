//! Generates one chunk from a rule file and writes it as a text grid (developer tool).
//!
//! The library generates a world of many chunks around moving focus points; this binary is the
//! smallest thing that drives it, so that a rule set can be checked by hand and the result rendered
//! with `wfc-render`.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use wave_forge::{Builder, ChunkCoord, ChunkShape, FocusPoint, Prior, Ruleset, WorldExtent};
use wfc_devtools::TileGrid;

#[derive(Parser)]
#[command(about = "Generate a grid from a rule file (developer tool)")]
struct Args {
    /// RON or vox rule file describing the tiles and what may sit next to what.
    #[arg(short, long, value_name = "FILE")]
    rule_file: PathBuf,
    /// Cells along x.
    #[arg(long, default_value_t = 8)]
    width: u32,
    /// Cells along y.
    #[arg(long, default_value_t = 8)]
    height: u32,
    /// Cells along z.
    #[arg(long, default_value_t = 8)]
    depth: u32,
    /// Every choice derives from this.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// File to write the grid to.
    #[arg(short, long, value_name = "FILE", default_value = "output.txt")]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let file = wfc_rules::loader::load_rule_file(&args.rule_file)
        .with_context(|| format!("loading {}", args.rule_file.display()))?;
    let ruleset = Ruleset::new(file.rules(), &file.tileset().weights)?;
    let tiles = ruleset.num_tiles();

    let chunk = ChunkShape {
        x: args.width,
        y: args.height,
        z: args.depth,
    };
    let at = ChunkCoord::new(0, 0, 0);
    let mut world = Builder::new(ruleset, Prior::open(tiles))
        .seed(args.seed)
        .extent(
            WorldExtent::new(chunk)
                .with_x(0..1)
                .with_y(0..1)
                .with_z(0..1),
        )
        .build()?;
    world.request(&[FocusPoint::new(at, 0)]);
    let events = world.run_until_idle()?;

    let solved = world
        .chunk(at)
        .with_context(|| format!("the chunk was not generated: {events:?}"))?;
    let grid = TileGrid::new(
        args.width as usize,
        args.height as usize,
        args.depth as usize,
        solved.tiles.iter().map(|&tile| tile as usize).collect(),
    )
    .map_err(anyhow::Error::msg)?;
    std::fs::write(&args.output, grid.to_text())
        .with_context(|| format!("writing {}", args.output.display()))?;

    let stats = world.stats();
    println!(
        "{}x{}x{} from {} tiles in {:.1} ms; wrote {}",
        args.width,
        args.height,
        args.depth,
        tiles,
        stats.solver_ms,
        args.output.display()
    );
    Ok(())
}

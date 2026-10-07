//! Renders a pack's height field seen from straight above as shaded relief (developer tool), for
//! judging a landform's shape: water blue below zero, land from green through brown to white with
//! height, lit from the north-west. A pixel is one of the stage's columns, so a coarse stage's
//! picture covers its scale times as much ground, and `--chunks` and `--corner` count its chunks.
//!
//! ```text
//! cargo run -p wfc-devtools --release --bin wfc-relief -- examples/presets/hills.world.ron \
//!     --chunks 32 --out hills.png [--stage height] [--seed 1] [--param hills=0.8]
//! ```

use anyhow::{Context, Result, anyhow, ensure};
use clap::Parser;
use image::{Rgb, RgbImage};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use wave_forge::stages::{Pack, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

#[derive(Parser)]
#[command(about = "Render a pack's height field as shaded relief (developer tool)")]
struct Args {
    /// The pack, a `.world.ron` file.
    pack: PathBuf,
    /// PNG file to write, a pixel per column.
    #[arg(short, long, default_value = "relief.png")]
    out: PathBuf,
    /// The Field stage to draw.
    #[arg(long, default_value = "height")]
    stage: String,
    /// Chunks along each side of the square drawn.
    #[arg(long, default_value_t = 32)]
    chunks: i32,
    /// The chunk at the square's lower corner, along x and y.
    #[arg(long, num_args = 2, default_values_t = [0, 0], allow_negative_numbers = true)]
    corner: Vec<i32>,
    /// Columns along each side of a chunk.
    #[arg(long, default_value_t = 8)]
    chunk_size: u32,
    /// The world's seed.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// The height that draws white; the land's colours run from 0 to it.
    #[arg(long, default_value_t = 30.0)]
    top: f32,
    /// A parameter of the pack, `name=value`. Repeatable.
    #[arg(long = "param", value_parser = parse_param)]
    params: Vec<(String, f32)>,
}

fn parse_param(text: &str) -> Result<(String, f32), String> {
    let (name, value) = text
        .split_once('=')
        .ok_or_else(|| format!("{text:?} is not name=value"))?;
    let value: f32 = value
        .parse()
        .map_err(|error| format!("{value:?} is not a number: {error}"))?;
    Ok((name.to_owned(), value))
}

/// The colour of land or water at `height`, before shading.
fn colour(height: f32, top: f32) -> [f32; 3] {
    if height < 0.0 {
        let depth = (-height / 10.0).min(1.0);
        return [0.25 - 0.15 * depth, 0.45 - 0.2 * depth, 0.75 - 0.25 * depth];
    }
    let stops = [
        (0.0, [0.35, 0.6, 0.3]),
        (0.4, [0.55, 0.6, 0.35]),
        (0.75, [0.5, 0.4, 0.3]),
        (1.0, [0.95, 0.95, 0.95]),
    ];
    let t = (height / top).clamp(0.0, 1.0);
    let after = stops
        .iter()
        .position(|stop| stop.0 >= t)
        .unwrap_or(stops.len() - 1)
        .max(1);
    let ((t0, c0), (t1, c1)) = (stops[after - 1], stops[after]);
    let k = (t - t0) / (t1 - t0);
    [0, 1, 2].map(|i| c0[i] + (c1[i] - c0[i]) * k)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.chunks > 0, "--chunks has to be above zero");
    let text = std::fs::read_to_string(&args.pack)
        .with_context(|| format!("reading {}", args.pack.display()))?;
    let pack = Pack::parse(&text).map_err(|error| anyhow!("{}: {error}", args.pack.display()))?;
    // Slopes are shaded per cell, so a coarse stage lights as the same ground at full detail would.
    let scale = pack
        .scale(&args.stage)
        .ok_or_else(|| anyhow!("{} is no stage of the pack", args.stage))? as f32;
    let mut runtime = Runtime::new(Arc::new(pack), args.seed, [args.chunk_size; 2]);
    let params: BTreeMap<String, f32> = args.params.iter().cloned().collect();
    runtime
        .set_params(&params)
        .map_err(|error| anyhow!("{error}"))?;
    let chunks: Vec<ChunkCoord> = (0..args.chunks)
        .flat_map(|y| (0..args.chunks).map(move |x| (x, y)))
        .map(|(x, y)| ChunkCoord::new(args.corner[0] + x, args.corner[1] + y, 0))
        .collect();
    // A focus is in the lattice's chunks; a coarse stage's chunk covers `scale` of them each way.
    let step = scale as i32;
    let focus: Vec<FocusPoint> = chunks
        .iter()
        .map(|chunk| FocusPoint::new(ChunkCoord::new(chunk.x * step, chunk.y * step, 0), 0))
        .collect();
    runtime
        .request(&focus, &[args.stage.as_str()])
        .map_err(|error| anyhow!("{error}"))?;
    runtime
        .run_until_idle()
        .map_err(|error| anyhow!("{error}"))?;

    let side = args.chunks as u32 * args.chunk_size;
    let mut heights = vec![0.0f32; (side * side) as usize];
    for &chunk in &chunks {
        let field = runtime
            .field(&args.stage, chunk)
            .ok_or_else(|| anyhow!("{} is not a Field stage", args.stage))?;
        let x0 = (chunk.x - args.corner[0]) as u32 * args.chunk_size;
        let y0 = (chunk.y - args.corner[1]) as u32 * args.chunk_size;
        for (i, &value) in field.values.iter().enumerate() {
            let (x, y) = (
                x0 + i as u32 % args.chunk_size,
                y0 + i as u32 / args.chunk_size,
            );
            heights[(y * side + x) as usize] = value;
        }
    }
    let at = |x: u32, y: u32| heights[(y.min(side - 1) * side + x.min(side - 1)) as usize];
    let light = [-0.5f32, 0.5, std::f32::consts::FRAC_1_SQRT_2];
    // The image's rows run from the square's far side, +y, down to its near one.
    let image = RgbImage::from_fn(side, side, |px, py| {
        let (x, y) = (px, side - 1 - py);
        let h = at(x, y);
        let dx = (at(x + 1, y) - at(x.saturating_sub(1), y)) / (2.0 * scale);
        let dy = (at(x, y + 1) - at(x, y.saturating_sub(1))) / (2.0 * scale);
        let normal = [-dx, -dy, 1.0];
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + 1.0).sqrt();
        let lit = (normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2]) / length;
        let shade = 0.35 + 0.65 * lit.max(0.0);
        let base = colour(h, args.top);
        Rgb(base.map(|c| (c * shade * 255.0).clamp(0.0, 255.0) as u8))
    });
    image
        .save(&args.out)
        .with_context(|| format!("writing {}", args.out.display()))?;
    let (low, high) = heights
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &h| {
            (low.min(h), high.max(h))
        });
    let mean = heights.iter().sum::<f32>() / heights.len() as f32;
    println!(
        "wrote {}, {side} by {side} columns, heights {low:.1} to {high:.1}, {mean:.3} on average",
        args.out.display()
    );
    Ok(())
}

//! Writes the city's module models as binary glTF files a game engine imports (developer tool).
//!
//! One `<module>.glb` per module of `examples/city.ron`, unrotated, centred on the origin and one
//! unit across, in a Y-up engine's frame: a game places the model named by a tile at the cell's
//! centre and turns it by the tile's rotation (`wave_forge::YUpSpace::yaw`). The models are the
//! city's crude voxel stand-ins, meant to be replaced by authored ones.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use wfc_devtools::{city, models};

#[derive(Parser)]
#[command(about = "Write the city's module models as glTF (developer tool)")]
struct Args {
    /// Directory to write the `.glb` files into; created if missing.
    #[arg(short, long, value_name = "DIR")]
    out: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;
    let city = city::city();
    let mut written = 0;
    let mut triangles = 0;
    for (name, model) in city.prototype_models() {
        let mesh = models::mesh(model);
        let path = args.out.join(format!("{name}.glb"));
        std::fs::write(&path, models::glb(&mesh, name))
            .with_context(|| format!("writing {}", path.display()))?;
        written += 1;
        triangles += mesh.triangles();
    }
    println!(
        "wrote {written} models, {triangles} triangles in all, to {}",
        args.out.display()
    );
    Ok(())
}

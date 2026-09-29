//! P6's measurement (docs/product/user-stories.md): a changed parameter regenerates only the
//! stages downstream of it, and a 3×3-chunk preview updates within 200 ms, measured per stage.
//!
//! ```text
//! cargo test --release --test interactive_edit -- --ignored --nocapture
//! ```
//!
//! The preview is the islands preset over 3 by 3 chunks of 16 by 16 columns, a game's chunk size.
//! Each parameter is changed five times, each time to another value in its range, and the time
//! from the change until the preview is generated again is printed with what each stage cost.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;
use wave_forge::stages::{Pack, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [16, 16];
const STAGES: [&str; 3] = ["height", "surface", "trees"];

#[test]
#[ignore = "a measurement: run in release with --ignored --nocapture"]
fn a_changed_parameter_updates_a_three_by_three_preview() {
    let text = std::fs::read_to_string(format!(
        "{}/examples/presets/islands.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the islands preset");
    let mut runtime = Runtime::new(Arc::new(Pack::parse(&text).expect("a valid pack")), 7, SIZE);
    let preview: Vec<FocusPoint> = (-1..=1)
        .flat_map(|y| (-1..=1).map(move |x| FocusPoint::new(ChunkCoord::new(x, y, 0), 0)))
        .collect();
    runtime.request(&preview, &STAGES).expect("the stages");
    let started = Instant::now();
    runtime.run_until_idle().expect("the stages run");
    eprintln!(
        "interactive_edit: first preview {:.2} ms",
        started.elapsed().as_secs_f64() * 1000.0
    );

    for name in ["trees", "roughness", "land"] {
        let mut updates = Vec::new();
        let mut regenerated: BTreeMap<String, (u64, f64)> = BTreeMap::new();
        for step in 1..=5 {
            let value = 0.1 + 0.15 * step as f32;
            let before: BTreeMap<String, (u64, f64)> = runtime
                .timings()
                .into_iter()
                .map(|(stage, timing)| (stage, (timing.products, timing.ms)))
                .collect();

            let started = Instant::now();
            runtime
                .set_params(&BTreeMap::from([(name.to_owned(), value)]))
                .expect("a value in range");
            runtime.request(&preview, &STAGES).expect("the stages");
            runtime.run_until_idle().expect("the stages run");
            updates.push(started.elapsed().as_secs_f64() * 1000.0);

            for (stage, timing) in runtime.timings() {
                let (products, ms) = before[&stage];
                let entry = regenerated.entry(stage).or_default();
                entry.0 += timing.products - products;
                entry.1 += timing.ms - ms;
            }
        }
        updates.sort_by(f64::total_cmp);
        let stages: Vec<String> = regenerated
            .iter()
            .filter(|(_, (products, _))| *products > 0)
            .map(|(stage, (products, ms))| {
                format!("{stage} {} chunks {:.2} ms", products / 5, ms / 5.0)
            })
            .collect();
        eprintln!(
            "interactive_edit: {name}: median {:.2} ms, max {:.2} ms; per change: {}",
            updates[2],
            updates[4],
            stages.join(", ")
        );
    }
}

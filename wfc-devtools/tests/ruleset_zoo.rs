//! The rule-set zoo: the same grid and the same seeds under rules of different *kind*, so thrashing
//! can be attributed to what a rule is rather than to how big the problem is.
//!
//! Every fixture we had varied rule set *size* and tightness; all of them were adjacency rules. That
//! answers "does a bigger problem thrash more" and nothing about "does this kind of rule thrash more".
//! Each test here changes exactly one thing against [`zoo_control`]: same 12x12x6 city, same module
//! set, same weights, same seed, differing only in the global constraint applied. See
//! docs/thrashing.md for the questions this is instrumenting.
//!
//! Run one at a time, in release mode, with a seed, and **with `WFC_SWEEP=1`**:
//!
//! ```text
//! WFC_SWEEP=1 WFC_SEED=8 WFC_REPORT_SEARCH=1 \
//!   cargo test -p wfc-devtools --release --test ruleset_zoo -- --ignored --nocapture zoo_counting
//! ```
//!
//! `WFC_SWEEP` is not optional for comparisons. Without it `common::solve_rules` gives a constrained
//! run an iteration budget 25 times larger than an unconstrained one (`cells * 50` against
//! `cells * 2`), so a control and a treatment would differ in budget as well as in rule kind, and any
//! difference measured would be uninterpretable. Under sweep both get `cells * 4`.
//!
//! The 12x12x6 city is deliberate: it is the instrument the thrashing study is calibrated on, with a
//! 48-seed baseline recorded in docs/solver-fit.md, so a number here can be compared against something.

mod common;

use std::sync::Arc;
use wfc_core::BoundaryCondition;
use wfc_core::constraint::{Cell, CountingConstraint, GlobalConstraint, RangeExclusionConstraint};
use wfc_core::grid::PossibilityGrid;
use wfc_devtools::city;

/// Applies several constraints as one, since the solver takes a single global constraint.
///
/// Our constraint types name one tile each, but the city has 81 rotated variants, so a rule about
/// "roads" is a rule about every road variant. Composing here keeps that out of the core API until a
/// measurement shows what shape it actually wants.
///
/// Concatenating the changed cells preserves the property the solver depends on: each part reports a
/// cell only when it truly cleared a bit, so the combination does too. A constraint that reported an
/// untouched cell would spin the solver's uncapped fixpoint loop without advancing its iteration
/// counter.
struct All(Vec<Box<dyn GlobalConstraint>>);

impl GlobalConstraint for All {
    fn apply(&self, grid: &mut PossibilityGrid) -> Result<Vec<Cell>, Cell> {
        let mut changed = Vec::new();
        for constraint in &self.0 {
            changed.extend(constraint.apply(grid)?);
        }
        Ok(changed)
    }
}

/// Solves the standard zoo city, optionally under a global constraint, and reports what it took.
///
/// Search statistics (seed, backtracks, where contradictions landed, undo depths) are printed by the
/// accelerator itself under `WFC_REPORT_SEARCH`; this only adds the wall-clock outcome.
async fn run_zoo(kind: &str, constraint: Option<Arc<dyn GlobalConstraint>>) {
    let city = city::city();
    let m = &city.modules;
    let (width, height, depth) = (12, 12, 6);
    let mut initial = PossibilityGrid::new(width, height, depth, m.variants.len());
    city::constrain_city(&mut initial, &city);

    let solved = common::solve_rules(
        &initial,
        &m.rules,
        Some(&m.tileset.weights),
        constraint,
        BoundaryCondition::Finite,
        1,
    )
    .await;
    eprintln!(
        "zoo: kind={kind} cells={} attempt={} run_s={:.3} total_s={:.3}",
        width * height * depth,
        solved.attempts,
        solved.solve_time.as_secs_f64(),
        solved.total_time.as_secs_f64(),
    );
}

/// Adjacency only. The baseline every other test in this file is measured against.
#[tokio::test]
#[ignore = "rule-set zoo; run with --ignored in release mode, one at a time, with WFC_SWEEP=1"]
async fn zoo_control_adjacency_only() {
    run_zoo("control", None).await;
}

/// Bounded, directional, non-adjacent: no elevated walkway within two cells directly above a road.
///
/// The cheapest kind of non-local rule — one cell and a fixed offset decide it, with no graph and no
/// whole-grid analysis. If this thrashes like connectivity does, the cause is non-locality itself
/// rather than the cost of checking it.
///
/// Walkways over roads is a pairing that can genuinely occur here, unlike (say) roofs over roads,
/// which the module set already makes impossible: a vacuous constraint would measure nothing.
#[tokio::test]
#[ignore = "rule-set zoo; run with --ignored in release mode, one at a time, with WFC_SWEEP=1"]
async fn zoo_range_exclusion() {
    let city = city::city();
    let m = &city.modules;
    let roads = m.variants_tagged("road");
    let walkways = m.variants_tagged("walkway");
    assert!(
        !roads.is_empty() && !walkways.is_empty(),
        "the rule needs both tags to exist, or it constrains nothing"
    );

    let parts: Vec<Box<dyn GlobalConstraint>> = roads
        .iter()
        .flat_map(|&road| {
            walkways.iter().map(move |&walkway| {
                Box::new(RangeExclusionConstraint::new(
                    road,
                    walkway,
                    [(0, 0, 1), (0, 0, 2)],
                )) as Box<dyn GlobalConstraint>
            })
        })
        .collect();
    eprintln!("zoo: range-exclusion over {} tile pairs", parts.len());
    run_zoo("range_exclusion", Some(Arc::new(All(parts)))).await;
}

/// Bounded but non-local in every direction: every door needs a road within two cells.
///
/// Counting sits between adjacency and connectivity. Like connectivity its failures can surface away
/// from their cause; unlike connectivity its radius is bounded, so no whole-grid analysis is needed.
/// That makes it the test of whether *unbounded* reach is what makes connectivity expensive, or
/// merely non-locality.
///
/// Deliberately loose: one road within two cells of a door, which the street layout should satisfy
/// almost everywhere. A rule that is unsatisfiable measures the failure path, not thrashing.
#[tokio::test]
#[ignore = "rule-set zoo; run with --ignored in release mode, one at a time, with WFC_SWEEP=1"]
async fn zoo_counting() {
    let city = city::city();
    let m = &city.modules;
    let roads = m.variants_tagged("road");
    let doors: Vec<usize> = (0..m.variants.len())
        .filter(|&tile| m.prototype_of(tile).name == "building_door")
        .collect();
    assert!(
        !roads.is_empty() && !doors.is_empty(),
        "the rule needs both roads and doors to exist, or it constrains nothing"
    );
    eprintln!("zoo: counting over {} doors, {} roads", doors.len(), roads.len());

    let constraint = CountingConstraint::new(roads, doors, 2, 1);
    run_zoo("counting", Some(Arc::new(constraint))).await;
}

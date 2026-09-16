//! The rule-set zoo: the same grid and the same seeds under rules of different *kind*, so thrashing
//! can be attributed to what a rule is rather than to how big the problem is.
//!
//! Every fixture we had varied rule set *size* and tightness; all of them were adjacency rules. That
//! answers "does a bigger problem thrash more" and nothing about "does this kind of rule thrash more".
//! Each test here changes exactly one thing against [`zoo_control_adjacency_only`]: same 12x12x6 city,
//! same module set, same weights, same seed, differing only in the global constraint applied. See
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
//! 48-seed baseline recorded in docs/solver-fit.md, so a number here can be compared against
//! something. The control is verified to reproduce the e2e `small_city` search line exactly.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
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

/// Tallies how much a constraint actually did, so "no effect" can be told from "never fired".
///
/// This is not a nicety. A counting rule that never triggers produces a search line byte-identical to
/// the control, which reads exactly like "this kind of rule is harmless" while in fact measuring
/// nothing at all. One configuration of the counting rule did precisely that (docs/thrashing.md), and
/// without a prune count the only way to notice was to be suspicious of a too-perfect match.
struct Counted {
    inner: Box<dyn GlobalConstraint>,
    prunes: Arc<AtomicUsize>,
    failures: Arc<AtomicUsize>,
}

impl GlobalConstraint for Counted {
    fn apply(&self, grid: &mut PossibilityGrid) -> Result<Vec<Cell>, Cell> {
        match self.inner.apply(grid) {
            Ok(changed) => {
                self.prunes.fetch_add(changed.len(), Ordering::Relaxed);
                Ok(changed)
            }
            Err(cell) => {
                self.failures.fetch_add(1, Ordering::Relaxed);
                Err(cell)
            }
        }
    }
}

/// Wraps a constraint in its tally counters.
fn counted(inner: Box<dyn GlobalConstraint>) -> (Arc<dyn GlobalConstraint>, Tally) {
    let tally = Tally {
        prunes: Arc::new(AtomicUsize::new(0)),
        failures: Arc::new(AtomicUsize::new(0)),
    };
    let constraint = Arc::new(Counted {
        inner,
        prunes: Arc::clone(&tally.prunes),
        failures: Arc::clone(&tally.failures),
    });
    (constraint, tally)
}

/// What a constraint did over a run: cells narrowed, and times it declared the grid unsatisfiable.
struct Tally {
    prunes: Arc<AtomicUsize>,
    failures: Arc<AtomicUsize>,
}

/// Solves the standard zoo city, optionally under a global constraint, and reports what it took.
///
/// Search statistics (seed, backtracks, where contradictions landed, undo depths) are printed by the
/// accelerator itself under `WFC_REPORT_SEARCH`; this adds the wall-clock outcome and, when there is a
/// constraint, proof of whether it did anything.
async fn run_zoo(kind: &str, constraint: Option<(Arc<dyn GlobalConstraint>, Tally)>) {
    let city = city::city();
    let m = &city.modules;
    let (width, height, depth) = (12, 12, 6);
    let mut initial = PossibilityGrid::new(width, height, depth, m.variants.len());
    city::constrain_city(&mut initial, &city);

    let (constraint, tally) = match constraint {
        Some((constraint, tally)) => (Some(constraint), Some(tally)),
        None => (None, None),
    };
    let solved = common::solve_rules(
        &initial,
        &m.rules,
        Some(&m.tileset.weights),
        constraint,
        BoundaryCondition::Finite,
        1,
    )
    .await;
    let effect = match &tally {
        Some(tally) => format!(
            " prunes={} constraint_failures={}",
            tally.prunes.load(Ordering::Relaxed),
            tally.failures.load(Ordering::Relaxed)
        ),
        None => String::new(),
    };
    eprintln!(
        "zoo: kind={kind} cells={} attempt={} run_s={:.3} total_s={:.3}{effect}",
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
/// whole-grid analysis. Measured at seed 8 it costs 80 backtracks against the control's 8, so
/// non-locality alone makes the search much harder even when checking the rule is trivial.
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
    run_zoo("range_exclusion", Some(counted(Box::new(All(parts))))).await;
}

/// Bounded but non-local in every direction: every building needs a road within three cells.
///
/// Counting sits between adjacency and connectivity. Like connectivity its failures can surface away
/// from their cause; unlike connectivity its radius is bounded, so no whole-grid analysis is needed.
/// That makes it the test of whether *unbounded* reach is what makes connectivity expensive, or
/// merely non-locality.
///
/// Choosing parameters for this rule took four attempts, and the failures are worth recording because
/// each was a different way to measure nothing.
///
/// `CountingConstraint` prunes only when the candidates in a ball *exactly equal* the required count,
/// since that is when every candidate becomes load-bearing. Above that it is sound but silent; below
/// it, the grid is already unsatisfiable. The operating band is narrow, and both edges were hit:
///
/// - a road within *two* cells of a door, and a road within *three* cells of a building, each produced
///   a search line byte-identical to the control with `prunes=0` — a ball that size almost always
///   holds more possible roads than the count demands, so slack never reached zero;
/// - three roads within one cell of every *building* was not tight but impossible. Roads exist only at
///   street level while `variants_tagged("building")` spans every height, so a building at z=3 could
///   never satisfy it. The run ground through 1911 backtracks, collapsed 174 of 864 cells, and hit its
///   iteration cap.
///
/// The subject is therefore doors, which are street-level by construction and, being optional, let the
/// search avoid them where the rule cannot be met — so the constraint binds without being impossible.
/// A count of three within one cell asks a door to face reasonably open street: at street level only
/// about eight cells are in reach, so three is near enough the boundary for a zero-slack propagator to
/// do visible work.
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
        "the rule needs both doors and roads to exist, or it constrains nothing"
    );
    eprintln!("zoo: counting over {} doors, {} roads", doors.len(), roads.len());

    let constraint = CountingConstraint::new(roads, doors, 1, 3);
    run_zoo("counting", Some(counted(Box::new(constraint)))).await;
}

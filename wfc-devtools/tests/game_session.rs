//! Can a game like marian42's infinite city run on the library: a player walking and running
//! through an unbounded city that is generated around them while they move?
//!
//! ```text
//! cargo test -p wfc-devtools --release --test game_session -- --ignored --nocapture
//! --test-threads=1
//! ```
//!
//! One session is played in wall-clock time, the way a game would drive the library: a frame loop
//! at 60 Hz hands the player's position to a [`Worker`] whenever it enters another chunk, drains
//! the worker's events, and never waits for generation. The player follows a fixed route whatever
//! the generator does, so a generator that falls behind shows up as a chunk missing from view
//! rather than as a slower walk. Every test below reads that one session and checks one thing a
//! player would notice.
//!
//! The tests are `#[ignore]`d: they need a compute device, and the session takes about four
//! minutes because that is how long the route takes to walk. A timing describes one build on one
//! machine and driver stack; docs/research/measurements.md records what the numbers mean.

mod kernels;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use wave_forge::{
    Builder, ChunkCoord, ChunkEvent, ChunkShape, FocusPoint, GeneratorStats, Ruleset, Worker,
    WorldExtent,
};
use wfc_devtools::city::{
    self, City, FaceMeetings, city_prior, disconnected_walkable_cells, walk_continuity,
    walkable_tiles,
};
use wfc_devtools::{BoundaryCondition, TileGrid, adjacency_violations};

/// 8 cells of 2 m: a 16 m chunk, eight storeys tall.
const CHUNK: ChunkShape = ChunkShape::cube(8);
const CELL_M: f64 = 2.0;
const CHUNK_M: f64 = CELL_M * CHUNK.x as f64;
const SEED: u64 = 11;

/// How far a player sees, and so how far ahead the city has to exist: marian42's generation range.
const VIEW_M: f64 = 30.0;
/// Chunks generated around the player's chunk. A new row of chunks at this distance is asked for
/// when the player enters a chunk, and comes within [`VIEW_M`] after 18 m more: 12.9 s at a walk
/// and 4.3 s at a run. A radius of 2 would leave half a second at a run.
const GENERATE_RADIUS: u32 = 3;
/// Chunks further than the radius plus this are dropped. Below one, a chunk about to be asked for
/// again would be dropped, because the chunks a focus needs reach one beyond its radius.
const EVICT_MARGIN: u32 = 1;

const WALK_M_S: f64 = 1.4;
/// marian42's run multiplier is three times the walking speed.
const RUN_M_S: f64 = 3.0 * WALK_M_S;
const FRAME: Duration = Duration::from_micros(16_667);
/// Building the device, compiling every kernel and generating the first view is loading, not play.
const LOAD_TIMEOUT: Duration = Duration::from_secs(180);
/// Where the player starts: the middle of chunk (0, 0).
const START: (f64, f64) = (8.0, 8.0);
/// The route, as legs to a point at a pace. It walks, runs, turns by 45 and 90 degrees, runs a
/// diagonal (which brings the most new chunks per metre), and ends back at the start after the
/// start has been dropped, so the chunks there are generated a second time.
const ROUTE: [((f64, f64), f64); 6] = [
    ((72.0, 8.0), WALK_M_S),
    ((200.0, 8.0), RUN_M_S),
    ((296.0, 104.0), RUN_M_S),
    ((296.0, 136.0), WALK_M_S),
    ((8.0, 136.0), RUN_M_S),
    (START, RUN_M_S),
];

/// Where the player is `t` seconds into the route, or `None` once it is walked.
fn position_at(t: f64) -> Option<(f64, f64)> {
    let mut from = START;
    let mut left = t;
    for (to, pace) in ROUTE {
        let length = (to.0 - from.0).hypot(to.1 - from.1);
        let duration = length / pace;
        if left < duration {
            let share = left / duration;
            return Some((
                from.0 + share * (to.0 - from.0),
                from.1 + share * (to.1 - from.1),
            ));
        }
        left -= duration;
        from = to;
    }
    None
}

fn route_length_m() -> f64 {
    let mut from = START;
    let mut length = 0.0;
    for (to, _) in ROUTE {
        length += (to.0 - from.0).hypot(to.1 - from.1);
        from = to;
    }
    length
}

fn chunk_at((x, y): (f64, f64)) -> ChunkCoord {
    ChunkCoord::new(
        (x / CHUNK_M).floor() as i32,
        (y / CHUNK_M).floor() as i32,
        0,
    )
}

/// The chunks whose ground comes within [`VIEW_M`] of the player.
fn in_view((x, y): (f64, f64)) -> Vec<ChunkCoord> {
    let reach = (VIEW_M / CHUNK_M).ceil() as i32 + 1;
    let centre = chunk_at((x, y));
    let mut chunks = Vec::new();
    for cy in centre.y - reach..=centre.y + reach {
        for cx in centre.x - reach..=centre.x + reach {
            let (x0, y0) = (f64::from(cx) * CHUNK_M, f64::from(cy) * CHUNK_M);
            let dx = (x0 - x).max(x - (x0 + CHUNK_M)).max(0.0);
            let dy = (y0 - y).max(y - (y0 + CHUNK_M)).max(0.0);
            if dx.hypot(dy) <= VIEW_M {
                chunks.push(ChunkCoord::new(cx, cy, 0));
            }
        }
    }
    chunks
}

/// The four chunks sharing a face with `chunk`, with the axis that leads to each.
fn face_neighbours(chunk: ChunkCoord) -> [ChunkCoord; 4] {
    [
        ChunkCoord::new(chunk.x + 1, chunk.y, 0),
        ChunkCoord::new(chunk.x - 1, chunk.y, 0),
        ChunkCoord::new(chunk.x, chunk.y + 1, 0),
        ChunkCoord::new(chunk.x, chunk.y - 1, 0),
    ]
}

/// A chunk's tiles as the session saw them, and whether they are the tiles its coordinate alone
/// decides.
///
/// Without repairs, a chunk of the first parity generated with no neighbour present, and a chunk of
/// the second parity generated against four such neighbours, hold the tiles their coordinates
/// decide (the determinism section of the library's docs). A repair, a failed neighbour, or a
/// neighbour that stayed while this chunk was dropped makes them depend on history instead.
struct Life {
    tiles: Box<[u16]>,
    from_coordinate: bool,
}

/// Everything the session measured.
struct Session {
    load: Duration,
    frames: usize,
    walked: Duration,
    crossings: usize,
    stats: GeneratorStats,
    distinct: usize,
    max_resident: usize,
    /// Frames times chunks: every frame a chunk in view had no tiles yet, and was not given up on.
    late_chunk_frames: usize,
    /// The longest a chunk in view went without tiles, in seconds.
    worst_late_s: f64,
    first_late: Option<(ChunkCoord, f64)>,
    /// Chunks in view that could not be placed, with when and where the player first saw the hole.
    holes: BTreeMap<ChunkCoord, (f64, (f64, f64))>,
    /// The main thread's cost of one frame: asking for chunks, dropping them, taking the events.
    frame_costs: Vec<Duration>,
    busiest_frame_events: usize,
    violations_inside: usize,
    violations_across_seams: usize,
    seams_checked: usize,
    revisits_compared: usize,
    revisits_different: Vec<ChunkCoord>,
    revisits_not_comparable: usize,
    walk_inside: FaceMeetings,
    walk_across_seams: FaceMeetings,
    largest_network_share: Option<f64>,
}

/// What the session keeps about the chunks it has been told about.
struct Tracker<'a> {
    city: &'a City,
    /// The chunks the worker holds, in the order its events said so.
    resident: HashMap<ChunkCoord, Life>,
    /// The last life of every dropped chunk.
    earlier: HashMap<ChunkCoord, Life>,
    failed: BTreeSet<ChunkCoord>,
    seen: BTreeSet<ChunkCoord>,
    walk_inside: HashMap<ChunkCoord, FaceMeetings>,
    walk_across_seams: HashMap<(ChunkCoord, ChunkCoord), FaceMeetings>,
    violations_inside: usize,
    violations_across_seams: usize,
    seams_checked: usize,
    revisits_compared: usize,
    revisits_different: Vec<ChunkCoord>,
    revisits_not_comparable: usize,
}

impl<'a> Tracker<'a> {
    fn new(city: &'a City) -> Self {
        Self {
            city,
            resident: HashMap::new(),
            earlier: HashMap::new(),
            failed: BTreeSet::new(),
            seen: BTreeSet::new(),
            walk_inside: HashMap::new(),
            walk_across_seams: HashMap::new(),
            violations_inside: 0,
            violations_across_seams: 0,
            seams_checked: 0,
            revisits_compared: 0,
            revisits_different: Vec::new(),
            revisits_not_comparable: 0,
        }
    }

    /// Takes one drain's events, in order, then checks every chunk they changed against the
    /// chunks beside it as the worker now holds them.
    fn take(&mut self, worker: &Worker, events: &[ChunkEvent], repaired: bool) {
        let mut changed: BTreeSet<ChunkCoord> = BTreeSet::new();
        for event in events {
            match *event {
                ChunkEvent::Updated(chunk) => {
                    let Some(tiles) = worker.chunk(chunk) else {
                        // Dropped again later in the same drain: nothing of it reached the player.
                        continue;
                    };
                    let from_coordinate = !repaired && self.decided_by_coordinate(chunk);
                    let rewritten = self.resident.contains_key(&chunk);
                    let life = Life {
                        tiles: tiles.tiles.clone(),
                        from_coordinate: from_coordinate && !rewritten,
                    };
                    if !rewritten {
                        self.compare_with_earlier(chunk, &life);
                    }
                    self.resident.insert(chunk, life);
                    self.seen.insert(chunk);
                    changed.insert(chunk);
                }
                ChunkEvent::Evicted(chunk) => {
                    if let Some(life) = self.resident.remove(&chunk) {
                        self.earlier.insert(chunk, life);
                    }
                    changed.remove(&chunk);
                }
                ChunkEvent::Failed { chunk, .. } => {
                    self.failed.insert(chunk);
                }
            }
        }
        for chunk in changed {
            self.check(worker, chunk);
        }
    }

    fn decided_by_coordinate(&self, chunk: ChunkCoord) -> bool {
        let neighbours = face_neighbours(chunk);
        if neighbours.iter().any(|n| self.failed.contains(n)) {
            return false;
        }
        if chunk.parity() == 0 {
            neighbours.iter().all(|n| !self.resident.contains_key(n))
        } else {
            neighbours.iter().all(|n| {
                self.resident
                    .get(n)
                    .is_some_and(|life| life.from_coordinate)
            })
        }
    }

    fn compare_with_earlier(&mut self, chunk: ChunkCoord, life: &Life) {
        let Some(earlier) = self.earlier.get(&chunk) else {
            return;
        };
        if !(earlier.from_coordinate && life.from_coordinate) {
            self.revisits_not_comparable += 1;
            return;
        }
        self.revisits_compared += 1;
        if earlier.tiles != life.tiles {
            self.revisits_different.push(chunk);
        }
    }

    /// Rule violations inside `chunk` and across each face it shares with a chunk the worker holds,
    /// and how often walkable faces meet there.
    fn check(&mut self, worker: &Worker, chunk: ChunkCoord) {
        let rules = &self.city.modules.rules;
        let this = worker.chunk(chunk).expect("a chunk the last drain updated");
        let (grid, _) =
            TileGrid::from_chunks(CHUNK, [this], self.city.air).expect("one whole chunk");
        self.violations_inside +=
            adjacency_violations(&grid, rules, BoundaryCondition::Finite).len();
        self.walk_inside.insert(
            chunk,
            walk_continuity(&grid, self.city, (CHUNK.x as usize, CHUNK.y as usize)).inside,
        );

        for neighbour in face_neighbours(chunk) {
            let Some(other) = worker.chunk(neighbour) else {
                continue;
            };
            let (grid, lowest) =
                TileGrid::from_chunks(CHUNK, [this, other], self.city.air).expect("two chunks");
            let side = |cell: (usize, usize, usize)| {
                (cell.0 / CHUNK.x as usize, cell.1 / CHUNK.y as usize)
            };
            self.violations_across_seams +=
                adjacency_violations(&grid, rules, BoundaryCondition::Finite)
                    .iter()
                    .filter(|v| side(v.cell) != side(v.neighbor))
                    .count();
            self.seams_checked += 1;
            let key = if lowest == chunk {
                (chunk, neighbour)
            } else {
                (neighbour, chunk)
            };
            self.walk_across_seams.insert(
                key,
                walk_continuity(&grid, self.city, (CHUNK.x as usize, CHUNK.y as usize))
                    .across_seams,
            );
        }
    }
}

fn session() -> &'static Session {
    static SESSION: OnceLock<Result<Session, String>> = OnceLock::new();
    match SESSION.get_or_init(play) {
        Ok(session) => session,
        Err(reason) => panic!("the session could not be played: {reason}"),
    }
}

/// Loads the city, walks the route in wall-clock time, and reports what happened.
fn play() -> Result<Session, String> {
    let city = city::city();
    let ruleset = Ruleset::from_modules(&city.modules).map_err(|e| e.to_string())?;
    let prior = city_prior(&city, CHUNK.z);
    let loading = Instant::now();
    let mut worker = Worker::spawn(move || {
        let mut world = Builder::new(ruleset, prior)
            .seed(SEED)
            .extent(WorldExtent::new(CHUNK).with_z(0..1))
            .halo(1)
            .build()?;
        kernels::warm(&mut world, GENERATE_RADIUS);
        Ok(world)
    });
    let mut tracker = Tracker::new(&city);

    let focus = |chunk: ChunkCoord| [FocusPoint::new(chunk, GENERATE_RADIUS)];
    let mut player_chunk = chunk_at(START);
    worker.request(&focus(player_chunk));
    loop {
        let before = worker.stats().repaired;
        let events = worker.drain();
        tracker.take(&worker, &events, worker.stats().repaired > before);
        if let Some(reason) = worker.failure() {
            return Err(reason.to_owned());
        }
        let loaded = in_view(START)
            .iter()
            .all(|c| tracker.resident.contains_key(c) || tracker.failed.contains(c));
        if loaded {
            break;
        }
        if loading.elapsed() > LOAD_TIMEOUT {
            return Err(format!(
                "the first view was not there after {LOAD_TIMEOUT:?}"
            ));
        }
        std::thread::sleep(FRAME);
    }
    let load = loading.elapsed();

    let mut frame_costs = Vec::new();
    let mut busiest_frame_events = 0;
    let mut crossings = 0;
    let mut max_resident = 0;
    let mut late_chunk_frames = 0;
    let mut missing_since: HashMap<ChunkCoord, f64> = HashMap::new();
    let mut worst_late_s: f64 = 0.0;
    let mut first_late = None;
    let mut holes = BTreeMap::new();
    let started = Instant::now();
    let mut deadline = started;
    while let Some(position) = position_at(started.elapsed().as_secs_f64()) {
        let t = started.elapsed().as_secs_f64();

        // What a game's frame does, and all that is timed.
        let frame = Instant::now();
        let here = chunk_at(position);
        if here != player_chunk {
            player_chunk = here;
            crossings += 1;
            worker.request(&focus(here));
            worker.evict_outside(&focus(here), EVICT_MARGIN);
        }
        let before = worker.stats().repaired;
        let events = worker.drain();
        let cost = frame.elapsed();
        frame_costs.push(cost);
        if cost == *frame_costs.iter().max().expect("this frame's cost") {
            busiest_frame_events = events.len();
        }

        tracker.take(&worker, &events, worker.stats().repaired > before);
        if let Some(reason) = worker.failure() {
            return Err(reason.to_owned());
        }
        max_resident = max_resident.max(tracker.resident.len());
        for chunk in in_view(position) {
            if tracker.resident.contains_key(&chunk) {
                if let Some(since) = missing_since.remove(&chunk) {
                    worst_late_s = worst_late_s.max(t - since);
                }
            } else if tracker.failed.contains(&chunk) {
                holes.entry(chunk).or_insert((t, position));
            } else {
                late_chunk_frames += 1;
                missing_since.entry(chunk).or_insert(t);
                first_late.get_or_insert((chunk, t));
            }
        }

        // A game's frame loop paces itself against the clock; so must this one, because the
        // generator runs on its own thread in real time and no simulated clock can speed it up.
        deadline += FRAME;
        std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
    }
    let walked = started.elapsed();
    for since in missing_since.values() {
        worst_late_s = worst_late_s.max(walked.as_secs_f64() - since);
    }

    let (grid, _) = TileGrid::from_chunks(CHUNK, worker.chunks(), city.air)
        .map_err(|e| format!("the world at the end: {e}"))?;
    let walkable = walkable_tiles(&city.modules);
    let walkable_cells = (0..grid.depth)
        .flat_map(|z| (0..grid.height).flat_map(move |y| (0..grid.width).map(move |x| (x, y, z))))
        .filter(|&(x, y, z)| walkable.contains(&grid.get(x, y, z)))
        .count();
    let largest_network_share = (walkable_cells > 0).then(|| {
        1.0 - disconnected_walkable_cells(&grid, &city).len() as f64 / walkable_cells as f64
    });
    let path = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("game_session.png");
    wfc_devtools::render::render_voxel_isometric(&grid, &city.voxels, 2)
        .save(&path)
        .map_err(|e| e.to_string())?;

    let session = Session {
        load,
        frames: frame_costs.len(),
        walked,
        crossings,
        stats: *worker.stats(),
        distinct: tracker.seen.len(),
        max_resident,
        late_chunk_frames,
        worst_late_s,
        first_late,
        holes,
        frame_costs,
        busiest_frame_events,
        violations_inside: tracker.violations_inside,
        violations_across_seams: tracker.violations_across_seams,
        seams_checked: tracker.seams_checked,
        revisits_compared: tracker.revisits_compared,
        revisits_different: tracker.revisits_different.clone(),
        revisits_not_comparable: tracker.revisits_not_comparable,
        walk_inside: tracker
            .walk_inside
            .values()
            .fold(FaceMeetings::default(), |sum, m| sum + *m),
        walk_across_seams: tracker
            .walk_across_seams
            .values()
            .fold(FaceMeetings::default(), |sum, m| sum + *m),
        largest_network_share,
    };
    session.print(&path);
    Ok(session)
}

impl Session {
    /// The frame cost at quantile `q`, from 0 to 1.
    fn frame_cost(&self, q: f64) -> Duration {
        let mut costs = self.frame_costs.clone();
        costs.sort();
        costs[((costs.len() - 1) as f64 * q).round() as usize]
    }

    fn print(&self, render: &std::path::Path) {
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        let share = |m: FaceMeetings| {
            m.share()
                .map_or_else(|| "none".to_owned(), |s| format!("{s:.3}"))
        };
        eprintln!(
            "game session: loaded in {:.1} s; walked {:.0} m in {:.1} s over {} frames, {} chunk \
             crossings",
            self.load.as_secs_f64(),
            route_length_m(),
            self.walked.as_secs_f64(),
            self.frames,
            self.crossings,
        );
        eprintln!(
            "game session: {:?}; {} distinct chunks, at most {} held at once",
            self.stats, self.distinct, self.max_resident
        );
        eprintln!(
            "game session: late chunk-frames {}, worst lateness {:.2} s, first late {:?}",
            self.late_chunk_frames, self.worst_late_s, self.first_late
        );
        eprintln!(
            "game session: {} holes in view: {:?}",
            self.holes.len(),
            self.holes
        );
        eprintln!(
            "game session: main thread per frame p50 {:.3} ms, p99 {:.3} ms, max {:.3} ms ({} \
             events in the busiest frame)",
            ms(self.frame_cost(0.5)),
            ms(self.frame_cost(0.99)),
            ms(self.frame_cost(1.0)),
            self.busiest_frame_events,
        );
        eprintln!(
            "game session: violations inside chunks {}, across {} seam checks {}",
            self.violations_inside, self.seams_checked, self.violations_across_seams
        );
        eprintln!(
            "game session: chunks walked back to: {} compared, {} different {:?}, {} not decided \
             by their coordinate alone",
            self.revisits_compared,
            self.revisits_different.len(),
            self.revisits_different,
            self.revisits_not_comparable,
        );
        eprintln!(
            "game session: walkable faces continued inside chunks {} of {} ({}), across seams {} \
             of {} ({}); largest network holds {} of the walkable cells at the end",
            self.walk_inside.continued,
            self.walk_inside.walkable,
            share(self.walk_inside),
            self.walk_across_seams.continued,
            self.walk_across_seams.walkable,
            share(self.walk_across_seams),
            self.largest_network_share
                .map_or_else(|| "none".to_owned(), |s| format!("{s:.3}")),
        );
        eprintln!("game session: rendered {}", render.display());
    }
}

#[test]
#[ignore = "needs a compute device; run with --ignored in release mode"]
fn no_chunk_in_view_is_ever_missing_while_it_generates() {
    let session = session();

    assert_eq!(
        session.late_chunk_frames, 0,
        "a chunk in view had no tiles yet: first {:?}, worst for {:.2} s",
        session.first_late, session.worst_late_s
    );
}

#[test]
#[ignore = "needs a compute device; run with --ignored in release mode"]
fn no_chunk_in_view_is_a_hole() {
    let session = session();

    assert!(
        session.holes.is_empty(),
        "{} chunks in view could not be placed: {:?}",
        session.holes.len(),
        session.holes
    );
}

#[test]
#[ignore = "needs a compute device; run with --ignored in release mode"]
fn no_generated_cell_breaks_a_rule_inside_a_chunk_or_across_a_seam() {
    let session = session();

    assert!(session.seams_checked > 0, "no seam was checked");
    assert_eq!(
        (session.violations_inside, session.violations_across_seams),
        (0, 0)
    );
}

#[test]
#[ignore = "needs a compute device; run with --ignored in release mode"]
fn generation_costs_the_main_thread_under_a_millisecond_a_frame() {
    let session = session();

    let (p99, max) = (session.frame_cost(0.99), session.frame_cost(1.0));

    assert!(
        p99 <= Duration::from_millis(1) && max <= Duration::from_millis(4),
        "p99 {p99:?}, max {max:?} of a {FRAME:?} frame"
    );
}

#[test]
#[ignore = "needs a compute device; run with --ignored in release mode"]
fn memory_stays_bounded_while_the_player_travels() {
    let session = session();
    // A chunk is kept within the radius plus the margin of the focus it was asked for by, or, while
    // a repair needs it, within the repair reach beyond the chunks asked for (the radius and the
    // face neighbours they are solved against); and the player may have moved on by one chunk.
    let kept = (GENERATE_RADIUS + EVICT_MARGIN).max(GENERATE_RADIUS + 1 + wave_forge::REPAIR_REACH);
    let side = 2 * (kept + 1) as usize + 1;

    assert!(
        session.max_resident <= side * side,
        "{} chunks held at once",
        session.max_resident
    );
    assert!(
        session.distinct >= 3 * session.max_resident,
        "the route generated only {} chunks, so dropping them was never needed",
        session.distinct
    );
}

#[test]
#[ignore = "needs a compute device; run with --ignored in release mode"]
fn a_chunk_walked_back_to_comes_back_the_same() {
    let session = session();

    assert!(
        session.revisits_compared >= 20,
        "only {} chunks could be compared",
        session.revisits_compared
    );
    assert!(
        session.revisits_different.is_empty(),
        "{:?}",
        session.revisits_different
    );
}

#[test]
#[ignore = "needs a compute device; run with --ignored in release mode"]
fn seams_cut_no_more_paths_than_chunk_interiors() {
    let session = session();

    let inside = session.walk_inside.share().expect("walkable faces inside");
    let across = session
        .walk_across_seams
        .share()
        .expect("walkable faces across seams");

    assert!(
        across >= inside - 0.05,
        "paths continue through {across:.3} of the walkable faces at seams against {inside:.3} \
         inside chunks"
    );
}

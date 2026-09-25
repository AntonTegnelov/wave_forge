//! Generating a pack's stages around focus points, providers first.
//!
//! A stage's output for a chunk is a pure function of the world seed, the stage, the chunk and what
//! its inputs hold within the stage's reach of that chunk (docs/architecture/stages.md, "The
//! execution contract"). The runtime makes that hold by construction: before a stage runs for a
//! chunk, every chunk of its inputs that the chunk's area grown by the reach overlaps is generated,
//! and the stage reads them only through a view bounded by that area. Whatever order chunks are
//! asked for in, each is computed from the same inputs and comes out the same.

use super::assemble::Growth;
use super::edits::{Edit, Edits};
use super::evaluate::{Leaves, evaluate, holds};
use super::facts::{Facts, Row, RowId, Table};
use super::network::SitePaths;
use super::pack::{
    Column, Expr, LocationKind, MAX_SCATTER_SLOTS, Output, Pack, Persist, Profile, Reach, Stage,
    StageKind, TableKind, point_stage_id, salt,
};
use super::regions::{Attempt, Curve, CurveId, RegionInput, RegionJob, region_of};
use super::rivers::DownhillRivers;
use super::save::{FrozenChunk, Save};
use super::town_thread::{Done, Job, Stopped, TownKey, TownThread};
use crate::noise::NoiseConfig;
use crate::products::InstanceId;
use crate::scheduler::FocusPoint;
use crate::towns::{Town, TownSolver};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;
use wfc_core::ChunkCoord;
use wfc_core::hash::pcg3d;

/// A value per cell column of one chunk.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Field {
    pub chunk: ChunkCoord,
    /// Columns along the lattice's x and y.
    pub size: [u32; 2],
    /// Row by row, x fastest.
    pub values: Vec<f32>,
}

impl Field {
    /// The value of the column at `x`, `y` within the chunk.
    ///
    /// # Panics
    /// If the column is outside the chunk.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> f32 {
        assert!(x < self.size[0] && y < self.size[1], "column ({x}, {y})");
        self.values[(y * self.size[0] + x) as usize]
    }
}

/// What names a site, and the town on it.
#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Deserialize, serde::Serialize,
)]
pub enum SiteId {
    /// A Sites stage's site: the region that owns it, which has at most one.
    Region(i32, i32),
    /// A TableSites stage's site: the row it stands for.
    Row(RowId),
    /// A Locations stage's site: its region and its place in the order the region placed its
    /// sites.
    Location { region: (i32, i32), index: u32 },
}

/// A settlement site: a rectangle of whole chunks at one height.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Site {
    pub id: SiteId,
    /// The kind a location table gave it; none for a Sites or TableSites stage's site.
    pub kind: Option<Arc<str>>,
    /// The chunks it covers, from `min` up to but not including `max`, along the lattice's x and y.
    pub min: (i32, i32),
    pub max: (i32, i32),
    /// The height its ground is levelled to.
    pub height: f32,
}

/// A place's name for a game to show: a translation key and the arguments a translation may use,
/// never a finished string, so every language names places its own way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaceName {
    /// `wf-place-<kind>`, the kind's name with its underscores as hyphens: lowercase ASCII with
    /// hyphens, which a Fluent message id, a `.po` msgid and a CSV key all accept.
    pub key: String,
    /// `region_x` and `region_y`, the region that placed the site, and `index`, its place among
    /// the region's sites, so a translation can tell sites of one kind apart.
    pub args: Vec<(&'static str, i64)>,
}

impl Site {
    /// The site's name, for a site a location table placed; a Sites or TableSites stage's site
    /// has none, and a row's name is the game's.
    #[must_use]
    pub fn name(&self) -> Option<PlaceName> {
        let SiteId::Location { region, index } = self.id else {
            return None;
        };
        let kind = self.kind.as_deref()?;
        Some(PlaceName {
            key: format!("wf-place-{}", kind.replace('_', "-")),
            args: vec![
                ("region_x", i64::from(region.0)),
                ("region_y", i64::from(region.1)),
                ("index", i64::from(index)),
            ],
        })
    }

    /// How far the world column `x`, `y` is from the site's footprint, in columns, for chunks of
    /// `size` columns; zero inside it.
    #[must_use]
    pub fn distance(&self, x: i64, y: i64, size: [u32; 2]) -> f32 {
        let span = |at: i64, min: i32, max: i32, size: u32| {
            let low = i64::from(min) * i64::from(size);
            let high = i64::from(max) * i64::from(size) - 1;
            (low - at).max(at - high).max(0) as f32
        };
        let dx = span(x, self.min.0, self.max.0, size[0]);
        let dy = span(y, self.min.1, self.max.1, size[1]);
        (dx * dx + dy * dy).sqrt()
    }

    fn overlaps(&self, chunk: ChunkCoord) -> bool {
        (self.min.0..self.max.0).contains(&chunk.x) && (self.min.1..self.max.1).contains(&chunk.y)
    }
}

/// One chunk of a site's town.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct TownChunk {
    /// The site's id, which names the town.
    pub site: SiteId,
    /// The site's levelled height, where an engine puts the town's lowest layer.
    pub height: f32,
    /// The chunk's tiles, x fastest, then y, then z, as a WFC chunk stores them.
    pub tiles: Arc<[u16]>,
}

/// A piece an Assemble stage placed: a prefab an engine binds a scene to by the piece's name.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Stamp {
    /// Positional: the chunk and the cell of its footprint's centre, the stage, and its place in
    /// its assembly's growth, so it never changes when other assemblies do.
    pub id: InstanceId,
    /// The site it was grown on.
    pub site: SiteId,
    /// The piece's name in the stage.
    pub piece: Arc<str>,
    /// The centre of its footprint along the lattice's x and y, and its floor's height, in cells.
    pub position: [f32; 3],
    /// Its turn about the vertical from the piece as authored, as a fraction of a whole turn: 0,
    /// 0.25, 0.5 or 0.75, from +x toward +y.
    pub turn: f32,
    /// The columns it covers, from `min` up to but not including `max`.
    pub min: [i64; 2],
    pub max: [i64; 2],
}

impl Stamp {
    /// Its turn in a Y-up engine's axes (the lattice's x, the height, the lattice's y), as the rows
    /// of a 3x3 matrix, exact for each quarter turn: a quarter turn takes the piece's +x to the
    /// lattice's +y.
    #[must_use]
    pub fn y_up_basis(&self) -> [[f32; 3]; 3] {
        let quarter = (self.turn * 4.0).round() as i32;
        let (sin, cos) = match quarter.rem_euclid(4) {
            0 => (0.0, 1.0),
            1 => (1.0, 0.0),
            2 => (0.0, -1.0),
            _ => (-1.0, 0.0),
        };
        [[cos, 0.0, -sin], [0.0, 1.0, 0.0], [sin, 0.0, cos]]
    }
}

/// A point a Scatter stage placed.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Point {
    /// Positional: the chunk, the stage and the column, so it never changes when other points do.
    pub id: InstanceId,
    /// What an engine binds a scene or a model to.
    pub kind: Arc<str>,
    /// Where it stands, in cells: x and y along the lattice's ground, z the height field's value.
    pub position: [f32; 3],
    /// Its turn about its own up, as a fraction of a whole turn.
    pub turn: f32,
    /// How much larger than its model it stands, 1 for as large.
    pub scale: f32,
    /// Its up, a unit vector along the lattice's x, y and height: straight up unless the stage
    /// tilts it or stands it along the ground.
    pub up: [f32; 3],
}

impl Point {
    /// Its rotation and scale in a Y-up engine's axes (the lattice's x, the height, the lattice's
    /// y), as the rows of a 3x3 matrix: turned by `turn` about the vertical, leant so the vertical
    /// becomes `up`, and scaled by `scale`.
    #[must_use]
    pub fn y_up_basis(&self) -> [[f32; 3]; 3] {
        let (sin, cos) = (self.turn * std::f32::consts::TAU).sin_cos();
        let turned = [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]];
        // The rotation taking the vertical to `up` about the axis perpendicular to both.
        let up = [self.up[0], self.up[2], self.up[1]];
        let (axis, lean_sin, lean_cos) = {
            let cross = [up[2], 0.0, -up[0]];
            let length = cross[0].hypot(cross[2]);
            if length < 1e-6 {
                ([1.0, 0.0, 0.0], 0.0, up[1].signum())
            } else {
                ([cross[0] / length, 0.0, cross[2] / length], length, up[1])
            }
        };
        let [kx, ky, kz] = axis;
        let skew = [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]];
        let mut leant = [[0.0; 3]; 3];
        for (row, leant_row) in leant.iter_mut().enumerate() {
            for (col, value) in leant_row.iter_mut().enumerate() {
                let identity = f32::from(u8::from(row == col));
                let squared: f32 = (0..3).map(|k| skew[row][k] * skew[k][col]).sum();
                *value = identity + skew[row][col] * lean_sin + squared * (1.0 - lean_cos);
            }
        }
        let mut basis = [[0.0; 3]; 3];
        for (row, basis_row) in basis.iter_mut().enumerate() {
            for (col, value) in basis_row.iter_mut().enumerate() {
                let sum: f32 = (0..3).map(|k| leant[row][k] * turned[k][col]).sum();
                *value = sum * self.scale;
            }
        }
        basis
    }
}

/// A category per cell column of one chunk: an index into the categories its Rules stage names
/// ([`crate::stages::StageKind::categories`]).
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Categories {
    pub chunk: ChunkCoord,
    /// Columns along the lattice's x and y.
    pub size: [u32; 2],
    /// Row by row, x fastest.
    pub values: Vec<u8>,
}

impl Categories {
    /// The category of the column at `x`, `y` within the chunk.
    ///
    /// # Panics
    /// If the column is outside the chunk.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> u8 {
        assert!(x < self.size[0] && y < self.size[1], "column ({x}, {y})");
        self.values[(y * self.size[0] + x) as usize]
    }
}

/// What a stage holds for one chunk.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Product {
    Field(Field),
    Categories(Categories),
    /// The sites whose footprint overlaps the chunk.
    Sites(Vec<Site>),
    /// The chunk's part of a town, or nothing for a chunk outside every site.
    Tiles(Option<TownChunk>),
    /// The points whose column lies in the chunk.
    Points(Vec<Point>),
    /// The curves of the chunk's region that pass through the chunk.
    Curves(Vec<Curve>),
    /// The pieces of assemblies whose footprint overlaps the chunk.
    Stamps(Vec<Stamp>),
}

impl Product {
    /// The footprints of a sites stage's sites or an Assemble stage's pieces, each with what names
    /// it.
    fn levelled(&self, size: [u32; 2]) -> Vec<(LevelledId, Levelled)> {
        let columns = |chunk: (i32, i32)| {
            [
                i64::from(chunk.0) * i64::from(size[0]),
                i64::from(chunk.1) * i64::from(size[1]),
            ]
        };
        match self {
            Self::Sites(sites) => sites
                .iter()
                .map(|site| {
                    let footprint = Levelled {
                        min: columns(site.min),
                        max: columns(site.max),
                        height: site.height,
                    };
                    (LevelledId::Site(site.id.clone()), footprint)
                })
                .collect(),
            Self::Stamps(stamps) => stamps
                .iter()
                .map(|stamp| {
                    let footprint = Levelled {
                        min: stamp.min,
                        max: stamp.max,
                        height: stamp.position[2],
                    };
                    (LevelledId::Stamp(stamp.id), footprint)
                })
                .collect(),
            Self::Field(_)
            | Self::Categories(_)
            | Self::Tiles(_)
            | Self::Points(_)
            | Self::Curves(_) => {
                unreachable!("inputs are type checked when the pack loads")
            }
        }
    }

    fn sites(&self) -> &[Site] {
        match self {
            Self::Sites(sites) => sites,
            Self::Field(_)
            | Self::Categories(_)
            | Self::Tiles(_)
            | Self::Points(_)
            | Self::Curves(_)
            | Self::Stamps(_) => {
                unreachable!("inputs are type checked when the pack loads")
            }
        }
    }
}

/// A site and the pieces grown on it.
type Assembly = (Site, Arc<[Stamp]>);

/// What names a footprint, so one seen from several chunks is counted once.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum LevelledId {
    Site(SiteId),
    Stamp(InstanceId),
}

/// A rectangle of columns at a height: a site's or an assembled piece's, which Flatten levels the
/// ground under and Scatter keeps a margin from.
#[derive(Clone, Copy, Debug)]
struct Levelled {
    /// The columns it covers, from `min` up to but not including `max`.
    min: [i64; 2],
    max: [i64; 2],
    height: f32,
}

impl Levelled {
    /// How far the column `x`, `y` is from it, in columns; zero inside it.
    fn distance(&self, x: i64, y: i64) -> f32 {
        let span = |at: i64, low: i64, high: i64| (low - at).max(at - (high - 1)).max(0) as f32;
        let dx = span(x, self.min[0], self.max[0]);
        let dy = span(y, self.min[1], self.max[1]);
        (dx * dx + dy * dy).sqrt()
    }
}

/// What generating one stage has cost so far: how many products, and the milliseconds they took
/// in all and at most. A Solve stage's time includes the towns it solved.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StageTiming {
    pub products: u64,
    pub ms: f64,
    pub slowest_ms: f64,
}

/// Why generating a stage failed.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StageError {
    #[error("no stage is named {0:?}")]
    UnknownStage(String),
    /// A stage read an input further from its chunk than its reach allows. That is a bug in the
    /// stage: its reach has to be declared as large as what it reads.
    #[error(
        "stage {stage:?} read {input:?} {needed} cells beyond its chunk, but its reach is {reach}"
    )]
    OutOfReach {
        stage: String,
        input: String,
        reach: u32,
        needed: u32,
    },
    #[error("stage {0:?} solves towns, but the runtime was given no town solver")]
    NoTownSolver(String),
    #[error("stage {0:?} names a region job the runtime was not given")]
    NoRegionJob(String),
    /// Only stages computed column by column from what they read (Field, Rules, Blur, Delta and
    /// Area stages over such stages) can be sampled without chunks.
    #[error(
        "stage {0:?} cannot be sampled without chunks: it is not a field, rules, blur, delta or area \
         stage"
    )]
    NotSampled(String),
    #[error("stage {stage:?} gave up on region {region:?} after {} attempts: {}", log.len(), log.join("; "))]
    RegionRejected {
        stage: String,
        region: (i32, i32),
        log: Vec<String>,
    },
    #[error("the pack names no noise {0:?}")]
    UnknownNoise(String),
    #[error("an edit cannot be applied: {0}")]
    Edit(String),
    #[error("the pack gives the world no bound")]
    Unbounded,
    #[error("no table is named {0:?}")]
    UnknownTable(String),
    #[error("table {table:?}: {message}")]
    Table { table: String, message: String },
    #[error("a stage reads the focused row of table {0:?}, but no row of it is focused")]
    NoFocus(String),
    #[error("the runtime was given no facts, which a stage or a focus reads")]
    NoFacts,
    #[error("the facts were made for another pack or seed than the runtime's")]
    OtherFacts,
    #[error("stage {stage:?} cannot draw curve {curve:?}: {message}")]
    Curve {
        stage: String,
        curve: CurveId,
        message: String,
    },
    #[error("the town solver's chunks are {solver:?} columns, the runtime's {runtime:?}")]
    ChunkMismatch { solver: [u32; 2], runtime: [u32; 2] },
    #[error("the town solver's thread stopped")]
    TownsStopped,
    #[error("stage {stage:?} could not solve the town of {site:?}: {message}")]
    Town {
        stage: String,
        site: SiteId,
        message: String,
    },
    /// An Assemble stage's assembly on a site stayed under its least number of pieces after
    /// every reroll.
    #[error("stage {stage:?} grew fewer than {min} pieces on {site:?} after every reroll")]
    Assemble {
        stage: String,
        site: SiteId,
        min: u32,
    },
}

/// One input of a stage, readable only within the stage's reach of the chunk being generated: a
/// field's values, or a Rules stage's category indices.
pub struct FieldView<'a> {
    stage: &'a str,
    input: &'a str,
    /// The reach along the lattice's x, which is what errors report.
    reach: u32,
    size: [u32; 2],
    /// The columns the view may read, in world columns of the reading stage, inclusive.
    min: [i64; 2],
    max: [i64; 2],
    /// The input's columns per column of the reading stage: 1 at the same scale, more for a coarser
    /// input, which a field is read from between its columns and a category from the one it covers.
    ratio: u32,
    chunks: BTreeMap<(i32, i32), &'a Product>,
}

impl FieldView<'_> {
    /// The input's value at the world column `x`, `y`: a field's value, or the index of a
    /// category.
    ///
    /// # Errors
    /// [`StageError::OutOfReach`] if the column is further from the chunk than the reach.
    pub fn get(&self, x: i64, y: i64) -> Result<f32, StageError> {
        let beyond =
            |value: i64, axis: usize| (self.min[axis] - value).max(value - self.max[axis]).max(0);
        let outside = beyond(x, 0).max(beyond(y, 1));
        if outside > 0 {
            return Err(StageError::OutOfReach {
                stage: self.stage.to_owned(),
                input: self.input.to_owned(),
                reach: self.reach,
                needed: self.reach + u32::try_from(outside).expect("a small distance"),
            });
        }
        let categorical = matches!(self.chunks.values().next(), Some(Product::Categories(_)));
        between(self.ratio, categorical, x, y, |x, y| Ok(self.at(x, y)))
    }

    /// The input's own value at one of its columns, which the view holds.
    fn at(&self, x: i64, y: i64) -> f32 {
        let (cx, cy) = (i64::from(self.size[0]), i64::from(self.size[1]));
        let chunk = (
            i32::try_from(x.div_euclid(cx)).expect("a chunk coordinate"),
            i32::try_from(y.div_euclid(cy)).expect("a chunk coordinate"),
        );
        let product = self
            .chunks
            .get(&chunk)
            .expect("the runtime generates every chunk within reach first");
        let (x, y) = (x.rem_euclid(cx) as u32, y.rem_euclid(cy) as u32);
        match product {
            Product::Field(field) => field.get(x, y),
            Product::Categories(categories) => f32::from(categories.get(x, y)),
            Product::Sites(_)
            | Product::Tiles(_)
            | Product::Points(_)
            | Product::Curves(_)
            | Product::Stamps(_) => {
                unreachable!("inputs are type checked when the pack loads")
            }
        }
    }
}

/// A Region stage's index and one of its regions.
type RegionKey = (usize, (i32, i32));

/// How an expression reads its inputs: a stage's value at one of the reading stage's columns,
/// already brought to the reading stage's scale.
type Read<'r> = dyn Fn(&str, i64, i64) -> Result<f32, StageError> + 'r;

/// Generates a pack's stages for the chunks focus points ask for.
pub struct Runtime {
    pack: Arc<Pack>,
    seed: u64,
    /// Columns per chunk along the lattice's x and y.
    size: [u32; 2],
    /// Which chunks of each stage the current request needs.
    needed: BTreeMap<usize, BTreeSet<ChunkCoord>>,
    focus: Vec<FocusPoint>,
    products: BTreeMap<(usize, ChunkCoord), Arc<Product>>,
    towns: Option<TownThread>,
    /// Towns asked of the town thread and not back yet, by Solve stage and site, with the ticket
    /// of the request and the site, dropped as solved towns are.
    solving: BTreeMap<(usize, SiteId), (u64, Site)>,
    /// The ticket of the next town request.
    ticket: u64,
    /// Towns solved, by Solve stage and site, with the site, kept while a chunk it covers is
    /// needed.
    solved: BTreeMap<(usize, SiteId), (Site, Arc<Town>)>,
    /// Assemblies grown, by Assemble stage and site, with the site, kept as towns are.
    assembled: BTreeMap<(usize, SiteId), Assembly>,
    /// What each stage has cost, by stage index.
    timings: Vec<StageTiming>,
    /// The region jobs Region stages name, by name.
    region_jobs: BTreeMap<String, Box<dyn RegionJob>>,
    /// Regions computed, by Region stage and region, kept while a chunk of their region is needed.
    regions: BTreeMap<RegionKey, Arc<[Curve]>>,
    /// Location tables placed, by Locations stage and region, kept while a chunk of their region
    /// is needed.
    placed: BTreeMap<RegionKey, Arc<Placed>>,
    facts: Option<Facts>,
    /// The player's edits, folded from their log, applied to every product as it is generated.
    edits: Folded,
    /// The log the edits were folded from, which a save keeps.
    log: Edits,
    /// Every chunk of a frozen stage as it was first generated, before the edits, by stage index
    /// and chunk: what it is from then on, and what a save keeps.
    frozen: BTreeMap<(usize, ChunkCoord), Arc<Product>>,
    /// The noises Field expressions read by name: the pack's, with any the engine replaced.
    noises: BTreeMap<String, NoiseConfig>,
    /// The focused row of each table that has one, by table index.
    focused: BTreeMap<usize, Row>,
    /// Each TableSites stage's sites, by stage index: the row each stands for and the chunks it
    /// covers, from the facts.
    footprints: BTreeMap<usize, Arc<[Footprint]>>,
}

/// A location table placed in one region: its sites, and a line per kind on what it placed and
/// refused.
struct Placed {
    sites: Vec<Site>,
    log: Vec<String>,
}

/// A log of edits folded into what generation looks up.
#[derive(Clone, Debug, Default, PartialEq)]
struct Folded {
    removed: BTreeSet<InstanceId>,
    /// Where each moved point stands now, and its turn.
    moved: BTreeMap<InstanceId, ([f32; 3], f32)>,
    /// What each field stage's column has been raised by in all, by stage index and column.
    raised: BTreeMap<(usize, (i64, i64)), f32>,
    /// Where each removed or moved point stood when it was generated.
    stood: BTreeMap<InstanceId, [f32; 2]>,
}

/// Which chunks of a stage no longer hold what they would be generated as now.
enum Stale {
    All,
    Chunks(BTreeSet<ChunkCoord>),
}

/// Where a table's row puts its site: the row, and the chunks from `min` up to but not including
/// `max`.
#[derive(Clone, Debug, PartialEq)]
struct Footprint {
    row: RowId,
    min: (i32, i32),
    max: (i32, i32),
}

impl Runtime {
    /// A runtime for `pack`, with chunks of `size` columns.
    #[must_use]
    pub fn new(pack: Arc<Pack>, seed: u64, size: [u32; 2]) -> Self {
        Self {
            timings: vec![StageTiming::default(); pack.stages.len()],
            noises: pack.noises.clone(),
            region_jobs: BTreeMap::new(),
            regions: BTreeMap::new(),
            placed: BTreeMap::new(),
            pack,
            seed,
            size,
            needed: BTreeMap::new(),
            focus: Vec::new(),
            products: BTreeMap::new(),
            towns: None,
            solving: BTreeMap::new(),
            ticket: 0,
            solved: BTreeMap::new(),
            assembled: BTreeMap::new(),
            facts: None,
            edits: Folded::default(),
            log: Edits::default(),
            frozen: BTreeMap::new(),
            focused: BTreeMap::new(),
            footprints: BTreeMap::new(),
        }
    }

    /// Gives the runtime the facts its stages read, replacing any it had, and drops every product
    /// that read a table that changed, returning them as (stage, chunk) as
    /// [`Runtime::request`] does; the request regenerates them. On an error nothing changes.
    ///
    /// # Errors
    /// [`StageError::OtherFacts`] if the facts were made for another pack or seed, and
    /// [`StageError::Table`] if a focused row is not in its table any more, or a TableSites stage's
    /// row has a position that is not finite, a size that is not a whole number from 1 to its
    /// `max_size`, or a site within a chunk of another row's.
    pub fn set_facts(&mut self, facts: Facts) -> Result<Vec<(String, ChunkCoord)>, StageError> {
        if !facts.made_for(&self.pack, self.seed) {
            return Err(StageError::OtherFacts);
        }
        let mut footprints = BTreeMap::new();
        for (index, stage) in self.pack.stages.iter().enumerate() {
            match stage.kind {
                StageKind::TableSites { .. } => {
                    footprints.insert(index, self.footprints_of(stage, &facts)?);
                }
                StageKind::TableCurves { .. } => self.check_curves(index, &facts)?,
                _ => {}
            }
        }
        let mut focused = BTreeMap::new();
        for (&table, row) in &self.focused {
            let now = facts.tables()[table]
                .row(&row.id)
                .ok_or_else(|| StageError::Table {
                    table: self.pack.tables[table].name.clone(),
                    message: format!(
                        "the focused row {:?} is not in the table any more",
                        row.id.0
                    ),
                })?;
            focused.insert(table, now.clone());
        }
        let mut stale = BTreeMap::new();
        for (index, stage) in self.pack.stages.iter().enumerate() {
            let chunks = match &stage.kind {
                StageKind::TableSites { .. } => {
                    self.moved_sites(index, &facts, &footprints[&index])
                }
                StageKind::TableCurves { .. } => self.moved_curves(index, &facts),
                _ if stage
                    .tables
                    .iter()
                    .any(|table| self.focused.get(table) != focused.get(table)) =>
                {
                    Stale::All
                }
                _ => continue,
            };
            stale.insert(index, chunks);
        }
        self.facts = Some(facts);
        self.focused = focused;
        self.footprints = footprints;
        Ok(self.invalidate(stale))
    }

    /// Whether every row of TableCurves stage `index` makes a curve: a finite start and end, and a
    /// radius from 0 to the `max_radius` of every Apply stage that draws it.
    fn check_curves(&self, index: usize, facts: &Facts) -> Result<(), StageError> {
        let stage = &self.pack.stages[index];
        let StageKind::TableCurves { table, .. } = &stage.kind else {
            unreachable!("called for TableCurves stages")
        };
        let widest = self
            .pack
            .stages
            .iter()
            .filter_map(|reader| match reader.kind {
                StageKind::Apply { max_radius, .. }
                    if reader.inputs.iter().any(|&(input, _)| input == index) =>
                {
                    Some(max_radius)
                }
                _ => None,
            })
            .min();
        let rows = facts.table(table).expect("linked when loaded");
        for row in &rows.rows {
            let curve = self.table_curve(index, rows, row);
            let radius = curve.values[0];
            let finite = curve.points.iter().flatten().all(|value| value.is_finite());
            if !finite || !(0.0..=widest.map_or(f32::MAX, |widest| widest as f32)).contains(&radius)
            {
                return Err(StageError::Table {
                    table: table.clone(),
                    message: format!(
                        "stage {:?}: row {:?} makes a curve through {:?} of radius {radius}; \
                         points are finite and a radius is from 0 to {}",
                        stage.name,
                        row.id.0,
                        curve.points,
                        widest.map_or("any".to_owned(), |widest| widest.to_string())
                    ),
                });
            }
        }
        Ok(())
    }

    /// The straight curve TableCurves stage `index` makes of `row` of `rows`.
    fn table_curve(&self, index: usize, rows: &Table, row: &Row) -> Curve {
        let StageKind::TableCurves {
            from, to, radius, ..
        } = &self.pack.stages[index].kind
        else {
            unreachable!("called for TableCurves stages")
        };
        let value = |column: &str| row.values[rows.column(column).expect("checked when loaded")];
        let radius = value(radius);
        Curve {
            id: CurveId::Row(row.id.clone()),
            points: vec![
                [value(&from.0), value(&from.1)],
                [value(&to.0), value(&to.1)],
            ],
            values: vec![radius, radius],
        }
    }

    /// The chunks TableCurves stage `index` holds differently with `facts`: those every row added,
    /// removed or changed passes through, where it was and where it is.
    fn moved_curves(&self, index: usize, facts: &Facts) -> Stale {
        let Some(old) = &self.facts else {
            return Stale::All;
        };
        let StageKind::TableCurves { table, .. } = &self.pack.stages[index].kind else {
            unreachable!("called for TableCurves stages")
        };
        let (before, after) = (
            old.table(table).expect("linked when loaded"),
            facts.table(table).expect("linked when loaded"),
        );
        let mut chunks = BTreeSet::new();
        for (rows, others) in [(before, after), (after, before)] {
            for row in &rows.rows {
                if others.row(&row.id) == Some(row) {
                    continue;
                }
                chunks.extend(self.chunks_touched(&self.table_curve(index, rows, row)));
            }
        }
        Stale::Chunks(chunks)
    }

    /// The chunks whose columns a curve passes through, as a chunk's product counts them.
    fn chunks_touched(&self, curve: &Curve) -> Vec<ChunkCoord> {
        let [sx, sy] = self.size.map(|size| size as f32);
        let (mut low, mut high) = ([f32::MAX; 2], [f32::MIN; 2]);
        for point in &curve.points {
            for axis in 0..2 {
                low[axis] = low[axis].min(point[axis]);
                high[axis] = high[axis].max(point[axis]);
            }
        }
        let (x0, x1) = (
            (low[0] / sx).floor() as i32 - 1,
            (high[0] / sx).floor() as i32 + 1,
        );
        let (y0, y1) = (
            (low[1] / sy).floor() as i32 - 1,
            (high[1] / sy).floor() as i32 + 1,
        );
        (x0..=x1)
            .flat_map(|x| (y0..=y1).map(move |y| ChunkCoord::new(x, y, 0)))
            .filter(|&chunk| {
                let (min, max) = self.chunk_rect(chunk);
                curve.touches(min, max)
            })
            .collect()
    }

    /// A chunk's columns as `Curve::touches` takes them: from its lowest corner to its highest.
    fn chunk_rect(&self, chunk: ChunkCoord) -> ([f32; 2], [f32; 2]) {
        let [sx, sy] = self.size.map(|size| size as f32);
        let min = [chunk.x as f32 * sx, chunk.y as f32 * sy];
        (min, [min[0] + sx, min[1] + sy])
    }

    /// The chunks TableSites stage `index` holds differently with `facts`, whose sites are
    /// `footprints`: those of every row added, removed or changed, where it was and where it is.
    fn moved_sites(&self, index: usize, facts: &Facts, footprints: &[Footprint]) -> Stale {
        let (Some(old), Some(was)) = (&self.facts, self.footprints.get(&index)) else {
            return Stale::All;
        };
        let StageKind::TableSites { table, .. } = &self.pack.stages[index].kind else {
            unreachable!("called for TableSites stages")
        };
        let (before, after) = (
            old.table(table).expect("linked when loaded"),
            facts.table(table).expect("linked when loaded"),
        );
        let mut chunks = BTreeSet::new();
        for footprint in was.iter().chain(footprints) {
            if before.row(&footprint.row) != after.row(&footprint.row) {
                for x in footprint.min.0..footprint.max.0 {
                    for y in footprint.min.1..footprint.max.1 {
                        chunks.insert(ChunkCoord::new(x, y, 0));
                    }
                }
            }
        }
        Stale::Chunks(chunks)
    }

    /// The sites a TableSites stage's rows put down, checked: every position finite, every size a
    /// whole number from 1 to the stage's `max_size`, and no two sites within a chunk of each
    /// other.
    fn footprints_of(&self, stage: &Stage, facts: &Facts) -> Result<Arc<[Footprint]>, StageError> {
        let StageKind::TableSites {
            table,
            at,
            size,
            max_size,
            ..
        } = &stage.kind
        else {
            unreachable!("called for TableSites stages")
        };
        let rows = facts.table(table).expect("linked when loaded");
        let column = |name: &str| rows.column(name).expect("checked when loaded");
        let (x, y, side) = (column(&at.0), column(&at.1), column(size));
        let failed = |message: String| StageError::Table {
            table: table.clone(),
            message: format!("stage {:?}: {message}", stage.name),
        };
        let mut footprints: Vec<Footprint> = Vec::with_capacity(rows.rows.len());
        for row in &rows.rows {
            let (at, side) = ([row.values[x], row.values[y]], row.values[side]);
            if !(at[0].is_finite() && at[1].is_finite()) {
                return Err(failed(format!("row {:?} stands at {at:?}", row.id.0)));
            }
            if side.fract() != 0.0 || !(1.0..=*max_size as f32).contains(&side) {
                return Err(failed(format!(
                    "row {:?} has a site of {side} chunks; 1 to {max_size} are allowed",
                    row.id.0
                )));
            }
            let chunk = (
                (at[0] / self.size[0] as f32).floor() as i32,
                (at[1] / self.size[1] as f32).floor() as i32,
            );
            let side = side as i32;
            let min = (chunk.0 - (side - 1) / 2, chunk.1 - (side - 1) / 2);
            let footprint = Footprint {
                row: row.id.clone(),
                min,
                max: (min.0 + side, min.1 + side),
            };
            // Sites of a Sites stage keep a chunk between them, which Flatten and Solve rely on.
            if let Some(other) = footprints.iter().find(|other| {
                other.min.0 <= footprint.max.0
                    && footprint.min.0 <= other.max.0
                    && other.min.1 <= footprint.max.1
                    && footprint.min.1 <= other.max.1
            }) {
                return Err(failed(format!(
                    "the sites of rows {:?} and {:?} come within a chunk of each other",
                    other.row.0, row.id.0
                )));
            }
            footprints.push(footprint);
        }
        Ok(Arc::from(footprints))
    }

    /// Focuses the runtime on the row `id` of `table`, which stages read through
    /// [`crate::stages::Expr::Row`], and drops every product that read another row of it,
    /// returning them as [`Runtime::set_facts`] does.
    ///
    /// # Errors
    /// [`StageError::NoFacts`] before [`Runtime::set_facts`], [`StageError::UnknownTable`] for a
    /// table the pack does not name, and [`StageError::Table`] for a row it does not hold.
    pub fn focus(
        &mut self,
        table: &str,
        id: RowId,
    ) -> Result<Vec<(String, ChunkCoord)>, StageError> {
        let facts = self.facts.as_ref().ok_or(StageError::NoFacts)?;
        let index = *self
            .pack
            .table_by_name
            .get(table)
            .ok_or_else(|| StageError::UnknownTable(table.to_owned()))?;
        let row = facts.tables()[index]
            .row(&id)
            .ok_or_else(|| StageError::Table {
                table: table.to_owned(),
                message: format!("it has no row {:?}", id.0),
            })?;
        if self.focused.get(&index) == Some(row) {
            return Ok(Vec::new());
        }
        self.focused.insert(index, row.clone());
        let stale = self
            .pack
            .stages
            .iter()
            .enumerate()
            .filter(|(_, stage)| {
                stage.tables.contains(&index) && !matches!(stage.kind, StageKind::TableSites { .. })
            })
            .map(|(reader, _)| (reader, Stale::All))
            .collect();
        Ok(self.invalidate(stale))
    }

    /// Drops what `stale` names and everything generated from it, returning the products as
    /// (stage, chunk). A chunk of a stage is stale when its reach covers a stale chunk of one of
    /// its inputs, and a town or a region goes with any stale chunk it covers; what is still asked
    /// for is generated again.
    fn invalidate(&mut self, mut stale: BTreeMap<usize, Stale>) -> Vec<(String, ChunkCoord)> {
        // Inputs come first in `order`, so a stage's inputs are judged before it.
        for &index in &self.pack.order {
            if matches!(stale.get(&index), Some(Stale::All)) {
                continue;
            }
            let stage = &self.pack.stages[index];
            // A chunk asked for may already have a town or a region behind it, so those count
            // with the chunks held.
            let candidates: BTreeSet<ChunkCoord> = self
                .needed
                .get(&index)
                .into_iter()
                .flatten()
                .copied()
                .chain(
                    self.products
                        .keys()
                        .filter(|(held, _)| *held == index)
                        .map(|&(_, chunk)| chunk),
                )
                .collect();
            let mut chunks = match stale.remove(&index) {
                Some(Stale::Chunks(chunks)) => chunks,
                Some(Stale::All) => unreachable!("skipped above"),
                None => BTreeSet::new(),
            };
            let mut all = false;
            for &(input, reach) in &stage.inputs {
                match stale.get(&input) {
                    None => {}
                    Some(Stale::All) => all = true,
                    Some(Stale::Chunks(inputs)) => {
                        let cells = reach.cells(self.size);
                        for &chunk in &candidates {
                            let (min, max) = self.reader_box(chunk, cells);
                            if self
                                .covering(stage.scale, input, min, max)
                                .iter()
                                .any(|covered| inputs.contains(covered))
                            {
                                chunks.insert(chunk);
                            }
                        }
                    }
                }
            }
            if all {
                stale.insert(index, Stale::All);
            } else if !chunks.is_empty() {
                stale.insert(index, Stale::Chunks(chunks));
            }
        }
        let is_stale = |stage: usize, chunk: ChunkCoord| match stale.get(&stage) {
            None => false,
            Some(Stale::All) => true,
            Some(Stale::Chunks(chunks)) => chunks.contains(&chunk),
        };
        let mut dropped = Vec::new();
        self.products.retain(|&(stage, chunk), _| {
            let gone = is_stale(stage, chunk);
            if gone {
                dropped.push((self.pack.stages[stage].name.clone(), chunk));
            }
            !gone
        });
        let fresh = |stage: usize, site: &Site| match stale.get(&stage) {
            None => true,
            Some(Stale::All) => false,
            Some(Stale::Chunks(chunks)) => !chunks.iter().any(|&chunk| site.overlaps(chunk)),
        };
        self.solved
            .retain(|&(stage, _), (site, _)| fresh(stage, site));
        self.solving
            .retain(|&(stage, _), (_, site)| fresh(stage, site));
        self.assembled
            .retain(|&(stage, _), (site, _)| fresh(stage, site));
        let pack = Arc::clone(&self.pack);
        self.regions.retain(|&(stage, region), _| {
            let size = pack.stages[stage]
                .kind
                .job_region()
                .expect("only region jobs compute regions");
            match stale.get(&stage) {
                None => true,
                Some(Stale::All) => false,
                Some(Stale::Chunks(chunks)) => {
                    !chunks.iter().any(|&chunk| region_of(chunk, size) == region)
                }
            }
        });
        self.placed.retain(|&(stage, region), _| {
            let StageKind::Locations { region: size, .. } = pack.stages[stage].kind else {
                unreachable!("only Locations stages place locations")
            };
            match stale.get(&stage) {
                None => true,
                Some(Stale::All) => false,
                Some(Stale::Chunks(chunks)) => {
                    !chunks.iter().any(|&chunk| region_of(chunk, size) == region)
                }
            }
        });
        dropped
    }

    /// Replaces the noise the pack names `name` with `noise`, a Godot `FastNoiseLite` resource's
    /// configuration say, for every expression that reads it.
    ///
    /// # Errors
    /// [`StageError::UnknownNoise`] if the pack names no such noise.
    pub fn with_noise(mut self, name: &str, noise: NoiseConfig) -> Result<Self, StageError> {
        let known = self
            .noises
            .get_mut(name)
            .ok_or_else(|| StageError::UnknownNoise(name.to_owned()))?;
        *known = noise;
        Ok(self)
    }

    /// Gives the runtime the player's edits, replacing any it had, and drops every product the
    /// change reaches, returning them as [`Runtime::request`] does; the request generates them
    /// again, with the edits. Only the chunks whose edits changed, and what reads them within its
    /// reach, are dropped. On an error nothing changes.
    ///
    /// # Errors
    /// [`StageError::Edit`] if an edit raises a stage that is not a field or names a point of no
    /// Scatter stage of the pack.
    pub fn set_edits(&mut self, edits: &Edits) -> Result<Vec<(String, ChunkCoord)>, StageError> {
        let folded = self.fold(edits)?;
        let mut stale: BTreeMap<usize, BTreeSet<ChunkCoord>> = BTreeMap::new();
        let [sx, sy] = [i64::from(self.size[0]), i64::from(self.size[1])];
        let keys: BTreeSet<(usize, (i64, i64))> = self
            .edits
            .raised
            .keys()
            .chain(folded.raised.keys())
            .copied()
            .collect();
        for key in keys {
            if self.edits.raised.get(&key) != folded.raised.get(&key) {
                let (stage, (x, y)) = key;
                let chunk = ChunkCoord::new(
                    i32::try_from(x.div_euclid(sx)).expect("a chunk coordinate"),
                    i32::try_from(y.div_euclid(sy)).expect("a chunk coordinate"),
                    0,
                );
                stale.entry(stage).or_default().insert(chunk);
            }
        }
        let points: BTreeSet<InstanceId> = self
            .edits
            .stood
            .keys()
            .chain(folded.stood.keys())
            .copied()
            .collect();
        for id in points {
            let before = (self.edits.removed.contains(&id), self.edits.moved.get(&id));
            let after = (folded.removed.contains(&id), folded.moved.get(&id));
            if before != after {
                let at = folded.stood.get(&id).or_else(|| self.edits.stood.get(&id));
                let at = at.expect("a point edit knows where its point stood");
                let chunk = ChunkCoord::new(
                    (at[0] / sx as f32).floor() as i32,
                    (at[1] / sy as f32).floor() as i32,
                    0,
                );
                stale
                    .entry(
                        self.point_stage(id)
                            .expect("folded edits name a Scatter stage"),
                    )
                    .or_default()
                    .insert(chunk);
            }
        }
        self.edits = folded;
        self.log = edits.clone();
        Ok(self.invalidate(
            stale
                .into_iter()
                .map(|(stage, chunks)| (stage, Stale::Chunks(chunks)))
                .collect(),
        ))
    }

    /// What a save keeps of the world: the edits, less those of ephemeral stages, and every chunk
    /// of a frozen stage generated so far, with the Wave Forge version and the pack's digest.
    #[must_use]
    pub fn save(&self) -> Save {
        let ephemeral = |stage: usize| self.pack.stages[stage].persist == Persist::Ephemeral;
        let kept = |edit: &&Edit| match edit {
            Edit::Remove { point, .. } | Edit::Move { point, .. } => self
                .point_stage(InstanceId::from(*point))
                .is_ok_and(|stage| !ephemeral(stage)),
            Edit::Raise { stage, .. } => self
                .pack
                .index(stage)
                .is_some_and(|stage| !ephemeral(stage)),
        };
        Save {
            generator: env!("CARGO_PKG_VERSION").to_owned(),
            pack: self.pack.digest(),
            edits: Edits {
                log: self.log.log.iter().filter(kept).cloned().collect(),
            },
            frozen: self
                .frozen
                .iter()
                .map(|(&(stage, chunk), product)| FrozenChunk {
                    stage: self.pack.stages[stage].name.clone(),
                    chunk,
                    product: Product::clone(product),
                })
                .collect(),
        }
    }

    /// Brings a world back from `save`: its edits, and the chunks of the stages this pack
    /// freezes as they were first generated, even if the pack has changed since. A frozen chunk
    /// of a stage this pack does not freeze, or does not have, is left out. Drops what the save
    /// changes and returns it as [`Runtime::request`] does. On an error nothing changes.
    ///
    /// # Errors
    /// [`StageError::Edit`] as [`Runtime::set_edits`].
    pub fn load(&mut self, save: &Save) -> Result<Vec<(String, ChunkCoord)>, StageError> {
        let mut dropped = self.set_edits(&save.edits)?;
        let mut stale: BTreeMap<usize, Stale> = BTreeMap::new();
        for frozen in &save.frozen {
            let Some(index) = self
                .pack
                .index(&frozen.stage)
                .filter(|&index| self.pack.stages[index].persist == Persist::Frozen)
            else {
                continue;
            };
            self.frozen
                .insert((index, frozen.chunk), Arc::new(frozen.product.clone()));
            match stale
                .entry(index)
                .or_insert_with(|| Stale::Chunks(BTreeSet::new()))
            {
                Stale::Chunks(chunks) => {
                    chunks.insert(frozen.chunk);
                }
                Stale::All => unreachable!("frozen chunks are staled one by one"),
            }
        }
        dropped.extend(self.invalidate(stale));
        Ok(dropped)
    }

    /// The edits of `log` as lookups, checked against the pack.
    fn fold(&self, edits: &Edits) -> Result<Folded, StageError> {
        let mut folded = Folded::default();
        for edit in &edits.log {
            match edit {
                Edit::Remove { point, at } => {
                    let id = InstanceId::from(*point);
                    self.point_stage(id)?;
                    folded.moved.remove(&id);
                    folded.removed.insert(id);
                    folded.stood.entry(id).or_insert(*at);
                }
                Edit::Move {
                    point,
                    from,
                    to,
                    turn,
                } => {
                    let id = InstanceId::from(*point);
                    self.point_stage(id)?;
                    folded.removed.remove(&id);
                    folded.moved.insert(id, (*to, *turn));
                    folded.stood.entry(id).or_insert(*from);
                }
                Edit::Raise { stage, column, by } => {
                    let index = self
                        .pack
                        .index(stage)
                        .filter(|&index| self.pack.stages[index].kind.output() == Output::Field)
                        .ok_or_else(|| {
                            StageError::Edit(format!("{stage:?} is no field stage to raise"))
                        })?;
                    *folded.raised.entry((index, *column)).or_default() += by;
                }
            }
        }
        Ok(folded)
    }

    /// The Scatter stage that placed the point `id`.
    fn point_stage(&self, id: InstanceId) -> Result<usize, StageError> {
        self.pack
            .stages
            .iter()
            .position(|stage| {
                matches!(stage.kind, StageKind::Scatter { .. })
                    && point_stage_id(stage.salt) == id.stage()
            })
            .ok_or_else(|| StageError::Edit(format!("no Scatter stage placed the point {id:?}")))
    }

    /// `product` of stage `index` with the player's edits applied: raises added to a field, and a
    /// Scatter stage's points removed or moved.
    fn edited(&self, index: usize, product: Product) -> Product {
        match product {
            Product::Field(mut field) => {
                let [sx, sy] = [i64::from(field.size[0]), i64::from(field.size[1])];
                let (x0, y0) = (i64::from(field.chunk.x) * sx, i64::from(field.chunk.y) * sy);
                for (&(_, (x, y)), by) in self
                    .edits
                    .raised
                    .range((index, (i64::MIN, i64::MIN))..=(index, (i64::MAX, i64::MAX)))
                {
                    if (x0..x0 + sx).contains(&x) && (y0..y0 + sy).contains(&y) {
                        field.values[((y - y0) * sx + (x - x0)) as usize] += by;
                    }
                }
                Product::Field(field)
            }
            Product::Points(points) => Product::Points(
                points
                    .into_iter()
                    .filter(|point| !self.edits.removed.contains(&point.id))
                    .map(|mut point| {
                        if let Some(&(to, turn)) = self.edits.moved.get(&point.id) {
                            point.position = to;
                            point.turn = turn;
                        }
                        point
                    })
                    .collect(),
            ),
            other @ (Product::Categories(_)
            | Product::Sites(_)
            | Product::Tiles(_)
            | Product::Curves(_)
            | Product::Stamps(_)) => other,
        }
    }

    /// Gives Region stages that name `name` the job they run.
    #[must_use]
    pub fn with_region_job(mut self, name: &str, job: impl RegionJob + 'static) -> Self {
        self.region_jobs.insert(name.to_owned(), Box::new(job));
        self
    }

    /// Gives Solve stages the solver their towns are solved with.
    ///
    /// # Errors
    /// [`StageError::ChunkMismatch`] if the solver's chunks are not the runtime's.
    pub fn with_towns(mut self, towns: Box<dyn TownSolver>) -> Result<Self, StageError> {
        let shape = towns.chunk_shape();
        if [shape.x, shape.y] != self.size {
            return Err(StageError::ChunkMismatch {
                solver: [shape.x, shape.y],
                runtime: self.size,
            });
        }
        self.towns = Some(TownThread::spawn(towns));
        Ok(self)
    }

    /// Asks for the `targets` stages in the chunks around `focus`, replacing the previous request,
    /// and drops what the new request does not need, returning it as (stage, chunk).
    ///
    /// # Errors
    /// [`StageError::UnknownStage`] if no stage is named like one of `targets`.
    pub fn request(
        &mut self,
        focus: &[FocusPoint],
        targets: &[&str],
    ) -> Result<Vec<(String, ChunkCoord)>, StageError> {
        let targets: Vec<(&str, Option<u32>)> =
            targets.iter().map(|&target| (target, None)).collect();
        self.request_around(focus, &targets)
    }

    /// Asks for each of `targets` in the chunks within its own radius of every focus point, or the
    /// focus point's radius for a target given none: ground far out, locations nearer and clutter
    /// nearest, say. Otherwise as [`Runtime::request`].
    ///
    /// # Errors
    /// [`StageError::UnknownStage`] if no stage is named like one of `targets`.
    pub fn request_each(
        &mut self,
        focus: &[FocusPoint],
        targets: &[(&str, Option<u32>)],
    ) -> Result<Vec<(String, ChunkCoord)>, StageError> {
        self.request_around(focus, targets)
    }

    /// Each target within its own radius of every focus point, or the focus point's radius where
    /// it has none.
    fn request_around(
        &mut self,
        focus: &[FocusPoint],
        targets: &[(&str, Option<u32>)],
    ) -> Result<Vec<(String, ChunkCoord)>, StageError> {
        let targets = targets
            .iter()
            .map(|&(target, radius)| {
                self.pack
                    .index(target)
                    .map(|index| (index, radius))
                    .ok_or_else(|| StageError::UnknownStage(target.to_owned()))
            })
            .collect::<Result<Vec<(usize, Option<u32>)>, StageError>>()?;
        let around = |radius: Option<u32>| -> BTreeSet<ChunkCoord> {
            focus
                .iter()
                .flat_map(|focus| {
                    let radius = radius.unwrap_or(focus.radius) as i32;
                    let centre = focus.chunk;
                    (-radius..=radius).flat_map(move |x| {
                        (-radius..=radius)
                            .map(move |y| ChunkCoord::new(centre.x + x, centre.y + y, 0))
                    })
                })
                .collect()
        };
        // Focus points are in the WFC lattice's chunks; a coarser target's chunks cover several. A
        // target's chunk wholly outside the world's bound is never asked for.
        let mut needed: BTreeMap<usize, BTreeSet<ChunkCoord>> = targets
            .into_iter()
            .map(|(target, radius)| {
                let scale = self.pack.stages[target].scale as i32;
                let chunks = around(radius)
                    .iter()
                    .map(|c| ChunkCoord::new(c.x.div_euclid(scale), c.y.div_euclid(scale), 0))
                    .filter(|&chunk| self.within_bound(target, chunk))
                    .collect();
                (target, chunks)
            })
            .collect();
        // Consumers come after their inputs in `order`, so walking it backwards reaches a stage
        // only once everything that reads it has said which of its chunks it needs.
        for &index in self.pack.order.iter().rev() {
            let Some(chunks) = needed.get(&index).cloned() else {
                continue;
            };
            let scale = self.pack.stages[index].scale;
            for &(input, reach) in &self.pack.stages[index].inputs {
                let reach = reach.cells(self.size);
                let covered: BTreeSet<ChunkCoord> = chunks
                    .iter()
                    .flat_map(|&chunk| {
                        let (min, max) = self.reader_box(chunk, reach);
                        self.covering(scale, input, min, max)
                    })
                    .collect();
                needed.entry(input).or_default().extend(covered);
            }
        }
        let mut dropped = Vec::new();
        self.products.retain(|&(stage, chunk), _| {
            let keep = needed
                .get(&stage)
                .is_some_and(|chunks| chunks.contains(&chunk));
            if !keep {
                dropped.push((self.pack.stages[stage].name.clone(), chunk));
            }
            keep
        });
        let pack = Arc::clone(&self.pack);
        let covered = |stage: usize, site: &Site| {
            needed
                .get(&stage)
                .is_some_and(|chunks| chunks.iter().any(|&chunk| site.overlaps(chunk)))
        };
        self.solved
            .retain(|&(stage, _), (site, _)| covered(stage, site));
        self.solving
            .retain(|&(stage, _), (_, site)| covered(stage, site));
        self.assembled
            .retain(|&(stage, _), (site, _)| covered(stage, site));
        // A bounded world has few regions, so it keeps every one it has computed, as a finite
        // world computes them once before streaming.
        let finite = pack.bound.is_some();
        self.regions.retain(|&(stage, region), _| {
            let size = pack.stages[stage]
                .kind
                .job_region()
                .expect("only region jobs compute regions");
            finite
                || needed.get(&stage).is_some_and(|chunks| {
                    chunks.iter().any(|&chunk| region_of(chunk, size) == region)
                })
        });
        self.placed.retain(|&(stage, region), _| {
            let StageKind::Locations { region: size, .. } = pack.stages[stage].kind else {
                unreachable!("only Locations stages place locations")
            };
            finite
                || needed.get(&stage).is_some_and(|chunks| {
                    chunks.iter().any(|&chunk| region_of(chunk, size) == region)
                })
        });
        self.needed = needed;
        self.focus = focus.to_vec();
        Ok(dropped)
    }

    /// Asks for the `targets` stages in every chunk that meets the world's bound, replacing the
    /// previous request as [`Runtime::request`] does: how a finite world computes its regions,
    /// region jobs and location tables before play, which it then keeps.
    ///
    /// # Errors
    /// [`StageError::Unbounded`] if the pack has no bound, and [`StageError::UnknownStage`] as
    /// [`Runtime::request`].
    pub fn request_bound(
        &mut self,
        targets: &[&str],
    ) -> Result<Vec<(String, ChunkCoord)>, StageError> {
        let bound = *self.pack.bound.as_ref().ok_or(StageError::Unbounded)?;
        let (low, high) = bound.extent();
        let [sx, sy] = self.size.map(|size| size as f32);
        let chunk = |at: f32, side: f32| (at / side).floor() as i32;
        let focus: Vec<FocusPoint> = (chunk(low[1], sy)..=chunk(high[1], sy))
            .flat_map(|y| (chunk(low[0], sx)..=chunk(high[0], sx)).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                let min = [x as f32 * sx, y as f32 * sy];
                bound.meets(min, [min[0] + sx, min[1] + sy])
            })
            .map(|(x, y)| FocusPoint::new(ChunkCoord::new(x, y, 0), 0))
            .collect();
        self.request(&focus, targets)
    }

    /// Whether any column of `chunk` of stage `index` lies inside the world's bound; every chunk
    /// does without one.
    fn within_bound(&self, index: usize, chunk: ChunkCoord) -> bool {
        let Some(bound) = &self.pack.bound else {
            return true;
        };
        let scale = self.pack.stages[index].scale as f32;
        let [sx, sy] = self.size.map(|size| size as f32 * scale);
        let min = [chunk.x as f32 * sx, chunk.y as f32 * sy];
        bound.meets(min, [min[0] + sx, min[1] + sy])
    }

    /// Generates everything the request needs that is missing, inputs first and nearest first,
    /// and returns what it generated as (stage, chunk).
    ///
    /// # Errors
    /// A [`StageError`] from a stage, which is a bug in that stage.
    pub fn run_until_idle(&mut self) -> Result<Vec<(String, ChunkCoord)>, StageError> {
        let mut generated = self.step(usize::MAX)?;
        // Every other stage is generated; what is left waits for towns.
        while !self.solving.is_empty() {
            self.receive_towns(Duration::MAX)?;
            generated.extend(self.step(usize::MAX)?);
        }
        Ok(generated)
    }

    /// Whether everything the request needs is generated.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.needed.iter().all(|(&index, chunks)| {
            chunks
                .iter()
                .all(|chunk| self.products.contains_key(&(index, *chunk)))
        })
    }

    /// Generates at most `budget` of the missing products, stage by stage with inputs first and
    /// each stage's chunks nearest first, and returns them as (stage, chunk). A worker calls this
    /// between looking for new requests.
    ///
    /// # Errors
    /// A [`StageError`] from a stage, which is a bug in that stage.
    pub fn step(&mut self, budget: usize) -> Result<Vec<(String, ChunkCoord)>, StageError> {
        self.receive_towns(Duration::ZERO)?;
        let mut generated = Vec::new();
        for &index in &self.pack.order.clone() {
            if generated.len() >= budget {
                break;
            }
            let Some(chunks) = self.needed.get(&index) else {
                continue;
            };
            let mut missing: Vec<ChunkCoord> = chunks
                .iter()
                .copied()
                .filter(|chunk| !self.products.contains_key(&(index, *chunk)))
                .collect();
            let scale = self.pack.stages[index].scale as i32;
            missing.sort_by_key(|chunk| {
                // Focus points are in the WFC lattice's chunks, so a coarse chunk is measured from
                // its first one.
                let first = ChunkCoord::new(chunk.x * scale, chunk.y * scale, 0);
                let distance = self
                    .focus
                    .iter()
                    .map(|focus| focus.distance(first))
                    .min()
                    .unwrap_or(u32::MAX);
                (distance, *chunk)
            });
            for chunk in missing.into_iter().take(budget - generated.len()) {
                if !self.frozen.contains_key(&(index, chunk)) {
                    self.request_town_of(index, chunk)?;
                    if self.waits_for_town(index, chunk) {
                        continue;
                    }
                }
                let started = std::time::Instant::now();
                let product = match self.frozen.get(&(index, chunk)) {
                    Some(frozen) => Product::clone(frozen),
                    None => {
                        self.assemble_of(index, chunk)?;
                        self.run_region_of(index, chunk)?;
                        self.place_locations_of(index, chunk)?;
                        let product = self.generate(index, chunk)?;
                        if self.pack.stages[index].persist == Persist::Frozen {
                            self.frozen
                                .insert((index, chunk), Arc::new(product.clone()));
                        }
                        product
                    }
                };
                let product = self.edited(index, product);
                let ms = started.elapsed().as_secs_f64() * 1000.0;
                let timing = &mut self.timings[index];
                timing.products += 1;
                timing.ms += ms;
                timing.slowest_ms = timing.slowest_ms.max(ms);
                self.products.insert((index, chunk), Arc::new(product));
                generated.push((self.pack.stages[index].name.clone(), chunk));
            }
        }
        Ok(generated)
    }

    /// What each stage has cost since the runtime was made, in the order the pack lists them.
    #[must_use]
    pub fn timings(&self) -> Vec<(String, StageTiming)> {
        self.pack
            .stages
            .iter()
            .zip(&self.timings)
            .map(|(stage, timing)| (stage.name.clone(), *timing))
            .collect()
    }

    /// What `stage` holds for `chunk`, if it has been generated and is still needed.
    #[must_use]
    pub fn product(&self, stage: &str, chunk: ChunkCoord) -> Option<&Product> {
        self.shared(stage, chunk).map(Arc::as_ref)
    }

    /// The same product, shared, so a worker can hand it to another thread without a copy.
    pub(crate) fn shared(&self, stage: &str, chunk: ChunkCoord) -> Option<&Arc<Product>> {
        let index = self.pack.index(stage)?;
        self.products.get(&(index, chunk))
    }

    /// The field `stage` holds for `chunk`, if it is a field stage and the chunk is generated.
    #[must_use]
    pub fn field(&self, stage: &str, chunk: ChunkCoord) -> Option<&Field> {
        match self.product(stage, chunk)? {
            Product::Field(field) => Some(field),
            Product::Sites(_)
            | Product::Tiles(_)
            | Product::Points(_)
            | Product::Categories(_)
            | Product::Curves(_)
            | Product::Stamps(_) => None,
        }
    }

    /// The categories `stage` holds for `chunk`, if it is a Rules stage and the chunk is
    /// generated.
    #[must_use]
    pub fn categories(&self, stage: &str, chunk: ChunkCoord) -> Option<&Categories> {
        match self.product(stage, chunk)? {
            Product::Categories(categories) => Some(categories),
            Product::Field(_)
            | Product::Sites(_)
            | Product::Tiles(_)
            | Product::Points(_)
            | Product::Curves(_)
            | Product::Stamps(_) => None,
        }
    }

    /// A stage's value at a point in WFC cells without generating any chunk: a field's value, or a
    /// category's index, at the column of the stage the point lies in. It equals what the chunk
    /// holding that column would hold. Only Field, Rules, Blur, Delta and Area stages whose inputs
    /// are too can be sampled; the pack's other kinds need neighbouring chunks. A runtime made only to sample
    /// never holds a product, so a game can keep one on any thread.
    ///
    /// # Errors
    /// [`StageError::UnknownStage`] for a stage the pack does not name, and
    /// [`StageError::NotSampled`] for a stage that is, or reads, one of the other kinds.
    pub fn sample(&self, stage: &str, at: [f32; 2]) -> Result<f32, StageError> {
        let index = self
            .pack
            .index(stage)
            .ok_or_else(|| StageError::UnknownStage(stage.to_owned()))?;
        let scale = self.pack.stages[index].scale as f32;
        let column = [
            (at[0] / scale).floor() as i64,
            (at[1] / scale).floor() as i64,
        ];
        self.sample_column(index, column, &RefCell::new(HashMap::new()))
    }

    /// A stage's values over `size` of its own columns from `min`, row by row with x fastest,
    /// sampled as [`Runtime::sample`] samples them: a world map, one value per column of a coarse
    /// stage, for a game to read before play.
    ///
    /// # Errors
    /// As [`Runtime::sample`].
    pub fn atlas(
        &self,
        stage: &str,
        min: [i64; 2],
        size: [u32; 2],
    ) -> Result<Vec<f32>, StageError> {
        let index = self
            .pack
            .index(stage)
            .ok_or_else(|| StageError::UnknownStage(stage.to_owned()))?;
        let memo = RefCell::new(HashMap::new());
        let mut values = Vec::with_capacity((size[0] * size[1]) as usize);
        for y in 0..i64::from(size[1]) {
            for x in 0..i64::from(size[0]) {
                values.push(self.sample_column(index, [min[0] + x, min[1] + y], &memo)?);
            }
        }
        Ok(values)
    }

    /// Stage `index`'s value at one of its columns, from its inputs' values sampled the same way,
    /// each column computed once per call.
    fn sample_column(
        &self,
        index: usize,
        column: [i64; 2],
        memo: &RefCell<HashMap<(usize, [i64; 2]), f32>>,
    ) -> Result<f32, StageError> {
        if let Some(&value) = memo.borrow().get(&(index, column)) {
            return Ok(value);
        }
        let stage = &self.pack.stages[index];
        let read = |name: &str, x: i64, y: i64| {
            let input = self.pack.index(name).expect("linked when loaded");
            let ratio = self.pack.stages[input].scale / stage.scale;
            let categorical = self.pack.stages[input].kind.output() == Output::Categories;
            between(ratio, categorical, x, y, |x, y| {
                self.sample_column(input, [x, y], memo)
            })
        };
        let value = match &stage.kind {
            StageKind::Field(expr) => evaluate(expr, &self.place(index, column, &read))?,
            StageKind::Rules { .. } => f32::from(self.categorise(index, column, &read)?),
            StageKind::Blur { input, radius } => blur(input, *radius, column, &read)?,
            StageKind::Delta { input, radius } => delta(input, *radius, column, &read)?,
            StageKind::Area { input, distance } => {
                f32::from(area(input, *distance, column, &read)?)
            }
            StageKind::Sites { .. }
            | StageKind::TableSites { .. }
            | StageKind::Locations { .. }
            | StageKind::TableCurves { .. }
            | StageKind::Apply { .. }
            | StageKind::Flatten { .. }
            | StageKind::Solve { .. }
            | StageKind::Scatter { .. }
            | StageKind::Assemble { .. }
            | StageKind::Region { .. }
            | StageKind::Rivers { .. }
            | StageKind::Network { .. } => return Err(StageError::NotSampled(stage.name.clone())),
        };
        // A sample is what the chunk holds, raises included.
        let value = value
            + self
                .edits
                .raised
                .get(&(index, (column[0], column[1])))
                .copied()
                .unwrap_or(0.0);
        memo.borrow_mut().insert((index, column), value);
        Ok(value)
    }

    /// The curves `stage` holds for `chunk`: those of its region that pass through it, if it is a
    /// Region stage and the chunk is generated.
    #[must_use]
    pub fn curves(&self, stage: &str, chunk: ChunkCoord) -> Option<&[Curve]> {
        match self.product(stage, chunk)? {
            Product::Curves(curves) => Some(curves),
            Product::Field(_)
            | Product::Categories(_)
            | Product::Sites(_)
            | Product::Tiles(_)
            | Product::Points(_)
            | Product::Stamps(_) => None,
        }
    }

    /// The pieces `stage` placed whose footprint overlaps `chunk`, if it is an Assemble stage and
    /// the chunk is generated.
    #[must_use]
    pub fn stamps(&self, stage: &str, chunk: ChunkCoord) -> Option<&[Stamp]> {
        match self.product(stage, chunk)? {
            Product::Stamps(stamps) => Some(stamps),
            Product::Field(_)
            | Product::Categories(_)
            | Product::Sites(_)
            | Product::Tiles(_)
            | Product::Points(_)
            | Product::Curves(_) => None,
        }
    }

    /// The sites `stage` holds for `chunk`, if it is a sites stage and the chunk is generated.
    #[must_use]
    pub fn sites(&self, stage: &str, chunk: ChunkCoord) -> Option<&[Site]> {
        match self.product(stage, chunk)? {
            Product::Sites(sites) => Some(sites),
            Product::Field(_)
            | Product::Tiles(_)
            | Product::Points(_)
            | Product::Categories(_)
            | Product::Curves(_)
            | Product::Stamps(_) => None,
        }
    }

    /// The town chunk `stage` holds for `chunk`: `None` if it is not a Solve stage, the chunk is
    /// not generated, or it lies outside every site.
    #[must_use]
    pub fn tiles(&self, stage: &str, chunk: ChunkCoord) -> Option<&TownChunk> {
        match self.product(stage, chunk)? {
            Product::Tiles(town) => town.as_ref(),
            Product::Field(_)
            | Product::Sites(_)
            | Product::Points(_)
            | Product::Categories(_)
            | Product::Curves(_)
            | Product::Stamps(_) => None,
        }
    }

    /// The points `stage` placed in `chunk`, if it is a Scatter stage and the chunk is generated.
    #[must_use]
    pub fn points(&self, stage: &str, chunk: ChunkCoord) -> Option<&[Point]> {
        match self.product(stage, chunk)? {
            Product::Points(points) => Some(points),
            Product::Field(_)
            | Product::Sites(_)
            | Product::Tiles(_)
            | Product::Categories(_)
            | Product::Curves(_)
            | Product::Stamps(_) => None,
        }
    }

    /// The site of a Solve stage's chunk, if the chunk lies in one.
    fn site_of(&self, index: usize, chunk: ChunkCoord) -> Option<Site> {
        let (sites, reach) = self.pack.stages[index].inputs[0];
        self.inputs_within(index, chunk, sites, reach.cells(self.size))
            .find(|&(at, _)| at == chunk)
            .and_then(|(_, product)| product.sites().first().cloned())
    }

    /// Computes the region of Region stage `index` that `chunk` lies in, if it is not computed yet:
    /// the job's attempts in turn until one is accepted or the budget is spent.
    fn run_region_of(&mut self, index: usize, chunk: ChunkCoord) -> Result<(), StageError> {
        let stage = &self.pack.stages[index];
        let rivers;
        let paths;
        let (job, size, halo, budget): (&dyn RegionJob, &u32, &u32, &u32) = match &stage.kind {
            StageKind::Region {
                job,
                region,
                halo,
                budget,
                ..
            } => {
                let job = self
                    .region_jobs
                    .get(job)
                    .ok_or_else(|| StageError::NoRegionJob(stage.name.clone()))?;
                (job.as_ref(), region, halo, budget)
            }
            StageKind::Rivers {
                height,
                region,
                sources,
                sea,
                width,
                step,
            } => {
                rivers = DownhillRivers {
                    height,
                    sources: *sources,
                    sea: *sea,
                    width: *width,
                    step: *step,
                };
                (&rivers, region, &0, &1)
            }
            StageKind::Network {
                height,
                region,
                width,
                climb,
                dry,
                ..
            } => {
                paths = SitePaths {
                    height,
                    width: *width,
                    climb: *climb,
                    dry: *dry,
                };
                (&paths, region, &0, &1)
            }
            _ => return Ok(()),
        };
        let region = region_of(chunk, *size);
        if self.regions.contains_key(&(index, region)) {
            return Ok(());
        }
        let [sx, sy] = [i64::from(self.size[0]), i64::from(self.size[1])];
        let (size, halo) = (i64::from(*size), i64::from(*halo));
        // The region and its halo, in the stage's own columns.
        let min = [
            (i64::from(region.0) * size - halo) * sx,
            (i64::from(region.1) * size - halo) * sy,
        ];
        let max = [
            (i64::from(region.0) * size + size + halo) * sx - 1,
            (i64::from(region.1) * size + size + halo) * sy - 1,
        ];
        let fields =
            |&&(input, _): &&(usize, Reach)| self.pack.stages[input].kind.output() == Output::Field;
        let views = stage
            .inputs
            .iter()
            .filter(fields)
            .map(|&(input, _)| {
                let view = self.view_over(index, input, (min, max), (halo * sx) as u32);
                (self.pack.stages[input].name.as_str(), view)
            })
            .collect();
        // A Network stage's sites: those whose centre lies in the region, once each.
        let mut sites: BTreeMap<SiteId, Site> = BTreeMap::new();
        for &(input, _) in &stage.inputs {
            if self.pack.stages[input].kind.output() != Output::Sites {
                continue;
            }
            for y in 0..size {
                for x in 0..size {
                    let at = ChunkCoord::new(
                        (i64::from(region.0) * size + x) as i32,
                        (i64::from(region.1) * size + y) as i32,
                        0,
                    );
                    let product = self
                        .products
                        .get(&(input, at))
                        .expect("a region job's inputs are generated over its region first");
                    for site in product.sites() {
                        sites.insert(site.id.clone(), site.clone());
                    }
                }
            }
        }
        let inside = |site: &Site| {
            let centre = |low: i32, high: i32| (i64::from(low) + i64::from(high)) / 2;
            region_of(
                ChunkCoord::new(
                    centre(site.min.0, site.max.0) as i32,
                    centre(site.min.1, site.max.1) as i32,
                    0,
                ),
                size as u32,
            ) == region
        };
        let mut input = RegionInput {
            region,
            size: size as u32,
            chunk: self.size,
            retry: 0,
            world: (self.seed as u32) ^ ((self.seed >> 32) as u32),
            salt: stage.salt,
            views,
            sites: sites.into_values().filter(|site| inside(site)).collect(),
        };
        let mut log = Vec::new();
        for retry in 0..*budget {
            input.retry = retry;
            match job.run(&input)? {
                Attempt::Accepted(curves) => {
                    self.regions.insert((index, region), Arc::from(curves));
                    return Ok(());
                }
                Attempt::Rejected(reason) => log.push(reason),
            }
        }
        Err(StageError::RegionRejected {
            stage: stage.name.clone(),
            region,
            log,
        })
    }

    /// Places the location table of Locations stage `index` in the region `chunk` lies in, if it
    /// is not placed yet: the kinds in order of priority, each trying its footprints in turn.
    fn place_locations_of(&mut self, index: usize, chunk: ChunkCoord) -> Result<(), StageError> {
        let stage = &self.pack.stages[index];
        let StageKind::Locations {
            region: size,
            kinds,
            ..
        } = &stage.kind
        else {
            return Ok(());
        };
        let region = region_of(chunk, *size);
        if self.placed.contains_key(&(index, region)) {
            return Ok(());
        }
        let [sx, sy] = [i64::from(self.size[0]), i64::from(self.size[1])];
        let side = i64::from(*size);
        // The region's own columns: every footprint lies inside them, and so does its centre.
        let min = [
            i64::from(region.0) * side * sx,
            i64::from(region.1) * side * sy,
        ];
        let max = [min[0] + side * sx - 1, min[1] + side * sy - 1];
        let views: BTreeMap<usize, FieldView<'_>> = stage
            .inputs
            .iter()
            .map(|&(input, reach)| {
                let widen =
                    |axis: usize| i64::from(reach.cells(self.size)[axis]) - side * [sx, sy][axis];
                let (wx, wy) = (widen(0).max(0), widen(1).max(0));
                let area = ([min[0] - wx, min[1] - wy], [max[0] + wx, max[1] + wy]);
                (input, self.view_over(index, input, area, wx as u32))
            })
            .collect();
        let read = |name: &str, x: i64, y: i64| {
            views[&self.pack.index(name).expect("linked when loaded")].get(x, y)
        };
        let StageKind::Locations { height, .. } = &stage.kind else {
            unreachable!("matched above")
        };
        let heights = &views[&self.pack.index(height).expect("linked when loaded")];
        let world = (self.seed as u32) ^ ((self.seed >> 32) as u32);
        let mut order: Vec<&LocationKind> = kinds.iter().collect();
        order.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.name.cmp(&b.name))
        });
        let mut sites: Vec<Site> = Vec::new();
        let mut log = Vec::with_capacity(order.len());
        for kind in order {
            let name: Arc<str> = Arc::from(kind.name.as_str());
            let stream = world ^ stage.salt ^ salt(&kind.name);
            let (mut kept, mut crowded, mut near, mut unmet) = (0, 0, 0, 0);
            for attempt in 0..kind.tries {
                if kept == kind.quota {
                    break;
                }
                let [place, ..] = pcg3d([
                    stream,
                    region.0 as u32,
                    (region.1 as u32) ^ attempt.wrapping_mul(0x9E37_79B9),
                ]);
                let room = size - kind.size - 1;
                let offset = (1 + place % room, 1 + (place >> 16) % room);
                let at = (
                    region.0 * *size as i32 + offset.0 as i32,
                    region.1 * *size as i32 + offset.1 as i32,
                );
                let footprint = (at, (at.0 + kind.size as i32, at.1 + kind.size as i32));
                if sites.iter().any(|site| {
                    site.min.0 <= footprint.1.0
                        && footprint.0.0 <= site.max.0
                        && site.min.1 <= footprint.1.1
                        && footprint.0.1 <= site.max.1
                }) {
                    crowded += 1;
                    continue;
                }
                let centre = |min: (i32, i32), max: (i32, i32)| {
                    (
                        (i64::from(min.0) + i64::from(max.0)) * sx / 2,
                        (i64::from(min.1) + i64::from(max.1)) * sy / 2,
                    )
                };
                let here = centre(footprint.0, footprint.1);
                if sites.iter().any(|site| {
                    site.kind.as_deref() == Some(kind.name.as_str()) && {
                        let there = centre(site.min, site.max);
                        ((here.0 - there.0) as f32).hypot((here.1 - there.1) as f32) < kind.apart
                    }
                }) {
                    near += 1;
                    continue;
                }
                let place = self.place(index, [here.0, here.1], &read);
                let mut meets = true;
                for condition in &kind.when {
                    if !holds(condition, &place)? {
                        meets = false;
                        break;
                    }
                }
                if !meets {
                    unmet += 1;
                    continue;
                }
                sites.push(Site {
                    id: SiteId::Location {
                        region,
                        index: sites.len() as u32,
                    },
                    kind: Some(Arc::clone(&name)),
                    min: footprint.0,
                    max: footprint.1,
                    height: self.footprint_height(footprint.0, footprint.1, heights)?,
                });
                kept += 1;
            }
            log.push(format!(
                "{}: placed {kept} of {}; refused {crowded} crowded, {near} near its kind, {unmet} \
                 failing its conditions",
                kind.name, kind.quota
            ));
        }
        self.placed
            .insert((index, region), Arc::new(Placed { sites, log }));
        Ok(())
    }

    /// What a Locations stage placed in the region `chunk` lies in and why it refused the rest:
    /// a line per kind, in the order they were placed, if the region is placed and still needed.
    #[must_use]
    pub fn location_log(&self, stage: &str, chunk: ChunkCoord) -> Option<&[String]> {
        let index = self.pack.index(stage)?;
        let StageKind::Locations { region, .. } = self.pack.stages[index].kind else {
            return None;
        };
        let placed = self.placed.get(&(index, region_of(chunk, region)))?;
        Some(&placed.log)
    }

    /// Asks the town thread for the town of `chunk`'s site for Solve stage `index`, unless it is
    /// solved or asked for already, or the chunk lies in no site.
    fn request_town_of(&mut self, index: usize, chunk: ChunkCoord) -> Result<(), StageError> {
        let stage = &self.pack.stages[index];
        let StageKind::Solve {
            rules,
            by,
            bottom,
            top,
            ..
        } = &stage.kind
        else {
            return Ok(());
        };
        let Some(site) = self.site_of(index, chunk) else {
            return Ok(());
        };
        let key = (index, site.id.clone());
        if self.solved.contains_key(&key) || self.solving.contains_key(&key) {
            return Ok(());
        }
        let rules = match (&site.id, by) {
            (SiteId::Row(row), Some((column, cases))) => {
                let name = self.name_in_row(stage.inputs[0].0, row, column);
                cases
                    .iter()
                    .find(|(case, _)| *case == name)
                    .map_or(rules, |(_, rules)| rules)
            }
            _ => rules,
        };
        let towns = self
            .towns
            .as_ref()
            .ok_or_else(|| StageError::NoTownSolver(stage.name.clone()))?;
        let [high, low, _] = site_hash(self.seed, stage.salt, &site.id);
        self.ticket += 1;
        towns.send(Job {
            key: TownKey {
                stage: index,
                site: site.id.clone(),
                ticket: self.ticket,
            },
            rules: rules.clone(),
            seed: (u64::from(high) << 32) | u64::from(low),
            size: (
                (site.max.0 - site.min.0) as u32,
                (site.max.1 - site.min.1) as u32,
            ),
            bottom: bottom.clone(),
            top: top.clone(),
        });
        self.solving
            .insert((index, site.id.clone()), (self.ticket, site));
        Ok(())
    }

    /// Whether `chunk` of stage `index` lies in a site whose town is not back from the town
    /// thread yet.
    fn waits_for_town(&self, index: usize, chunk: ChunkCoord) -> bool {
        matches!(self.pack.stages[index].kind, StageKind::Solve { .. })
            && self
                .site_of(index, chunk)
                .is_some_and(|site| !self.solved.contains_key(&(index, site.id)))
    }

    /// Takes the towns the town thread has solved, waiting up to `wait` for the first, and keeps
    /// those still asked for. A Solve stage's time includes the towns it solved.
    fn receive_towns(&mut self, wait: Duration) -> Result<(), StageError> {
        let Some(towns) = &self.towns else {
            return Ok(());
        };
        let done = towns
            .take(wait)
            .map_err(|Stopped| StageError::TownsStopped)?;
        for Done { key, town, ms } in done {
            let at = (key.stage, key.site.clone());
            if !self
                .solving
                .get(&at)
                .is_some_and(|&(ticket, _)| ticket == key.ticket)
            {
                continue;
            }
            let (_, site) = self.solving.remove(&at).expect("looked up above");
            let town = town.map_err(|error| StageError::Town {
                stage: self.pack.stages[key.stage].name.clone(),
                site: key.site,
                message: error.to_string(),
            })?;
            let timing = &mut self.timings[at.0];
            timing.ms += ms;
            timing.slowest_ms = timing.slowest_ms.max(ms);
            self.solved.insert(at, (site, Arc::new(town)));
        }
        Ok(())
    }

    /// Waits up to `wait` for a town the runtime asked its town thread for, when stepping has
    /// nothing else left to do; a worker calls this rather than stepping again at once.
    ///
    /// # Errors
    /// [`StageError::Town`] if the town could not be solved, and [`StageError::TownsStopped`] if
    /// the town thread stopped.
    pub fn wait_for_towns(&mut self, wait: Duration) -> Result<(), StageError> {
        self.receive_towns(wait)
    }

    /// The site of an Assemble stage's chunk, if the chunk lies in one of the kinds it grows on.
    fn assembly_site(&self, index: usize, chunk: ChunkCoord) -> Option<Site> {
        let StageKind::Assemble { kinds, .. } = &self.pack.stages[index].kind else {
            unreachable!("only an Assemble stage's chunks lie in assemblies")
        };
        self.site_of(index, chunk).filter(|site| {
            kinds.is_empty()
                || site
                    .kind
                    .as_deref()
                    .is_some_and(|kind| kinds.iter().any(|known| known == kind))
        })
    }

    /// Grows the assembly of `chunk`'s site for Assemble stage `index`, unless it is grown already
    /// or the chunk lies in no site it grows on.
    fn assemble_of(&mut self, index: usize, chunk: ChunkCoord) -> Result<(), StageError> {
        let stage = &self.pack.stages[index];
        let StageKind::Assemble {
            start,
            pieces,
            max,
            min,
            tries,
            rerolls,
            lift,
            ..
        } = &stage.kind
        else {
            return Ok(());
        };
        let Some(site) = self.assembly_site(index, chunk) else {
            return Ok(());
        };
        if self.assembled.contains_key(&(index, site.id.clone())) {
            return Ok(());
        }
        let growth = Growth {
            pieces,
            start: pieces
                .iter()
                .position(|piece| piece.name == *start)
                .expect("checked when loaded"),
            max: *max,
            min: *min,
            tries: *tries,
            rerolls: *rerolls,
        };
        let [sx, sy] = [i64::from(self.size[0]), i64::from(self.size[1])];
        let [high, low, _] = site_hash(self.seed, stage.salt, &site.id);
        let placed = growth
            .grow(
                [i64::from(site.min.0) * sx, i64::from(site.min.1) * sy],
                [i64::from(site.max.0) * sx, i64::from(site.max.1) * sy],
                [high, low],
            )
            .ok_or_else(|| StageError::Assemble {
                stage: stage.name.clone(),
                site: site.id.clone(),
                min: *min,
            })?;
        let id_stage = point_stage_id(stage.salt);
        let stamps: Arc<[Stamp]> = placed
            .iter()
            .enumerate()
            .map(|(slot, placed)| {
                let [x, y, level] = placed.min;
                let [width, depth, _] = placed.size.map(i64::from);
                let centre = (x + width / 2, y + depth / 2);
                let owner = ChunkCoord::new(
                    i32::try_from(centre.0.div_euclid(sx)).expect("a chunk coordinate"),
                    i32::try_from(centre.1.div_euclid(sy)).expect("a chunk coordinate"),
                    0,
                );
                let cell = (centre.1.rem_euclid(sy) * sx + centre.0.rem_euclid(sx)) as u32;
                Stamp {
                    id: InstanceId::new(owner, id_stage, cell, slot as u16),
                    site: site.id.clone(),
                    piece: Arc::from(pieces[placed.piece].name.as_str()),
                    position: [
                        x as f32 + width as f32 / 2.0,
                        y as f32 + depth as f32 / 2.0,
                        site.height + lift + level as f32,
                    ],
                    turn: f32::from(placed.turn) / 4.0,
                    min: [x, y],
                    max: [x + width, y + depth],
                }
            })
            .collect();
        self.assembled
            .insert((index, site.id.clone()), (site, stamps));
        Ok(())
    }

    /// The name a TableSites stage's row holds in a names column of its table.
    fn name_in_row(&self, sites: usize, row: &RowId, column: &str) -> &str {
        let StageKind::TableSites { table, .. } = &self.pack.stages[sites].kind else {
            unreachable!(
                "a Solve stage chooses by a column only over TableSites, checked when loaded"
            )
        };
        let Some(TableKind::Given { columns }) = self.pack.table(table) else {
            unreachable!("checked when loaded")
        };
        let Some((_, Column::Names(names))) = columns.iter().find(|(name, _)| name == column)
        else {
            unreachable!("checked when loaded")
        };
        let rows = self
            .facts
            .as_ref()
            .and_then(|facts| facts.table(table))
            .expect("a site of a table's row was placed from the facts");
        let value = rows.row(row).expect("a site's row is in its table").values
            [rows.column(column).expect("checked when loaded")];
        &names[value as usize]
    }

    /// How many chunks the runtime holds, over all stages.
    #[must_use]
    pub fn held(&self) -> usize {
        self.products.len()
    }

    /// The columns within `reach` columns of `chunk`'s, in the chunk's own stage's columns, from
    /// the lowest corner to the highest, both included.
    fn reader_box(&self, chunk: ChunkCoord, reach: [u32; 2]) -> ([i64; 2], [i64; 2]) {
        let span = |axis: usize, at: i32| {
            let size = i64::from(self.size[axis]);
            let reach = i64::from(reach[axis]);
            (
                i64::from(at) * size - reach,
                (i64::from(at) + 1) * size - 1 + reach,
            )
        };
        let (x0, x1) = span(0, chunk.x);
        let (y0, y1) = span(1, chunk.y);
        ([x0, y0], [x1, y1])
    }

    /// The chunks of `input` that hold the columns `min..=max` of a stage of `scale`, and the
    /// column around them that an input coarser than the reader is read between.
    fn covering(&self, scale: u32, input: usize, min: [i64; 2], max: [i64; 2]) -> Vec<ChunkCoord> {
        let coarser = self.pack.stages[input].scale;
        let (scale, coarser) = (i64::from(scale), i64::from(coarser));
        let between = if coarser > scale { coarser } else { 0 };
        let span = |axis: usize| {
            let cells = i64::from(self.size[axis]) * coarser;
            let low = (min[axis] * scale - between).div_euclid(cells);
            let high = ((max[axis] + 1) * scale - 1 + between).div_euclid(cells);
            (
                i32::try_from(low).expect("a chunk"),
                i32::try_from(high).expect("a chunk"),
            )
        };
        let ((x0, x1), (y0, y1)) = (span(0), span(1));
        (x0..=x1)
            .flat_map(|x| (y0..=y1).map(move |y| ChunkCoord::new(x, y, 0)))
            .collect()
    }

    /// The products of `input` within `reach` of `chunk` of stage `reader`.
    fn inputs_within(
        &self,
        reader: usize,
        chunk: ChunkCoord,
        input: usize,
        reach: [u32; 2],
    ) -> impl Iterator<Item = (ChunkCoord, &Product)> {
        let (min, max) = self.reader_box(chunk, reach);
        self.covering(self.pack.stages[reader].scale, input, min, max)
            .into_iter()
            .map(move |at| {
                let product = self
                    .products
                    .get(&(input, at))
                    .expect("inputs are generated before the stages that read them");
                (at, product.as_ref())
            })
    }

    /// Stage `stage`'s view of `input` over its own columns `min..=max`.
    fn view_over(
        &self,
        stage: usize,
        input: usize,
        (min, max): ([i64; 2], [i64; 2]),
        reach: u32,
    ) -> FieldView<'_> {
        let scale = self.pack.stages[stage].scale;
        let chunks = self
            .covering(scale, input, min, max)
            .into_iter()
            .map(|at| {
                let product = self
                    .products
                    .get(&(input, at))
                    .expect("inputs are generated before the stages that read them");
                ((at.x, at.y), product.as_ref())
            })
            .collect();
        FieldView {
            stage: &self.pack.stages[stage].name,
            input: &self.pack.stages[input].name,
            reach,
            size: self.size,
            min,
            max,
            ratio: self.pack.stages[input].scale / scale,
            chunks,
        }
    }

    fn view(&self, stage: usize, chunk: ChunkCoord, input: usize, reach: Reach) -> FieldView<'_> {
        let reach = reach.cells(self.size);
        self.view_over(stage, input, self.reader_box(chunk, reach), reach[0])
    }

    /// Every footprint of `input`, a sites stage or an Assemble stage, within `reach` of `chunk`,
    /// once each.
    fn levelled_near(
        &self,
        index: usize,
        chunk: ChunkCoord,
        input: usize,
        reach: Reach,
    ) -> Vec<Levelled> {
        let mut footprints: BTreeMap<LevelledId, Levelled> = BTreeMap::new();
        for (_, product) in self.inputs_within(index, chunk, input, reach.cells(self.size)) {
            footprints.extend(product.levelled(self.size));
        }
        footprints.into_values().collect()
    }

    fn generate(&self, index: usize, chunk: ChunkCoord) -> Result<Product, StageError> {
        let stage = &self.pack.stages[index];
        if let StageKind::Solve { .. } = &stage.kind {
            return Ok(Product::Tiles(self.site_of(index, chunk).map(|site| {
                let (_, town) = &self.solved[&(index, site.id.clone())];
                let (x, y) = (chunk.x - site.min.0, chunk.y - site.min.1);
                TownChunk {
                    site: site.id.clone(),
                    height: site.height,
                    tiles: Arc::clone(&town.chunks[(y as u32 * town.size.0 + x as u32) as usize]),
                }
            })));
        }
        if let StageKind::Assemble { .. } = &stage.kind {
            let [sx, sy] = [i64::from(self.size[0]), i64::from(self.size[1])];
            let (x0, y0) = (i64::from(chunk.x) * sx, i64::from(chunk.y) * sy);
            let overlaps = |stamp: &&Stamp| {
                stamp.min[0] < x0 + sx
                    && x0 < stamp.max[0]
                    && stamp.min[1] < y0 + sy
                    && y0 < stamp.max[1]
            };
            return Ok(Product::Stamps(
                self.assembly_site(index, chunk)
                    .map(|site| {
                        let (_, stamps) = &self.assembled[&(index, site.id)];
                        stamps.iter().filter(overlaps).cloned().collect()
                    })
                    .unwrap_or_default(),
            ));
        }
        if let StageKind::Locations { region, .. } = &stage.kind {
            let placed = &self.placed[&(index, region_of(chunk, *region))];
            return Ok(Product::Sites(
                placed
                    .sites
                    .iter()
                    .filter(|site| site.overlaps(chunk))
                    .cloned()
                    .collect(),
            ));
        }
        if let StageKind::TableSites { .. } = &stage.kind {
            let (height, reach) = stage.inputs[0];
            let view = self.view(index, chunk, height, reach);
            let mut sites = Vec::new();
            let footprints = self.footprints.get(&index).ok_or(StageError::NoFacts)?;
            for footprint in footprints.iter() {
                let site = Site {
                    id: SiteId::Row(footprint.row.clone()),
                    kind: None,
                    min: footprint.min,
                    max: footprint.max,
                    height: 0.0,
                };
                if site.overlaps(chunk) {
                    sites.push(Site {
                        height: self.footprint_height(site.min, site.max, &view)?,
                        ..site
                    });
                }
            }
            return Ok(Product::Sites(sites));
        }
        if let StageKind::Sites {
            region,
            size,
            chance,
            ..
        } = &stage.kind
        {
            let (height, reach) = stage.inputs[0];
            let view = self.view(index, chunk, height, reach);
            let owner = (
                chunk.x.div_euclid(*region as i32),
                chunk.y.div_euclid(*region as i32),
            );
            let site = self.site(stage.salt, owner, *region, *size, *chance, &view)?;
            return Ok(Product::Sites(
                site.filter(|site| site.overlaps(chunk))
                    .into_iter()
                    .collect(),
            ));
        }
        if let StageKind::Scatter { .. } = &stage.kind {
            return self.scatter(index, chunk).map(Product::Points);
        }
        if let Some(region) = stage.kind.job_region() {
            let curves = &self.regions[&(index, region_of(chunk, region))];
            let (min, max) = self.chunk_rect(chunk);
            return Ok(Product::Curves(
                curves
                    .iter()
                    .filter(|curve| curve.touches(min, max))
                    .cloned()
                    .collect(),
            ));
        }
        if let StageKind::TableCurves { table, .. } = &stage.kind {
            let rows = self
                .facts
                .as_ref()
                .ok_or(StageError::NoFacts)?
                .table(table)
                .expect("linked when loaded");
            let (min, max) = self.chunk_rect(chunk);
            return Ok(Product::Curves(
                rows.rows
                    .iter()
                    .map(|row| self.table_curve(index, rows, row))
                    .filter(|curve| curve.touches(min, max))
                    .collect(),
            ));
        }
        if let StageKind::Apply { .. } = &stage.kind {
            return self.apply(index, chunk).map(Product::Field);
        }
        let views: BTreeMap<usize, FieldView<'_>> = stage
            .inputs
            .iter()
            .filter(|&&(input, _)| {
                matches!(
                    self.pack.stages[input].kind.output(),
                    Output::Field | Output::Categories
                )
            })
            .map(|&(input, reach)| (input, self.view(index, chunk, input, reach)))
            .collect();
        let read = |name: &str, x: i64, y: i64| {
            views[&self.pack.index(name).expect("linked when loaded")].get(x, y)
        };
        if let StageKind::Rules { .. } | StageKind::Area { .. } = &stage.kind {
            let [sx, sy] = self.size;
            let mut values = Vec::with_capacity((sx * sy) as usize);
            for y in 0..sy {
                for x in 0..sx {
                    let column = [
                        i64::from(chunk.x) * i64::from(sx) + i64::from(x),
                        i64::from(chunk.y) * i64::from(sy) + i64::from(y),
                    ];
                    values.push(match &stage.kind {
                        StageKind::Area { input, distance } => {
                            area(input, *distance, column, &read)?
                        }
                        _ => self.categorise(index, column, &read)?,
                    });
                }
            }
            return Ok(Product::Categories(Categories {
                chunk,
                size: self.size,
                values,
            }));
        }
        let near: Vec<Levelled> = match &stage.kind {
            StageKind::Flatten { .. } => {
                let (sites, reach) = stage.inputs[1];
                self.levelled_near(index, chunk, sites, reach)
            }
            _ => Vec::new(),
        };
        let [sx, sy] = self.size;
        let mut values = Vec::with_capacity((sx * sy) as usize);
        for y in 0..sy {
            for x in 0..sx {
                let column = [
                    i64::from(chunk.x) * i64::from(sx) + i64::from(x),
                    i64::from(chunk.y) * i64::from(sy) + i64::from(y),
                ];
                let value = match &stage.kind {
                    StageKind::Field(expr) => evaluate(expr, &self.place(index, column, &read))?,
                    StageKind::Blur { input, radius } => blur(input, *radius, column, &read)?,
                    StageKind::Delta { input, radius } => delta(input, *radius, column, &read)?,
                    StageKind::Flatten { height, blend, .. } => {
                        let view = &views[&self.pack.index(height).expect("linked when loaded")];
                        let base = view.get(column[0], column[1])?;
                        let nearest = near
                            .iter()
                            .map(|footprint| (footprint.distance(column[0], column[1]), footprint))
                            .min_by(|a, b| a.0.total_cmp(&b.0));
                        match nearest {
                            Some((0.0, footprint)) => footprint.height,
                            Some((distance, footprint)) if distance < *blend as f32 => {
                                let t = distance / *blend as f32;
                                let t = t * t * (3.0 - 2.0 * t);
                                footprint.height + (base - footprint.height) * t
                            }
                            _ => base,
                        }
                    }
                    StageKind::Sites { .. }
                    | StageKind::Area { .. }
                    | StageKind::TableSites { .. }
                    | StageKind::Locations { .. }
                    | StageKind::TableCurves { .. }
                    | StageKind::Apply { .. }
                    | StageKind::Solve { .. }
                    | StageKind::Scatter { .. }
                    | StageKind::Assemble { .. }
                    | StageKind::Rules { .. }
                    | StageKind::Region { .. }
                    | StageKind::Rivers { .. }
                    | StageKind::Network { .. } => {
                        unreachable!("handled above")
                    }
                };
                values.push(value);
            }
        }
        Ok(Product::Field(Field {
            chunk,
            size: self.size,
            values,
        }))
    }

    /// Apply stage `index`'s field for `chunk`: its height with the curves near each column drawn
    /// in.
    fn apply(&self, index: usize, chunk: ChunkCoord) -> Result<Field, StageError> {
        let stage = &self.pack.stages[index];
        let StageKind::Apply {
            max_radius,
            blend,
            profile,
            ..
        } = &stage.kind
        else {
            unreachable!("called for Apply stages")
        };
        let (height, reach) = stage.inputs[0];
        let heights = self.view(index, chunk, height, reach);
        let (input, reach) = stage.inputs[1];
        // Every curve within reach once, in the order of their ids, so a tie always goes the same
        // way.
        let mut curves: BTreeMap<CurveId, &Curve> = BTreeMap::new();
        for (_, product) in self.inputs_within(index, chunk, input, reach.cells(self.size)) {
            let Product::Curves(near) = product else {
                unreachable!("inputs are type checked when the pack loads")
            };
            for curve in near {
                if let Some(wide) = curve
                    .values
                    .iter()
                    .find(|&&r| !(0.0..=*max_radius as f32).contains(&r))
                {
                    return Err(StageError::Curve {
                        stage: stage.name.clone(),
                        curve: curve.id.clone(),
                        message: format!("a radius of {wide}; 0 to {max_radius} are allowed"),
                    });
                }
                curves.insert(curve.id.clone(), curve);
            }
        }
        let blend = *blend as f32;
        let [sx, sy] = self.size;
        let mut values = Vec::with_capacity((sx * sy) as usize);
        for y in 0..sy {
            for x in 0..sx {
                let column = [
                    i64::from(chunk.x) * i64::from(sx) + i64::from(x),
                    i64::from(chunk.y) * i64::from(sy) + i64::from(y),
                ];
                let base = heights.get(column[0], column[1])?;
                let at = [column[0] as f32 + 0.5, column[1] as f32 + 0.5];
                // The curve that weighs most here, and the point of its centre line nearest.
                let mut best: Option<(f32, [f32; 2])> = None;
                for curve in curves.values() {
                    for (i, pair) in curve.points.windows(2).enumerate() {
                        let (a, b) = (pair[0], pair[1]);
                        let along = [b[0] - a[0], b[1] - a[1]];
                        let length = along[0] * along[0] + along[1] * along[1];
                        let t = if length == 0.0 {
                            0.0
                        } else {
                            (((at[0] - a[0]) * along[0] + (at[1] - a[1]) * along[1]) / length)
                                .clamp(0.0, 1.0)
                        };
                        let nearest = [a[0] + along[0] * t, a[1] + along[1] * t];
                        let distance = (at[0] - nearest[0]).hypot(at[1] - nearest[1]);
                        let radius = curve.values[i] + (curve.values[i + 1] - curve.values[i]) * t;
                        let weight = if distance <= radius {
                            1.0
                        } else if distance < radius + blend {
                            let s = (distance - radius) / blend;
                            1.0 - s * s * (3.0 - 2.0 * s)
                        } else {
                            continue;
                        };
                        if best.is_none_or(|(known, _)| weight > known) {
                            best = Some((weight, nearest));
                        }
                    }
                }
                values.push(match best {
                    None => base,
                    Some((weight, nearest)) => {
                        let centre =
                            heights.get(nearest[0].floor() as i64, nearest[1].floor() as i64)?;
                        let target = match profile {
                            Profile::Level => centre,
                            Profile::Carve(depth) => centre - depth,
                        };
                        base + (target - base) * weight
                    }
                });
            }
        }
        Ok(Field {
            chunk,
            size: self.size,
            values,
        })
    }

    /// The points of Scatter stage `index` whose column lies in `chunk`.
    ///
    /// Every block of `spacing` columns has `count` candidates, each at a hashed column, with a
    /// hashed priority. A candidate passes its own tests (chance, height, slope, conditions,
    /// water, sites) from what lies at and around its column, and is kept if it passes and no
    /// candidate that also passes, of higher priority, lies closer than `apart`. A kept candidate
    /// is the first point of its group; the others scatter around it and each passes the same tests
    /// at its own column. Everything a decision reads is within the stage's reach, so a chunk's
    /// points are the same whatever else has been generated.
    fn scatter(&self, index: usize, chunk: ChunkCoord) -> Result<Vec<Point>, StageError> {
        let stage = &self.pack.stages[index];
        let StageKind::Scatter {
            kind,
            height,
            spacing,
            count,
            group,
            chance,
            between,
            max_slope,
            when,
            water,
            avoid,
            block,
            apart,
            scale,
            tilt,
            align,
        } = &stage.kind
        else {
            unreachable!("called for Scatter stages")
        };
        let views: BTreeMap<usize, FieldView<'_>> = stage
            .inputs
            .iter()
            .filter(|&&(input, _)| {
                matches!(
                    self.pack.stages[input].kind.output(),
                    Output::Field | Output::Categories
                )
            })
            .map(|&(input, reach)| (input, self.view(index, chunk, input, reach)))
            .collect();
        let read = |name: &str, x: i64, y: i64| {
            views[&self.pack.index(name).expect("linked when loaded")].get(x, y)
        };
        let heights = &views[&self.pack.index(height).expect("linked when loaded")];
        let avoided = match avoid {
            Some((name, _)) => {
                let input = self.pack.index(name).expect("linked when loaded");
                let (_, reach) = *stage
                    .inputs
                    .iter()
                    .find(|&&(known, _)| known == input)
                    .expect("linked when loaded");
                self.levelled_near(index, chunk, input, reach)
            }
            None => Vec::new(),
        };
        let margin = avoid.as_ref().map_or(0, |(_, margin)| *margin);
        // Every point of a blocking stage within reach, with the clearance kept from it.
        let mut blockers: Vec<([f32; 2], f32)> = Vec::new();
        for (name, clearance) in block {
            let input = self.pack.index(name).expect("linked when loaded");
            let (_, reach) = *stage
                .inputs
                .iter()
                .find(|&&(known, _)| known == input)
                .expect("linked when loaded");
            for (_, product) in self.inputs_within(index, chunk, input, reach.cells(self.size)) {
                let Product::Points(points) = product else {
                    unreachable!("inputs are type checked when the pack loads")
                };
                blockers.extend(
                    points
                        .iter()
                        .map(|point| ([point.position[0], point.position[1]], *clearance)),
                );
            }
        }
        let world = (self.seed as u32) ^ ((self.seed >> 32) as u32);
        let spacing = i64::from(*spacing);
        let threshold = (f64::from(*chance) * 4_294_967_296.0) as u64;
        let [sx, sy] = [i64::from(self.size[0]), i64::from(self.size[1])];
        let (x0, y0) = (i64::from(chunk.x) * sx, i64::from(chunk.y) * sy);
        let apart = i64::from(*apart);
        let radius = group.map_or(0.0, |group| group.radius);
        // Candidates up to a group's radius beyond the chunk can have points in it, and those are
        // crowded by candidates up to `apart` further out.
        let reach = apart + radius.ceil() as i64;
        let blocks = |low: i64, high: i64| low.div_euclid(spacing)..=high.div_euclid(spacing);
        let mut candidates: Vec<Candidate> = Vec::new();
        for by in blocks(y0 - reach, y0 + sy - 1 + reach) {
            for bx in blocks(x0 - reach, x0 + sx - 1 + reach) {
                let many = count.0
                    + pcg3d([world ^ stage.salt ^ COUNT_STREAM, bx as u32, by as u32])[0]
                        % (count.1 - count.0 + 1);
                for slot in 0..many {
                    let [place, rank, turn] = pcg3d([
                        world ^ stage.salt,
                        bx as u32,
                        (by as u32) ^ 0x5bd1_e995 ^ slot.wrapping_mul(0x9E37_79B9),
                    ]);
                    let column = (
                        bx * spacing + i64::from(place) % spacing,
                        by * spacing + i64::from(place >> 16) % spacing,
                    );
                    let near = (x0 - reach..=x0 + sx - 1 + reach).contains(&column.0)
                        && (y0 - reach..=y0 + sy - 1 + reach).contains(&column.1);
                    if !near {
                        continue;
                    }
                    let fraction = (
                        f32::from((rank & 0xFF) as u8) / 256.0,
                        f32::from(((rank >> 8) & 0xFF) as u8) / 256.0,
                    );
                    candidates.push(Candidate {
                        column,
                        at: (column.0 as f32 + fraction.0, column.1 as f32 + fraction.1),
                        priority: (rank, bx, by, slot),
                        slot,
                        turn: unit(turn),
                        exists: u64::from(turn) < threshold,
                    });
                }
            }
        }
        // The height of a point standing at `at` in `column`, if it passes the stage's tests there.
        let passes = |column: (i64, i64), at: (f32, f32)| -> Result<Option<f32>, StageError> {
            let (x, y) = column;
            let here = heights.get(x, y)?;
            if between.is_some_and(|(low, high)| !(low..=high).contains(&here)) {
                return Ok(None);
            }
            if let Some(limit) = max_slope {
                let along_x = (heights.get(x + 1, y)? - heights.get(x - 1, y)?).abs() / 2.0;
                let along_y = (heights.get(x, y + 1)? - heights.get(x, y - 1)?).abs() / 2.0;
                if along_x.max(along_y) > *limit {
                    return Ok(None);
                }
            }
            let place = self.place(index, [x, y], &read);
            for condition in when {
                if !holds(condition, &place)? {
                    return Ok(None);
                }
            }
            if water.is_some_and(|water| {
                !(water.depth.0..=water.depth.1).contains(&(water.level - here))
            }) {
                return Ok(None);
            }
            if avoided
                .iter()
                .any(|footprint| footprint.distance(x, y) < margin as f32)
            {
                return Ok(None);
            }
            if blockers
                .iter()
                .any(|&(point, clearance)| (point[0] - at.0).hypot(point[1] - at.1) < clearance)
            {
                return Ok(None);
            }
            Ok(Some(here))
        };
        let mut heights_of: Vec<Option<f32>> = Vec::with_capacity(candidates.len());
        for candidate in &candidates {
            heights_of.push(if candidate.exists {
                passes(candidate.column, candidate.at)?
            } else {
                None
            });
        }
        let point_stage = point_stage_id(stage.salt);
        let kind: Arc<str> = Arc::from(kind.as_str());
        let inside = |(x, y): (i64, i64)| (x0..x0 + sx).contains(&x) && (y0..y0 + sy).contains(&y);
        let mut points = Vec::new();
        for (i, candidate) in candidates.iter().enumerate() {
            let Some(anchor_height) = heights_of[i] else {
                continue;
            };
            let crowded = candidates.iter().enumerate().any(|(j, other)| {
                j != i
                    && heights_of[j].is_some()
                    && other.priority > candidate.priority
                    && ((other.at.0 - candidate.at.0).powi(2)
                        + (other.at.1 - candidate.at.1).powi(2))
                    .sqrt()
                        < apart as f32
            });
            if crowded {
                continue;
            }
            let (column, stream) = (
                candidate.column,
                [candidate.column.0 as u32, candidate.column.1 as u32],
            );
            let members = group.map_or(1, |group| {
                group.size.0
                    + pcg3d([world ^ stage.salt ^ GROUP_STREAM, stream[0], stream[1]])[0]
                        % (group.size.1 - group.size.0 + 1)
            });
            // Ids name the candidate's own column, so a member standing in the next chunk still
            // has an id no other point shares.
            let owner = ChunkCoord::new(
                i32::try_from(column.0.div_euclid(sx)).expect("a chunk coordinate"),
                i32::try_from(column.1.div_euclid(sy)).expect("a chunk coordinate"),
                0,
            );
            let cell = (column.1.rem_euclid(sy) * sx + column.0.rem_euclid(sx)) as u32;
            for member in 0..members {
                let slot = candidate.slot * MAX_SCATTER_SLOTS + member;
                let key = stream[1] ^ slot.wrapping_mul(0x2545_F491);
                let [angle, distance, turn] =
                    pcg3d([world ^ stage.salt ^ GROUP_STREAM, stream[0], key]);
                let (at, turn) = if member == 0 {
                    (candidate.at, candidate.turn)
                } else {
                    let (sin, cos) = (unit(angle) * std::f32::consts::TAU).sin_cos();
                    let away = radius * unit(distance).sqrt();
                    (
                        (candidate.at.0 + cos * away, candidate.at.1 + sin * away),
                        unit(turn),
                    )
                };
                let standing = (at.0.floor() as i64, at.1.floor() as i64);
                if !inside(standing) {
                    continue;
                }
                let z = if member == 0 {
                    anchor_height
                } else {
                    match passes(standing, at)? {
                        Some(z) => z,
                        None => continue,
                    }
                };
                let [size, lean, aligned] =
                    pcg3d([world ^ stage.salt ^ LOOK_STREAM, stream[0], key]);
                let up = if unit(aligned) < *align {
                    let (x, y) = standing;
                    let along_x = (heights.get(x + 1, y)? - heights.get(x - 1, y)?) / 2.0;
                    let along_y = (heights.get(x, y + 1)? - heights.get(x, y - 1)?) / 2.0;
                    let length = (along_x * along_x + along_y * along_y + 1.0).sqrt();
                    [-along_x / length, -along_y / length, 1.0 / length]
                } else {
                    let degrees = tilt.0 + (tilt.1 - tilt.0) * unit(lean);
                    let (sin, cos) = degrees.to_radians().sin_cos();
                    let (towards_y, towards_x) =
                        (unit(lean.rotate_left(16)) * std::f32::consts::TAU).sin_cos();
                    [sin * towards_x, sin * towards_y, cos]
                };
                points.push(Point {
                    id: InstanceId::new(owner, point_stage, cell, slot as u16),
                    kind: Arc::clone(&kind),
                    position: [at.0, at.1, z],
                    turn,
                    scale: scale.0 + (scale.1 - scale.0) * unit(size),
                    up,
                });
            }
        }
        Ok(points)
    }

    /// The site of `owner`'s region, if it has one, from the stage's own hash stream: whether it
    /// exists, its size and where it sits inside the region, one chunk in from every edge, and its
    /// height from `height` over its footprint's centre and inner corners.
    fn site(
        &self,
        salt: u32,
        owner: (i32, i32),
        region: u32,
        size: (u32, u32),
        chance: f32,
        height: &FieldView<'_>,
    ) -> Result<Option<Site>, StageError> {
        let world = (self.seed as u32) ^ ((self.seed >> 32) as u32);
        let [exists, dims, place] = pcg3d([world ^ salt, owner.0 as u32, owner.1 as u32]);
        // Existence is an integer comparison, so it is the same on every machine.
        if u64::from(exists) >= (f64::from(chance) * 4_294_967_296.0) as u64 {
            return Ok(None);
        }
        let sizes = size.1 - size.0 + 1;
        let (w, h) = (size.0 + dims % sizes, size.0 + (dims >> 16) % sizes);
        let (ox, oy) = (
            1 + place % (region - w - 1),
            1 + (place >> 16) % (region - h - 1),
        );
        let min = (
            owner.0 * region as i32 + ox as i32,
            owner.1 * region as i32 + oy as i32,
        );
        let max = (min.0 + w as i32, min.1 + h as i32);
        Ok(Some(Site {
            id: SiteId::Region(owner.0, owner.1),
            kind: None,
            min,
            max,
            height: self.footprint_height(min, max, height)?,
        }))
    }

    /// The height a site over the chunks `min..max` is levelled to: the mean of `height` at its
    /// footprint's centre and inner corners.
    fn footprint_height(
        &self,
        min: (i32, i32),
        max: (i32, i32),
        height: &FieldView<'_>,
    ) -> Result<f32, StageError> {
        let [sx, sy] = [i64::from(self.size[0]), i64::from(self.size[1])];
        let (x0, y0) = (i64::from(min.0) * sx, i64::from(min.1) * sy);
        let (x1, y1) = (i64::from(max.0) * sx - 1, i64::from(max.1) * sy - 1);
        let samples = [
            ((x0 + x1) / 2, (y0 + y1) / 2),
            (x0 + 1, y0 + 1),
            (x1 - 1, y0 + 1),
            (x0 + 1, y1 - 1),
            (x1 - 1, y1 - 1),
        ];
        let mut sum = 0.0;
        for (x, y) in samples {
            sum += height.get(x, y)?;
        }
        Ok(sum / samples.len() as f32)
    }

    /// Where stage `index`'s expressions are evaluated at `column`, reading inputs through `read`.
    fn place<'p, 'r>(
        &'p self,
        index: usize,
        column: [i64; 2],
        read: &'p Read<'r>,
    ) -> ColumnPlace<'p, 'r> {
        let stage = &self.pack.stages[index];
        ColumnPlace {
            runtime: self,
            salt: stage.salt,
            scale: stage.scale,
            column,
            read,
        }
    }
}

/// One column of a stage, where its expressions' leaves read noise, inputs and the position.
struct ColumnPlace<'p, 'r> {
    runtime: &'p Runtime,
    salt: u32,
    scale: u32,
    column: [i64; 2],
    read: &'p Read<'r>,
}

impl Leaves for ColumnPlace<'_, '_> {
    fn leaf(&self, expr: &Expr) -> Result<f32, StageError> {
        let (column, read, pack) = (self.column, self.read, &self.runtime.pack);
        // The column's centre in WFC cells, so a formula means the same at every scale.
        let centre = [
            (column[0] as f32 + 0.5) * self.scale as f32,
            (column[1] as f32 + 0.5) * self.scale as f32,
        ];
        Ok(match expr {
            Expr::Noise {
                frequency,
                octaves,
                name,
            } => {
                let stream = name.as_deref().map_or(self.salt, noise_stream);
                value_noise(self.runtime.seed, stream, *frequency, *octaves, centre)
            }
            Expr::FastNoise(name) => self.runtime.noises[name].sample(centre[0], centre[1]),
            Expr::Input(name) => read(name, column[0], column[1])?,
            Expr::X => centre[0],
            Expr::Y => centre[1],
            Expr::Distance((x, y)) => (centre[0] - x).hypot(centre[1] - y),
            Expr::Angle((x, y)) => {
                let turn = (centre[1] - y).atan2(centre[0] - x) / std::f32::consts::TAU;
                turn.rem_euclid(1.0)
            }
            Expr::Is(stage, names) => {
                let index = pack.index(stage).expect("linked when loaded");
                let known = pack.stages[index].kind.categories();
                let here = read(stage, column[0], column[1])? as usize;
                f32::from(u8::from(names.iter().any(|name| name == known[here])))
            }
            Expr::Match {
                input: stage,
                cases,
                otherwise,
                blend,
            } => {
                let index = pack.index(stage).expect("linked when loaded");
                let names = pack.stages[index].kind.categories();
                let reach = i64::from(*blend);
                // A tent in each direction, so a category's weight falls off smoothly with its
                // distance from the column and the blend moves by a small step per column.
                let mut weights: Vec<(usize, f32)> = Vec::new();
                for dy in -reach..=reach {
                    for dx in -reach..=reach {
                        let weight = ((reach + 1 - dx.abs()) * (reach + 1 - dy.abs())) as f32;
                        let category = read(stage, column[0] + dx, column[1] + dy)? as usize;
                        match weights.iter_mut().find(|(known, _)| *known == category) {
                            Some((_, sum)) => *sum += weight,
                            None => weights.push((category, weight)),
                        }
                    }
                }
                let total: f32 = weights.iter().map(|(_, weight)| weight).sum();
                let mut blended = 0.0;
                for (category, weight) in weights {
                    let case = cases
                        .iter()
                        .find(|(name, _)| name == names[category])
                        .map_or(otherwise.as_ref(), |(_, expr)| expr);
                    blended += weight / total * evaluate(case, self)?;
                }
                blended
            }
            Expr::Row(table, column) => {
                let index = pack.table_by_name[table];
                let row = self
                    .runtime
                    .focused
                    .get(&index)
                    .ok_or_else(|| StageError::NoFocus(table.clone()))?;
                row.values[pack.tables[index]
                    .column(column)
                    .expect("checked when loaded")]
            }
            Expr::Parent(_) | Expr::Random(..) | Expr::Index | Expr::Count | Expr::Share(_) => {
                unreachable!("a stage's expressions are checked when loaded")
            }
            Expr::Constant(_)
            | Expr::Add(..)
            | Expr::Sub(..)
            | Expr::Mul(..)
            | Expr::Min(..)
            | Expr::Max(..)
            | Expr::Abs(_)
            | Expr::Floor(_)
            | Expr::Sin(_)
            | Expr::Clamp(..)
            | Expr::Smoothstep(..)
            | Expr::Remap(..)
            | Expr::Curve(..)
            | Expr::Select { .. } => unreachable!("evaluate combines these itself"),
        })
    }
}

impl Runtime {
    /// The category Rules stage `index` gives `column`: the first rule whose conditions all hold,
    /// or its fallback.
    fn categorise(
        &self,
        index: usize,
        column: [i64; 2],
        read: &Read<'_>,
    ) -> Result<u8, StageError> {
        let stage = &self.pack.stages[index];
        let StageKind::Rules { rules, otherwise } = &stage.kind else {
            unreachable!("only Rules stages categorise")
        };
        let names = stage.kind.categories();
        let index_of = |name: &str| {
            names
                .iter()
                .position(|known| *known == name)
                .expect("every category is named") as u8
        };
        let place = self.place(index, column, read);
        for rule in rules {
            let mut all = true;
            for condition in &rule.when {
                if !holds(condition, &place)? {
                    all = false;
                    break;
                }
            }
            if all {
                return Ok(index_of(&rule.category));
            }
        }
        Ok(index_of(otherwise))
    }
}

/// The average of `input` over the square of `radius` columns around `column`.
fn blur(input: &str, radius: u32, column: [i64; 2], read: &Read<'_>) -> Result<f32, StageError> {
    let r = i64::from(radius);
    let mut sum = 0.0_f32;
    for dy in -r..=r {
        for dx in -r..=r {
            sum += read(input, column[0] + dx, column[1] + dy)?;
        }
    }
    Ok(sum / ((2 * r + 1) * (2 * r + 1)) as f32)
}

/// The highest value of `input` less its lowest over the square of `radius` columns around
/// `column`.
fn delta(input: &str, radius: u32, column: [i64; 2], read: &Read<'_>) -> Result<f32, StageError> {
    let r = i64::from(radius);
    let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
    for dy in -r..=r {
        for dx in -r..=r {
            let value = read(input, column[0] + dx, column[1] + dy)?;
            low = low.min(value);
            high = high.max(value);
        }
    }
    Ok(high - low)
}

/// 1, `edge`, where one of the eight columns `distance` columns from `column` along the axes and
/// the diagonals has another category of `input` than `column`; 0, `median`, where none has.
fn area(input: &str, distance: u32, column: [i64; 2], read: &Read<'_>) -> Result<u8, StageError> {
    let d = i64::from(distance);
    let here = read(input, column[0], column[1])?;
    for (dx, dy) in [
        (-d, -d),
        (0, -d),
        (d, -d),
        (-d, 0),
        (d, 0),
        (-d, d),
        (0, d),
        (d, d),
    ] {
        if read(input, column[0] + dx, column[1] + dy)? != here {
            return Ok(1);
        }
    }
    Ok(0)
}

/// A reading stage's column `x`, `y` in an input `ratio` times coarser, from the input's own
/// columns through `at`: the input's value itself at the same scale, the column it lies in for a
/// category, and linearly between the four columns around its centre for a field.
fn between(
    ratio: u32,
    categorical: bool,
    x: i64,
    y: i64,
    at: impl Fn(i64, i64) -> Result<f32, StageError>,
) -> Result<f32, StageError> {
    if ratio == 1 {
        return at(x, y);
    }
    let ratio = i64::from(ratio);
    if categorical {
        return at(x.div_euclid(ratio), y.div_euclid(ratio));
    }
    let place = |column: i64| (column as f64 + 0.5) / ratio as f64 - 0.5;
    let (u, v) = (place(x), place(y));
    let (i, j) = (u.floor() as i64, v.floor() as i64);
    let (s, t) = ((u - i as f64) as f32, (v - j as f64) as f32);
    let bottom = at(i, j)? + (at(i + 1, j)? - at(i, j)?) * s;
    let top = at(i, j + 1)? + (at(i + 1, j + 1)? - at(i, j + 1)?) * s;
    Ok(bottom + (top - bottom) * t)
}

/// The stream of a named noise: the same wherever the name appears, and apart from any stage's
/// own stream, whose salt is the stage's name alone.
fn noise_stream(name: &str) -> u32 {
    salt(name) ^ 0x6E6F_6973
}

/// One Scatter candidate: its column, where in it the point stands, its priority (a hash, with the
/// block and the candidate's place in it breaking ties), that place, its turn, and whether the
/// chance test lets it exist at all.
struct Candidate {
    column: (i64, i64),
    at: (f32, f32),
    priority: (u32, i64, i64, u32),
    slot: u32,
    turn: f32,
    exists: bool,
}

/// Mixed into a Scatter stage's stream for how many candidates a block has.
const COUNT_STREAM: u32 = 0x636E_7473;
/// Mixed into a Scatter stage's stream for a group's size and its members' places.
const GROUP_STREAM: u32 = 0x6772_7570;
/// Mixed into a Scatter stage's stream for a point's scale, tilt and alignment.
const LOOK_STREAM: u32 = 0x6C6F_6F6B;

/// A hash as a number from 0 up to but not including 1.
fn unit(hash: u32) -> f32 {
    (hash >> 8) as f32 / (1u32 << 24) as f32
}

/// A random number in 0..1 for one lattice point of one octave, from the world seed and the
/// stage's salt: a named stream, so no other stage or octave shares it.
fn lattice(seed: u64, salt: u32, octave: u32, x: i64, y: i64) -> f32 {
    let world = (seed as u32) ^ ((seed >> 32) as u32);
    let [a, ..] = pcg3d([world ^ salt, x as u32, y as u32]);
    let [b, ..] = pcg3d([
        a ^ octave.wrapping_mul(0x9E37_79B9),
        (x >> 32) as u32,
        (y >> 32) as u32,
    ]);
    (b >> 8) as f32 / (1u32 << 24) as f32
}

/// Fractal value noise at a point in WFC cells, in 0..1: every point depends on its position alone,
/// so chunks meet without seams and every scale samples the same noise.
fn value_noise(seed: u64, salt: u32, frequency: f32, octaves: u32, at: [f32; 2]) -> f32 {
    let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
    let mut sum = 0.0;
    let mut weight = 1.0;
    let mut total = 0.0;
    let mut scale = frequency;
    for octave in 0..octaves {
        let fx = at[0] * scale;
        let fy = at[1] * scale;
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (smooth(fx - x0), smooth(fy - y0));
        let (ix, iy) = (x0 as i64, y0 as i64);
        let corner = |dx: i64, dy: i64| lattice(seed, salt, octave, ix + dx, iy + dy);
        let bottom = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * tx;
        let top = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * tx;
        sum += (bottom + (top - bottom) * ty) * weight;
        total += weight;
        weight *= 0.5;
        scale *= 2.0;
    }
    sum / total
}

/// The hash a Solve or Assemble stage seeds a site's town or assembly from: the world's seed, the
/// stage's salt and the site's id, so a site grows the same whichever chunk asks for it.
fn site_hash(seed: u64, salt: u32, id: &SiteId) -> [u32; 3] {
    let world = (seed as u32) ^ ((seed >> 32) as u32);
    match id {
        SiteId::Region(x, y) => pcg3d([world ^ salt, *x as u32, *y as u32]),
        SiteId::Location { region, index } => pcg3d([
            world ^ salt ^ 0x6C6F_6361,
            region.0 as u32,
            (region.1 as u32) ^ index.wrapping_mul(0x9E37_79B9),
        ]),
        SiteId::Row(row) => {
            let hash = row
                .0
                .iter()
                .fold(world ^ salt ^ 0x524F_5753, |hash, &part| {
                    pcg3d([hash, part as u32, (part >> 32) as u32])[0]
                });
            pcg3d([hash, row.0.len() as u32, 0])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime(pack: &str) -> Runtime {
        Runtime::new(
            Arc::new(Pack::parse(pack).expect("a valid pack")),
            7,
            [8, 8],
        )
    }

    const PACK: &str = r#"(
        version: 1,
        stages: [
            (name: "rough", kind: Field(Noise(frequency: 0.2, octaves: 3))),
            (name: "smooth", kind: Blur(input: "rough", radius: 3)),
            (name: "height", kind: Field(Mul(Input("smooth"), Constant(8.0)))),
        ],
    )"#;

    #[test]
    fn a_stage_is_generated_only_after_every_input_chunk_within_its_reach() {
        let mut runtime = runtime(PACK);
        runtime
            .request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 0)], &["height"])
            .expect("a stage");

        let generated = runtime.run_until_idle().expect("the stages run");

        // A blur of 3 columns around one chunk of 8 reaches into all eight neighbours.
        let rough: Vec<_> = generated
            .iter()
            .filter(|(stage, _)| stage == "rough")
            .collect();
        assert_eq!(rough.len(), 9);
        let first_smooth = generated.iter().position(|(stage, _)| stage == "smooth");
        let last_rough = generated.iter().rposition(|(stage, _)| stage == "rough");
        assert!(last_rough < first_smooth);
    }

    #[test]
    fn a_blurred_field_is_the_mean_of_its_input_around_each_column() {
        let mut runtime = runtime(PACK);
        runtime
            .request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 0)], &["smooth"])
            .expect("a stage");
        runtime.run_until_idle().expect("the stages run");

        let rough = |x: i64, y: i64| {
            let chunk = ChunkCoord::new(x.div_euclid(8) as i32, y.div_euclid(8) as i32, 0);
            runtime
                .field("rough", chunk)
                .expect("generated")
                .get(x.rem_euclid(8) as u32, y.rem_euclid(8) as u32)
        };
        let mut sum = 0.0_f32;
        for dy in -3..=3 {
            for dx in -3..=3 {
                sum += rough(dx, dy);
            }
        }

        let smooth = runtime
            .field("smooth", ChunkCoord::new(0, 0, 0))
            .expect("generated");
        assert_eq!(smooth.get(0, 0), sum / 49.0);
    }

    #[test]
    fn a_read_beyond_the_reach_is_an_error_naming_the_stage() {
        let field = Product::Field(Field {
            chunk: ChunkCoord::new(0, 0, 0),
            size: [4, 4],
            values: vec![0.0; 16],
        });
        let view = FieldView {
            stage: "reader",
            input: "source",
            reach: 1,
            size: [4, 4],
            min: [-1, -1],
            max: [4, 4],
            ratio: 1,
            chunks: BTreeMap::from([((0, 0), &field)]),
        };

        let result = view.get(6, 0);

        assert_eq!(
            result,
            Err(StageError::OutOfReach {
                stage: "reader".to_owned(),
                input: "source".to_owned(),
                reach: 1,
                needed: 3
            })
        );
    }

    #[test]
    fn what_a_new_request_does_not_need_is_dropped() {
        let mut runtime = runtime(PACK);
        runtime
            .request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 0)], &["height"])
            .expect("a stage");
        runtime.run_until_idle().expect("the stages run");

        runtime
            .request(
                &[FocusPoint::new(ChunkCoord::new(10, 0, 0), 0)],
                &["height"],
            )
            .expect("a stage");

        assert_eq!(runtime.held(), 0);
    }

    #[test]
    fn noise_stays_in_the_unit_interval() {
        let values: Vec<f32> = (0..2000)
            .map(|i| {
                let at = [(i * 7 - 5000) as f32 + 0.5, (i * 13 - 9000) as f32 + 0.5];
                value_noise(3, 11, 0.37, 4, at)
            })
            .collect();

        assert!(values.iter().all(|v| (0.0..1.0).contains(v)), "{values:?}");
    }
}

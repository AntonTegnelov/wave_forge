//! Region jobs: bounded computations over a whole square region of chunks at a time.
//!
//! Some generation cannot be done one chunk at a time: a river has to know where it enters and
//! leaves, a location table has to count what it placed. A Region stage runs a [`RegionJob`] once
//! per region, over its inputs across the region and a halo around it, and every chunk of the
//! region reads the job's curves. The job is code the game gives the runtime, the code tier of
//! docs/architecture/stages.md, and it stays pure: everything it may depend on is in its
//! [`RegionInput`].
//!
//! Neighbouring regions never read each other. Where they must agree, at a river's crossing point
//! say, both compute the same value from [`RegionInput::edge_hash`], which depends only on the
//! edge. A job may reject an attempt, a river that runs uphill say; the runtime then tries again
//! with the next retry index, which changes [`RegionInput::hash`] but never the edge hashes, and
//! gives up with every reason once the stage's budget is spent.

use super::facts::RowId;
use super::runtime::{FieldView, Site, StageError};
use std::collections::BTreeMap;
use wfc_core::hash::pcg3d;

/// A polyline a region job produced: points in world columns along the lattice's x and y, and a
/// value per point for the job to give meaning to, a river's width say.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Curve {
    pub id: CurveId,
    pub points: Vec<[f32; 2]>,
    pub values: Vec<f32>,
}

/// Which curve it is. Positional, so the same in every run and whatever order curves are computed
/// in.
#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Deserialize, serde::Serialize,
)]
pub enum CurveId {
    /// A region job's curve: the region that made it and its place in that region's list.
    Region { region: (i32, i32), index: u32 },
    /// A TableCurves stage's curve: the row it stands for.
    Row(RowId),
}

impl Curve {
    /// Whether any segment of the curve comes within the columns `min..=max`.
    pub(crate) fn touches(&self, min: [f32; 2], max: [f32; 2]) -> bool {
        let inside =
            |p: [f32; 2]| (min[0]..=max[0]).contains(&p[0]) && (min[1]..=max[1]).contains(&p[1]);
        self.points.iter().copied().any(inside)
            || self.points.windows(2).any(|pair| {
                let low = [pair[0][0].min(pair[1][0]), pair[0][1].min(pair[1][1])];
                let high = [pair[0][0].max(pair[1][0]), pair[0][1].max(pair[1][1])];
                low[0] <= max[0] && high[0] >= min[0] && low[1] <= max[1] && high[1] >= min[1]
            })
    }
}

/// One of the four edges of a region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    /// Towards -x.
    West,
    /// Towards +x.
    East,
    /// Towards -y.
    South,
    /// Towards +y.
    North,
}

/// What an attempt at a region came to.
#[derive(Clone, Debug, PartialEq)]
pub enum Attempt {
    /// The region's curves.
    Accepted(Vec<Curve>),
    /// Why this attempt is not good enough; the runtime tries again with the next retry index.
    Rejected(String),
}

/// A computation over a whole region, which a Region stage names.
///
/// It must be pure: the same [`RegionInput`] gives the same [`Attempt`]. It reads inputs only
/// through [`RegionInput::field`], which refuses a column outside the region and its halo.
pub trait RegionJob: Send {
    /// One attempt at the region.
    ///
    /// # Errors
    /// A [`StageError`] from reading an input, which ends generation.
    fn run(&self, input: &RegionInput<'_>) -> Result<Attempt, StageError>;
}

/// Everything a region job may depend on.
pub struct RegionInput<'a> {
    pub(crate) region: (i32, i32),
    /// Chunks along each side of the region.
    pub(crate) size: u32,
    /// Columns per chunk.
    pub(crate) chunk: [u32; 2],
    pub(crate) retry: u32,
    pub(crate) world: u32,
    pub(crate) salt: u32,
    pub(crate) views: BTreeMap<&'a str, FieldView<'a>>,
    /// The sites whose centre lies in the region, for the stages that read sites.
    pub(crate) sites: Vec<Site>,
}

impl RegionInput<'_> {
    /// The region's key: its position on the lattice of regions.
    #[must_use]
    pub const fn region(&self) -> (i32, i32) {
        self.region
    }

    /// The columns the region covers, from its lowest corner to its highest, both included.
    #[must_use]
    pub fn columns(&self) -> ([i64; 2], [i64; 2]) {
        let span = |at: i32, axis: usize| {
            let width = i64::from(self.size) * i64::from(self.chunk[axis]);
            (i64::from(at) * width, i64::from(at) * width + width - 1)
        };
        let (x0, x1) = span(self.region.0, 0);
        let (y0, y1) = span(self.region.1, 1);
        ([x0, y0], [x1, y1])
    }

    /// Columns per chunk along the lattice's x and y.
    pub(crate) const fn chunk(&self) -> [u32; 2] {
        self.chunk
    }

    /// The sites whose centre lies in the region, in the order of their ids.
    pub(crate) fn sites(&self) -> &[Site] {
        &self.sites
    }

    /// Which attempt this is, from 0.
    #[must_use]
    pub const fn retry(&self) -> u32 {
        self.retry
    }

    /// An input field's value at a world column.
    ///
    /// # Errors
    /// [`StageError::UnknownStage`] for a stage the Region stage does not read, and
    /// [`StageError::OutOfReach`] for a column outside the region and its halo.
    pub fn field(&self, stage: &str, x: i64, y: i64) -> Result<f32, StageError> {
        self.views
            .get(stage)
            .ok_or_else(|| StageError::UnknownStage(stage.to_owned()))?
            .get(x, y)
    }

    /// A hash of the region, this attempt and `purpose`: the job's own random numbers.
    #[must_use]
    pub fn hash(&self, purpose: u32) -> u32 {
        pcg3d([
            self.world ^ self.salt,
            (self.region.0 as u32) ^ purpose.wrapping_mul(0x9E37_79B9),
            (self.region.1 as u32) ^ self.retry.wrapping_mul(0x85EB_CA6B),
        ])[0]
    }

    /// A hash of one of the region's edges and `purpose`, which the neighbour across that edge
    /// computes too, whatever either region's retry index: how two regions agree on what crosses
    /// between them without reading each other.
    #[must_use]
    pub fn edge_hash(&self, edge: Edge, purpose: u32) -> u32 {
        let (x, y) = self.region;
        // An edge is named by the region on its lower side and its axis, from either side.
        let (low, axis) = match edge {
            Edge::West => ((x - 1, y), 0),
            Edge::East => ((x, y), 0),
            Edge::South => ((x, y - 1), 1),
            Edge::North => ((x, y), 1),
        };
        pcg3d([
            self.world ^ self.salt ^ 0x6564_6765,
            (low.0 as u32) ^ purpose.wrapping_mul(0x9E37_79B9),
            (low.1 as u32) ^ (axis + 1),
        ])[0]
    }
}

/// The region of a Region stage with regions of `size` chunks that `chunk` lies in.
pub(crate) const fn region_of(chunk: wfc_core::ChunkCoord, size: u32) -> (i32, i32) {
    (
        chunk.x.div_euclid(size as i32),
        chunk.y.div_euclid(size as i32),
    )
}

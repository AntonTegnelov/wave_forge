//! The model a world is generated from, and the seam a solver plugs into.
//!
//! A world is a lattice of chunks of one shape ([`ChunkShape`]). A chunk is solved as a *region*:
//! itself widened by a halo that is solved and thrown away ([`Region`]). What the world allows is a
//! [`Ruleset`] (adjacency packed into words, one integer weight per tile) and a [`Prior`] (what a
//! cell may hold before anything is decided, by layer, by world face or cell by cell). What it has
//! decided is a [`ChunkStore`]. [`Solver`] is what turns a batch of regions into tiles, on a GPU or
//! on the CPU reference behind the `reference` feature.
//!
//! Nothing here allocates per cell, is async, or knows about a device: see
//! docs/architecture/solver.md, "The model".

pub mod chunk;
pub mod domains;
pub mod hash;
pub mod prior;
#[cfg(feature = "reference")]
pub mod reference;
pub mod rules;
pub mod solver;
pub mod store;

pub use crate::chunk::{ChunkCoord, ChunkShape, Region, RegionShape, WorldCell};
pub use crate::domains::Domains;
pub use crate::prior::Prior;
pub use crate::rules::{MAX_TILES, RuleTable, Ruleset, TileMask};
pub use crate::solver::{
    BatchResult, JobId, RegionBatch, RegionStats, RegionStatus, SolveBudget, Solver, SolverError,
};
pub use crate::store::{Chunk, ChunkStore, WorldExtent, region_init, region_init_by};

use thiserror::Error;

/// What a rule set, a world or a region can be wrong about.
#[derive(Error, Debug)]
pub enum ModelError {
    /// A rule set the solver cannot represent.
    #[error("a rule set of {tiles} tiles is outside the supported range of 1 to {max}")]
    TileCount {
        /// How many tiles the rule set has.
        tiles: usize,
        /// The most tiles a solver supports, [`MAX_TILES`].
        max: u32,
    },
    /// Rules defined over the wrong number of directions.
    #[error("rules over {axes} axes, but a solver needs the 6 of a cubic grid")]
    AxisCount {
        /// How many axes the rules are defined over.
        axes: usize,
    },
    /// One weight per tile is required.
    #[error("{tiles} tiles but {weights} weights")]
    WeightCount {
        /// How many tiles the rule set has.
        tiles: usize,
        /// How many weights were given.
        weights: usize,
    },
    /// A weight that cannot be used.
    #[error("tile {tile} has weight {weight}, which is not a finite, non-negative number")]
    Weight {
        /// The tile with the weight.
        tile: usize,
        /// The weight, negative or not finite.
        weight: f32,
    },
    /// Every tile has weight zero, so nothing could ever be chosen.
    #[error("no tile has a positive weight")]
    NoPositiveWeight,
    /// Domain words that do not match the cells they describe.
    #[error("{expected} words describe these cells, but {got} were given")]
    WordCount {
        /// How many words the cells take.
        expected: usize,
        /// How many words were given.
        got: usize,
    },
    /// A chunk outside the world's extent.
    #[error("chunk ({}, {}, {}) is outside the world", chunk.x, chunk.y, chunk.z)]
    OutsideWorld {
        /// The chunk.
        chunk: crate::chunk::ChunkCoord,
    },
    /// A chunk with the wrong number of cells.
    #[error("a chunk of this world holds {expected} cells, but {got} were given")]
    ChunkCells {
        /// How many cells a chunk of this world holds.
        expected: usize,
        /// How many cells were given.
        got: usize,
    },
    /// A chunk holding a tile the rule set does not have.
    #[error("chunk ({}, {}, {}) holds tile {tile}, but the rule set has {tiles}", chunk.x, chunk.y, chunk.z)]
    UnknownTile {
        /// The chunk.
        chunk: crate::chunk::ChunkCoord,
        /// The first tile in the chunk that the rule set does not have.
        tile: u16,
        /// How many tiles the rule set has.
        tiles: u32,
    },
    /// A cell that should have been decided by now.
    #[error("cell ({}, {}, {}) is not decided", cell[0], cell[1], cell[2])]
    Undecided {
        /// The cell, in world coordinates.
        cell: crate::chunk::WorldCell,
    },
}

//! Towns: bounded WFC worlds, solved whole, one per settlement site.
//!
//! In a world whose ground rises and falls, WFC runs per site rather than as one lattice across the
//! terrain (docs/architecture/stages.md, "How WFC joins"): each town is a bounded world the size of
//! its site's footprint, solved all at once from a seed of its own, so its tiles are a function of
//! the site alone. [`TownSolver`] is the seam to whatever solves it; [`WfcTowns`] is the one this
//! library offers, over any [`Solver`].

use crate::loader::RuleFile;
use crate::{Builder, ChunkCoord, FocusPoint, Prior, Ruleset, Solver, WorldExtent};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use wfc_core::{ChunkShape, TileMask};
use wfc_rules::modules::{Face, NEG_X, NEG_Y, POS_X, POS_Y};

/// A set of tiles a rule file names: every variant of the modules with a tag, or of one module.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum Selector {
    Tagged(String),
    Named(String),
}

impl Selector {
    fn mask(&self, file: &RuleFile) -> Result<TileMask, TownError> {
        let tiles = match self {
            Self::Tagged(tag) => file.tiles_tagged(tag),
            Self::Named(name) => file.tiles_named(name),
        };
        if tiles.is_empty() {
            return Err(TownError::NoTiles(self.clone()));
        }
        Ok(tiles.into_iter().fold(TileMask::EMPTY, |mask, tile| {
            mask.union(TileMask::single(
                u32::try_from(tile).expect("at most 256 tiles"),
            ))
        }))
    }
}

/// Why a town could not be solved.
#[derive(Debug, thiserror::Error)]
pub enum TownError {
    #[error("no rule set is named {0:?}")]
    UnknownRules(String),
    #[error("the rule file has no tiles for {0:?}")]
    NoTiles(Selector),
    #[error(
        "a town needs at least 3 layers, a street, a roof and air above, but chunks are {0} tall"
    )]
    TooShallow(u32),
    #[error(transparent)]
    Generation(#[from] crate::Error),
    /// The rule set could not place some chunks of the town, even with repairs.
    #[error("chunks {0:?} of the town could not be placed")]
    Unplaced(Vec<(i32, i32)>),
}

/// What a bounded town allows where, for `file` and chunks `depth` cells tall: `bottom` on the
/// lowest layer, `top` on the highest, anything between, and no tile whose face asks for a
/// walkable neighbour facing out of the town's sides, where nothing would continue it.
///
/// # Errors
/// [`TownError::TooShallow`] under three layers; [`TownError::NoTiles`] for a selector that names
/// no tile.
pub fn town_prior(
    file: &RuleFile,
    depth: u32,
    bottom: Option<&Selector>,
    top: Option<&Selector>,
) -> Result<Prior, TownError> {
    if depth < 3 {
        return Err(TownError::TooShallow(depth));
    }
    let tiles = u32::try_from(file.num_tiles()).expect("at most 256 tiles");
    let mut layers = vec![TileMask::all(tiles); depth as usize];
    if let Some(bottom) = bottom {
        layers[0] = bottom.mask(file)?;
    }
    if let Some(top) = top {
        layers[depth as usize - 1] = top.mask(file)?;
    }
    let paths_out = |axis: usize| {
        let Some(modules) = file.modules() else {
            return TileMask::EMPTY;
        };
        (0..tiles)
            .filter(|&tile| {
                matches!(modules.face(tile as usize, axis), Face::Horizontal(face) if face.enforce_walkable_neighbor)
            })
            .fold(TileMask::EMPTY, |mask, tile| mask.union(TileMask::single(tile)))
    };
    Ok([POS_X, NEG_X, POS_Y, NEG_Y]
        .iter()
        .fold(Prior::open(tiles).with_layers(layers), |prior, &axis| {
            prior.with_face_ban(axis, paths_out(axis))
        }))
}

/// One town to solve.
#[derive(Clone, Copy, Debug)]
pub struct TownRequest<'a> {
    /// The rule set, by the name the solver was given it under.
    pub rules: &'a str,
    pub seed: u64,
    /// Chunks along the lattice's x and y.
    pub size: (u32, u32),
    pub bottom: Option<&'a Selector>,
    pub top: Option<&'a Selector>,
}

/// A solved town: each chunk's tiles, row by row with x fastest, from the town's lowest corner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Town {
    pub size: (u32, u32),
    pub chunks: Vec<Arc<[u16]>>,
}

/// Solves towns; the seam between the stage runtime and a WFC solver, on a GPU or not.
pub trait TownSolver: Send {
    /// The shape every chunk of every town has.
    fn chunk_shape(&self) -> ChunkShape;

    /// Solves one town whole.
    ///
    /// # Errors
    /// A [`TownError`] naming what is wrong.
    fn solve(&mut self, request: &TownRequest<'_>) -> Result<Town, TownError>;
}

/// A solver on a GPU device of its own for `ruleset`, as [`crate::Builder::build`] makes one: what
/// [`WfcTowns::with_rules`] takes when towns are solved on a GPU.
///
/// # Errors
/// If no device is available or the rule set does not fit one.
#[cfg(feature = "wgpu")]
pub fn gpu_solver(
    ruleset: Arc<Ruleset>,
) -> Result<crate::BlockSolver<crate::WgpuBackend>, crate::Error> {
    let backend = crate::WgpuBackend::from_env().map_err(wfc_gpu::error::GpuError::from)?;
    Ok(crate::BlockSolver::new(
        backend,
        ruleset,
        crate::SolverConfig::default(),
    )?)
}

/// A solver on a GPU device of its own for `ruleset`, as [`gpu_solver`] makes one, that keeps its
/// compiled kernels in `dir` across runs: compiling one takes seconds on some drivers, and a town's
/// first solve would otherwise pay for it every time a game starts.
///
/// # Errors
/// If no device is available, the rule set does not fit one, or the cache directory cannot be made
/// or read.
#[cfg(feature = "wgpu")]
pub fn gpu_solver_cached(
    ruleset: Arc<Ruleset>,
    dir: &std::path::Path,
) -> Result<crate::BlockSolver<crate::WgpuBackend>, crate::Error> {
    let backend = crate::WgpuBackend::from_env()
        .and_then(|backend| backend.cache_pipelines_in(dir))
        .map_err(wfc_gpu::error::GpuError::from)?;
    Ok(crate::BlockSolver::new(
        backend,
        ruleset,
        crate::SolverConfig::default(),
    )?)
}

struct RuleSet<S> {
    file: RuleFile,
    ruleset: Ruleset,
    /// Taken while a town is being solved.
    solver: Option<S>,
}

/// Towns solved by the library's own generator on any [`Solver`], one solver per rule set.
pub struct WfcTowns<S: Solver> {
    chunk: ChunkShape,
    sets: BTreeMap<String, RuleSet<S>>,
}

impl<S: Solver + Send> WfcTowns<S> {
    /// Towns of chunks of `chunk` cells, with no rule set yet.
    #[must_use]
    pub const fn new(chunk: ChunkShape) -> Self {
        Self {
            chunk,
            sets: BTreeMap::new(),
        }
    }

    /// Adds a rule set under `name`, with the solver `solver` builds for its compiled rules.
    ///
    /// # Errors
    /// If the rule file does not compile into a rule set.
    pub fn with_rules(
        mut self,
        name: &str,
        file: RuleFile,
        solver: impl FnOnce(Arc<Ruleset>) -> Result<S, crate::Error>,
    ) -> Result<Self, crate::Error> {
        let ruleset = Ruleset::new(file.rules(), &file.tileset().weights)?;
        let solver = solver(Arc::new(ruleset.clone()))?;
        self.sets.insert(
            name.to_owned(),
            RuleSet {
                file,
                ruleset,
                solver: Some(solver),
            },
        );
        Ok(self)
    }
}

impl<S: Solver> WfcTowns<S> {
    /// The solver of the rule set `name`, to inspect, for example what compiling its kernels has
    /// cost; `None` for a rule set it does not have.
    #[must_use]
    pub fn solver(&self, name: &str) -> Option<&S> {
        self.sets.get(name)?.solver.as_ref()
    }
}

impl<S: Solver + Send> TownSolver for WfcTowns<S> {
    fn chunk_shape(&self) -> ChunkShape {
        self.chunk
    }

    fn solve(&mut self, request: &TownRequest<'_>) -> Result<Town, TownError> {
        let set = self
            .sets
            .get_mut(request.rules)
            .ok_or_else(|| TownError::UnknownRules(request.rules.to_owned()))?;
        let prior = town_prior(&set.file, self.chunk.z, request.bottom, request.top)?;
        let (w, h) = request.size;
        let extent = town_extent(self.chunk, request.size);
        let solver = set
            .solver
            .take()
            .expect("a rule set's solver is back after every town");
        let mut world = Builder::new(set.ruleset.clone(), prior)
            .seed(request.seed)
            .extent(extent)
            .halo(1)
            .build_with(solver);
        // Every kernel the town runs is compiled at once before it starts, rather than one after
        // another as its batches first need them.
        let shapes = world.kernel_shapes(w.max(h));
        let centre = ChunkCoord::new(w as i32 / 2, h as i32 / 2, 0);
        world.request(&[FocusPoint::new(centre, w.max(h))]);
        let outcome = world
            .solver_mut()
            .warm(&shapes)
            .map_err(crate::Error::from)
            .and_then(|()| world.run_until_idle());
        let mut unplaced: Vec<(i32, i32)> = Vec::new();
        let mut chunks = Vec::with_capacity((w * h) as usize);
        if outcome.is_ok() {
            for y in 0..h as i32 {
                for x in 0..w as i32 {
                    match world.chunk(ChunkCoord::new(x, y, 0)) {
                        Some(chunk) => chunks.push(Arc::from(chunk.tiles.as_ref())),
                        None => unplaced.push((x, y)),
                    }
                }
            }
        }
        set.solver = Some(world.into_solver());
        if let Err(error) = outcome {
            return Err(TownError::Generation(error));
        }
        if !unplaced.is_empty() {
            return Err(TownError::Unplaced(unplaced));
        }
        Ok(Town {
            size: request.size,
            chunks,
        })
    }
}

/// The bounded world a town of `size` chunks is solved in.
fn town_extent(chunk: ChunkShape, size: (u32, u32)) -> WorldExtent {
    WorldExtent::new(chunk)
        .with_x(0..size.0 as i32)
        .with_y(0..size.1 as i32)
        .with_z(0..1)
}

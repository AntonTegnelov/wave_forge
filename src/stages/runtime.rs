//! Generating a pack's stages around focus points, providers first.
//!
//! A stage's output for a chunk is a pure function of the world seed, the stage, the chunk and what
//! its inputs hold within the stage's reach of that chunk (docs/architecture/stages.md, "The
//! execution contract"). The runtime makes that hold by construction: before a stage runs for a
//! chunk, every chunk of its inputs that the chunk's area grown by the reach overlaps is generated,
//! and the stage reads them only through a view bounded by that area. Whatever order chunks are
//! asked for in, each is computed from the same inputs and comes out the same.

use super::pack::{Condition, Expr, Output, Pack, Reach, StageKind, point_stage_id, salt};
use crate::products::InstanceId;
use crate::scheduler::FocusPoint;
use crate::towns::{Town, TownRequest, TownSolver};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use wfc_core::ChunkCoord;
use wfc_core::hash::pcg3d;

/// A value per cell column of one chunk.
#[derive(Clone, Debug, PartialEq)]
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

/// A settlement site: a rectangle of whole chunks at one height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Site {
    /// The region that owns it, which also names it: a region has at most one site.
    pub region: (i32, i32),
    /// The chunks it covers, from `min` up to but not including `max`, along the lattice's x and y.
    pub min: (i32, i32),
    pub max: (i32, i32),
    /// The height its ground is levelled to.
    pub height: f32,
}

impl Site {
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
#[derive(Clone, Debug, PartialEq)]
pub struct TownChunk {
    /// The site's region, which names the town.
    pub region: (i32, i32),
    /// The site's levelled height, where an engine puts the town's lowest layer.
    pub height: f32,
    /// The chunk's tiles, x fastest, then y, then z, as a WFC chunk stores them.
    pub tiles: Arc<[u16]>,
}

/// A point a Scatter stage placed.
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    /// Positional: the chunk, the stage and the column, so it never changes when other points do.
    pub id: InstanceId,
    /// What an engine binds a scene or a model to.
    pub kind: Arc<str>,
    /// Where it stands, in cells: x and y along the lattice's ground, z the height field's value.
    pub position: [f32; 3],
    /// Its turn about the vertical, as a fraction of a whole turn.
    pub turn: f32,
}

/// What a stage holds for one chunk.
#[derive(Clone, Debug, PartialEq)]
pub enum Product {
    Field(Field),
    /// The sites whose footprint overlaps the chunk.
    Sites(Vec<Site>),
    /// The chunk's part of a town, or nothing for a chunk outside every site.
    Tiles(Option<TownChunk>),
    /// The points whose column lies in the chunk.
    Points(Vec<Point>),
}

impl Product {
    fn field(&self) -> &Field {
        match self {
            Self::Field(field) => field,
            Self::Sites(_) | Self::Tiles(_) | Self::Points(_) => {
                unreachable!("inputs are type checked when the pack loads")
            }
        }
    }

    fn sites(&self) -> &[Site] {
        match self {
            Self::Sites(sites) => sites,
            Self::Field(_) | Self::Tiles(_) | Self::Points(_) => {
                unreachable!("inputs are type checked when the pack loads")
            }
        }
    }
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
    #[error("the town solver's chunks are {solver:?} columns, the runtime's {runtime:?}")]
    ChunkMismatch { solver: [u32; 2], runtime: [u32; 2] },
    #[error("stage {stage:?} could not solve the town of region {region:?}: {message}")]
    Town {
        stage: String,
        region: (i32, i32),
        message: String,
    },
}

/// One input of a stage, readable only within the stage's reach of the chunk being generated.
pub struct FieldView<'a> {
    stage: &'a str,
    input: &'a str,
    /// The reach along the lattice's x, which is what errors report.
    reach: u32,
    size: [u32; 2],
    /// The columns the view may read, in world columns, inclusive.
    min: [i64; 2],
    max: [i64; 2],
    chunks: BTreeMap<(i32, i32), &'a Field>,
}

impl FieldView<'_> {
    /// The input's value at the world column `x`, `y`.
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
        let (cx, cy) = (i64::from(self.size[0]), i64::from(self.size[1]));
        let chunk = (
            i32::try_from(x.div_euclid(cx)).expect("a chunk coordinate"),
            i32::try_from(y.div_euclid(cy)).expect("a chunk coordinate"),
        );
        let field = self
            .chunks
            .get(&chunk)
            .expect("the runtime generates every chunk within reach first");
        Ok(field.get(x.rem_euclid(cx) as u32, y.rem_euclid(cy) as u32))
    }
}

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
    towns: Option<Box<dyn TownSolver>>,
    /// Towns solved, by Solve stage and site region, kept while a chunk of their region is needed.
    solved: BTreeMap<(usize, (i32, i32)), Arc<Town>>,
}

impl Runtime {
    /// A runtime for `pack`, with chunks of `size` columns.
    #[must_use]
    pub fn new(pack: Arc<Pack>, seed: u64, size: [u32; 2]) -> Self {
        Self {
            pack,
            seed,
            size,
            needed: BTreeMap::new(),
            focus: Vec::new(),
            products: BTreeMap::new(),
            towns: None,
            solved: BTreeMap::new(),
        }
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
        self.towns = Some(towns);
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
        let targets = targets
            .iter()
            .map(|target| {
                self.pack
                    .index(target)
                    .ok_or_else(|| StageError::UnknownStage((*target).to_owned()))
            })
            .collect::<Result<Vec<usize>, StageError>>()?;
        let asked: BTreeSet<ChunkCoord> = focus
            .iter()
            .flat_map(|focus| {
                let radius = focus.radius as i32;
                let centre = focus.chunk;
                (-radius..=radius).flat_map(move |x| {
                    (-radius..=radius).map(move |y| ChunkCoord::new(centre.x + x, centre.y + y, 0))
                })
            })
            .collect();
        let mut needed: BTreeMap<usize, BTreeSet<ChunkCoord>> = targets
            .into_iter()
            .map(|target| (target, asked.clone()))
            .collect();
        // Consumers come after their inputs in `order`, so walking it backwards reaches a stage
        // only once everything that reads it has said which of its chunks it needs.
        for &index in self.pack.order.iter().rev() {
            let Some(chunks) = needed.get(&index).cloned() else {
                continue;
            };
            for &(input, reach) in &self.pack.stages[index].inputs {
                let reach = reach.cells(self.size);
                let covered: BTreeSet<ChunkCoord> = chunks
                    .iter()
                    .flat_map(|&chunk| self.chunks_within(chunk, reach))
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
        self.solved.retain(|&(stage, owner), _| {
            let StageKind::Solve { sites, .. } = &pack.stages[stage].kind else {
                unreachable!("only Solve stages solve towns")
            };
            let sites = pack.index(sites).expect("linked when loaded");
            let StageKind::Sites { region, .. } = pack.stages[sites].kind else {
                unreachable!("a Solve stage reads a Sites stage")
            };
            needed.get(&stage).is_some_and(|chunks| {
                chunks.iter().any(|chunk| {
                    (
                        chunk.x.div_euclid(region as i32),
                        chunk.y.div_euclid(region as i32),
                    ) == owner
                })
            })
        });
        self.needed = needed;
        self.focus = focus.to_vec();
        Ok(dropped)
    }

    /// Generates everything the request needs that is missing, inputs first and nearest first,
    /// and returns what it generated as (stage, chunk).
    ///
    /// # Errors
    /// A [`StageError`] from a stage, which is a bug in that stage.
    pub fn run_until_idle(&mut self) -> Result<Vec<(String, ChunkCoord)>, StageError> {
        self.step(usize::MAX)
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
            missing.sort_by_key(|chunk| {
                let distance = self
                    .focus
                    .iter()
                    .map(|focus| focus.distance(*chunk))
                    .min()
                    .unwrap_or(u32::MAX);
                (distance, *chunk)
            });
            for chunk in missing.into_iter().take(budget - generated.len()) {
                self.solve_town_of(index, chunk)?;
                let product = self.generate(index, chunk)?;
                self.products.insert((index, chunk), Arc::new(product));
                generated.push((self.pack.stages[index].name.clone(), chunk));
            }
        }
        Ok(generated)
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
            Product::Sites(_) | Product::Tiles(_) | Product::Points(_) => None,
        }
    }

    /// The sites `stage` holds for `chunk`, if it is a sites stage and the chunk is generated.
    #[must_use]
    pub fn sites(&self, stage: &str, chunk: ChunkCoord) -> Option<&[Site]> {
        match self.product(stage, chunk)? {
            Product::Sites(sites) => Some(sites),
            Product::Field(_) | Product::Tiles(_) | Product::Points(_) => None,
        }
    }

    /// The town chunk `stage` holds for `chunk`: `None` if it is not a Solve stage, the chunk is
    /// not generated, or it lies outside every site.
    #[must_use]
    pub fn tiles(&self, stage: &str, chunk: ChunkCoord) -> Option<&TownChunk> {
        match self.product(stage, chunk)? {
            Product::Tiles(town) => town.as_ref(),
            Product::Field(_) | Product::Sites(_) | Product::Points(_) => None,
        }
    }

    /// The points `stage` placed in `chunk`, if it is a Scatter stage and the chunk is generated.
    #[must_use]
    pub fn points(&self, stage: &str, chunk: ChunkCoord) -> Option<&[Point]> {
        match self.product(stage, chunk)? {
            Product::Points(points) => Some(points),
            Product::Field(_) | Product::Sites(_) | Product::Tiles(_) => None,
        }
    }

    /// The site of a Solve stage's chunk, if the chunk lies in one.
    fn site_of(&self, index: usize, chunk: ChunkCoord) -> Option<Site> {
        let (sites, reach) = self.pack.stages[index].inputs[0];
        self.inputs_within(chunk, sites, reach.cells(self.size))
            .find(|&(at, _)| at == chunk)
            .and_then(|(_, product)| product.sites().first().copied())
    }

    /// Solves the town of `chunk`'s site for Solve stage `index`, unless it is solved already or
    /// the chunk lies in no site.
    fn solve_town_of(&mut self, index: usize, chunk: ChunkCoord) -> Result<(), StageError> {
        let stage = &self.pack.stages[index];
        let StageKind::Solve {
            rules, bottom, top, ..
        } = &stage.kind
        else {
            return Ok(());
        };
        let Some(site) = self.site_of(index, chunk) else {
            return Ok(());
        };
        if self.solved.contains_key(&(index, site.region)) {
            return Ok(());
        }
        let towns = self
            .towns
            .as_mut()
            .ok_or_else(|| StageError::NoTownSolver(stage.name.clone()))?;
        let world = (self.seed as u32) ^ ((self.seed >> 32) as u32);
        let [high, low, _] = pcg3d([
            world ^ stage.salt,
            site.region.0 as u32,
            site.region.1 as u32,
        ]);
        let request = TownRequest {
            rules,
            seed: (u64::from(high) << 32) | u64::from(low),
            size: (
                (site.max.0 - site.min.0) as u32,
                (site.max.1 - site.min.1) as u32,
            ),
            bottom: bottom.as_ref(),
            top: top.as_ref(),
        };
        let town = towns.solve(&request).map_err(|error| StageError::Town {
            stage: stage.name.clone(),
            region: site.region,
            message: error.to_string(),
        })?;
        self.solved.insert((index, site.region), Arc::new(town));
        Ok(())
    }

    /// How many chunks the runtime holds, over all stages.
    #[must_use]
    pub fn held(&self) -> usize {
        self.products.len()
    }

    /// The chunks whose columns lie within `reach` columns of `chunk`'s, along each axis.
    fn chunks_within(&self, chunk: ChunkCoord, reach: [u32; 2]) -> Vec<ChunkCoord> {
        let span = |axis: usize, at: i32| {
            let size = i64::from(self.size[axis]);
            let low = i64::from(at) * size - i64::from(reach[axis]);
            let high = (i64::from(at) + 1) * size - 1 + i64::from(reach[axis]);
            let chunk = |column: i64| i32::try_from(column.div_euclid(size)).expect("a chunk");
            chunk(low)..=chunk(high)
        };
        span(0, chunk.x)
            .flat_map(|x| span(1, chunk.y).map(move |y| ChunkCoord::new(x, y, 0)))
            .collect()
    }

    /// The products of `input` within `reach` of `chunk`.
    fn inputs_within(
        &self,
        chunk: ChunkCoord,
        input: usize,
        reach: [u32; 2],
    ) -> impl Iterator<Item = (ChunkCoord, &Product)> {
        self.chunks_within(chunk, reach).into_iter().map(move |at| {
            let product = self
                .products
                .get(&(input, at))
                .expect("inputs are generated before the stages that read them");
            (at, product.as_ref())
        })
    }

    fn view(&self, stage: usize, chunk: ChunkCoord, input: usize, reach: Reach) -> FieldView<'_> {
        let reach = reach.cells(self.size);
        let origin = [
            i64::from(chunk.x) * i64::from(self.size[0]),
            i64::from(chunk.y) * i64::from(self.size[1]),
        ];
        let chunks = self
            .inputs_within(chunk, input, reach)
            .map(|(at, product)| ((at.x, at.y), product.field()))
            .collect();
        FieldView {
            stage: &self.pack.stages[stage].name,
            input: &self.pack.stages[input].name,
            reach: reach[0],
            size: self.size,
            min: [
                origin[0] - i64::from(reach[0]),
                origin[1] - i64::from(reach[1]),
            ],
            max: [
                origin[0] + i64::from(self.size[0]) - 1 + i64::from(reach[0]),
                origin[1] + i64::from(self.size[1]) - 1 + i64::from(reach[1]),
            ],
            chunks,
        }
    }

    /// Every site of `input` within `reach` of `chunk`, once each.
    fn sites_near(&self, chunk: ChunkCoord, input: usize, reach: Reach) -> Vec<Site> {
        let mut sites: BTreeMap<(i32, i32), Site> = BTreeMap::new();
        for (_, product) in self.inputs_within(chunk, input, reach.cells(self.size)) {
            for site in product.sites() {
                sites.insert(site.region, *site);
            }
        }
        sites.into_values().collect()
    }

    fn generate(&self, index: usize, chunk: ChunkCoord) -> Result<Product, StageError> {
        let stage = &self.pack.stages[index];
        if let StageKind::Solve { .. } = &stage.kind {
            return Ok(Product::Tiles(self.site_of(index, chunk).map(|site| {
                let town = &self.solved[&(index, site.region)];
                let (x, y) = (chunk.x - site.min.0, chunk.y - site.min.1);
                TownChunk {
                    region: site.region,
                    height: site.height,
                    tiles: Arc::clone(&town.chunks[(y as u32 * town.size.0 + x as u32) as usize]),
                }
            })));
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
        let views: BTreeMap<usize, FieldView<'_>> = stage
            .inputs
            .iter()
            .filter(|&&(input, _)| self.pack.stages[input].kind.output() == Output::Field)
            .map(|&(input, reach)| (input, self.view(index, chunk, input, reach)))
            .collect();
        let near: Vec<Site> = match &stage.kind {
            StageKind::Flatten { .. } => {
                let (sites, reach) = stage.inputs[1];
                self.sites_near(chunk, sites, reach)
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
                    StageKind::Field(expr) => self.evaluate(expr, stage.salt, column, &|name| {
                        &views[&self.pack.index(name).expect("linked when loaded")]
                    })?,
                    StageKind::Blur { input, radius } => {
                        let view = &views[&self.pack.index(input).expect("linked when loaded")];
                        let r = i64::from(*radius);
                        let mut sum = 0.0_f32;
                        for dy in -r..=r {
                            for dx in -r..=r {
                                sum += view.get(column[0] + dx, column[1] + dy)?;
                            }
                        }
                        sum / ((2 * r + 1) * (2 * r + 1)) as f32
                    }
                    StageKind::Flatten { height, blend, .. } => {
                        let view = &views[&self.pack.index(height).expect("linked when loaded")];
                        let base = view.get(column[0], column[1])?;
                        let nearest = near
                            .iter()
                            .map(|site| (site.distance(column[0], column[1], self.size), site))
                            .min_by(|a, b| a.0.total_cmp(&b.0));
                        match nearest {
                            Some((0.0, site)) => site.height,
                            Some((distance, site)) if distance < *blend as f32 => {
                                let t = distance / *blend as f32;
                                let t = t * t * (3.0 - 2.0 * t);
                                site.height + (base - site.height) * t
                            }
                            _ => base,
                        }
                    }
                    StageKind::Sites { .. }
                    | StageKind::Solve { .. }
                    | StageKind::Scatter { .. } => {
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

    /// The points of Scatter stage `index` whose column lies in `chunk`.
    ///
    /// Every block of `spacing` columns has one candidate at a hashed column, with a hashed
    /// priority. A candidate passes its own tests (chance, height, slope, sites) from what lies
    /// at and around its column, and is kept if it passes and no candidate that also passes, of
    /// higher priority, lies closer than `apart`. Everything a decision reads is within the
    /// stage's reach, so a chunk's points are the same whatever else has been generated.
    fn scatter(&self, index: usize, chunk: ChunkCoord) -> Result<Vec<Point>, StageError> {
        let stage = &self.pack.stages[index];
        let StageKind::Scatter {
            kind,
            spacing,
            chance,
            between,
            max_slope,
            avoid,
            apart,
            ..
        } = &stage.kind
        else {
            unreachable!("called for Scatter stages")
        };
        let (height, reach) = stage.inputs[0];
        let heights = self.view(index, chunk, height, reach);
        let sites = match avoid {
            Some(_) => {
                let (sites, reach) = stage.inputs[1];
                self.sites_near(chunk, sites, reach)
            }
            None => Vec::new(),
        };
        let margin = avoid.as_ref().map_or(0, |(_, margin)| *margin);
        let world = (self.seed as u32) ^ ((self.seed >> 32) as u32);
        let spacing = i64::from(*spacing);
        let threshold = (f64::from(*chance) * 4_294_967_296.0) as u64;
        let [sx, sy] = [i64::from(self.size[0]), i64::from(self.size[1])];
        let (x0, y0) = (i64::from(chunk.x) * sx, i64::from(chunk.y) * sy);
        let apart = i64::from(*apart);
        // The blocks that can hold a candidate within `apart` of the chunk's columns.
        let blocks = |low: i64, high: i64| low.div_euclid(spacing)..=high.div_euclid(spacing);
        let mut candidates: Vec<Candidate> = Vec::new();
        for by in blocks(y0 - apart, y0 + sy - 1 + apart) {
            for bx in blocks(x0 - apart, x0 + sx - 1 + apart) {
                let [place, rank, turn] =
                    pcg3d([world ^ stage.salt, bx as u32, (by as u32) ^ 0x5bd1_e995]);
                let column = (
                    bx * spacing + i64::from(place) % spacing,
                    by * spacing + i64::from(place >> 16) % spacing,
                );
                // Only a candidate within `apart` columns of the chunk can crowd one inside it: a
                // point stands within its column, so one further out is more than `apart` away.
                let near = (x0 - apart..=x0 + sx - 1 + apart).contains(&column.0)
                    && (y0 - apart..=y0 + sy - 1 + apart).contains(&column.1);
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
                    priority: (rank, bx, by),
                    turn: (turn >> 8) as f32 / (1u32 << 24) as f32,
                    exists: u64::from(turn) < threshold,
                });
            }
        }
        let passes = |candidate: &Candidate| -> Result<Option<f32>, StageError> {
            if !candidate.exists {
                return Ok(None);
            }
            let (x, y) = candidate.column;
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
            if sites
                .iter()
                .any(|site| site.distance(x, y, self.size) < margin as f32)
            {
                return Ok(None);
            }
            Ok(Some(here))
        };
        let mut heights_of: Vec<Option<f32>> = Vec::with_capacity(candidates.len());
        for candidate in &candidates {
            heights_of.push(passes(candidate)?);
        }
        let point_stage = point_stage_id(stage.salt);
        let kind: Arc<str> = Arc::from(kind.as_str());
        let mut points = Vec::new();
        for (i, candidate) in candidates.iter().enumerate() {
            let (x, y) = candidate.column;
            let inside = (x0..x0 + sx).contains(&x) && (y0..y0 + sy).contains(&y);
            let Some(z) = heights_of[i].filter(|_| inside) else {
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
            let cell = ((y - y0) * sx + (x - x0)) as u32;
            points.push(Point {
                id: InstanceId::new(chunk, point_stage, cell, 0),
                kind: Arc::clone(&kind),
                position: [candidate.at.0, candidate.at.1, z],
                turn: candidate.turn,
            });
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
        Ok(Some(Site {
            region: owner,
            min,
            max,
            height: sum / samples.len() as f32,
        }))
    }

    fn evaluate<'v>(
        &self,
        expr: &Expr,
        salt: u32,
        column: [i64; 2],
        input: &dyn Fn(&str) -> &'v FieldView<'v>,
    ) -> Result<f32, StageError> {
        let value = |expr: &Expr| self.evaluate(expr, salt, column, input);
        let centre = [column[0] as f32 + 0.5, column[1] as f32 + 0.5];
        Ok(match expr {
            Expr::Constant(value) => *value,
            Expr::Noise {
                frequency,
                octaves,
                name,
            } => {
                let stream = name.as_deref().map_or(salt, noise_stream);
                value_noise(self.seed, stream, *frequency, *octaves, column)
            }
            Expr::Input(name) => input(name).get(column[0], column[1])?,
            Expr::X => centre[0],
            Expr::Y => centre[1],
            Expr::Distance((x, y)) => (centre[0] - x).hypot(centre[1] - y),
            Expr::Angle((x, y)) => {
                let turn = (centre[1] - y).atan2(centre[0] - x) / std::f32::consts::TAU;
                turn.rem_euclid(1.0)
            }
            Expr::Add(a, b) => value(a)? + value(b)?,
            Expr::Sub(a, b) => value(a)? - value(b)?,
            Expr::Mul(a, b) => value(a)? * value(b)?,
            Expr::Min(a, b) => value(a)?.min(value(b)?),
            Expr::Max(a, b) => value(a)?.max(value(b)?),
            Expr::Abs(a) => value(a)?.abs(),
            Expr::Floor(a) => value(a)?.floor(),
            Expr::Clamp(a, low, high) => value(a)?.clamp(*low, *high),
            Expr::Smoothstep(low, high, a) => {
                let t = ((value(a)? - low) / (high - low)).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            }
            Expr::Remap(a, (from_low, from_high), (to_low, to_high)) => {
                to_low + (value(a)? - from_low) / (from_high - from_low) * (to_high - to_low)
            }
            Expr::Curve(a, points) => curve(points, value(a)?),
            Expr::Select {
                when,
                then,
                otherwise,
            } => {
                let holds = match when {
                    Condition::Less(a, b) => value(a)? < value(b)?,
                    Condition::Greater(a, b) => value(a)? > value(b)?,
                };
                value(if holds { then } else { otherwise })?
            }
        })
    }
}

/// The stream of a named noise: the same wherever the name appears, and apart from any stage's
/// own stream, whose salt is the stage's name alone.
fn noise_stream(name: &str) -> u32 {
    salt(name) ^ 0x6E6F_6973
}

/// A piecewise-linear curve at `x`: level beyond its first and last points, linear between.
/// The points are in increasing x and at least two, which loading checks.
fn curve(points: &[(f32, f32)], x: f32) -> f32 {
    let (first, last) = (points[0], points[points.len() - 1]);
    if x <= first.0 {
        return first.1;
    }
    if x >= last.0 {
        return last.1;
    }
    let after = points.partition_point(|point| point.0 <= x);
    let ((x0, y0), (x1, y1)) = (points[after - 1], points[after]);
    y0 + (x - x0) / (x1 - x0) * (y1 - y0)
}

/// One Scatter candidate: its column, where in it the point stands, its priority (a hash, with the
/// block breaking ties), its turn, and whether the chance test lets it exist at all.
struct Candidate {
    column: (i64, i64),
    at: (f32, f32),
    priority: (u32, i64, i64),
    turn: f32,
    exists: bool,
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

/// Fractal value noise at a world column, in 0..1: every column depends on its position alone,
/// so chunks meet without seams.
fn value_noise(seed: u64, salt: u32, frequency: f32, octaves: u32, column: [i64; 2]) -> f32 {
    let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
    let mut sum = 0.0;
    let mut weight = 1.0;
    let mut total = 0.0;
    let mut scale = frequency;
    for octave in 0..octaves {
        let fx = (column[0] as f32 + 0.5) * scale;
        let fy = (column[1] as f32 + 0.5) * scale;
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
        let field = Field {
            chunk: ChunkCoord::new(0, 0, 0),
            size: [4, 4],
            values: vec![0.0; 16],
        };
        let view = FieldView {
            stage: "reader",
            input: "source",
            reach: 1,
            size: [4, 4],
            min: [-1, -1],
            max: [4, 4],
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
            .map(|i| value_noise(3, 11, 0.37, 4, [i * 7 - 5000, i * 13 - 9000]))
            .collect();

        assert!(values.iter().all(|v| (0.0..1.0).contains(v)), "{values:?}");
    }
}

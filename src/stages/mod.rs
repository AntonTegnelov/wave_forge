//! Generation as a pack of stages (docs/architecture/stages.md, and docs/reference/packs.md for the
//! format): the pack a world is described by, and the runtime that generates its stages around
//! focus points, providers first.

mod assemble;
pub mod brushes;
mod caves;
mod codec;
mod connected;
mod droplets;
pub mod edits;
mod erode;
mod evaluate;
pub mod facts;
mod lakes;
mod network;
pub mod pack;
pub mod regions;
mod rivers;
pub mod runtime;
pub mod save;
mod talus;
mod town_thread;
pub mod worker;
mod world_run;

pub use edits::{Edit, Edits, PointId};
pub use facts::{AUTHORED, Facts, GivenRow, MAX_SHARED, Row, RowId, Table, Value};
pub use pack::{
    AmbienceDef, AmbienceKind, AuthoredRow, Biome, Bound, Column, Condition, Door, DropletsGive,
    Expr, Facing, Group, Level, LocationKind, MAX_APPEARANCE, MAX_BLEND, MAX_CATEGORIES,
    MAX_CAVE_ROOMS, MAX_CHILDREN, MAX_DEPOSITS, MAX_PIECES, MAX_SCATTER_SLOTS, MAX_SPAWN_BUDGET,
    MAX_TRIES, MAX_VARIANTS, Materials, PACK_VERSION, Pack, PackError, PackFile, PackWater,
    ParamDef, Pattern, Persist, Piece, Profile, Room, Rule, ScatterKind, SolveMask, Spawnable,
    StageDef, StageKind, TableDef, TableKind, Water,
};
pub use runtime::{
    Categories, Field, FieldView, Judgement, PlaceName, Point, Product, Readings, Rejection,
    Runtime, Site, SiteId, StageError, StageTiming, Stamp, TownChunk, Volume,
};
pub use save::{FrozenChunk, Save};
pub use worker::{StageEvent, StageWorker};
pub use world_run::RunProgress;

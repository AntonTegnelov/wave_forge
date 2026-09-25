//! Generation as a pack of stages (docs/architecture/stages.md, and docs/reference/packs.md for the
//! format): the pack a world is described by, and the runtime that generates its stages around
//! focus points, providers first.

mod assemble;
pub mod edits;
mod evaluate;
pub mod facts;
mod lakes;
mod network;
pub mod pack;
pub mod regions;
mod rivers;
pub mod runtime;
pub mod save;
mod town_thread;
pub mod worker;

pub use edits::{Edit, Edits, PointId};
pub use facts::{Facts, GivenRow, MAX_SHARED, Row, RowId, Table, Value};
pub use pack::{
    Bound, Column, Condition, Door, Expr, Facing, Group, LocationKind, MAX_BLEND, MAX_CATEGORIES,
    MAX_CHILDREN, MAX_PIECES, MAX_SCATTER_SLOTS, MAX_TRIES, PACK_VERSION, Pack, PackError,
    PackFile, PackWater, Persist, Piece, Profile, Rule, StageDef, StageKind, TableDef, TableKind,
    Water,
};
pub use runtime::{
    Categories, Field, FieldView, PlaceName, Point, Product, Runtime, Site, SiteId, StageError,
    StageTiming, Stamp, TownChunk,
};
pub use save::{FrozenChunk, Save};
pub use worker::{StageEvent, StageWorker};

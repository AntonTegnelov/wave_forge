//! Generation as a pack of stages (docs/architecture/stages.md, and docs/reference/packs.md for the
//! format): the pack a world is described by, and the runtime that generates its stages around
//! focus points, providers first.

mod evaluate;
pub mod facts;
pub mod pack;
pub mod regions;
pub mod runtime;
pub mod worker;

pub use facts::{Facts, GivenRow, MAX_SHARED, Row, RowId, Table, Value};
pub use pack::{
    Column, Condition, Expr, MAX_BLEND, MAX_CATEGORIES, MAX_CHILDREN, PACK_VERSION, Pack,
    PackError, PackFile, Rule, StageDef, StageKind, TableDef, TableKind,
};
pub use runtime::{
    Categories, Field, FieldView, Point, Product, Runtime, Site, StageError, StageTiming, TownChunk,
};
pub use worker::{StageEvent, StageWorker};

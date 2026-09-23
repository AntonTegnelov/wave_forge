//! Generation as a pack of stages (docs/architecture/stages.md, and docs/reference/packs.md for the
//! format): the pack a world is described by, and the runtime that generates its stages around
//! focus points, providers first.

pub mod pack;
pub mod runtime;
pub mod worker;

pub use pack::{Condition, Expr, PACK_VERSION, Pack, PackError, PackFile, StageDef, StageKind};
pub use runtime::{Field, FieldView, Point, Product, Runtime, Site, StageError, TownChunk};
pub use worker::{StageEvent, StageWorker};

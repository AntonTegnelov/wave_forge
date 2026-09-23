//! Generation as a pack of stages (docs/generation-model.md): the pack a world is described by,
//! and the runtime that generates its stages around focus points, providers first.

pub mod pack;
pub mod runtime;

pub use pack::{Expr, PACK_VERSION, Pack, PackError, PackFile, StageDef, StageKind};
pub use runtime::{Field, FieldView, Product, Runtime, Site, StageError, TownChunk};

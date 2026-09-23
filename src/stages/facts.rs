//! Tables of facts: rows the game gives, and rows generated from a parent table's.
//!
//! Some of what shapes a world is not generation at all: a history the game simulates before play,
//! a galaxy's systems a player travels between (docs/architecture/stages.md, "History and other
//! facts from the game"). A pack names such tables, and [`Facts`] holds them for one seed: the
//! given tables as the game last gave them, and the generated ones computed from the pack, the seed
//! and their parents. A runtime reads the facts it is given ([`crate::stages::Runtime::set_facts`]),
//! so the world is a function of the pack, the seed and the facts.

use super::evaluate::{Leaves, evaluate};
use super::pack::{Column, Expr, MAX_CHILDREN, Pack, Schema, TableKind, salt};
use super::runtime::StageError;
use std::collections::BTreeMap;
use std::sync::Arc;
use wfc_core::hash::pcg3d;

/// A row's id. A given row's is the game's own id alone; a generated row's is its parent's id
/// followed by its index among the parent's children, or its index alone without a parent. Either
/// way, adding or changing other rows never changes it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RowId(pub Vec<u64>);

/// One row of a table: its id and a value for each of the table's columns, in their order.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub id: RowId,
    pub values: Vec<f32>,
}

/// A table's rows, in the order of their ids.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Table {
    /// The columns' names, in the order a row holds their values.
    pub columns: Vec<String>,
    pub rows: Vec<Row>,
}

impl Table {
    /// The row with `id`, if the table has one.
    #[must_use]
    pub fn row(&self, id: &RowId) -> Option<&Row> {
        self.rows
            .binary_search_by(|row| row.id.cmp(id))
            .ok()
            .map(|at| &self.rows[at])
    }

    /// Where the column named `name` is in a row's values, if the table has one.
    #[must_use]
    pub fn column(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|column| column == name)
    }
}

/// A value the game gives for one column of a row.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// For a `Number` column.
    Number(f32),
    /// For a `Names` column: one of its names.
    Name(String),
}

/// A row the game gives: its own id, and a value for every column of the table.
#[derive(Clone, Debug, PartialEq)]
pub struct GivenRow {
    pub id: u64,
    pub values: BTreeMap<String, Value>,
}

/// Every table of a pack for one seed. Cloning is cheap, since the tables are shared, so a game
/// can keep its own and hand a copy to each runtime.
#[derive(Clone, Debug)]
pub struct Facts {
    pack: Arc<Pack>,
    seed: u64,
    /// By the pack's table index.
    tables: Vec<Arc<Table>>,
}

impl Facts {
    /// The pack's tables for `seed`: the given ones empty, the generated ones computed.
    ///
    /// # Errors
    /// [`StageError::Table`] if a generated table's count is out of range, or a share splits a
    /// value that is not a whole number.
    pub fn new(pack: Arc<Pack>, seed: u64) -> Result<Self, StageError> {
        let tables = pack
            .tables
            .iter()
            .map(|schema| {
                Arc::new(Table {
                    columns: schema.columns.clone(),
                    rows: Vec::new(),
                })
            })
            .collect();
        let mut facts = Self { pack, seed, tables };
        for &index in &facts.pack.table_order.clone() {
            if matches!(facts.pack.tables[index].kind, TableKind::Generated { .. }) {
                facts.tables[index] = Arc::new(facts.generate(index)?);
            }
        }
        Ok(facts)
    }

    /// Replaces the rows of the given table named `table` and recomputes every generated table
    /// below it. On an error nothing changes.
    ///
    /// # Errors
    /// [`StageError::UnknownTable`] if the pack names no such table, and [`StageError::Table`] if
    /// it is not a given table, two rows share an id, a row lacks a column or has one the table
    /// does not, a number is not finite or a name is not one of its column's, or recomputing a
    /// table below fails.
    pub fn give(&mut self, table: &str, rows: Vec<GivenRow>) -> Result<(), StageError> {
        let index = self
            .pack
            .table_by_name
            .get(table)
            .copied()
            .ok_or_else(|| StageError::UnknownTable(table.to_owned()))?;
        let mut next = self.clone();
        next.tables[index] = Arc::new(given_rows(&self.pack.tables[index], rows)?);
        for &below in &self.pack.table_order {
            if self.descends(below, index) {
                next.tables[below] = Arc::new(next.generate(below)?);
            }
        }
        *self = next;
        Ok(())
    }

    /// The table named `name`, if the pack has one.
    #[must_use]
    pub fn table(&self, name: &str) -> Option<&Table> {
        let index = *self.pack.table_by_name.get(name)?;
        Some(&self.tables[index])
    }

    /// Whether these facts were made for `pack` and `seed`.
    pub(crate) fn made_for(&self, pack: &Arc<Pack>, seed: u64) -> bool {
        Arc::ptr_eq(&self.pack, pack) && self.seed == seed
    }

    /// The tables, by the pack's table index, shared so a runtime can tell which changed.
    pub(crate) fn tables(&self) -> &[Arc<Table>] {
        &self.tables
    }

    /// Whether table `index` lies below table `ancestor`.
    fn descends(&self, index: usize, ancestor: usize) -> bool {
        let mut at = self.pack.tables[index].parent;
        while let Some(parent) = at {
            if parent == ancestor {
                return true;
            }
            at = self.pack.tables[parent].parent;
        }
        false
    }

    /// Generated table `index`'s rows from its parent's as they are now.
    fn generate(&self, index: usize) -> Result<Table, StageError> {
        let schema = &self.pack.tables[index];
        let TableKind::Generated { count, columns, .. } = &schema.kind else {
            unreachable!("called for generated tables")
        };
        let failed = |message: String| StageError::Table {
            table: schema.name.clone(),
            message,
        };
        let world = (self.seed as u32) ^ ((self.seed >> 32) as u32);
        let parents: Vec<Option<(&Schema, &Row)>> = match schema.parent {
            None => vec![None],
            Some(parent) => self.tables[parent]
                .rows
                .iter()
                .map(|row| Some((&self.pack.tables[parent], row)))
                .collect(),
        };
        let mut shared: Vec<String> = Vec::new();
        for (_, expr) in columns {
            expr.visit(&mut |node| {
                if let Expr::Share(column) = node
                    && !shared.contains(column)
                {
                    shared.push(column.clone());
                }
            });
        }
        let mut rows = Vec::new();
        for parent in parents {
            let id: &[u64] = parent.map_or(&[], |(_, row)| &row.id.0);
            let place = RowPlace {
                world,
                stream: schema.salt ^ COUNT_STREAM,
                id,
                parent,
                index: 0,
                count: 0,
                shares: &BTreeMap::new(),
            };
            let children = evaluate(count, &place)?.floor();
            if !(0.0..=MAX_CHILDREN as f32).contains(&children) {
                return Err(failed(format!(
                    "a count of {children} for parent {id:?}; 0 to {MAX_CHILDREN} are allowed"
                )));
            }
            let children = children as u32;
            let mut shares = BTreeMap::new();
            if let Some((parent_schema, parent_row)) = parent.filter(|_| children > 0) {
                for column in &shared {
                    let at = parent_schema.column(column).expect("checked when loaded");
                    let budget = parent_row.values[at];
                    let parts = share(world, schema.salt ^ salt(column), id, budget, children)
                        .ok_or_else(|| {
                            failed(format!(
                                "parent {id:?} shares {column:?} of {budget}, which is not a \
                                 whole number from 0 to {MAX_SHARED}"
                            ))
                        })?;
                    shares.insert(column.clone(), parts);
                }
            }
            for child in 0..children {
                let mut path = id.to_vec();
                path.push(u64::from(child));
                let mut values = Vec::with_capacity(columns.len());
                for (name, expr) in columns {
                    let place = RowPlace {
                        world,
                        stream: schema.salt ^ salt(name),
                        id: &path,
                        parent,
                        index: child,
                        count: children,
                        shares: &shares,
                    };
                    values.push(evaluate(expr, &place)?);
                }
                rows.push(Row {
                    id: RowId(path),
                    values,
                });
            }
        }
        Ok(Table {
            columns: schema.columns.clone(),
            rows,
        })
    }
}

/// Mixed into a generated table's stream for its count, apart from its columns' streams.
const COUNT_STREAM: u32 = 0x636F_756E;

/// The largest budget a share splits: every whole number up to it is exact in an `f32`.
pub const MAX_SHARED: f32 = 16_777_216.0;

/// A given table's rows, checked against its columns, in the order of their ids.
fn given_rows(schema: &Schema, rows: Vec<GivenRow>) -> Result<Table, StageError> {
    let failed = |message: String| StageError::Table {
        table: schema.name.clone(),
        message,
    };
    let TableKind::Given { columns } = &schema.kind else {
        return Err(failed(
            "it is generated; only a given table takes rows".to_owned(),
        ));
    };
    let mut checked = Vec::with_capacity(rows.len());
    for row in rows {
        if let Some(unknown) = row
            .values
            .keys()
            .find(|name| !columns.iter().any(|(column, _)| column == *name))
        {
            return Err(failed(format!("row {} has no column {unknown:?}", row.id)));
        }
        let mut values = Vec::with_capacity(columns.len());
        for (name, column) in columns {
            let value = row
                .values
                .get(name)
                .ok_or_else(|| failed(format!("row {} gives no {name:?}", row.id)))?;
            values.push(match (column, value) {
                (Column::Number, Value::Number(number)) if number.is_finite() => *number,
                (Column::Names(names), Value::Name(given)) => {
                    let at = names
                        .iter()
                        .position(|known| known == given)
                        .ok_or_else(|| {
                            failed(format!(
                                "row {} gives {name:?} as {given:?}, which is not one of {names:?}",
                                row.id
                            ))
                        })?;
                    at as f32
                }
                _ => {
                    return Err(failed(format!(
                        "row {} gives {name:?} as {value:?}, but the column holds {column:?}",
                        row.id
                    )));
                }
            });
        }
        checked.push(Row {
            id: RowId(vec![row.id]),
            values,
        });
    }
    checked.sort_by(|a, b| a.id.cmp(&b.id));
    if let Some(pair) = checked.windows(2).find(|pair| pair[0].id == pair[1].id) {
        return Err(failed(format!("two rows have id {:?}", pair[0].id.0)));
    }
    Ok(Table {
        columns: schema.columns.clone(),
        rows: checked,
    })
}

/// `budget` split among `children` by hashed weights into whole numbers that add up to it
/// exactly, or `None` if it is not a whole number from 0 to [`MAX_SHARED`].
///
/// Each child's part is the difference of two rounded-down points along the budget, at its
/// weights' running sum before and after it, so the parts add up to the whole by construction.
fn share(world: u32, stream: u32, id: &[u64], budget: f32, children: u32) -> Option<Vec<f32>> {
    if !(0.0..=MAX_SHARED).contains(&budget) || budget.fract() != 0.0 {
        return None;
    }
    let budget = budget as u64;
    let weights: Vec<u64> = (0..children)
        .map(|child| 1 + u64::from(hash(world, stream, id, u64::from(child)) >> 16))
        .collect();
    let total: u64 = weights.iter().sum();
    let mut before = 0;
    let mut parts = Vec::with_capacity(weights.len());
    for weight in weights {
        let after = before + weight;
        parts.push((budget * after / total - budget * before / total) as f32);
        before = after;
    }
    Some(parts)
}

/// A hash of a row's id and `last`, in `stream`.
fn hash(world: u32, stream: u32, id: &[u64], last: u64) -> u32 {
    id.iter()
        .chain([&last])
        .fold(world ^ stream, |hash, &part| {
            pcg3d([hash, part as u32, (part >> 32) as u32])[0]
        })
}

/// A generated row, or a parent row's count of children, where their expressions' leaves read the
/// parent, the row's hash stream and its place among its siblings.
struct RowPlace<'a> {
    world: u32,
    /// The table's and the column's salts, which name the stream `Random` draws from.
    stream: u32,
    id: &'a [u64],
    parent: Option<(&'a Schema, &'a Row)>,
    index: u32,
    count: u32,
    /// Each shared column's parts, by the parent's column name.
    shares: &'a BTreeMap<String, Vec<f32>>,
}

impl Leaves for RowPlace<'_> {
    fn leaf(&self, expr: &Expr) -> Result<f32, StageError> {
        Ok(match expr {
            Expr::Parent(column) => {
                let (schema, row) = self.parent.expect("checked when loaded");
                row.values[schema.column(column).expect("checked when loaded")]
            }
            Expr::Random(low, high) => {
                let unit =
                    (hash(self.world, self.stream, self.id, 0) >> 8) as f32 / (1u32 << 24) as f32;
                low + unit * (high - low)
            }
            Expr::Index => self.index as f32,
            Expr::Count => self.count as f32,
            Expr::Share(column) => self.shares[column][self.index as usize],
            Expr::Noise { .. }
            | Expr::Input(_)
            | Expr::X
            | Expr::Y
            | Expr::Distance(_)
            | Expr::Angle(_)
            | Expr::Is(..)
            | Expr::Match { .. }
            | Expr::Row(..) => unreachable!("a table's expressions are checked when loaded"),
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

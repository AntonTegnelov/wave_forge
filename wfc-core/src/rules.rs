//! Compiled rules: a tile set's adjacencies as word-aligned bitmasks, and its weights as integers.
//!
//! Every solver reads the same table, in the same layout, on CPU and GPU: one
//! [`words_per_cell`](RuleTable::words_per_cell)-word mask per `(axis, tile)` holding the tiles
//! allowed in the neighbour along that axis. Propagation is then a union of masks and an AND, with
//! no lookups into a hash map and no branching per tile pair.

use crate::ModelError;
use wfc_rules::AdjacencyRules;
use wfc_rules::modules::CompiledModules;

/// Tiles per mask word.
pub const TILES_PER_WORD: u32 = 32;
/// Words a mask holds. The GPU kernel needs a constant size for its function-local arrays.
pub const MAX_WORDS: usize = 8;
/// Tiles a rule set may have.
pub const MAX_TILES: u32 = MAX_WORDS as u32 * TILES_PER_WORD;
/// Neighbour directions: `+x, -x, +y, -y, +z, -z`, matching [`wfc_rules::AdjacencyRules`].
pub const AXES: usize = 6;
/// The largest weight after quantisation. Weights are integers so that a choice is the same on
/// every backend: a float sum may be contracted into a fused multiply-add by one driver and not
/// another, which would make the same seed pick differently.
pub const WEIGHT_SCALE: u32 = 65_535;

/// The set of tiles a cell may still hold, one bit per tile.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct TileMask {
    words: [u32; MAX_WORDS],
}

impl TileMask {
    /// A mask with no tile.
    pub const EMPTY: Self = Self {
        words: [0; MAX_WORDS],
    };

    /// Every tile below `num_tiles`.
    #[must_use]
    pub fn all(num_tiles: u32) -> Self {
        let mut mask = Self::EMPTY;
        for tile in 0..num_tiles.min(MAX_TILES) {
            mask.insert(tile);
        }
        mask
    }

    /// Just `tile`.
    #[must_use]
    pub fn single(tile: u32) -> Self {
        let mut mask = Self::EMPTY;
        mask.insert(tile);
        mask
    }

    /// The mask these words describe. Words past [`MAX_WORDS`] are ignored.
    #[must_use]
    pub fn from_words(words: &[u32]) -> Self {
        let mut mask = Self::EMPTY;
        for (slot, word) in mask.words.iter_mut().zip(words) {
            *slot = *word;
        }
        mask
    }

    /// The raw words, low tiles first.
    #[must_use]
    pub const fn words(&self) -> &[u32; MAX_WORDS] {
        &self.words
    }

    /// Whether `tile` is in the mask.
    #[must_use]
    pub fn contains(&self, tile: u32) -> bool {
        tile < MAX_TILES && self.words[(tile / TILES_PER_WORD) as usize] & bit(tile) != 0
    }

    /// Adds `tile`.
    ///
    /// # Panics
    /// If `tile` is at or above [`MAX_TILES`].
    pub fn insert(&mut self, tile: u32) {
        assert!(
            tile < MAX_TILES,
            "tile {tile} is above the {MAX_TILES}-tile limit"
        );
        self.words[(tile / TILES_PER_WORD) as usize] |= bit(tile);
    }

    /// Removes `tile`.
    pub fn remove(&mut self, tile: u32) {
        if tile < MAX_TILES {
            self.words[(tile / TILES_PER_WORD) as usize] &= !bit(tile);
        }
    }

    /// How many tiles the mask holds.
    #[must_use]
    pub fn count(&self) -> u32 {
        self.words.iter().map(|word| word.count_ones()).sum()
    }

    /// Whether no tile is left.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    /// The only tile left, if the cell is decided.
    #[must_use]
    pub fn decided(&self) -> Option<u32> {
        (self.count() == 1).then(|| self.iter().next().expect("one tile"))
    }

    /// The tiles in the mask, lowest first.
    pub fn iter(&self) -> impl Iterator<Item = u32> + use<> {
        let words = self.words;
        (0..MAX_WORDS).flat_map(move |word| {
            let mut bits = words[word];
            std::iter::from_fn(move || {
                (bits != 0).then(|| {
                    let tile = word as u32 * TILES_PER_WORD + bits.trailing_zeros();
                    bits &= bits - 1;
                    tile
                })
            })
        })
    }

    /// The tiles in both masks.
    #[must_use]
    pub fn intersect(mut self, other: Self) -> Self {
        for (word, other) in self.words.iter_mut().zip(other.words) {
            *word &= other;
        }
        self
    }

    /// The tiles in either mask.
    #[must_use]
    pub fn union(mut self, other: Self) -> Self {
        for (word, other) in self.words.iter_mut().zip(other.words) {
            *word |= other;
        }
        self
    }

    /// The tiles of this mask that `other` does not hold.
    #[must_use]
    pub fn subtract(mut self, other: Self) -> Self {
        for (word, other) in self.words.iter_mut().zip(other.words) {
            *word &= !other;
        }
        self
    }
}

fn bit(tile: u32) -> u32 {
    1 << (tile % TILES_PER_WORD)
}

/// Adjacency rules as word-aligned masks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuleTable {
    num_tiles: u32,
    words_per_cell: u32,
    /// `[(axis * num_tiles + tile) * words_per_cell + word]`.
    words: Vec<u32>,
}

impl RuleTable {
    /// Packs `rules`.
    ///
    /// # Errors
    /// If the rule set has more than [`MAX_TILES`] tiles, no tiles, or is not defined over the six
    /// axes the solver knows.
    pub fn pack(rules: &AdjacencyRules) -> Result<Self, ModelError> {
        let num_tiles = rules.num_tiles();
        if num_tiles == 0 || num_tiles as u32 > MAX_TILES {
            return Err(ModelError::TileCount {
                tiles: num_tiles,
                max: MAX_TILES,
            });
        }
        if rules.num_axes() != AXES {
            return Err(ModelError::AxisCount {
                axes: rules.num_axes(),
            });
        }
        let num_tiles = num_tiles as u32;
        let words_per_cell = num_tiles.div_ceil(TILES_PER_WORD);
        let mut words = vec![0u32; AXES * (num_tiles * words_per_cell) as usize];
        for axis in 0..AXES {
            for tile in 0..num_tiles {
                let row = (axis as u32 * num_tiles + tile) * words_per_cell;
                for other in 0..num_tiles {
                    if rules.check(tile as usize, other as usize, axis) {
                        words[(row + other / TILES_PER_WORD) as usize] |= bit(other);
                    }
                }
            }
        }
        Ok(Self {
            num_tiles,
            words_per_cell,
            words,
        })
    }

    /// How many tiles the rule set has.
    #[must_use]
    pub const fn num_tiles(&self) -> u32 {
        self.num_tiles
    }

    /// Words a cell's mask needs.
    #[must_use]
    pub const fn words_per_cell(&self) -> u32 {
        self.words_per_cell
    }

    /// The tiles allowed in the neighbour along `axis` of a cell holding `tile`.
    ///
    /// # Panics
    /// If `axis` is at or above [`AXES`] or `tile` is at or above the rule set's tile count.
    #[must_use]
    pub fn row(&self, axis: usize, tile: u32) -> &[u32] {
        assert!(
            axis < AXES && tile < self.num_tiles,
            "no rule row for axis {axis}, tile {tile}"
        );
        let row = ((axis as u32 * self.num_tiles + tile) * self.words_per_cell) as usize;
        &self.words[row..row + self.words_per_cell as usize]
    }

    /// [`RuleTable::row`] as a mask.
    #[must_use]
    pub fn allowed_next_to(&self, tile: u32, axis: usize) -> TileMask {
        TileMask::from_words(self.row(axis, tile))
    }

    /// The tiles allowed along `axis` of a cell holding any tile of `mask`.
    #[must_use]
    pub fn allowed_by(&self, mask: TileMask, axis: usize) -> TileMask {
        mask.iter().fold(TileMask::EMPTY, |allowed, tile| {
            allowed.union(self.allowed_next_to(tile, axis))
        })
    }

    /// The whole table, in the layout a GPU buffer expects.
    #[must_use]
    pub fn as_words(&self) -> &[u32] {
        &self.words
    }
}

/// The opposite of `axis`: a neighbour along `axis` sees this cell along its opposite.
#[must_use]
pub const fn opposite(axis: usize) -> usize {
    axis ^ 1
}

/// The offset of a neighbour along `axis`, in cells.
#[must_use]
pub const fn axis_offset(axis: usize) -> [i32; 3] {
    match axis {
        0 => [1, 0, 0],
        1 => [-1, 0, 0],
        2 => [0, 1, 0],
        3 => [0, -1, 0],
        4 => [0, 0, 1],
        _ => [0, 0, -1],
    }
}

/// A rule set ready to solve with: adjacencies plus one integer weight per tile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ruleset {
    table: RuleTable,
    weights: Vec<u32>,
}

impl Ruleset {
    /// Packs `rules` and quantises `weights`, keeping their proportions and giving every tile with
    /// a positive weight at least one unit, so no legal tile becomes unreachable.
    ///
    /// # Errors
    /// If the rules cannot be packed, `weights` does not have one entry per tile, a weight is not
    /// finite or is negative, or no weight is positive.
    pub fn new(rules: &AdjacencyRules, weights: &[f32]) -> Result<Self, ModelError> {
        let table = RuleTable::pack(rules)?;
        if weights.len() != table.num_tiles() as usize {
            return Err(ModelError::WeightCount {
                tiles: table.num_tiles() as usize,
                weights: weights.len(),
            });
        }
        let mut largest = 0.0f32;
        for (tile, &weight) in weights.iter().enumerate() {
            if !weight.is_finite() || weight < 0.0 {
                return Err(ModelError::Weight { tile, weight });
            }
            largest = largest.max(weight);
        }
        if largest <= 0.0 {
            return Err(ModelError::NoPositiveWeight);
        }
        let quantised = weights
            .iter()
            .map(|&weight| {
                let scaled = (weight / largest * WEIGHT_SCALE as f32).round() as u32;
                if weight > 0.0 { scaled.max(1) } else { 0 }
            })
            .collect();
        Ok(Self {
            table,
            weights: quantised,
        })
    }

    /// The rule set a connector-built module set compiles to.
    ///
    /// # Errors
    /// As [`Ruleset::new`].
    pub fn from_modules(modules: &CompiledModules) -> Result<Self, ModelError> {
        Self::new(&modules.rules, &modules.tileset.weights)
    }

    /// The packed adjacencies.
    #[must_use]
    pub const fn table(&self) -> &RuleTable {
        &self.table
    }

    /// One weight per tile, quantised to at most [`WEIGHT_SCALE`].
    #[must_use]
    pub fn weights(&self) -> &[u32] {
        &self.weights
    }

    /// How many tiles the rule set has.
    #[must_use]
    pub const fn num_tiles(&self) -> u32 {
        self.table.num_tiles()
    }

    /// Words a cell's mask needs.
    #[must_use]
    pub const fn words_per_cell(&self) -> u32 {
        self.table.words_per_cell()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiles 0 and 1, where 0 may sit east of 1 and nothing else is allowed.
    fn two_tiles() -> AdjacencyRules {
        AdjacencyRules::from_allowed_tuples(2, AXES, [(0, 1, 0)])
    }

    #[test]
    fn a_mask_holds_tiles_across_words() {
        let mut mask = TileMask::EMPTY;

        mask.insert(0);
        mask.insert(31);
        mask.insert(32);
        mask.insert(200);

        assert_eq!(mask.count(), 4);
        assert_eq!(mask.iter().collect::<Vec<_>>(), vec![0, 31, 32, 200]);
        assert!(mask.contains(200));
        assert!(!mask.contains(201));
        assert_eq!(mask.words()[0], 0x8000_0001);
        assert_eq!(mask.words()[1], 1);
    }

    #[test]
    fn a_decided_mask_reports_its_tile() {
        assert_eq!(TileMask::single(7).decided(), Some(7));
        assert_eq!(TileMask::all(3).decided(), None);
        assert_eq!(TileMask::EMPTY.decided(), None);
    }

    #[test]
    fn packing_puts_each_row_at_axis_times_tiles_plus_tile() {
        let table = RuleTable::pack(&two_tiles()).expect("two tiles pack");

        assert_eq!(table.num_tiles(), 2);
        assert_eq!(table.words_per_cell(), 1);
        // The row of (axis 0, tile 1) holds tile 0, because 0 may sit east of 1.
        assert_eq!(table.row(0, 1), &[1]);
        assert_eq!(table.row(0, 0), &[0]);
        assert_eq!(table.as_words().len(), AXES * 2);
        // The row of (axis 0, tile 1) starts at word (0 * num_tiles + 1) * words_per_cell.
        assert_eq!(table.as_words()[1], 1);
    }

    #[test]
    fn a_row_is_the_union_over_a_mask() {
        let table = RuleTable::pack(&two_tiles()).expect("two tiles pack");

        let allowed = table.allowed_by(TileMask::all(2), 0);

        assert_eq!(allowed, TileMask::single(0));
    }

    #[test]
    fn weights_keep_their_proportions_and_stay_reachable() {
        let rules = two_tiles();

        let ruleset = Ruleset::new(&rules, &[10.0, 0.01]).expect("valid weights");

        assert_eq!(ruleset.weights()[0], WEIGHT_SCALE);
        assert_eq!(ruleset.weights()[1], 66);
        assert!(
            ruleset.weights()[1] > 0,
            "a positive weight stays reachable"
        );
    }

    #[test]
    fn rule_sets_that_cannot_be_solved_are_rejected() {
        let rules = two_tiles();

        assert!(matches!(
            Ruleset::new(&rules, &[1.0]),
            Err(ModelError::WeightCount {
                tiles: 2,
                weights: 1
            })
        ));
        assert!(matches!(
            Ruleset::new(&rules, &[1.0, f32::NAN]),
            Err(ModelError::Weight { tile: 1, .. })
        ));
        assert!(matches!(
            Ruleset::new(&rules, &[0.0, 0.0]),
            Err(ModelError::NoPositiveWeight)
        ));
        let four_axes = AdjacencyRules::from_allowed_tuples(2, 4, [(0, 1, 0)]);
        assert!(matches!(
            RuleTable::pack(&four_axes),
            Err(ModelError::AxisCount { axes: 4 })
        ));
    }
}

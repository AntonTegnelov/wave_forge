//! Possibility storage: every cell's mask in one contiguous word array.
//!
//! The layout is `[cell * words_per_cell + word]`, the same on the CPU and inside a GPU buffer, so
//! a region's domains are uploaded and read back without repacking.

use crate::ModelError;
use crate::rules::{TILES_PER_WORD, TileMask};

/// The possible tiles of a run of cells.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Domains {
    cells: u32,
    words_per_cell: u32,
    words: Vec<u32>,
}

impl Domains {
    /// `cells` cells, each holding every tile below `num_tiles`.
    #[must_use]
    pub fn filled(cells: u32, num_tiles: u32) -> Self {
        let words_per_cell = num_tiles.div_ceil(TILES_PER_WORD).max(1);
        let mut domains = Self {
            cells,
            words_per_cell,
            words: vec![0; (cells * words_per_cell) as usize],
        };
        let all = TileMask::all(num_tiles);
        for cell in 0..cells {
            domains.set(cell, all);
        }
        domains
    }

    /// The domains of the masks `masks` yields, in order.
    #[must_use]
    pub fn from_masks(words_per_cell: u32, masks: impl IntoIterator<Item = TileMask>) -> Self {
        let mut words = Vec::new();
        let mut cells = 0;
        for mask in masks {
            words.extend_from_slice(&mask.words()[..words_per_cell as usize]);
            cells += 1;
        }
        Self {
            cells,
            words_per_cell,
            words,
        }
    }

    /// Domains from words already in the storage layout.
    ///
    /// # Errors
    /// If `words` does not hold exactly `cells * words_per_cell` entries.
    pub fn from_words(
        cells: u32,
        words_per_cell: u32,
        words: Vec<u32>,
    ) -> Result<Self, ModelError> {
        let expected = (cells * words_per_cell) as usize;
        if words.len() != expected {
            return Err(ModelError::WordCount {
                expected,
                got: words.len(),
            });
        }
        Ok(Self {
            cells,
            words_per_cell,
            words,
        })
    }

    /// How many cells these domains cover.
    #[must_use]
    pub const fn cells(&self) -> u32 {
        self.cells
    }

    /// Words per cell.
    #[must_use]
    pub const fn words_per_cell(&self) -> u32 {
        self.words_per_cell
    }

    /// One cell's words.
    ///
    /// # Panics
    /// If `cell` is out of range.
    #[must_use]
    pub fn cell(&self, cell: u32) -> &[u32] {
        let start = self.start(cell);
        &self.words[start..start + self.words_per_cell as usize]
    }

    /// One cell's words, mutably.
    ///
    /// # Panics
    /// If `cell` is out of range.
    pub fn cell_mut(&mut self, cell: u32) -> &mut [u32] {
        let start = self.start(cell);
        let end = start + self.words_per_cell as usize;
        &mut self.words[start..end]
    }

    /// One cell's mask.
    ///
    /// # Panics
    /// If `cell` is out of range.
    #[must_use]
    pub fn mask(&self, cell: u32) -> TileMask {
        TileMask::from_words(self.cell(cell))
    }

    /// Replaces one cell's mask.
    ///
    /// # Panics
    /// If `cell` is out of range.
    pub fn set(&mut self, cell: u32, mask: TileMask) {
        let words = self.words_per_cell as usize;
        self.cell_mut(cell).copy_from_slice(&mask.words()[..words]);
    }

    /// How many tiles a cell may still hold.
    ///
    /// # Panics
    /// If `cell` is out of range.
    #[must_use]
    pub fn count(&self, cell: u32) -> u32 {
        self.cell(cell).iter().map(|word| word.count_ones()).sum()
    }

    /// The tile a cell has been decided to, if it has.
    ///
    /// # Panics
    /// If `cell` is out of range.
    #[must_use]
    pub fn decided(&self, cell: u32) -> Option<u32> {
        self.mask(cell).decided()
    }

    /// Whether every cell holds exactly one tile.
    #[must_use]
    pub fn all_decided(&self) -> bool {
        (0..self.cells).all(|cell| self.count(cell) == 1)
    }

    /// Every word, in the layout a GPU buffer expects.
    #[must_use]
    pub fn as_words(&self) -> &[u32] {
        &self.words
    }

    /// Appends `other`'s cells, which must have the same words per cell.
    ///
    /// # Panics
    /// If the two disagree on words per cell.
    pub fn append(&mut self, other: &Self) {
        assert_eq!(
            self.words_per_cell, other.words_per_cell,
            "words per cell differ"
        );
        self.words.extend_from_slice(&other.words);
        self.cells += other.cells;
    }

    /// A copy of the `index`-th run of `cells` cells, as a batch of equal-sized regions holds them.
    ///
    /// # Panics
    /// If the run is out of range.
    #[must_use]
    pub fn chunk(&self, index: u32, cells: u32) -> Self {
        let start = (index * cells * self.words_per_cell) as usize;
        let end = start + (cells * self.words_per_cell) as usize;
        Self {
            cells,
            words_per_cell: self.words_per_cell,
            words: self.words[start..end].to_vec(),
        }
    }

    fn start(&self, cell: u32) -> usize {
        assert!(
            cell < self.cells,
            "cell {cell} of {} is out of range",
            self.cells
        );
        (cell * self.words_per_cell) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filled_domains_hold_every_tile_once_per_cell() {
        let domains = Domains::filled(4, 40);

        assert_eq!(domains.words_per_cell(), 2);
        assert_eq!(domains.as_words().len(), 8);
        assert_eq!(domains.count(3), 40);
        assert!(!domains.all_decided());
    }

    #[test]
    fn a_cell_keeps_its_own_words() {
        let mut domains = Domains::filled(3, 33);

        domains.set(1, TileMask::single(32));

        assert_eq!(domains.decided(1), Some(32));
        assert_eq!(domains.count(0), 33);
        assert_eq!(domains.cell(1), &[0, 1]);
    }

    #[test]
    fn a_batch_splits_into_its_regions() {
        let mut batch = Domains::filled(2, 8);
        batch.set(0, TileMask::single(1));
        batch.set(1, TileMask::single(2));
        let second = Domains::from_masks(1, [TileMask::single(3), TileMask::single(4)]);
        batch.append(&second);

        let regions: Vec<Domains> = (0..2).map(|region| batch.chunk(region, 2)).collect();

        assert_eq!(batch.cells(), 4);
        assert_eq!(regions[0].decided(0), Some(1));
        assert_eq!(regions[1].decided(1), Some(4));
    }

    #[test]
    fn words_that_do_not_fit_the_cells_are_rejected() {
        assert!(matches!(
            Domains::from_words(2, 2, vec![0; 3]),
            Err(ModelError::WordCount {
                expected: 4,
                got: 3
            })
        ));
    }
}

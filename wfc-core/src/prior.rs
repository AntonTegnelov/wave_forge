//! What a world says about its cells before any of them is solved.
//!
//! A prior is how anything outside the solver constrains it: which tiles a layer allows (street
//! level at the bottom, air on top), which tiles are forbidden where the world ends (a path may not
//! lead out of a bounded world), and cells another generation stage has already decided. It is the
//! only door into the solver, and the one other stages drive WFC through
//! (docs/architecture/solver.md, "The prior").

use crate::chunk::WorldCell;
use crate::rules::{AXES, TileMask, axis_offset};
use crate::store::WorldExtent;
use std::collections::HashMap;

/// The starting domains of a world.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prior {
    /// Allowed tiles per layer, indexed from the world's lowest cell. The last entry covers
    /// everything above it.
    layers: Vec<TileMask>,
    /// Tiles removed from a cell whose neighbour along that axis would be outside the world.
    face_bans: [TileMask; AXES],
    /// Cells another layer has narrowed or decided.
    overrides: HashMap<WorldCell, TileMask>,
}

impl Prior {
    /// A world where every tile is allowed everywhere.
    #[must_use]
    pub fn open(num_tiles: u32) -> Self {
        Self {
            layers: vec![TileMask::all(num_tiles)],
            face_bans: [TileMask::EMPTY; AXES],
            overrides: HashMap::new(),
        }
    }

    /// Replaces the per-layer masks, lowest world layer first.
    ///
    /// # Panics
    /// If `layers` is empty.
    #[must_use]
    pub fn with_layers(mut self, layers: Vec<TileMask>) -> Self {
        assert!(!layers.is_empty(), "a world needs at least one layer mask");
        self.layers = layers;
        self
    }

    /// Forbids `tiles` in cells on the world's face along `axis`.
    ///
    /// # Panics
    /// If `axis` is at or above [`AXES`].
    #[must_use]
    pub fn with_face_ban(mut self, axis: usize, tiles: TileMask) -> Self {
        assert!(axis < AXES, "no axis {axis}");
        self.face_bans[axis] = tiles;
        self
    }

    /// Narrows one cell, for example to a single tile another layer has placed.
    #[must_use]
    pub fn with_override(mut self, at: WorldCell, tiles: TileMask) -> Self {
        self.overrides.insert(at, tiles);
        self
    }

    /// Narrows several cells.
    #[must_use]
    pub fn with_overrides(
        mut self,
        cells: impl IntoIterator<Item = (WorldCell, TileMask)>,
    ) -> Self {
        self.overrides.extend(cells);
        self
    }

    /// How many layer masks the prior holds.
    #[must_use]
    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// The tiles `at` may hold, before anything is solved.
    #[must_use]
    pub fn domain(&self, at: WorldCell, extent: &WorldExtent) -> TileMask {
        let mut mask = self.layer(at, extent);
        for axis in 0..AXES {
            let [dx, dy, dz] = axis_offset(axis);
            let neighbour = [at[0] + dx, at[1] + dy, at[2] + dz];
            if !extent.contains_cell(neighbour) {
                mask = mask.subtract(self.face_bans[axis]);
            }
        }
        if let Some(narrowed) = self.overrides.get(&at) {
            mask = mask.intersect(*narrowed);
        }
        mask
    }

    /// The tiles `at` may hold when it lies outside the world: the layer alone.
    ///
    /// A region's halo can reach past the world's edge, and those cells still have to be solved so
    /// the chunk's own border tiles are completable. They are never committed.
    #[must_use]
    pub fn open_domain(&self, at: WorldCell, extent: &WorldExtent) -> TileMask {
        self.layer(at, extent)
    }

    fn layer(&self, at: WorldCell, extent: &WorldExtent) -> TileMask {
        let index = at[2] - extent.lowest_cell_z();
        let index = index.clamp(0, self.layers.len() as i32 - 1) as usize;
        self.layers[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkShape;

    const CHUNK: ChunkShape = ChunkShape { x: 4, y: 4, z: 3 };

    fn extent() -> WorldExtent {
        WorldExtent::new(CHUNK)
            .with_x(0..2)
            .with_y(0..2)
            .with_z(0..1)
    }

    fn prior() -> Prior {
        Prior::open(8)
            .with_layers(vec![
                TileMask::single(0),
                TileMask::all(8),
                TileMask::single(7),
            ])
            .with_face_ban(0, TileMask::single(3))
    }

    #[test]
    fn layers_pin_the_bottom_and_top_of_the_world() {
        let (prior, extent) = (prior(), extent());

        assert_eq!(prior.domain([1, 1, 0], &extent), TileMask::single(0));
        assert_eq!(prior.domain([1, 1, 2], &extent), TileMask::single(7));
        assert_eq!(prior.domain([1, 1, 1], &extent).count(), 8);
    }

    #[test]
    fn a_face_ban_applies_only_where_the_world_ends() {
        let (prior, extent) = (prior(), extent());

        // x = 7 is the last cell of the last chunk along x, so its +x neighbour is outside.
        assert!(!prior.domain([7, 1, 1], &extent).contains(3));
        assert!(prior.domain([6, 1, 1], &extent).contains(3));
    }

    #[test]
    fn an_unbounded_axis_has_no_face() {
        let unbounded = WorldExtent::new(CHUNK).with_z(0..1);

        assert!(prior().domain([7, 1, 1], &unbounded).contains(3));
    }

    #[test]
    fn an_override_narrows_one_cell() {
        let prior = prior().with_override([2, 2, 1], TileMask::single(5));

        assert_eq!(prior.domain([2, 2, 1], &extent()), TileMask::single(5));
        assert_eq!(prior.domain([2, 3, 1], &extent()).count(), 8);
    }

    #[test]
    fn a_cell_outside_the_world_keeps_its_layer_without_bans() {
        let (prior, extent) = (prior(), extent());

        let outside = prior.open_domain([-1, 1, 1], &extent);

        assert_eq!(
            outside.count(),
            8,
            "no face ban applies to a cell that is not in the world"
        );
    }
}

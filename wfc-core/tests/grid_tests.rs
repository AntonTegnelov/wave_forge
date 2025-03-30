use wfc_core::grid::Grid;

#[test]
fn test_grid_new() {
    let grid: Grid<usize> = Grid::new(3, 4, 5);
    assert_eq!(grid.width, 3);
    assert_eq!(grid.height, 4);
    assert_eq!(grid.depth, 5);
    // Check if initialized with default (0 for usize) by accessing a cell
    assert_eq!(*grid.get(0, 0, 0).expect("Cell (0,0,0) should exist"), 0);
    assert_eq!(*grid.get(2, 3, 4).expect("Cell (2,3,4) should exist"), 0);
}

#[test]
fn test_grid_get() {
    let mut grid: Grid<usize> = Grid::new(2, 2, 2);
    // Set a value using the public API
    *grid.get_mut(1, 0, 1).unwrap() = 42;

    // Test getting the set value
    assert_eq!(*grid.get(1, 0, 1).unwrap(), 42);
    // Test getting a default value
    assert_eq!(*grid.get(0, 0, 0).unwrap(), 0);

    // Out of bounds checks
    assert!(grid.get(2, 0, 0).is_none()); // x out of bounds
    assert!(grid.get(0, 2, 0).is_none()); // y out of bounds
    assert!(grid.get(0, 0, 2).is_none()); // z out of bounds
}

#[test]
fn test_grid_get_mut() {
    let mut grid: Grid<usize> = Grid::new(2, 3, 4);

    // Get mutable reference and modify
    if let Some(cell) = grid.get_mut(1, 2, 3) {
        *cell = 99;
    }
    assert_eq!(*grid.get(1, 2, 3).unwrap(), 99);

    // Get mutable reference to default value and modify
    if let Some(cell) = grid.get_mut(0, 0, 0) {
        *cell = 1;
    }
    assert_eq!(*grid.get(0, 0, 0).unwrap(), 1);

    // Out of bounds checks
    assert!(grid.get_mut(2, 0, 0).is_none()); // x out of bounds
    assert!(grid.get_mut(0, 3, 0).is_none()); // y out of bounds
    assert!(grid.get_mut(0, 0, 4).is_none()); // z out of bounds
}

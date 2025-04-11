/// Converts 3D coordinates to a flattened 1D index.
///
/// # Arguments
///
/// * `x` - The x-coordinate
/// * `y` - The y-coordinate
/// * `z` - The z-coordinate
/// * `width` - The width of the grid
/// * `height` - The height of the grid
///
/// # Returns
///
/// The 1D index corresponding to the 3D coordinates
pub fn coords_to_index(x: usize, y: usize, z: usize, width: usize, height: usize) -> u32 {
    // Convert a 3D coordinate (x, y, z) to a flattened 1D index
    // using row-major ordering (x + y * width + z * width * height)
    (x + y * width + z * width * height) as u32
}

/// Converts a flattened 1D index back to 3D coordinates.
///
/// # Arguments
///
/// * `index` - The flattened 1D index
/// * `width` - The width of the grid
/// * `height` - The height of the grid
///
/// # Returns
///
/// The 3D coordinates (x, y, z) corresponding to the 1D index
pub fn index_to_coords(index: usize, width: usize, height: usize) -> (usize, usize, usize) {
    // Calculate 3D coordinates from a flattened index
    let area = width * height;
    let z = index / area;
    let remainder = index % area;
    let y = remainder / width;
    let x = remainder % width;
    (x, y, z)
}

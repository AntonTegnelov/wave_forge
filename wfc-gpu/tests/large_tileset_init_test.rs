use wfc_gpu::test_utils::test_large_tileset_init;

#[test]
fn test_large_tileset_initialization() {
    // Test with 200 tiles, well above the previous 128 limit
    let result = test_large_tileset_init(16, 16, 200);

    // Assert that initialization succeeds
    assert!(
        result.is_ok(),
        "Failed to initialize GPU accelerator with 200 tiles: {:?}",
        result.err()
    );

    // Access the accelerator to verify it was created correctly
    let accelerator = result.unwrap();

    // Verify that the tile count matches what we requested
    assert_eq!(
        accelerator.num_tiles(),
        200,
        "Accelerator's num_tiles doesn't match the expected value"
    );

    // Cleanup happens automatically when accelerator goes out of scope
}

use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::fs;
use std::process::Command; // Run programs
use tempfile::tempdir; // Create temporary directories for testing

// Helper function to create a dummy rule file matching RonRuleFile format
fn create_dummy_rules(dir: &tempfile::TempDir, filename: &str) -> std::path::PathBuf {
    let file_path = dir.path().join(filename);
    // Define Tile0 adjacent to itself on all axes
    let rules_content = r#"
        (
            tiles: [
                ( name: "Tile0", weight: 1.0 ),
            ],
            adjacency: [
                ("Tile0", "Tile0", "+x"),
                ("Tile0", "Tile0", "-x"),
                ("Tile0", "Tile0", "+y"),
                ("Tile0", "Tile0", "-y"),
                ("Tile0", "Tile0", "+z"),
                ("Tile0", "Tile0", "-z"),
            ],
        )
        "#;
    fs::write(&file_path, rules_content).expect("Failed to write dummy rule file");
    file_path
}

#[test]
fn test_basic_cpu_run() -> Result<(), Box<dyn std::error::Error>> {
    let tmp_dir = tempdir()?;
    let rule_file = create_dummy_rules(&tmp_dir, "cpu_rules.ron");
    let output_file = tmp_dir.path().join("cpu_output.txt");

    let mut cmd = Command::cargo_bin("wave-forge")?;
    cmd.env("RUST_LOG", "info"); // Set log level for test run

    cmd.arg("--rule-file")
        .arg(rule_file)
        .arg("--width")
        .arg("3")
        .arg("--height")
        .arg("3")
        .arg("--depth")
        .arg("3")
        .arg("--output-path")
        .arg(&output_file)
        .arg("--cpu-only");

    cmd.assert()
        .success()
        .stderr(predicate::str::contains("CPU WFC completed successfully."));

    // Check if output file was created (basic check)
    assert!(output_file.exists(), "Output file was not created");
    // TODO: Check output file content?

    Ok(())
}

// Only run GPU test if feature is enabled
#[cfg(feature = "gpu")]
#[test]
fn test_basic_gpu_run() -> Result<(), Box<dyn std::error::Error>> {
    let tmp_dir = tempdir()?;
    let rule_file = create_dummy_rules(&tmp_dir, "gpu_rules.ron");
    let output_file = tmp_dir.path().join("gpu_output.txt");

    let mut cmd = Command::cargo_bin("wave-forge")?;
    cmd.env("RUST_LOG", "info"); // Set log level for test run

    // Run without --cpu-only to test GPU path
    cmd.arg("--rule-file")
        .arg(rule_file)
        .arg("--width")
        .arg("3")
        .arg("--height")
        .arg("3")
        .arg("--depth")
        .arg("3")
        .arg("--output-path")
        .arg(&output_file);

    // Now expect success or a normal WFC error, not the specific skipped message
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("GPU WFC completed successfully."));

    // Check if output file was created
    assert!(output_file.exists(), "Output file was not created");

    Ok(())
}

// Main entry point that forwards to the wave-forge-app binary
fn main() {
    // Exit with the same code as the app
    std::process::exit(match wave_forge_app::main() {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    });
}

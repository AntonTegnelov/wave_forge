fn main() -> anyhow::Result<()> {
    // Use the wave_forge_app binary directly
    // The tokio runtime is already set up inside wave_forge_app's main function
    wave_forge_app::main()
}

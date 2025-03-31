// wave-forge-app/src/main.rs

pub mod config;
use clap::Parser;
use config::AppConfig;

fn main() {
    // Parse command-line arguments
    let config = AppConfig::parse();

    // TODO: Initialize logging (e.g., env_logger)

    println!("Wave Forge App");
    println!("Config: {:?}", config);

    // TODO: Implement main application logic using the parsed config
    //  - Load rules
    //  - Initialize grid
    //  - Select CPU/GPU components based on config.cpu_only and feature flags
    //  - Run WFC
    //  - Save output
}

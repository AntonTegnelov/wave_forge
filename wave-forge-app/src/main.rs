// wave-forge-app/src/main.rs

mod config;

use config::AppConfig;

fn main() {
    let config = AppConfig::from_args();
    println!("Wave Forge App - Loaded Config: {:?}", config);
    // TODO: Implement main application logic based on config
}

#[cfg(test)]
mod tests {
    use std::env;
    use wfc_gpu::shader::shaders::{ShaderManager, ShaderType};

    #[test]
    fn test_shader_loading_behavior() {
        // Set up logging
        env::set_var("RUST_LOG", "debug");
        env_logger::builder()
            .format_timestamp(None)
            .format_target(false)
            .init();

        // Create shader manager
        let mut manager = ShaderManager::new().expect("Failed to create ShaderManager");

        // Test different feature combinations
        let feature_combinations = vec![
            (vec![], "base variant"),
            (vec!["atomics"], "atomics variant"),
            (vec!["subgrid"], "subgrid variant"),
            (vec!["atomics", "subgrid"], "combined variant"),
        ];

        for (features, desc) in feature_combinations {
            println!("\n=== Testing {} ===", desc);

            // Try loading entropy shader
            let entropy_result = manager.load_shader_variant(ShaderType::Entropy, &features);
            assert!(
                entropy_result.is_ok(),
                "Failed to load entropy shader for {}",
                desc
            );

            // Try loading propagation shader
            let prop_result = manager.load_shader_variant(ShaderType::Propagation, &features);
            assert!(
                prop_result.is_ok(),
                "Failed to load propagation shader for {}",
                desc
            );
        }
    }
}

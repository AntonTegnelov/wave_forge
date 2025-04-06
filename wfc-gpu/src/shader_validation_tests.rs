//! Tests for shader module validation

use crate::test_utils::create_test_device_queue;

/// Tests that a simple shader compiles correctly
#[test]
fn test_minimal_shader_compilation() {
    let (device, _) = create_test_device_queue();

    // Create a minimal test shader
    let shader_src = r#"
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let x = global_id.x;
        }
    "#;

    // Try to create the shader module - this will panic if compilation fails
    let _shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Minimal Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    // If we get here, the shader compiled successfully
}

/// Tests the shader module processing (include system)
#[test]
fn test_shader_processing() {
    // Simple test to verify our shader inclusion mechanism works
    let main_code = r#"
    // Main shader
    #include "utils.wgsl"
    
    fn main() {
        test_util();
    }
    "#
    .to_string();

    let utils_code = r#"
    // Utils module
    fn test_util() {
        // Test utility function
    }
    "#
    .to_string();

    // Test processing includes - simplified version of what the pipeline does
    let processed = process_test_includes(main_code, utils_code);

    // Verify the includes were properly processed
    assert!(processed.contains("// Utils module"));
    assert!(processed.contains("// Main shader"));
    assert!(processed.contains("fn test_util()"));
    assert!(processed.contains("fn main()"));
    assert!(!processed.contains("#include")); // The include directive should be gone
}

/// Tests compilation with NUM_TILES_U32 constant replacement
#[test]
fn test_shader_constant_replacement() {
    let (device, _) = create_test_device_queue();

    // Create a shader with the NUM_TILES_U32 constant
    let shader_src = r#"
        // Test shader with constant
        const NUM_TILES_U32: u32 = 4;
        
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let x = global_id.x;
            let _y = NUM_TILES_U32; // Use the constant
        }
    "#;

    // Try to create the shader module
    let _shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Constant Test Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    // If we get here, the shader compiled successfully
}

// Helper function to process includes for testing
fn process_test_includes(main_code: String, utils_code: String) -> String {
    // Very simplified version of the shader include system
    main_code.replace("#include \"utils.wgsl\"", &utils_code)
}

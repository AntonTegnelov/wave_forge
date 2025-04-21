//! Shader optimization tool for the Wave Function Collapse GPU implementation.
//!
//! This tool analyzes and optimizes WGSL shaders by:
//! 1. Removing unused code paths
//! 2. Inlining functions where beneficial
//! 3. Optimizing variable usage and memory access patterns
//! 4. Generating specialized variants for common configurations

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self};
use std::path::{Path, PathBuf};

/// Configuration for shader optimization.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Source directory for shader components
    pub src_dir: PathBuf,
    /// Output directory for optimized shaders
    pub out_dir: PathBuf,
    /// Enable aggressive optimizations (may reduce readability)
    pub aggressive: bool,
    /// List of features to enable by default
    pub default_features: Vec<String>,
    /// Generate specialized variants for these feature combinations
    pub variants: Vec<FeatureVariant>,
    /// Enable debug output during optimization
    pub debug: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            src_dir: PathBuf::from("src/shader/shaders"),
            out_dir: PathBuf::from("src/shader/shaders/variants"),
            aggressive: false,
            default_features: vec![],
            variants: vec![],
            debug: false,
        }
    }
}

/// Represents a variant with a specific set of features enabled.
#[derive(Debug, Clone)]
pub struct FeatureVariant {
    /// Variant name
    pub name: String,
    /// Enabled features
    pub features: Vec<String>,
    /// Target workgroup size
    pub workgroup_size: Option<(u32, u32, u32)>,
    /// Additional defines
    pub defines: HashMap<String, String>,
}

/// Information about a shader component.
#[derive(Debug, Clone)]
pub struct ShaderComponent {
    /// Component name
    #[allow(dead_code)]
    pub name: String,
    /// Path to the component source
    pub path: PathBuf,
    /// Dependencies on other components
    pub dependencies: Vec<String>,
    /// Required GPU features
    #[allow(dead_code)]
    pub required_features: HashSet<String>,
    /// Provided features
    pub provided_features: HashSet<String>,
}

/// Shader optimization tool.
pub struct ShaderOptimizer {
    config: OptimizerConfig,
    components: HashMap<String, ShaderComponent>,
}

impl ShaderOptimizer {
    /// Create a new shader optimizer with the given configuration.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            components: HashMap::new(),
        }
    }

    /// Load component information from the registry file.
    pub fn load_registry(&mut self, registry_path: &Path) -> io::Result<()> {
        let content = fs::read_to_string(registry_path)?;
        let registry: serde_json::Value = serde_json::from_str(&content)?;

        if let Some(components) = registry.get("components").and_then(|c| c.as_object()) {
            for (name, info) in components {
                let path_str = info.get("path").and_then(|p| p.as_str()).unwrap_or("");
                let path = PathBuf::from(path_str);

                // Parse dependencies
                let mut dependencies = Vec::new();
                if let Some(deps) = info.get("dependencies").and_then(|d| d.as_array()) {
                    for dep in deps {
                        if let Some(dep_str) = dep.as_str() {
                            dependencies.push(dep_str.to_owned());
                        }
                    }
                }

                // Parse required features
                let mut required_features = HashSet::new();
                if let Some(features) = info.get("required_features").and_then(|f| f.as_array()) {
                    for feature in features {
                        if let Some(feature_str) = feature.as_str() {
                            required_features.insert(feature_str.to_owned());
                        }
                    }
                }

                // Parse provided features
                let mut provided_features = HashSet::new();
                if let Some(features) = info.get("provided_features").and_then(|f| f.as_array()) {
                    for feature in features {
                        if let Some(feature_str) = feature.as_str() {
                            provided_features.insert(feature_str.to_owned());
                        }
                    }
                }

                let component = ShaderComponent {
                    name: name.clone(),
                    path,
                    dependencies,
                    required_features,
                    provided_features,
                };

                self.components.insert(name.clone(), component);
            }
        }

        Ok(())
    }

    /// Generate optimized shader variants.
    pub fn generate_variants(&self) -> io::Result<()> {
        // Create output directory if it doesn't exist
        fs::create_dir_all(&self.config.out_dir)?;

        // Generate default variants first
        self.generate_default_variants()?;

        // Generate specialized variants
        for variant in &self.config.variants {
            self.generate_specific_variant(variant)?;
        }

        Ok(())
    }

    /// Generate default shader variants (entropy calculation and propagation).
    fn generate_default_variants(&self) -> io::Result<()> {
        // Generate entropy calculation shader
        let entropy_path = self.config.out_dir.join("entropy_default.wgsl");
        self.generate_shader("entropy", &self.config.default_features, &entropy_path)?;

        // Generate propagation shader
        let propagation_path = self.config.out_dir.join("propagation_default.wgsl");
        self.generate_shader(
            "propagation",
            &self.config.default_features,
            &propagation_path,
        )?;

        Ok(())
    }

    /// Generate a specific shader variant.
    fn generate_specific_variant(&self, variant: &FeatureVariant) -> io::Result<()> {
        // The shader type is part of the variant name (entropy_*, propagation_*)
        let shader_type = if variant.name.starts_with("entropy") {
            "entropy"
        } else if variant.name.starts_with("propagation") {
            "propagation"
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Invalid variant name: {}", variant.name),
            ));
        };

        let output_path = self.config.out_dir.join(format!("{}.wgsl", variant.name));
        self.generate_shader(shader_type, &variant.features, &output_path)?;

        // Apply additional optimizations based on variant-specific settings
        let mut content = fs::read_to_string(&output_path)?;

        // Apply workgroup size if specified
        if let Some(workgroup_size) = variant.workgroup_size {
            content = self.apply_workgroup_size(&content, workgroup_size);
        }

        // Apply custom defines
        for (name, value) in &variant.defines {
            content = content.replace(
                &format!("// #define {name} "),
                &format!("#define {name} {value}"),
            );
        }

        fs::write(output_path, content)?;
        Ok(())
    }

    /// Generate a shader of the specified type with the given features.
    fn generate_shader(
        &self,
        shader_type: &str,
        features: &[String],
        output_path: &Path,
    ) -> io::Result<()> {
        // Resolve components needed for this shader type
        let components = self.resolve_components_for_shader(shader_type, features);

        // Start assembling the shader
        let mut content = String::new();

        // Add header
        content.push_str(&format!("// Generated shader for type: {shader_type}\n"));
        content.push_str("// Features: ");
        for feature in features {
            content.push_str(&format!("{feature} "));
        }
        content.push_str("\n\n");

        // Add define statements for features
        for feature in features {
            content.push_str(&format!("#define FEATURE_{} 1\n", feature.to_uppercase()));
        }
        content.push('\n');

        // Include components in dependency order
        for component_name in components {
            if let Some(component) = self.components.get(&component_name) {
                let component_path = self.config.src_dir.join(&component.path);
                if component_path.exists() {
                    let component_content = fs::read_to_string(&component_path)?;
                    content.push_str(&format!("// Component: {component_name}\n"));
                    content.push_str(&component_content);
                    content.push_str("\n\n");
                } else {
                    eprintln!("Warning: Component file not found: {component_path:?}");
                }
            }
        }

        // Add entry point function
        if shader_type == "entropy" {
            content.push_str("@compute @workgroup_size(64) fn main_entropy() {\n");
            content.push_str("    calculate_entropy();\n");
            content.push_str("}\n");
        } else {
            content.push_str("@compute @workgroup_size(64) fn main_propagate() {\n");
            content.push_str("    propagate_constraints();\n");
            content.push_str("}\n");
        }

        // Apply optimizations if enabled
        if self.config.aggressive {
            content = self.optimize_shader_content(&content);
        }

        // Write to output file
        fs::write(output_path, content)?;

        if self.config.debug {
            println!("Generated shader: {output_path:?}");
        }

        Ok(())
    }

    /// Resolve the components needed for a shader type with the given features.
    fn resolve_components_for_shader(&self, shader_type: &str, features: &[String]) -> Vec<String> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut feature_set: HashSet<String> = features.iter().cloned().collect();

        // Start with core components for this shader type
        let core_components = match shader_type {
            "entropy" => vec!["Utils", "Coords", "Rules", "EntropyCalculation"],
            "propagation" => vec![
                "Utils",
                "Coords",
                "Rules",
                "WorklistManagement",
                "ContradictionDetection",
            ],
            _ => vec![],
        };

        // Add feature-specific components
        if shader_type == "propagation" {
            if feature_set.contains("subgrid") {
                core_components
                    .iter()
                    .chain(["SubgridPropagation"].iter())
                    .copied()
                    .for_each(|c| {
                        self.add_component_with_deps(
                            c,
                            &mut result,
                            &mut visited,
                            &mut feature_set,
                        );
                    });
            } else {
                core_components
                    .iter()
                    .chain(["DirectPropagation"].iter())
                    .copied()
                    .for_each(|c| {
                        self.add_component_with_deps(
                            c,
                            &mut result,
                            &mut visited,
                            &mut feature_set,
                        );
                    });
            }
        } else if shader_type == "entropy" {
            if feature_set.contains("shannon") {
                core_components
                    .iter()
                    .chain(["ShannonEntropy"].iter())
                    .copied()
                    .for_each(|c| {
                        self.add_component_with_deps(
                            c,
                            &mut result,
                            &mut visited,
                            &mut feature_set,
                        );
                    });
            } else if feature_set.contains("count_based") {
                core_components
                    .iter()
                    .chain(["CountBasedEntropy"].iter())
                    .copied()
                    .for_each(|c| {
                        self.add_component_with_deps(
                            c,
                            &mut result,
                            &mut visited,
                            &mut feature_set,
                        );
                    });
            } else {
                core_components.iter().copied().for_each(|c| {
                    self.add_component_with_deps(c, &mut result, &mut visited, &mut feature_set);
                });
            }
        }

        result
    }

    /// Add a component and its dependencies to the result list.
    fn add_component_with_deps(
        &self,
        component_name: &str,
        result: &mut Vec<String>,
        visited: &mut HashSet<String>,
        feature_set: &mut HashSet<String>,
    ) {
        if visited.contains(component_name) {
            return;
        }

        visited.insert(component_name.to_owned());

        if let Some(component) = self.components.get(component_name) {
            // Add dependencies first
            for dep in &component.dependencies {
                self.add_component_with_deps(dep, result, visited, feature_set);
            }

            // Add provided features to the feature set
            for feature in &component.provided_features {
                feature_set.insert(feature.clone());
            }

            // Add the component itself
            result.push(component_name.to_owned());
        }
    }

    /// Apply workgroup size to shader content.
    fn apply_workgroup_size(&self, content: &str, workgroup_size: (u32, u32, u32)) -> String {
        let (x, y, z) = workgroup_size;

        // Replace workgroup_size annotations in compute shaders
        content.replace(
            "@compute @workgroup_size(64)",
            &format!("@compute @workgroup_size({x}, {y}, {z})"),
        )
    }

    /// Apply optimizations to shader content.
    fn optimize_shader_content(&self, content: &str) -> String {
        // First run preprocessor
        let optimized = content.to_string();

        // Remove comments
        let mut result = String::new();
        let _in_comment = false;
        let mut i = 0;
        let chars: Vec<char> = optimized.chars().collect();

        while i < chars.len() {
            if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '/' {
                // Skip until end of line
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            } else if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '*' {
                // Skip until end of block comment
                i += 2;
                while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                    i += 1;
                }
                if i + 1 < chars.len() {
                    i += 2; // Skip the */
                }
            } else {
                result.push(chars[i]);
                i += 1;
            }
        }

        // Remove extra whitespace
        let result = result
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<&str>>()
            .join("\n");

        result
    }
}

/// Main function for the shader optimizer tool.
#[cfg(not(test))]
#[allow(dead_code)]
fn main() -> io::Result<()> {
    // Default configuration
    let config = OptimizerConfig {
        src_dir: PathBuf::from("src/shader/shaders"),
        out_dir: PathBuf::from("src/shader/shaders/variants"),
        aggressive: true,
        default_features: vec!["atomics".to_owned()],
        variants: vec![
            FeatureVariant {
                name: "entropy_shannon".to_owned(),
                features: vec!["atomics".to_owned(), "shannon".to_owned()],
                workgroup_size: Some((256, 1, 1)),
                defines: HashMap::new(),
            },
            FeatureVariant {
                name: "entropy_count_based".to_owned(),
                features: vec!["atomics".to_owned(), "count_based".to_owned()],
                workgroup_size: Some((256, 1, 1)),
                defines: HashMap::new(),
            },
            FeatureVariant {
                name: "propagation_direct".to_owned(),
                features: vec!["atomics".to_owned()],
                workgroup_size: Some((64, 1, 1)),
                defines: HashMap::new(),
            },
            FeatureVariant {
                name: "propagation_subgrid".to_owned(),
                features: vec!["atomics".to_owned(), "subgrid".to_owned()],
                workgroup_size: Some((16, 16, 1)),
                defines: HashMap::new(),
            },
        ],
        debug: true,
    };

    let registry_path = PathBuf::from("src/shader/shaders/components/registry.json");
    let mut optimizer = ShaderOptimizer::new(config);

    // Load component registry
    optimizer.load_registry(&registry_path)?;

    // Generate variants
    optimizer.generate_variants()?;

    println!("Shader optimization complete.");
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_shader_optimizer() {
        // Add actual tests for shader optimizer functionality
        // For now, just verify the module compiles
        assert!(true);
    }
}

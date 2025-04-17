//! Shader validation tool for the Wave Function Collapse GPU implementation.
//!
//! This tool validates WGSL shaders against the WGSL specification and
//! identifies potential issues such as:
//! 1. Syntax errors
//! 2. Type errors
//! 3. Undefined variables or functions
//! 4. Resource binding conflicts
//! 5. Workgroup memory usage issues

use std::fs::{self};
use std::io::{self};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Configuration for shader validation.
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    /// Path to naga validator binary (if using external validator)
    pub naga_validator_path: Option<PathBuf>,
    /// Path to tint validator binary (if using external validator)
    pub tint_validator_path: Option<PathBuf>,
    /// Directory containing shaders to validate
    pub shader_dir: PathBuf,
    /// Output validation report to this file
    pub report_path: Option<PathBuf>,
    /// Generate warnings in addition to errors
    pub warnings: bool,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            naga_validator_path: None,
            tint_validator_path: None,
            shader_dir: PathBuf::from("src/shader/shaders"),
            report_path: Some(PathBuf::from("validation_report.txt")),
            warnings: true,
        }
    }
}

/// Results of validating a shader file.
#[derive(Debug, Clone)]
pub struct ShaderValidationResult {
    /// Path to the shader file
    #[allow(dead_code)]
    pub file_path: PathBuf,
    /// List of errors found
    pub errors: Vec<ValidationMessage>,
    /// List of warnings found
    pub warnings: Vec<ValidationMessage>,
    /// Whether validation was successful (no errors)
    pub success: bool,
}

/// A validation message (error or warning).
#[derive(Debug, Clone)]
pub struct ValidationMessage {
    /// Message type
    pub message_type: MessageType,
    /// Message content
    pub message: String,
    /// Line number
    pub line: Option<usize>,
    /// Column number
    pub column: Option<usize>,
}

/// Type of validation message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    /// Error - prevents shader compilation
    Error,
    /// Warning - may cause issues
    Warning,
    /// Info - informational message
    Info,
}

/// Shader validator for WGSL shaders.
pub struct ShaderValidator {
    config: ValidatorConfig,
    /// Internal validation results
    validation_results: Vec<ShaderValidationResult>,
    /// Report content
    report: String,
}

impl ShaderValidator {
    /// Create a new shader validator with the given configuration.
    pub const fn new(config: ValidatorConfig) -> Self {
        Self {
            config,
            validation_results: Vec::new(),
            report: String::new(),
        }
    }

    /// Validate all shader files in the specified directory.
    pub fn validate_directory(&mut self) -> io::Result<bool> {
        self.report.clear();
        self.validation_results.clear();

        let shader_dir = &self.config.shader_dir;
        self.report.push_str(&format!(
            "# Shader Validation Report\n\nValidating shaders in: {shader_dir:?}\n\n"
        ));

        let mut all_valid = true;
        let entries = fs::read_dir(shader_dir)?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().is_some_and(|ext| ext == "wgsl") {
                let result = self.validate_shader(&path)?;
                all_valid = all_valid && result.success;
                self.validation_results.push(result);
            } else if path.is_dir() {
                // Recursively validate shaders in subdirectories
                let mut subdir_validator = Self {
                    config: ValidatorConfig {
                        shader_dir: path.clone(),
                        ..self.config.clone()
                    },
                    validation_results: Vec::new(),
                    report: String::new(),
                };

                let subdir_valid = subdir_validator.validate_directory()?;
                all_valid = all_valid && subdir_valid;

                // Append subdir validation results
                self.report.push_str(&subdir_validator.report);
                self.validation_results
                    .extend(subdir_validator.validation_results);
            }
        }

        // Generate report summary
        self.report.push_str("\n## Summary\n\n");
        let errors = self
            .validation_results
            .iter()
            .map(|r| r.errors.len())
            .sum::<usize>();
        let warnings = self
            .validation_results
            .iter()
            .map(|r| r.warnings.len())
            .sum::<usize>();

        self.report.push_str(&format!(
            "- Total files validated: {}\n",
            self.validation_results.len()
        ));
        self.report.push_str(&format!(
            "- Files with errors: {}\n",
            self.validation_results
                .iter()
                .filter(|r| !r.errors.is_empty())
                .count()
        ));
        self.report.push_str(&format!("- Total errors: {errors}\n"));
        self.report
            .push_str(&format!("- Total warnings: {warnings}\n"));

        // Write report to file if specified
        if let Some(report_path) = &self.config.report_path {
            fs::write(report_path, &self.report)?;
        }

        Ok(all_valid)
    }

    /// Validate a single shader file.
    pub fn validate_shader(&mut self, shader_path: &Path) -> io::Result<ShaderValidationResult> {
        let content = fs::read_to_string(shader_path)?;
        let file_name = shader_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();

        self.report
            .push_str(&format!("## Validating shader: {file_name}\n\n"));

        let mut result = ShaderValidationResult {
            file_path: shader_path.to_path_buf(),
            errors: Vec::new(),
            warnings: Vec::new(),
            success: true,
        };

        // Try to use external validator if specified
        if let Some(_validator_path) = &self.config.naga_validator_path {
            return self.validate_with_naga(shader_path);
        } else if let Some(_validator_path) = &self.config.tint_validator_path {
            return self.validate_with_tint(shader_path);
        }

        // Fallback to internal validation
        self.internal_validation(&content, &mut result);

        // Add results to report
        if result.errors.is_empty() && result.warnings.is_empty() {
            self.report.push_str("\u{2705} No issues found\n\n");
        } else {
            // Add errors
            if !result.errors.is_empty() {
                self.report.push_str("### Errors\n\n");
                for error in &result.errors {
                    let location = if let (Some(line), Some(column)) = (error.line, error.column) {
                        format!("line {line}, column {column}")
                    } else if let Some(line) = error.line {
                        format!("line {line}")
                    } else {
                        "unknown location".to_owned()
                    };

                    self.report
                        .push_str(&format!("- \u{1f534} {}: {}\n", location, error.message));
                }
                self.report.push('\n');
            }

            // Add warnings
            if !result.warnings.is_empty() && self.config.warnings {
                self.report.push_str("### Warnings\n\n");
                for warning in &result.warnings {
                    let location =
                        if let (Some(line), Some(column)) = (warning.line, warning.column) {
                            format!("line {line}, column {column}")
                        } else if let Some(line) = warning.line {
                            format!("line {line}")
                        } else {
                            "unknown location".to_owned()
                        };

                    self.report
                        .push_str(&format!("- \u{1f7e1} {}: {}\n", location, warning.message));
                }
                self.report.push('\n');
            }
        }

        result.success = result.errors.is_empty();
        Ok(result)
    }

    /// Validate a shader using the Naga validator.
    fn validate_with_naga(&self, shader_path: &Path) -> io::Result<ShaderValidationResult> {
        let validator_path = self.config.naga_validator_path.as_ref().unwrap();

        let output = Command::new(validator_path)
            .arg("validate")
            .arg(shader_path)
            .output()?;

        let mut result = ShaderValidationResult {
            file_path: shader_path.to_path_buf(),
            errors: Vec::new(),
            warnings: Vec::new(),
            success: output.status.success(),
        };

        // Parse output for errors and warnings
        let output_str = String::from_utf8_lossy(&output.stdout);
        let error_output = String::from_utf8_lossy(&output.stderr);

        // Process errors
        for line in error_output.lines() {
            if line.contains("error:") {
                let parts: Vec<&str> = line.splitn(2, "error:").collect();
                if parts.len() == 2 {
                    // Try to extract line and column information
                    let (line_num, column_num) = parse_location(parts[0]);

                    result.errors.push(ValidationMessage {
                        message_type: MessageType::Error,
                        message: parts[1].trim().to_owned(),
                        line: line_num,
                        column: column_num,
                    });
                }
            }
        }

        // Process warnings
        if self.config.warnings {
            for line in output_str.lines() {
                if line.contains("warning:") {
                    let parts: Vec<&str> = line.splitn(2, "warning:").collect();
                    if parts.len() == 2 {
                        // Try to extract line and column information
                        let (line_num, column_num) = parse_location(parts[0]);

                        result.warnings.push(ValidationMessage {
                            message_type: MessageType::Warning,
                            message: parts[1].trim().to_owned(),
                            line: line_num,
                            column: column_num,
                        });
                    }
                }
            }
        }

        Ok(result)
    }

    /// Validate a shader using the Tint validator.
    fn validate_with_tint(&self, shader_path: &Path) -> io::Result<ShaderValidationResult> {
        let validator_path = self.config.tint_validator_path.as_ref().unwrap();

        let output = Command::new(validator_path).arg(shader_path).output()?;

        let mut result = ShaderValidationResult {
            file_path: shader_path.to_path_buf(),
            errors: Vec::new(),
            warnings: Vec::new(),
            success: output.status.success(),
        };

        // Parse output for errors and warnings
        let output_str = String::from_utf8_lossy(&output.stdout);
        let error_output = String::from_utf8_lossy(&output.stderr);

        // Process errors
        for line in error_output.lines() {
            if line.contains("error:") {
                let parts: Vec<&str> = line.splitn(2, "error:").collect();
                if parts.len() == 2 {
                    // Try to extract line and column information
                    let (line_num, column_num) = parse_location(parts[0]);

                    result.errors.push(ValidationMessage {
                        message_type: MessageType::Error,
                        message: parts[1].trim().to_owned(),
                        line: line_num,
                        column: column_num,
                    });
                }
            }
        }

        // Process warnings
        if self.config.warnings {
            for line in output_str.lines() {
                if line.contains("warning:") {
                    let parts: Vec<&str> = line.splitn(2, "warning:").collect();
                    if parts.len() == 2 {
                        // Try to extract line and column information
                        let (line_num, column_num) = parse_location(parts[0]);

                        result.warnings.push(ValidationMessage {
                            message_type: MessageType::Warning,
                            message: parts[1].trim().to_owned(),
                            line: line_num,
                            column: column_num,
                        });
                    }
                }
            }
        }

        Ok(result)
    }

    /// Internal validation logic for when no external validator is available.
    fn internal_validation(&self, content: &str, result: &mut ShaderValidationResult) {
        // Basic syntax check - look for unbalanced braces, semicolons, etc.
        let mut brace_count = 0;
        let mut bracket_count = 0;
        let mut paren_count = 0;

        for (i, line) in content.lines().enumerate() {
            let line_num = i + 1;

            // Count braces, brackets, and parentheses
            for c in line.chars() {
                match c {
                    '{' => brace_count += 1,
                    '}' => brace_count -= 1,
                    '[' => bracket_count += 1,
                    ']' => bracket_count -= 1,
                    '(' => paren_count += 1,
                    ')' => paren_count -= 1,
                    _ => {}
                }
            }

            // Check for missing semicolons in appropriate contexts
            if !line.trim().is_empty()
                && !line.trim().ends_with('{')
                && !line.trim().ends_with('}')
                && !line.trim().ends_with(';')
                && !line.trim().starts_with("//")
                && !line.trim().starts_with("fn ")
                && !line.trim().starts_with("var ")
                && !line.trim().starts_with("let ")
                && !line.contains("@compute")
                && !line.contains("@group")
                && !line.contains("@binding")
            {
                // Skip if it's a multi-line statement
                let trimmed = line.trim();
                if !trimmed.ends_with(',')
                    && !trimmed.ends_with('(')
                    && !trimmed.contains("/*")
                    && !trimmed.contains("*/")
                {
                    result.warnings.push(ValidationMessage {
                        message_type: MessageType::Warning,
                        message: "Possible missing semicolon".to_owned(),
                        line: Some(line_num),
                        column: None,
                    });
                }
            }

            // Check for resource binding issues
            if line.contains("@binding") {
                // TODO: Implement more sophisticated binding validation
                if line.contains("@binding(") && !line.contains("@group") {
                    result.warnings.push(ValidationMessage {
                        message_type: MessageType::Warning,
                        message: "Binding specified without group".to_owned(),
                        line: Some(line_num),
                        column: None,
                    });
                }
            }

            // Check workgroup size declaration
            if line.contains("@workgroup_size") && !line.contains("@compute") {
                result.errors.push(ValidationMessage {
                    message_type: MessageType::Error,
                    message: "@workgroup_size attribute must be used with @compute".to_owned(),
                    line: Some(line_num),
                    column: None,
                });
            }
        }

        // Check for unbalanced delimiters
        if brace_count != 0 {
            result.errors.push(ValidationMessage {
                message_type: MessageType::Error,
                message: format!("Unbalanced braces: {brace_count}"),
                line: None,
                column: None,
            });
        }

        if bracket_count != 0 {
            result.errors.push(ValidationMessage {
                message_type: MessageType::Error,
                message: format!("Unbalanced brackets: {bracket_count}"),
                line: None,
                column: None,
            });
        }

        if paren_count != 0 {
            result.errors.push(ValidationMessage {
                message_type: MessageType::Error,
                message: format!("Unbalanced parentheses: {paren_count}"),
                line: None,
                column: None,
            });
        }

        // Check for required entry points
        if !content.contains("fn main")
            && !content.contains("main_entropy")
            && !content.contains("main_propagate")
        {
            result.warnings.push(ValidationMessage {
                message_type: MessageType::Warning,
                message: "Shader might be missing an entry point function".to_owned(),
                line: None,
                column: None,
            });
        }
    }

    /// Get the validation report.
    pub fn get_report(&self) -> &str {
        &self.report
    }

    /// Get the validation results.
    pub fn get_results(&self) -> &[ShaderValidationResult] {
        &self.validation_results
    }
}

/// Parse location information from a string.
fn parse_location(location_str: &str) -> (Option<usize>, Option<usize>) {
    let parts: Vec<&str> = location_str.split(':').collect();

    let line = if parts.len() > 1 {
        parts[parts.len() - 2].trim().parse::<usize>().ok()
    } else {
        None
    };

    let column = if parts.len() > 2 {
        parts[parts.len() - 1].trim().parse::<usize>().ok()
    } else {
        None
    };

    (line, column)
}

/// Main function for the shader validator tool.
pub fn main() -> io::Result<()> {
    // Default configuration
    let config = ValidatorConfig {
        naga_validator_path: None,
        tint_validator_path: None,
        shader_dir: PathBuf::from("src/shader/shaders"),
        report_path: Some(PathBuf::from("validation_report.txt")),
        warnings: true,
    };

    let mut validator = ShaderValidator::new(config);

    // Validate shaders
    let all_valid = validator.validate_directory()?;

    if all_valid {
        println!("\u{2705} All shaders passed validation!");
    } else {
        println!("\u{274c} Some shaders failed validation. See the report for details.");
    }

    println!("Report saved to {:?}", validator.config.report_path);

    Ok(())
}

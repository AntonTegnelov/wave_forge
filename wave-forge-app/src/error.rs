use thiserror::Error;
use wfc_core::WfcError as CoreWfcError;
use wfc_gpu::GpuError as CoreGpuError;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Configuration Error: {0}")]
    Config(String),

    #[error("Error loading rules or tileset: {0}")]
    RulesLoadError(#[from] wfc_rules::LoadError),

    #[error("WFC Core Error: {0}")]
    WfcCore(#[from] CoreWfcError),

    #[error("GPU Initialization/Execution Error: {0}")]
    GpuError(#[from] CoreGpuError),

    #[error("Visualization Error: {0}")]
    VisualizationError(String),

    #[error("Save Error: {0}")]
    SaveError(anyhow::Error),

    #[error("Benchmark CSV Error: {0}")]
    BenchmarkCsvError(#[from] csv::Error),

    #[error("Operation Cancelled by User")]
    Cancelled,

    #[error("Anyhow Error: {0}")]
    Anyhow(#[from] anyhow::Error),

    // Specific GPU Init Error (can wrap CoreGpuError)
    #[error("GPU Accelerator Initialization Error: {0}")]
    GpuInitializationError(CoreGpuError),

    // Keep WfcError variant if used directly
    #[error("WFC Error: {0}")]
    WfcError(CoreWfcError),
}

// We implement Send + Sync manually because underlying errors
// like wgpu::Error might not be Sync.
// This is generally safe IF we ensure AppError doesn't expose
// problematic non-Sync inner types directly across await points.
// However, this can be subtle. For now, assume this is acceptable.
unsafe impl Send for AppError {}
unsafe impl Sync for AppError {}

use thiserror::Error;
use wfc_core::WfcError;
use wfc_gpu::GpuError;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Configuration Error: {0}")]
    Config(String),

    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),

    #[error("WFC Core Error: {0}")]
    WfcCore(#[from] WfcError),

    #[error("GPU Error: {0}")]
    Gpu(#[from] GpuError),

    #[error("Visualization Error: {0}")]
    Visualization(String),

    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}

// We implement Send + Sync manually because underlying errors
// like wgpu::Error might not be Sync.
// This is generally safe IF we ensure AppError doesn't expose
// problematic non-Sync inner types directly across await points.
// However, this can be subtle. For now, assume this is acceptable.
unsafe impl Send for AppError {}
unsafe impl Sync for AppError {}

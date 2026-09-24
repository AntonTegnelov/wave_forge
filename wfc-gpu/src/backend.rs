//! What the block solver needs from a compute API, and nothing more.
//!
//! The solver runs one kernel over a batch of regions and reads two buffers back. That is the whole
//! surface, so a backend is small enough for an engine to supply its own: a Bevy plugin hands over
//! the device Bevy already owns, and a Godot extension can implement this over `RenderingDevice`
//! without the solver knowing (docs/architecture/solver.md, "The backend seam").
//!
//! Nothing here is async. A submission is started, polled and waited on, so an engine decides where
//! blocking happens: on a frame, or on a worker thread.

use thiserror::Error;

/// What a buffer is for. A backend maps these onto its own usage flags.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BufferUsage {
    /// Written by the host, read by the kernel.
    Input,
    /// Written by the kernel and copied out after a dispatch; the host may reset it beforehand.
    Output,
    /// Written by the kernel and read by it again; never leaves the device.
    Scratch,
    /// The kernel's parameters.
    Uniform,
    /// Host-readable, the target of a copy from an [`BufferUsage::Output`] buffer.
    Readback,
}

/// The limits a kernel has to fit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BackendLimits {
    /// Bytes of workgroup memory one workgroup may declare. This bounds chunk size times tiles.
    pub workgroup_storage: u32,
    /// Invocations one workgroup may have.
    pub invocations_per_workgroup: u32,
    /// Workgroups one dispatch may have, which bounds a batch.
    pub workgroups: u32,
}

/// Why a backend could not do something.
#[derive(Error, Debug)]
pub enum BackendError {
    #[error("no compute device is available: {0}")]
    NoDevice(String),
    #[error("the kernel did not compile: {0}")]
    Kernel(String),
    #[error("a buffer of {bytes} B could not be created: {reason}")]
    Buffer { bytes: u64, reason: String },
    #[error("the dispatch failed: {0}")]
    Dispatch(String),
    #[error("reading a buffer back failed: {0}")]
    Readback(String),
    #[error("the pipeline cache at {path} could not be read or written: {reason}")]
    Cache { path: String, reason: String },
}

/// A compute API the block solver can run on.
///
/// The solver creates one pipeline per region shape, buffers per batch capacity, writes the inputs,
/// and dispatches one workgroup per region. Implementations are free to be as simple as they like:
/// there is no bind group management and no queue here, and caching compiled pipelines across runs
/// is the implementation's own business. The solver compiles several pipelines at once from as many
/// threads, so a backend is shared between threads and its pipelines cross them.
pub trait ComputeBackend: Sync {
    /// A compiled kernel.
    type Pipeline: Send;
    /// A set of buffers bound to a pipeline, in binding order.
    type Binding;
    /// Device memory.
    type Buffer;
    /// Work in flight.
    type Submission;

    /// What a kernel has to fit.
    fn limits(&self) -> BackendLimits;

    /// Compiles WGSL.
    ///
    /// # Errors
    /// If the source does not compile, or the backend cannot translate WGSL.
    fn create_pipeline(&self, wgsl: &str, entry: &str) -> Result<Self::Pipeline, BackendError>;

    /// Allocates `bytes` of device memory.
    ///
    /// # Errors
    /// If the allocation fails.
    fn create_buffer(&self, bytes: u64, usage: BufferUsage) -> Result<Self::Buffer, BackendError>;

    /// Binds `buffers` to `pipeline`, in binding order.
    ///
    /// # Errors
    /// If the buffers do not match what the pipeline declares.
    fn create_binding(
        &self,
        pipeline: &Self::Pipeline,
        buffers: &[&Self::Buffer],
    ) -> Result<Self::Binding, BackendError>;

    /// Writes host data into a buffer.
    fn write(&self, buffer: &Self::Buffer, offset: u64, words: &[u32]);

    /// Runs `workgroups` workgroups of `pipeline` and copies each `readbacks` pair from its output
    /// buffer into its readback buffer.
    ///
    /// # Errors
    /// If the work could not be submitted.
    fn dispatch(
        &self,
        pipeline: &Self::Pipeline,
        binding: &Self::Binding,
        workgroups: u32,
        readbacks: &[(&Self::Buffer, &Self::Buffer)],
    ) -> Result<Self::Submission, BackendError>;

    /// Whether the work has finished, without blocking.
    ///
    /// A backend that cannot answer without blocking returns `false` until [`ComputeBackend::wait`]
    /// is called, and says so through [`ComputeBackend::can_poll`].
    ///
    /// # Errors
    /// If the device reports a failure.
    fn is_done(&self, submission: &Self::Submission) -> Result<bool, BackendError>;

    /// Blocks until the work has finished.
    ///
    /// # Errors
    /// If the device reports a failure.
    fn wait(&self, submission: &Self::Submission) -> Result<(), BackendError>;

    /// Copies a readback buffer into `words`, which must be exactly its size.
    ///
    /// Only valid once the submission that filled it has finished.
    ///
    /// # Errors
    /// If the buffer cannot be read.
    fn read(&self, readback: &Self::Buffer, words: &mut [u32]) -> Result<(), BackendError>;

    /// Whether [`ComputeBackend::is_done`] can answer without blocking. A caller that needs to stay
    /// responsive and gets `false` here should drive the solver from a worker thread.
    fn can_poll(&self) -> bool {
        true
    }
}

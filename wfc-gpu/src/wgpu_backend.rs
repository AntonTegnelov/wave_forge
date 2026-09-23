//! The block solver on wgpu, either on its own device or on one an engine already owns.
//!
//! A Bevy plugin passes Bevy's device and queue to [`WgpuBackend::from_device`], so the solver
//! shares the engine's device instead of competing with it for the GPU. A standalone program or a
//! test uses [`WgpuBackend::from_env`].
//!
//! Compiling a kernel takes seconds on some drivers, so a backend can keep compiled pipelines in a
//! file with [`WgpuBackend::cache_pipelines_in`]: a later run on the same adapter and driver starts
//! from them.

use crate::backend::{BackendError, BackendLimits, BufferUsage, ComputeBackend};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The workgroup memory the solver asks for when it makes its own device. Chunks of 8x8x8 with a
/// one-cell halo at 81 tiles need about 23 KiB, over the 16 KiB every device must offer.
const WANTED_WORKGROUP_STORAGE: u32 = 32_768;
/// The invocations per workgroup the solver asks for.
const WANTED_INVOCATIONS: u32 = 512;
/// Storage buffers the kernel binds, which is more than the eight a default device promises.
const STORAGE_BUFFERS: u32 = 10;

/// Compute on wgpu.
#[derive(Debug)]
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: BackendLimits,
    /// The adapter behind the device, when this backend chose it.
    adapter: Option<wgpu::AdapterInfo>,
    /// Compiled pipelines kept across runs, and the file they are kept in.
    cache: Option<(wgpu::PipelineCache, PathBuf)>,
}

/// Work submitted to wgpu, and how many of its readbacks have been mapped.
#[derive(Debug)]
pub struct WgpuSubmission {
    mapped: Arc<AtomicUsize>,
    readbacks: usize,
}

impl WgpuBackend {
    /// A backend on a device of its own.
    ///
    /// The instance is built from the environment so `WGPU_*` variables apply; the dev container
    /// reaches its GPU through a translation layer wgpu hides unless one of them is set.
    /// `WGPU_ADAPTER_NAME` picks the first adapter whose name contains it, ignoring case, which is
    /// how a test run is pointed at a software device; without it the high-performance adapter is
    /// used.
    ///
    /// # Errors
    /// If no adapter or device can be had, or `WGPU_ADAPTER_NAME` matches no adapter.
    pub fn from_env() -> Result<Self, BackendError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = match std::env::var("WGPU_ADAPTER_NAME") {
            Ok(wanted) => adapter_named(&instance, &wanted)?,
            Err(_) => pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            }))
            .map_err(|error| BackendError::NoDevice(error.to_string()))?,
        };
        let offered = adapter.limits();
        let limits = wgpu::Limits {
            max_compute_workgroup_storage_size: offered
                .max_compute_workgroup_storage_size
                .min(WANTED_WORKGROUP_STORAGE),
            max_compute_invocations_per_workgroup: offered
                .max_compute_invocations_per_workgroup
                .min(WANTED_INVOCATIONS),
            max_compute_workgroup_size_x: offered
                .max_compute_workgroup_size_x
                .min(WANTED_INVOCATIONS),
            max_storage_buffers_per_shader_stage: offered
                .max_storage_buffers_per_shader_stage
                .max(STORAGE_BUFFERS),
            ..wgpu::Limits::default()
        };
        // Asked for where offered, so pipelines can be cached across runs.
        let features = adapter.features() & wgpu::Features::PIPELINE_CACHE;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("wave forge block solver"),
            required_features: features,
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|error| BackendError::NoDevice(error.to_string()))?;
        Ok(Self {
            adapter: Some(adapter.get_info()),
            ..Self::from_device(device, queue)
        })
    }

    /// A backend on a device an engine owns.
    #[must_use]
    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let granted = device.limits();
        let limits = BackendLimits {
            workgroup_storage: granted.max_compute_workgroup_storage_size,
            invocations_per_workgroup: granted.max_compute_invocations_per_workgroup,
            workgroups: granted.max_compute_workgroups_per_dimension,
        };
        Self {
            device,
            queue,
            limits,
            adapter: None,
            cache: None,
        }
    }

    /// Keeps compiled pipelines in a file in `dir`, named for the adapter and driver, loading what
    /// an earlier run on them left there and writing it again after every compile. A backend that
    /// cannot cache (an engine's device, whose adapter it does not know, a device without the
    /// feature, or a graphics API wgpu does not cache on) is returned as it is;
    /// [`WgpuBackend::caches_pipelines`] says which.
    ///
    /// # Errors
    /// [`BackendError::Cache`] if the directory cannot be made or an existing file cannot be read.
    pub fn cache_pipelines_in(mut self, dir: &Path) -> Result<Self, BackendError> {
        let Some(key) = self
            .adapter
            .as_ref()
            .filter(|_| {
                self.device
                    .features()
                    .contains(wgpu::Features::PIPELINE_CACHE)
            })
            .and_then(wgpu::util::pipeline_cache_key)
        else {
            return Ok(self);
        };
        let path = dir.join(key);
        let cache_error = |reason: std::io::Error| BackendError::Cache {
            path: path.display().to_string(),
            reason: reason.to_string(),
        };
        std::fs::create_dir_all(dir).map_err(cache_error)?;
        let data = match std::fs::read(&path) {
            Ok(data) => Some(data),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(cache_error(error)),
        };
        // SAFETY: wgpu requires the data to come from an earlier cache of this adapter and driver.
        // The file is named by `pipeline_cache_key`, which is derived from exactly those, so only
        // such a cache is ever loaded; `fallback` makes wgpu start empty if the data is stale.
        let cache = unsafe {
            self.device
                .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("block solver"),
                    data: data.as_deref(),
                    fallback: true,
                })
        };
        self.cache = Some((cache, path));
        Ok(self)
    }

    /// Whether compiled pipelines are kept in a file across runs.
    #[must_use]
    pub const fn caches_pipelines(&self) -> bool {
        self.cache.is_some()
    }

    /// The device it runs on, for a caller that also renders with it.
    #[must_use]
    pub const fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// What it runs on, for a benchmark's or a test run's report: the adapter and driver when this
    /// backend chose them, and the workgroup storage it was granted.
    #[must_use]
    pub fn describe(&self) -> String {
        match &self.adapter {
            Some(adapter) => format!(
                "{} ({:?}, {} {}), {} B workgroup storage",
                adapter.name,
                adapter.backend,
                adapter.driver,
                adapter.driver_info,
                self.limits.workgroup_storage
            ),
            None => format!(
                "an engine's device, {} B workgroup storage",
                self.limits.workgroup_storage
            ),
        }
    }
}

/// The first adapter whose name contains `wanted`, ignoring case.
fn adapter_named(instance: &wgpu::Instance, wanted: &str) -> Result<wgpu::Adapter, BackendError> {
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    let wanted_lower = wanted.to_lowercase();
    let names: Vec<String> = adapters.iter().map(|a| a.get_info().name).collect();
    adapters
        .into_iter()
        .find(|adapter| {
            adapter
                .get_info()
                .name
                .to_lowercase()
                .contains(&wanted_lower)
        })
        .ok_or_else(|| {
            BackendError::NoDevice(format!(
                "WGPU_ADAPTER_NAME={wanted} matches none of the adapters: {}",
                names.join(", ")
            ))
        })
}

impl ComputeBackend for WgpuBackend {
    type Pipeline = wgpu::ComputePipeline;
    type Binding = wgpu::BindGroup;
    type Buffer = wgpu::Buffer;
    type Submission = WgpuSubmission;

    fn limits(&self) -> BackendLimits {
        self.limits
    }

    fn create_pipeline(&self, wgsl: &str, entry: &str) -> Result<Self::Pipeline, BackendError> {
        // Errors reach the device's error scope rather than this call, so the scope is what the
        // caller is told about: a kernel that does not compile must not look like a working one.
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("block solver"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("block solver"),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: self.cache.as_ref().map(|(cache, _)| cache),
            });
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(BackendError::Kernel(error.to_string()));
        }
        if let Some((cache, path)) = &self.cache
            && let Some(data) = cache.get_data()
        {
            // Written beside the file and renamed over it, so a crash never leaves half a cache.
            let partial = path.with_extension("partial");
            std::fs::write(&partial, data)
                .and_then(|()| std::fs::rename(&partial, path))
                .map_err(|reason| BackendError::Cache {
                    path: path.display().to_string(),
                    reason: reason.to_string(),
                })?;
        }
        Ok(pipeline)
    }

    fn create_buffer(&self, bytes: u64, usage: BufferUsage) -> Result<Self::Buffer, BackendError> {
        if bytes == 0 {
            return Err(BackendError::Buffer {
                bytes,
                reason: "a zero-sized binding is not valid".to_owned(),
            });
        }
        let usage = match usage {
            BufferUsage::Input => wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            BufferUsage::Output => {
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
            }
            BufferUsage::Scratch => wgpu::BufferUsages::STORAGE,
            BufferUsage::Uniform => wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            BufferUsage::Readback => wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        };
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage,
            mapped_at_creation: false,
        }))
    }

    fn create_binding(
        &self,
        pipeline: &Self::Pipeline,
        buffers: &[&Self::Buffer],
    ) -> Result<Self::Binding, BackendError> {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let binding = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("block solver"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        match pollster::block_on(scope.pop()) {
            Some(error) => Err(BackendError::Dispatch(error.to_string())),
            None => Ok(binding),
        }
    }

    fn write(&self, buffer: &Self::Buffer, offset: u64, words: &[u32]) {
        self.queue
            .write_buffer(buffer, offset, bytemuck::cast_slice(words));
    }

    fn dispatch(
        &self,
        pipeline: &Self::Pipeline,
        binding: &Self::Binding,
        workgroups: u32,
        readbacks: &[(&Self::Buffer, &Self::Buffer)],
    ) -> Result<Self::Submission, BackendError> {
        if workgroups > self.limits.workgroups {
            return Err(BackendError::Dispatch(format!(
                "{workgroups} workgroups exceed the device's {}",
                self.limits.workgroups
            )));
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, binding, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        for (from, to) in readbacks {
            encoder.copy_buffer_to_buffer(from, 0, to, 0, to.size());
        }
        self.queue.submit(Some(encoder.finish()));
        // Mapping is queued behind the submission, so the count reaching the readbacks is what says
        // the dispatch is done. It is the only completion signal that does not block.
        let mapped = Arc::new(AtomicUsize::new(0));
        for (_, to) in readbacks {
            let mapped = Arc::clone(&mapped);
            to.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    mapped.fetch_add(1, Ordering::Release);
                }
            });
        }
        Ok(WgpuSubmission {
            mapped,
            readbacks: readbacks.len(),
        })
    }

    fn is_done(&self, submission: &Self::Submission) -> Result<bool, BackendError> {
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| BackendError::Dispatch(error.to_string()))?;
        Ok(submission.mapped.load(Ordering::Acquire) >= submission.readbacks)
    }

    fn wait(&self, submission: &Self::Submission) -> Result<(), BackendError> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| BackendError::Dispatch(error.to_string()))?;
        if submission.mapped.load(Ordering::Acquire) < submission.readbacks {
            return Err(BackendError::Readback(
                "the device finished but a readback was not mapped".to_owned(),
            ));
        }
        Ok(())
    }

    fn read(&self, readback: &Self::Buffer, words: &mut [u32]) -> Result<(), BackendError> {
        let slice = readback.slice(..);
        let mapped = slice
            .get_mapped_range()
            .map_err(|error| BackendError::Readback(error.to_string()))?;
        let read: &[u32] = bytemuck::cast_slice(&mapped);
        if read.len() != words.len() {
            return Err(BackendError::Readback(format!(
                "{} words were read into room for {}",
                read.len(),
                words.len()
            )));
        }
        words.copy_from_slice(read);
        drop(mapped);
        readback.unmap();
        Ok(())
    }
}

//! The block solver on wgpu, either on its own device or on one an engine already owns.
//!
//! A Bevy plugin passes Bevy's device and queue to [`WgpuBackend::from_device`], so the solver
//! shares the engine's device instead of competing with it for the GPU. A standalone program or a
//! test uses [`WgpuBackend::from_env`].
//!
//! Compiling a kernel takes seconds on some drivers, so a backend can keep compiled pipelines in
//! files with [`WgpuBackend::cache_pipelines_in`]: a later run on the same adapter starts from them.
//! Each kernel has a file of its own, named for the adapter and the kernel's source, so backends of
//! several rule sets, or several processes, sharing a directory never overwrite each other's.
//! A driver is meant to refuse a cache an older version of it wrote, and native Vulkan drivers do,
//! but dozen, Vulkan over Direct3D 12, keeps its own identity when the Direct3D 12 driver under it
//! is updated; that driver then fails on the stale pipelines, as out of memory, and loses the
//! device. So such a failure deletes the file and is reported as [`BackendError::StaleCache`].

use crate::backend::{BackendError, BackendLimits, BufferUsage, ComputeBackend};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// The workgroup memory the solver asks for when it makes its own device. Chunks of 8x8x8 with a
/// one-cell halo at 81 tiles need about 23 KiB, over the 16 KiB every device must offer.
const WANTED_WORKGROUP_STORAGE: u32 = 32_768;
/// The invocations per workgroup the solver asks for.
const WANTED_INVOCATIONS: u32 = 512;
/// Numbers each partial cache file, so writers on several threads never share one; the process id
/// beside it keeps processes apart.
static PARTIALS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Compute on wgpu.
#[derive(Debug)]
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: BackendLimits,
    /// The adapter behind the device, when this backend chose it.
    adapter: Option<wgpu::AdapterInfo>,
    /// The directory compiled pipelines are kept in across runs, and the adapter's key their files
    /// are named by.
    cache: Option<(PathBuf, String)>,
    /// Whether the driver failed on a cache: its file is deleted, and no cache is written again,
    /// even by a pipeline another thread makes on the lost device.
    /// Held while a cache file is written or deleted, so a pipeline made on another thread
    /// never writes one back between the flag being set and the file going.
    stale_cache: Mutex<bool>,
}

/// Work submitted to wgpu, how many of its readbacks have been mapped, and why the first that
/// could not be was not.
#[derive(Debug)]
pub struct WgpuSubmission {
    mapped: Arc<AtomicUsize>,
    failed: Arc<Mutex<Option<String>>>,
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
        let instance = instance();
        let adapter = match std::env::var("WGPU_ADAPTER_NAME") {
            Ok(wanted) => adapter_named(instance, &wanted)?,
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
            // A full batch of a large rule set's checkpoints passes the default 128 MiB.
            max_storage_buffer_binding_size: offered.max_storage_buffer_binding_size,
            max_buffer_size: offered.max_buffer_size,
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
            storage_binding: granted
                .max_storage_buffer_binding_size
                .min(granted.max_buffer_size),
        };
        Self {
            device,
            queue,
            limits,
            adapter: None,
            cache: None,
            stale_cache: Mutex::new(false),
        }
    }

    /// Keeps compiled pipelines in files in `dir`, one a kernel, named for the adapter, the driver
    /// and the kernel's source, loading what an earlier run on them left there and writing it again
    /// after every compile. A backend that cannot cache (an engine's device, whose adapter it does
    /// not know, a device without the feature, or a graphics API wgpu does not cache on) is
    /// returned as it is; [`WgpuBackend::caches_pipelines`] says which.
    ///
    /// # Errors
    /// [`BackendError::Cache`] if the directory cannot be made.
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
        std::fs::create_dir_all(dir).map_err(|reason| BackendError::Cache {
            path: dir.display().to_string(),
            reason: reason.to_string(),
        })?;
        self.cache = Some((dir.to_owned(), key));
        Ok(self)
    }

    /// The cache for the kernel `wgsl` at `entry`, and the file it is kept in, from what an earlier
    /// run left there.
    ///
    /// # Errors
    /// [`BackendError::Cache`] if an existing file cannot be read.
    fn pipeline_cache(
        &self,
        wgsl: &str,
        entry: &str,
    ) -> Result<Option<(wgpu::PipelineCache, PathBuf)>, BackendError> {
        let Some((dir, key)) = &self.cache else {
            return Ok(None);
        };
        // FNV-1a over the source and the entry point: a kernel's file holds only its pipeline.
        let source = [wgsl.as_bytes(), b"\0", entry.as_bytes()].concat();
        let hash = source
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, &byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
            });
        let path = dir.join(format!("{key}-{hash:016x}"));
        let data = match std::fs::read(&path) {
            Ok(data) => Some(data),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(reason) => {
                return Err(BackendError::Cache {
                    path: path.display().to_string(),
                    reason: reason.to_string(),
                });
            }
        };
        // SAFETY: wgpu requires the data to come from an earlier cache of this adapter. The file is
        // named by `pipeline_cache_key`, from the adapter's vendor and device, so only such a cache
        // is ever loaded; `fallback` makes wgpu start empty when the driver sees the data is stale,
        // and `create_pipeline` deletes the file when the driver fails on it instead.
        let cache = unsafe {
            self.device
                .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("block solver"),
                    data: data.as_deref(),
                    fallback: true,
                })
        };
        Ok(Some((cache, path)))
    }

    /// Whether compiled pipelines are kept in files across runs.
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

/// Whether every readback of a submission whose mappings have all completed was mapped.
///
/// # Errors
/// [`BackendError::Readback`] with why the first that failed was not, a lost device say.
fn mapped_all(submission: &WgpuSubmission) -> Result<(), BackendError> {
    match submission
        .failed
        .lock()
        .expect("no thread panics holding the lock")
        .as_ref()
    {
        Some(reason) => Err(BackendError::Readback(format!(
            "a readback could not be mapped: {reason}"
        ))),
        None => Ok(()),
    }
}

/// The process's one wgpu instance, made on first use and never dropped. Dropping the last
/// instance lets the Vulkan loader unload its driver while a thread that used a device of it may
/// still be ending, and that thread's thread-local destructors, the driver's code, then fault as it
/// exits; Godot quitting after a world run did exactly that.
fn instance() -> &'static wgpu::Instance {
    static INSTANCE: OnceLock<wgpu::Instance> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env())
    })
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
        let cache = self.pipeline_cache(wgsl, entry)?;
        // Errors reach the device's error scopes rather than this call, so the scopes are what the
        // caller is told about: a kernel that does not compile must not look like a working one,
        // and a driver out of memory must not end the process.
        let scopes = [
            wgpu::ErrorFilter::Validation,
            wgpu::ErrorFilter::OutOfMemory,
            wgpu::ErrorFilter::Internal,
        ]
        .map(|filter| self.device.push_error_scope(filter));
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
                cache: cache.as_ref().map(|(cache, _)| cache),
            });
        // Popped innermost first; the first error the driver gave is the one reported.
        let mut first = None;
        for scope in scopes.into_iter().rev() {
            if let Some(error) = pollster::block_on(scope.pop()) {
                first.get_or_insert(error);
            }
        }
        match (first, &cache) {
            (None, _) => {}
            (
                Some(error @ (wgpu::Error::OutOfMemory { .. } | wgpu::Error::Internal { .. })),
                Some((_, path)),
            ) => {
                let mut stale = self
                    .stale_cache
                    .lock()
                    .expect("no thread panics holding the lock");
                *stale = true;
                return Err(match std::fs::remove_file(path) {
                    Err(reason) if reason.kind() != std::io::ErrorKind::NotFound => {
                        BackendError::Cache {
                            path: path.display().to_string(),
                            reason: reason.to_string(),
                        }
                    }
                    _ => BackendError::StaleCache {
                        path: path.display().to_string(),
                        reason: error.to_string(),
                    },
                });
            }
            (Some(error), _) => return Err(BackendError::Kernel(error.to_string())),
        }
        let stale = self
            .stale_cache
            .lock()
            .expect("no thread panics holding the lock");
        if let Some((cache, path)) = &cache
            && !*stale
            && let Some(data) = cache.get_data()
        {
            // Written beside the file and renamed over it, so a crash never leaves half a cache,
            // under a name no other thread or process writing the same kernel uses.
            let partial = path.with_extension(format!(
                "partial.{}.{}",
                std::process::id(),
                PARTIALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
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
        // As for a pipeline, an allocation's failure reaches the device's error scopes, and outside
        // one it would end the process.
        let scopes = [
            wgpu::ErrorFilter::Validation,
            wgpu::ErrorFilter::OutOfMemory,
        ]
        .map(|filter| self.device.push_error_scope(filter));
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        let mut first = None;
        for scope in scopes.into_iter().rev() {
            if let Some(error) = pollster::block_on(scope.pop()) {
                first.get_or_insert(error);
            }
        }
        match first {
            Some(error) => Err(BackendError::Buffer {
                bytes,
                reason: error.to_string(),
            }),
            None => Ok(buffer),
        }
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
        let failed = Arc::new(Mutex::new(None));
        for (_, to) in readbacks {
            let (mapped, failed) = (Arc::clone(&mapped), Arc::clone(&failed));
            to.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                if let Err(error) = result {
                    failed
                        .lock()
                        .expect("no thread panics holding the lock")
                        .get_or_insert(error.to_string());
                }
                mapped.fetch_add(1, Ordering::Release);
            });
        }
        Ok(WgpuSubmission {
            mapped,
            failed,
            readbacks: readbacks.len(),
        })
    }

    fn is_done(&self, submission: &Self::Submission) -> Result<bool, BackendError> {
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| BackendError::Dispatch(error.to_string()))?;
        if submission.mapped.load(Ordering::Acquire) < submission.readbacks {
            return Ok(false);
        }
        mapped_all(submission).map(|()| true)
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
        mapped_all(submission)
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

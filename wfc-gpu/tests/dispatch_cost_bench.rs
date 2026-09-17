//! Is the cost of GPU work per *dispatch* or per unit of work?
//!
//! The solver's wall-clock tracks its dispatch count almost exactly (docs/solver-redesign.md), and
//! batching dispatches into one submit did not help. That leaves one way out: do more per dispatch.
//! WGSL has no grid-wide barrier, but it has workgroup barriers, so a single workgroup can run a
//! sequential loop — which is what a block-local solve would need.
//!
//! This measures the same total work as (a) many dispatches of one iteration and (b) one dispatch
//! looping internally. If (b) is much cheaper, a block-local solver is worth building.
//!
//! ```text
//! cargo test -p wfc-gpu --release --test dispatch_cost_bench -- --ignored --nocapture
//! ```

use std::time::Instant;
use wfc_gpu::gpu::backend::{GpuBackend, WgpuBackend};

/// The loop a real block solve would run also reads adjacency rules from a storage buffer on every
/// step, which workgroup memory cannot hold. If storage reads inside a tight loop are slow, the win
/// measured above shrinks, so this variant does the same loop plus `reads` storage lookups per step.
const STORAGE_SHADER: &str = r#"
struct Params { iterations: u32, work: u32 };

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> rules: array<u32>;

var<workgroup> scratch: array<u32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    var acc = data[lane];
    let mask = arrayLength(&rules) - 1u;
    for (var i = 0u; i < params.iterations; i = i + 1u) {
        // Stand-in for unioning allowed-neighbour masks: scattered reads of the rule table.
        for (var w = 0u; w < params.work; w = w + 1u) {
            acc = acc * 1664525u + 1013904223u;
            acc = acc + rules[acc & mask];
        }
        scratch[lane] = acc;
        workgroupBarrier();
        acc = acc + scratch[(lane + 1u) % 64u];
        workgroupBarrier();
    }
    data[lane] = acc;
}
"#;

const SHADER: &str = r#"
struct Params { iterations: u32, work: u32 };

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

var<workgroup> scratch: array<u32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    var acc = data[lane];
    // One "iteration" stands in for one collapse step: some work, then an exchange with the rest of
    // the workgroup, which is what forces the barrier a sequential solve would need.
    for (var i = 0u; i < params.iterations; i = i + 1u) {
        for (var w = 0u; w < params.work; w = w + 1u) {
            acc = acc * 1664525u + 1013904223u;
        }
        scratch[lane] = acc;
        workgroupBarrier();
        acc = acc + scratch[(lane + 1u) % 64u];
        workgroupBarrier();
    }
    data[lane] = acc;
}
"#;

#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn work_per_dispatch_versus_dispatch_count() {
    let backend = WgpuBackend::new();
    let device = backend.device();
    let queue = backend.queue();

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("dispatch cost"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("dispatch cost"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let data = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("data"),
        size: 64 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: 8,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dispatch cost"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: data.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params.as_entire_binding(),
            },
        ],
    });

    // `work` is the per-iteration arithmetic; keep it small so the loop, not the maths, is measured.
    let work = 64u32;
    let total_iterations = 256u32;

    let run = |iterations: u32, dispatches: u32| -> std::time::Duration {
        queue.write_buffer(&params, 0, bytemuck::cast_slice(&[iterations, work]));
        let started = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        for _ in 0..dispatches {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        started.elapsed()
    };

    // The GPU boosts its clocks under load, so a cold first measurement is not comparable with a
    // later one. Warm up properly, then interleave the variants and take medians.
    for _ in 0..8 {
        run(total_iterations, 4);
    }
    let mut many_samples = Vec::new();
    let mut one_samples = Vec::new();
    for _ in 0..5 {
        many_samples.push(run(1, total_iterations));
        one_samples.push(run(total_iterations, 1));
    }
    many_samples.sort();
    one_samples.sort();
    let many = many_samples[many_samples.len() / 2];
    let one = one_samples[one_samples.len() / 2];
    // The same comparison, but each step also reads the rule table from storage.
    let storage_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("dispatch cost, storage reads"),
        source: wgpu::ShaderSource::Wgsl(STORAGE_SHADER.into()),
    });
    let storage_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("dispatch cost, storage reads"),
        layout: None,
        module: &storage_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    // A rule table the size of ours: 6 axes x 81 x 81 bits, rounded to a power of two for masking.
    let rules = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rules"),
        size: 4 * 4096,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let storage_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dispatch cost, storage reads"),
        layout: &storage_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: data.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: rules.as_entire_binding(),
            },
        ],
    });
    let run_storage = |iterations: u32, dispatches: u32| -> std::time::Duration {
        queue.write_buffer(&params, 0, bytemuck::cast_slice(&[iterations, work]));
        let started = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        for _ in 0..dispatches {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&storage_pipeline);
            pass.set_bind_group(0, &storage_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        started.elapsed()
    };
    for _ in 0..8 {
        run_storage(total_iterations, 4);
    }
    let mut storage_many_samples = Vec::new();
    let mut storage_one_samples = Vec::new();
    for _ in 0..5 {
        storage_many_samples.push(run_storage(1, total_iterations));
        storage_one_samples.push(run_storage(total_iterations, 1));
    }
    storage_many_samples.sort();
    storage_one_samples.sort();
    let storage_many = storage_many_samples[storage_many_samples.len() / 2];
    let storage_one = storage_one_samples[storage_one_samples.len() / 2];
    eprintln!(
        "bench: with storage reads per step ({work} per iteration)\n  \
         as {total_iterations} dispatches: {storage_many:?} ({:.3} ms per iteration)\n  \
         as 1 dispatch looping:            {storage_one:?} ({:.3} ms per iteration)\n  \
         ratio: {:.1}x cheaper inside one dispatch",
        storage_many.as_secs_f64() * 1000.0 / f64::from(total_iterations),
        storage_one.as_secs_f64() * 1000.0 / f64::from(total_iterations),
        storage_many.as_secs_f64() / storage_one.as_secs_f64().max(f64::EPSILON),
    );

    eprintln!(
        "bench: {total_iterations} iterations of identical work\n  \
         as {total_iterations} dispatches of 1 iteration: {many:?} ({:.3} ms per iteration)\n  \
         as 1 dispatch looping {total_iterations} times: {one:?} ({:.3} ms per iteration)\n  \
         ratio: {:.1}x cheaper inside one dispatch",
        many.as_secs_f64() * 1000.0 / f64::from(total_iterations),
        one.as_secs_f64() * 1000.0 / f64::from(total_iterations),
        many.as_secs_f64() / one.as_secs_f64().max(f64::EPSILON),
    );
}

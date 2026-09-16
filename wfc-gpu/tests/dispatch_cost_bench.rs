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
            wgpu::BindGroupEntry { binding: 0, resource: data.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
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

    // Warm up: the first submit pays pipeline and allocator costs.
    run(1, 1);

    let many = run(1, total_iterations);
    let one = run(total_iterations, 1);
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

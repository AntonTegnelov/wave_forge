//! Frame times of wgpu alone, with no Bevy, rendering about what `frame_times` draws: for telling
//! a graphics API's own frame pacing apart from Bevy's use of it
//! ([#318](https://github.com/AntonTegnelov/wave_forge/issues/318),
//! docs/guides/desktop-measurements.md).
//!
//! ```text
//! cargo run -p wave_forge_bevy --release --example wgpu_frame_times -- [--seconds S] [--out FILE]
//! ```
//!
//! Each frame writes a camera to a uniform buffer and draws `CUBES` instanced cubes into a 1920 by
//! 1080 colour target with depth, as `frame_times` draws a city's cells, then submits; a frame
//! waits for the one before it to finish, as Bevy's pipelined rendering does. It prints one line of
//! `key=value` fields in `frame_times`' format, phase `bare`, appended to `--out` as well when it is
//! given. `WGPU_BACKEND` picks the graphics API.

use bevy_math::{Mat4, Vec3, Vec4, proj};
use std::io::Write as _;
use std::time::{Duration, Instant};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
/// About as many cubes as `frame_times` draws when its view is full.
const CUBES: u32 = 20_000;
/// Cubes along each side of the square they stand in.
const SIDE: u32 = 142;

const SHADER: &str = r"
struct Camera { view_projection: mat4x4<f32> }
@group(0) @binding(0) var<uniform> camera: Camera;

struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) shade: f32,
}

@vertex
fn vertex(@builtin(vertex_index) corner: u32, @builtin(instance_index) cube: u32) -> Out {
    // A cube's 36 corners from its index: 6 faces of 2 triangles.
    let face = corner / 6u;
    let k = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
    )[corner % 6u];
    let sign = select(-1.0, 1.0, face % 2u == 0u);
    var local: vec3<f32>;
    switch face / 2u {
        case 0u: { local = vec3(sign, k.x, k.y); }
        case 1u: { local = vec3(k.x, sign, k.y); }
        default: { local = vec3(k.x, k.y, sign); }
    }
    let at = vec3(f32(cube % 142u) * 2.5, f32((cube * 7u) % 5u) * 2.0, f32(cube / 142u) * 2.5);
    var out: Out;
    out.position = camera.view_projection * vec4(at + local, 1.0);
    out.shade = 0.4 + 0.15 * f32(face);
    return out;
}

@fragment
fn fragment(in: Out) -> @location(0) vec4<f32> {
    return vec4(in.shade, in.shade * 0.95, in.shade * 0.9, 1.0);
}
";

fn main() {
    let mut seconds = 20.0_f32;
    let mut out: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--seconds" => {
                let text = value();
                seconds = text
                    .parse()
                    .unwrap_or_else(|_| panic!("--seconds takes a number, not {text}"));
            }
            "--out" => out = Some(value()),
            other => panic!("unknown option {other}"),
        }
    }

    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("an adapter");
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wgpu_frame_times"),
        ..Default::default()
    }))
    .expect("a device");

    let target = |format: wgpu::TextureFormat| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: WIDTH,
                    height: HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    };
    let colour = target(wgpu::TextureFormat::Rgba8UnormSrgb);
    let depth = target(wgpu::TextureFormat::Depth32Float);
    let camera = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("camera"),
        size: 64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cubes"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cubes"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vertex"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fragment"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::TextureFormat::Rgba8UnormSrgb.into())],
        }),
        multiview_mask: None,
        cache: None,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("camera"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera.as_entire_binding(),
        }],
    });

    let centre = Vec3::new(SIDE as f32 * 1.25, 0.0, SIDE as f32 * 1.25);
    let projection = proj::perspective(0.8, WIDTH as f32 / HEIGHT as f32, 0.5, 2000.0);
    let started = Instant::now();
    let mut frames = Vec::new();
    let mut previous: Option<wgpu::SubmissionIndex> = None;
    while started.elapsed() < Duration::from_secs_f32(seconds) {
        let frame = Instant::now();
        let turn = started.elapsed().as_secs_f32() * 0.2;
        let eye = centre + Vec3::new(turn.cos() * 150.0, 120.0, turn.sin() * 150.0);
        let view = look_at(eye, centre);
        let matrix: Vec<u8> = (projection * view)
            .to_cols_array()
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        queue.write_buffer(&camera, 0, &matrix);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cubes"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &colour,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.draw(0..36, 0..CUBES);
        }
        let submitted = queue.submit([encoder.finish()]);
        // One frame in flight, as Bevy's pipelined rendering keeps.
        if let Some(index) = previous.replace(submitted) {
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(index),
                    timeout: None,
                })
                .expect("the previous frame finishes");
        }
        frames.push(frame.elapsed().as_secs_f64() * 1000.0);
    }

    let mut sorted = frames.clone();
    sorted.sort_by(f64::total_cmp);
    let at = |fraction: f64| sorted[((sorted.len() - 1) as f64 * fraction).round() as usize];
    let line = format!(
        "frame_times adapter=\"{}\" device=bare max_batch=none speed=0 phase=bare frames={} p50_ms={:.2} p99_ms={:.2} max_ms={:.2} chunks=0",
        info.name,
        frames.len(),
        at(0.5),
        at(0.99),
        sorted[sorted.len() - 1],
    );
    println!("{line}");
    if let Some(out) = out {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&out)
            .unwrap_or_else(|error| panic!("{out} cannot be written: {error}"));
        writeln!(file, "{line}").unwrap_or_else(|error| panic!("{out} cannot be written: {error}"));
    }
}

/// The right-handed view from `eye` toward `target`, Y up.
fn look_at(eye: Vec3, target: Vec3) -> Mat4 {
    let forward = (target - eye).normalize();
    let side = forward.cross(Vec3::Y).normalize();
    let up = side.cross(forward);
    Mat4::from_cols(
        Vec4::new(side.x, up.x, -forward.x, 0.0),
        Vec4::new(side.y, up.y, -forward.y, 0.0),
        Vec4::new(side.z, up.z, -forward.z, 0.0),
        Vec4::new(-side.dot(eye), -up.dot(eye), forward.dot(eye), 1.0),
    )
}

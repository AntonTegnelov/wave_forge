# Debugging and observability

Practices and tools for understanding what a parallel, asynchronous, GPU-driven Wave Forge run is doing. Testing is covered in [testing.md](testing.md).

## Why this program is hard to debug

- **Work happens in three places at once:** CPU threads, async tasks that hop between threads at every `.await`, and GPU command queues that execute some time after they were submitted.
- **GPU errors surface late and far from their cause.** A wrong binding is reported when the command buffer is validated or executed, not where the mistake is in the code.
- **The failures are data races and silent corruption**, such as a shader updating a neighbour non-atomically, not crashes. Stepping through a debugger changes timing and usually hides them.
- **Results are random** (until seeding lands, A-6), so a failure seen once may not come back.

Stepping through code is therefore the last resort. The approach is: make runs reproducible, check invariants automatically, record structured timelines, and turn state into artifacts that can be inspected afterwards.

## Practices

### 1. Reproduce before you debug

- Save the inputs of a failing run: rule file, grid size, boundary mode and, once supported, the seed.
- Save the output grid (`--output-path`) and render it with `wfc-render` ([testing.md](testing.md#rendering-tools)).
- Shrink the case: the smallest grid and rule set that still fails is worth more than any amount of logging.

### 2. Let invariants find the bug

`wfc_devtools::adjacency_violations` lists every broken adjacency with coordinates, tiles and direction. A handful of violations scattered across a grid points to a race or a lost update. Violations along one axis point to a direction or indexing mistake. Violations everywhere point to a layout or binding mismatch.

Host and shader structs that must match are guarded by layout tests (`wfc-gpu/src/buffers/mod.rs`). If you add a field to one side, the test tells you to update the other.

### 3. Trace with spans, not log lines

Instrumentation uses [`tracing`](https://docs.rs/tracing) spans. The GPU run loop records:

| Span | Covers |
|---|---|
| `wfc_run` | The whole run (fields: grid size, tile count) |
| `initial_propagation` | Propagating cells constrained before the run |
| `iteration` | One observe → collapse → propagate round |
| `entropy_pass` | Dispatching the entropy compute shader |
| `select_cell` | Reading back the lowest-entropy cell |
| `upload_grid` / `download_grid` | Moving the whole grid between CPU and GPU |
| `propagate` | Propagating one collapse (fields: cell) |
| `propagation_pass` | One compute pass inside `propagate` (field: worklist size) |

Spans follow a task across `.await` points and threads, and they record *durations* and *nesting*, which log lines cannot. The rest of the code base still uses plain `log` calls (A-15).

Conventions:

- **One span per stage, not per cell.** Per-cell spans cost more than the work they measure.
- **Put identifying data in span fields** (iteration, cell coordinates, worklist size) instead of formatting it into messages, so tools can filter and aggregate.
- **Name spans after what the code does** (`propagation_pass`), so a timeline reads like the algorithm.
- **Instrument futures, don't enter spans across `.await`.** Use `future.instrument(span)` for async stages; an entered span guard held across an await point attaches unrelated work to the span.

### 4. Look at the timeline

The stress tests can write a Chrome-format trace of every span:

```bash
WFC_TRACE_CHROME=trace.json cargo test -p wfc-devtools --release --test stress -- --ignored --nocapture
```

Open `trace.json` in [Perfetto](https://ui.perfetto.dev) (or `chrome://tracing`). Things to look for: which stage dominates each iteration, how many propagation passes a collapse triggers, and gaps where the CPU is waiting on the GPU. The JSON is also easy to summarise in a script, which is much more useful to an LLM than raw logs.

For live profiling of long runs, [Tracy](https://github.com/wolfpld/tracy) via `tracing-tracy` is the planned next step alongside the performance work in [#7](https://github.com/AntonTegnelov/wave_forge/issues/7).

### 5. Inspect the async runtime when tasks stall

If a run hangs rather than fails, the question is which task is waiting on what. [tokio-console](https://github.com/tokio-rs/console) shows live tasks, their wake-ups and how long they have been idle. It is not wired in by default because it needs a special build flag. To use it temporarily in the CLI, add `console-subscriber`, call `console_subscriber::init()` at startup, build with `RUSTFLAGS="--cfg tokio_unstable"`, and run `tokio-console` alongside. The library crates are meant to stop depending on Tokio (A-12), so this applies to the CLI only.

### 6. GPU-specific checks

- **Validation errors are fatal by design.** wgpu panics with the failing call and resource label, so keep every buffer, bind group, pipeline and pass labelled.
- **Read state back and render it.** `GpuAccelerator::get_intermediate_result` downloads the current possibilities; render one layer to see how far propagation got.
- **Suspect races when results are nearly right.** A pass that updates shared buffers from many invocations must either be race-free or be followed by a check pass. The direct propagation strategy currently does the latter (A-11).
- **Check which adapter you got.** The CLI logs `Using GPU adapter: ...` at startup. In the dev container it should be `Microsoft Direct3D12 (NVIDIA GeForce RTX 3070)`; if it is `llvmpipe`, the instance was not built from the environment, so wgpu hid the non-conformant dozen adapter (`RUST_LOG=wgpu_hal=warn` shows "hiding adapter").
- **A `SIGSEGV` after all tests passed is the dozen unload bug, not our code.** A backtrace ends in `__nptl_deallocate_tsd` calling an unmapped address. See the known issue in [development.md](development.md#toolchain-and-environment) for the preload workaround.
- **Compare adapters when a result looks wrong.** dozen is a non-conformant translation layer. Rerun with `WGPU_ADAPTER_NAME=llvmpipe`: if the software device behaves correctly, suspect the driver stack before the shader.
- **Real hardware for performance.** Software Vulkan (llvmpipe) timings say nothing about GPU performance, and dozen timings include translation overhead.

### 7. CPU-side concurrency

For lock-free or atomic CPU code, use [loom](https://github.com/tokio-rs/loom) to explore thread interleavings in tests, and ThreadSanitizer on nightly (`RUSTFLAGS="-Zsanitizer=thread"`) for integration runs. Both are opt-in tools, not part of the default test run.

## When something goes wrong: a checklist

1. Does it reproduce on a smaller grid with the same rules?
2. What does `adjacency_violations` report, and is there a pattern (scattered, one axis, everywhere)?
3. What does the rendered output look like (layer view or four-view sheet)?
4. What does the timeline show: which stage, how many passes, any stalls?
5. Do the host/shader layout tests still pass?
6. Only then: add targeted spans or readbacks around the suspicious stage.

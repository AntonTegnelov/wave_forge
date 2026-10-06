# Measuring on a desktop

Every timing so far comes from the dev container, whose GPU is reached through a translated driver
(dozen) or a software one (lavapipe) ([environment.md](environment.md)). Some decisions need native
numbers, and one script takes all of them on a Windows desktop: `tools/measure_desktop.ps1`. This
page is how to run it and what it measures. What the numbers decide is written in the issues and
stories linked below; the numbers themselves go into
[measurements.md](../research/measurements.md) with the machine and driver they came from.

## What it measures

On Vulkan and on Direct3D 12:

| Run | Tool | For |
|---|---|---|
| A city streamed around a moving focus in Bevy and drawn at 1080p: on Bevy's device and on one of the solver's own, at 4.2 m/s and 30 m/s, and at 30 m/s with batches of at most 16 and 4 regions. An idle phase first, the cost of drawing alone | `wave_forge_bevy/examples/frame_times.rs` | where the solver runs in both engines, and whether a smaller batch buys smoother frames ([#39](https://github.com/AntonTegnelov/wave_forge/issues/39)) |
| The city walked in Godot with its module models drawn, on Forward+, at 4.2 m/s and 30 m/s, with the node's own time, each chunk drawn as a MultiMesh per module and as one merged mesh, and what drawing a chunk costs Godot's thread | `wave_forge_godot/godot/measure_city.gd` | the solver on a device of its own in a desktop Godot ([#39](https://github.com/AntonTegnelov/wave_forge/issues/39)), P1's frame rate, and how a chunk is drawn ([#203](https://github.com/AntonTegnelov/wave_forge/issues/203)) |
| Grass on 25 chunks with and without it, the ground's levels of detail, and the far ground beyond the near ground, each with a check for gaps, on Forward+ | `render_ground.gd`, `render_lods.gd`, `render_far.gd` | E45, E47 and E52 on a desktop ([#166](https://github.com/AntonTegnelov/wave_forge/issues/166), P2) |
| The maximal preset's continent played from the directory its world run baked, in a window of 1920 by 1080 on Forward+, its far ground out to 2 km: walking at 4.2 m/s, then flying at 30 m/s, with the most memory the process held and a check that no stage ran | `wave_forge_godot/godot/measure_playback.gd` | M1's frame rate and bounded memory ([#197](https://github.com/AntonTegnelov/wave_forge/issues/197), [#247](https://github.com/AntonTegnelov/wave_forge/issues/247)) |
| The small city preset walked by a camera at 4.2 m/s from the world's origin into the city, its town, grass, bodies and navigation built around it, in a window of 1920 by 1080 on Forward+, with the node's own time | `wave_forge_godot/godot/measure_city_preset.gd` | P1's bars on a stage world with a town drawn ([#307](https://github.com/AntonTegnelov/wave_forge/issues/307)) |

Once, headless, before the runs above: the whole continent baked from `continent.tscn` through the
node's world run, as the editor dock's World run bakes it, from an empty directory, with the time it
took and the most memory it held (`measure_world_run.gd`), for M1's ten minutes; the playback above
plays what it wrote. Also once and headless, a new world made from nothing as a game makes one,
the game's history and then the whole continent until it plays (`examples/new_world/measure.gd`),
for M2's ten minutes. Then, also once and headless: the history example's first towns, on the first run after building and on a second
one, for P3's time into a new world; `tests/golden_stages.rs`, which checks that the stages come
out on Windows bit for bit as recorded on Linux ([engine-integration.md](../architecture/engine-integration.md#noise-that-means-the-same-in-both-engines));
`tests/interactive_edit.rs`, how long a changed parameter takes to regenerate a 3×3-chunk preview,
for P6 ([measurements.md](../research/measurements.md) L33); `wfc-devtools/tests/continent.rs`, what part of
the maximal preset's continent costs, for M1 (L34); and, three times, `measure_volume.gd`, what a cave volume costs per chunk while a player walks
through it at 4.2 m/s, for P1 ([#71](https://github.com/AntonTegnelov/wave_forge/issues/71),
[measurements.md](../research/measurements.md) E53).

Each measured phase lasts 20 seconds and prints the median, 99th percentile and slowest frame. The
whole run takes about 80 minutes, most of it the first build and the continent's two whole runs.

It does not measure a second GPU vendor, which #39 also asks for.

## Before you start

On the Windows desktop, once:

1. **Git** ([git-scm.com](https://git-scm.com/download/win)).
2. **Rust** through [rustup](https://rustup.rs). It asks for the Visual Studio C++ build tools
   ("Desktop development with C++") if they are missing; install them. Open a new terminal
   afterwards so `cargo` is on the path. The repository's `rust-toolchain.toml` picks the version.
3. **The latest driver** for the GPU, from its vendor.
4. About **17 GB free** on the drive that holds `%LOCALAPPDATA%`, where the builds go, or pass
   `-BuildDir` to use another drive.

The script downloads Godot 4.7.2 itself.

## Running it

1. Clone a fresh copy outside the dev container's folder, on a drive with room, and switch to
   `develop`:

   ```powershell
   git clone https://github.com/AntonTegnelov/wave_forge.git C:\wave_forge-measure
   cd C:\wave_forge-measure
   git switch develop
   ```

   For a later run, `git pull` in that folder instead.

2. Close games, browsers with video and anything else that uses the GPU, plug a laptop in, and
   leave the machine alone while it runs: windows open and close by themselves, and anything else on
   the GPU shows up in the numbers.

3. Run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\measure_desktop.ps1
   ```

   It prints each step as it starts. The first build takes 10 to 20 minutes.

4. When it ends it prints where the results are: a folder `measurements\<date-time>\` in the
   clone and a zip next to it. Copy the zip into the `measurements\` folder of the dev container's
   repository on the host (the folder the container mounts), where the next session can read it, or
   attach it to [#39](https://github.com/AntonTegnelov/wave_forge/issues/39).

`summary.txt` in the results has the machine (GPU, driver, CPU, memory, commit) and one line per
measured phase; `logs\` has each run's full output, and the pictures the ground scripts drew are
beside them. The script ends with `every run passed`, or lists the runs that failed and exits with
a non-zero code; the others still ran.

### Options

| Option | Default | Use |
|---|---|---|
| `-Apis vulkan` or `-Apis d3d12` | both | measure on one graphics API only |
| `-Only bevy,godot,ground,history,m1,m2,preset,stages,volume` | all | run only some parts, for example again after a failure |
| `-Seconds 20` | 20 | how long each measured phase lasts |
| `-SkipBuild` | | use what the last run built |
| `-BuildDir D:\wave_forge-build` | `%LOCALAPPDATA%\wave_forge` | where Cargo builds and Godot is downloaded to |
| `-Godot C:\path\to\Godot_v4.7.2-stable_win64_console.exe` | downloaded | a Godot you already have; use the `_console` executable, which prints to the terminal |

## If something goes wrong

- **"running scripts is disabled on this system":** start it with `-ExecutionPolicy Bypass` as
  above, which changes nothing outside this run.
- **`cargo` is not recognised:** open a new terminal after installing Rust.
- **A build fails with a linker error:** the Visual Studio C++ build tools are missing; run the
  Visual Studio Installer and add "Desktop development with C++".
- **Godot fails on `vulkan` or `d3d12`:** update the GPU driver, or measure the other API with
  `-Apis`.
- **One run failed:** its log is in `logs\`, named after the run. Rerun only that part, for example
  `-Only godot -SkipBuild`, and send both zips.

The script also runs on Linux with PowerShell 7 (`pwsh`), on Vulkan only, with `-Godot` pointing at
a Godot binary and a display for the windowed runs; that is how it was checked in the dev
container, whose numbers do not count.

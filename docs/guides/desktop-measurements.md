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
| The city walked in Godot with its module models drawn, on Forward+, at 4.2 m/s and 30 m/s, with the node's own time | `wave_forge_godot/godot/measure_city.gd` | the solver on a device of its own in a desktop Godot ([#39](https://github.com/AntonTegnelov/wave_forge/issues/39)), P1's frame rate |
| Grass on 25 chunks with and without it, and the ground's levels of detail with a check for gaps, on Forward+ | `render_ground.gd`, `render_lods.gd` | E45 and E47 on a desktop ([#166](https://github.com/AntonTegnelov/wave_forge/issues/166)) |

Once, headless: the history example's first towns, on the first run after building and on a second
one, for P3's time into a new world.

Each measured phase lasts 20 seconds and prints the median, 99th percentile and slowest frame. The
whole run takes about 45 minutes, most of it the first build.

It does not measure a second GPU vendor, which #39 also asks for, nor merged chunk meshes against a
MultiMesh per module ([#38](https://github.com/AntonTegnelov/wave_forge/issues/38)), which has no
tool yet.

## Before you start

On the Windows desktop, once:

1. **Git** ([git-scm.com](https://git-scm.com/download/win)).
2. **Rust** through [rustup](https://rustup.rs). It asks for the Visual Studio C++ build tools
   ("Desktop development with C++") if they are missing; install them. Open a new terminal
   afterwards so `cargo` is on the path. The repository's `rust-toolchain.toml` picks the version.
3. **The latest driver** for the GPU, from its vendor.
4. About **15 GB free** on the drive that holds `%LOCALAPPDATA%`, where the builds go, or pass
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
| `-Only bevy,godot,ground,history` | all | run only some parts, for example again after a failure |
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

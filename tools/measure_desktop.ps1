<#
.SYNOPSIS
Builds Wave Forge and runs every measurement that waits on a desktop, then zips the results.

.DESCRIPTION
What it measures and why is in docs/guides/desktop-measurements.md, which is also the step-by-step
guide. In short, on each graphics API in -Apis:
  - Bevy (#39): frame times while a city streams, on Bevy's device and on one of the solver's own,
    at a walk and in flight, and with smaller batches (wave_forge_bevy/examples/frame_times.rs).
  - Godot (#39, P1): frame times while the city streams with its models drawn, on Forward+
    (wave_forge_godot/godot/measure_city.gd).
  - Godot (#166, P2): grass, ground levels of detail and far ground on Forward+ (render_ground.gd,
    render_lods.gd, render_far.gd).
And once, P3: the history example's first towns, on a first run and a second one; and the golden
stages test, which checks the stages come out on Windows bit for bit as recorded on Linux.

Everything goes into measurements/<date-time>/ in the repository, and a zip of it next to that
folder. A run that fails is recorded and the others go on; the script ends with a non-zero exit code
if any failed.

Works in Windows PowerShell 5.1 and PowerShell 7, on Windows and on Linux (where only Vulkan runs,
and the windowed runs need a display).

.PARAMETER Godot
The Godot 4.7.2 executable. On Windows use the *_console.exe, which prints to the terminal. Without
it, the script downloads Godot 4.7.2 into the build directory once.

.PARAMETER BuildDir
Where Cargo builds and Godot is downloaded to. Several GB; keep it off a nearly full drive. Defaults
to %LOCALAPPDATA%\wave_forge on Windows and ~/.cache/wave_forge elsewhere.

.PARAMETER Apis
The graphics APIs to measure on: vulkan, d3d12 or both (the default on Windows).

.PARAMETER Seconds
How long each measured phase lasts. Default 20.

.PARAMETER Only
Run only some parts: bevy, godot, ground, history, stages. Default all.

.PARAMETER SkipBuild
Use what an earlier run built.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File tools\measure_desktop.ps1
#>
[CmdletBinding()]
param(
    [string]$Godot = "",
    [string]$BuildDir = "",
    [string[]]$Apis = @(),
    [int]$Seconds = 20,
    [string[]]$Only = @("bevy", "godot", "ground", "history", "stages"),
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$GodotVersion = "4.7.2"
$OnWindows = ($PSVersionTable.PSEdition -eq "Desktop") -or $IsWindows
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ($BuildDir -eq "") {
    if ($OnWindows) { $BuildDir = Join-Path $env:LOCALAPPDATA "wave_forge" }
    else { $BuildDir = Join-Path $HOME ".cache/wave_forge" }
}
# `-File` hands a list as one string, `a,b`, so both forms are split here.
$Apis = @($Apis | ForEach-Object { $_ -split "," } | Where-Object { $_ -ne "" })
$Only = @($Only | ForEach-Object { $_ -split "," } | Where-Object { $_ -ne "" })
foreach ($part in $Only) {
    if (@("bevy", "godot", "ground", "history", "stages") -notcontains $part) {
        throw "-Only takes bevy, godot, ground, history and stages, not $part"
    }
}
if ($Apis.Count -eq 0) {
    if ($OnWindows) { $Apis = @("vulkan", "d3d12") } else { $Apis = @("vulkan") }
}
foreach ($api in $Apis) {
    if ($api -ne "vulkan" -and $api -ne "d3d12") { throw "-Apis takes vulkan and d3d12, not $api" }
}

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Results = Join-Path $Repo "measurements/$Stamp"
$Logs = Join-Path $Results "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$Failures = New-Object System.Collections.Generic.List[string]

function Say([string]$Message) {
    Write-Host "measure_desktop: $Message"
}

# Runs a program with its output going to a log (and the terminal); records a failure by name.
function Invoke-Logged([string]$Name, [string]$Program, [string[]]$Arguments) {
    $log = Join-Path $Logs "$Name.log"
    Say "$Name"
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Program @Arguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log | Out-Null
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    if ($code -ne 0) {
        $Failures.Add("$Name (exit $code, see logs/$Name.log)")
        Say "$Name failed with exit code $code"
    }
    return $code
}

function Invoke-Checked([string]$Name, [string]$Program, [string[]]$Arguments) {
    if ((Invoke-Logged $Name $Program $Arguments) -ne 0) {
        throw "$Name failed; see $(Join-Path $Logs "$Name.log")"
    }
}

# --- The machine -------------------------------------------------------------------------------

$system = New-Object System.Collections.Generic.List[string]
$system.Add("date: $(Get-Date -Format o)")
$system.Add("commit: $(git -C $Repo rev-parse HEAD)")
$system.Add("uncommitted changes: $((git -C $Repo status --short | Measure-Object).Count)")
$system.Add("rustc: $(rustc --version)")
if ($OnWindows) {
    $os = Get-CimInstance Win32_OperatingSystem
    $system.Add("os: $($os.Caption) $($os.Version)")
    foreach ($cpu in Get-CimInstance Win32_Processor) { $system.Add("cpu: $($cpu.Name)") }
    $system.Add("memory: $([math]::Round($os.TotalVisibleMemorySize / 1MB, 1)) GB")
    foreach ($gpu in Get-CimInstance Win32_VideoController) {
        $system.Add("gpu: $($gpu.Name), driver $($gpu.DriverVersion) ($($gpu.DriverDate))")
    }
} else {
    $system.Add("os: $(uname -srm)")
    $system.Add("cpu: $((Get-Content /proc/cpuinfo | Select-String '^model name' | Select-Object -First 1).ToString().Split(':')[1].Trim())")
}

# --- Godot -------------------------------------------------------------------------------------

if ($Godot -eq "") {
    if (-not $OnWindows) { throw "give -Godot: the script downloads Godot only on Windows" }
    $godotDir = Join-Path $BuildDir "godot-$GodotVersion"
    $Godot = Join-Path $godotDir "Godot_v$GodotVersion-stable_win64_console.exe"
    if (-not (Test-Path $Godot)) {
        New-Item -ItemType Directory -Force -Path $godotDir | Out-Null
        $zip = Join-Path $godotDir "godot.zip"
        $url = "https://github.com/godotengine/godot/releases/download/$GodotVersion-stable/Godot_v$GodotVersion-stable_win64.exe.zip"
        Say "downloading Godot $GodotVersion"
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
        Expand-Archive -Path $zip -DestinationPath $godotDir -Force
        Remove-Item $zip
    }
}
if (-not (Test-Path $Godot)) { throw "no Godot at $Godot" }
$system.Add("godot: $((& $Godot --version 2>&1 | Select-Object -Last 1).ToString())")
$system | Set-Content (Join-Path $Results "system.txt")

# --- Build -------------------------------------------------------------------------------------

$Targets = @{
    root  = Join-Path $BuildDir "target/wave_forge"
    godot = Join-Path $BuildDir "target/wave_forge_godot"
    bevy  = Join-Path $BuildDir "target/wave_forge_bevy"
}
if ($OnWindows) {
    $Library = "wave_forge_godot.dll"; $FrameTimes = "frame_times.exe"
} else {
    $Library = "libwave_forge_godot.so"; $FrameTimes = "frame_times"
}

# Puts what a Godot project in this repository loads next to it: the extension, the city rule set,
# the city's module models, and the extension list the editor would write.
function Initialize-GodotProject([string]$Project, [bool]$Valley) {
    $bin = Join-Path $Project "bin"
    New-Item -ItemType Directory -Force -Path $bin | Out-Null
    Copy-Item (Join-Path $Targets.godot "release/$Library") $bin -Force
    Copy-Item (Join-Path $Repo "examples/city.ron") (Join-Path $Project "city.ron") -Force
    if ($Valley) {
        Copy-Item (Join-Path $Repo "examples/valley.world.ron") (Join-Path $Project "valley.world.ron") -Force
    }
    $env:CARGO_TARGET_DIR = $Targets.root
    Invoke-Checked "export-models-$(Split-Path $Project -Leaf)" "cargo" @(
        "run", "--quiet", "--release", "--manifest-path", (Join-Path $Repo "Cargo.toml"),
        "-p", "wfc-devtools", "--bin", "wfc-export-models", "--", "--out", (Join-Path $Project "models"))
    $dotGodot = Join-Path $Project ".godot"
    New-Item -ItemType Directory -Force -Path $dotGodot | Out-Null
    Set-Content -Path (Join-Path $dotGodot "extension_list.cfg") -Value "res://wave_forge.gdextension"
}

$GodotProject = Join-Path $Repo "wave_forge_godot/godot"
$HistoryProject = Join-Path $Repo "examples/history"
if (-not $SkipBuild) {
    $env:CARGO_TARGET_DIR = $Targets.godot
    Invoke-Checked "build-godot" "cargo" @("build", "--release", "--manifest-path", (Join-Path $Repo "wave_forge_godot/Cargo.toml"))
    Initialize-GodotProject $GodotProject $true
    Initialize-GodotProject $HistoryProject $false
    if ($Only -contains "bevy") {
        $env:CARGO_TARGET_DIR = $Targets.bevy
        Invoke-Checked "build-bevy" "cargo" @("build", "--release", "--example", "frame_times", "--manifest-path", (Join-Path $Repo "wave_forge_bevy/Cargo.toml"))
    }
}
Remove-Item Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue

# --- Measure -----------------------------------------------------------------------------------

$seconds = "$Seconds"
foreach ($api in $Apis) {
    if ($Only -contains "bevy") {
        # wgpu names Direct3D 12 dx12.
        if ($api -eq "d3d12") { $env:WGPU_BACKEND = "dx12" } else { $env:WGPU_BACKEND = "vulkan" }
        $bevy = Join-Path $Targets.bevy "release/examples/$FrameTimes"
        $out = Join-Path $Results "bevy.txt"
        $runs = @(
            @("shared", "4.2", ""), @("own", "4.2", ""),
            @("shared", "30", ""), @("own", "30", ""),
            @("shared", "30", "16"), @("shared", "30", "4"))
        foreach ($run in $runs) {
            $arguments = @("--device", $run[0], "--speed", $run[1], "--seconds", $seconds, "--out", $out)
            $name = "bevy-$api-$($run[0])-$($run[1])"
            if ($run[2] -ne "") {
                $arguments += @("--max-batch", $run[2])
                $name += "-batch$($run[2])"
            }
            [void](Invoke-Logged $name $bevy $arguments)
        }
        Remove-Item Env:WGPU_BACKEND -ErrorAction SilentlyContinue
    }
    $renderer = @("--rendering-driver", $api, "--rendering-method", "forward_plus")
    if ($Only -contains "godot") {
        foreach ($speed in @("4.2", "30")) {
            [void](Invoke-Logged "godot-city-$api-$speed" $Godot (@("--path", $GodotProject) + $renderer + @(
                "--script", "measure_city.gd", "--", "--speed", $speed, "--seconds", $seconds,
                "--out", (Join-Path $Results "godot_city.txt"))))
        }
    }
    if ($Only -contains "ground") {
        foreach ($script in @("render_ground", "render_lods", "render_far")) {
            $name = "godot-$script-$api"
            [void](Invoke-Logged $name $Godot (@("--path", $GodotProject) + $renderer + @("--script", "$script.gd")))
            # The pictures a script saved, named in its output.
            foreach ($line in Get-Content (Join-Path $Logs "$name.log")) {
                foreach ($match in [regex]::Matches($line, "saved (\S+\.png)")) {
                    $picture = $match.Groups[1].Value
                    if (Test-Path $picture) {
                        Copy-Item $picture (Join-Path $Results "$name-$(Split-Path $picture -Leaf)") -Force
                    }
                }
            }
        }
    }
}
if ($Only -contains "history") {
    foreach ($attempt in @("first", "second")) {
        [void](Invoke-Logged "history-$attempt" $Godot @("--headless", "--path", $HistoryProject, "--script", "check.gd"))
    }
}

if ($Only -contains "stages") {
    $env:CARGO_TARGET_DIR = $Targets.root
    [void](Invoke-Logged "golden-stages" "cargo" @("test", "--release", "--manifest-path", (Join-Path $Repo "Cargo.toml"), "--test", "golden_stages"))
    Remove-Item Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue
}

# --- Summary -----------------------------------------------------------------------------------

$summary = New-Object System.Collections.Generic.List[string]
$summary.AddRange([string[]](Get-Content (Join-Path $Results "system.txt")))
foreach ($file in @("bevy.txt", "godot_city.txt")) {
    $path = Join-Path $Results $file
    if (Test-Path $path) { $summary.Add(""); $summary.AddRange([string[]](Get-Content $path)) }
}
foreach ($log in Get-ChildItem $Logs -Filter "*.log") {
    foreach ($line in Get-Content $log.FullName) {
        if ($line -match "^(render_ground|render_lods|render_far|check): ") { $summary.Add("$($log.BaseName): $line") }
    }
}
$summary.Add("")
if ($Failures.Count -eq 0) { $summary.Add("every run passed") }
else { $summary.Add("failed:"); $summary.AddRange($Failures) }
$summary | Set-Content (Join-Path $Results "summary.txt")

$zip = "$Results.zip"
Compress-Archive -Path (Join-Path $Results "*") -DestinationPath $zip -Force
Say "results in $Results"
Say "send $zip"
if ($Failures.Count -gt 0) {
    Say "$($Failures.Count) runs failed: $($Failures -join '; ')"
    exit 1
}

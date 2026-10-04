# A sample world

A sample world for a trailer, built on the small city preset with no Rust and almost no code
([story N10](../../docs/product/user-stories.md)): a city solved by WFC in rolling countryside,
grass and trees swaying in the wind, a day and a night going by, and townsfolk walking between the
buildings on the navigation map Wave Forge bakes for the city.

## Running it

You need Godot 4.7 and Rust (<https://rustup.rs>), on Linux.

1. From this folder, run `./prepare.sh`. It builds the Wave Forge extension, which takes a few
   minutes the first time, and installs it here as a game would: the library in `bin/` and the
   addon in `addons/wave_forge/`.
2. Open this folder in Godot and press play. Walk with W, A, S and D, click to look around with the
   mouse, jump with Space, and press Escape to free the mouse.

The first run takes a little longer: the towns' GPU kernels are compiled once and kept in Godot's
user folder.

## How it fits together

- `main.tscn` holds the whole scene. `City` is the addon's small city preset
  (`addons/wave_forge/presets/city.tscn`) as it ships, its view raised to 6 chunks so that its
  navigation, 4 chunks out, covers the whole city from where you start. `Walker` is the addon's
  first-person walker, at the edge of the city, facing it.
- `day_night.gd` turns the sun from sunrise to sunset over `day_seconds` (180 by default) and lights
  the night with the moon; the procedural sky draws the sun's disc from the sun's light, so it
  follows. Change `time` in the inspector to start at another hour.
- `townsfolk.gd` waits until the city's site is generated and every chunk of it has navigation,
  then puts `count` people at random points of the streets and sends each to another whenever it
  arrives. `person.tscn` is one of them, a capsule with a `NavigationAgent3D`; put a character of
  your own in its place.
- The wind is the extension's global shader parameter `wave_forge_wind`, which the grass and the
  trees read ([godot.md](../../docs/reference/godot.md#wind)); set it from a script with
  `RenderingServer.global_shader_parameter_set`.
- `check.gd` checks all of this headless: `godot --headless --path . --script check.gd`.

## Making a trailer in an hour

1. Run `./prepare.sh` and open the project, as above. Press play once and walk into the city, to
   see that it all runs.
2. Pick the look in the inspector: select `City` and move the preset's sliders, density for how
   much of the city is built, hills for the countryside, trees for the woods; or press Reroll seed
   for another city.
3. Pick the hour: select `DayNight` and set `time` (0.3 is the morning, 0.7 the evening) and
   `day_seconds` (30 makes a day go by in a shot).
4. Record: Godot's Movie Maker (Project Settings, Editor, Movie Writer, then the clapperboard
   button by the play button) writes every frame at a fixed rate, whatever the frame time. Walk a
   path through the countryside into the streets.
5. More townsfolk: select `Townsfolk` and raise `count`.

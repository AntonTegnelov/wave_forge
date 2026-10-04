# A new world

An example of generating a maximal world once, when a player starts a new game, and playing it
without further waits ([story M2](../../docs/product/user-stories.md)). The world is the maximal
preset's continent, 4 by 4 km of over a hundred stages with towns of eight cultures. Its history is the
game's, simulated over the continent's map in GDScript before anything is generated. The whole
world is then generated into the game's folder, with a progress screen and a button to stop, and
played from what was written.

## Running it

You need Godot 4.7 and Rust (<https://rustup.rs>), on Linux, and about 1 GB free in Godot's user
folder.

1. From this folder, run `./prepare.sh`. It builds the Wave Forge extension, which takes a few
   minutes the first time, installs it here with the Wave Forge addon, and copies in the
   continent's scene, pack and cultures.
2. Open this folder in Godot and press play. The screen shows how many of the continent's 65 536
   chunks are done and which stages have cost the most so far; the whole run takes about ten
   minutes, the first chunks after a minute or two.
3. Press Stop, or close the game, at any time. Press Resume, or start the game again, and the run
   goes on from what it wrote.
4. Once every chunk is there, the game plays the world from the folder, generating nothing: walk
   with W, A, S and D, click to look around, jump with Space and press Escape to free the mouse.

The world lives in Godot's user folder for the project, in `new_world/`. Delete that folder for a
new world.

## How it fits together

- `history.gd` is the game's history. It reads the continent with `sample`, which computes a stage
  at a point without generating anything, and returns the `settlements` table: where each
  settlement stands, how large it grew, when it was founded, its culture and what became of it.
- `main.gd` is the new-world flow. It starts the continent's node (`continent.tscn`, copied from
  the extension's own project), gives it the history and saves it as `new_world/history.json`,
  and runs the whole world with `run_world` into `new_world/world/`. `world_run_progress` drives
  the progress screen, `cancel_world_run` the Stop button, and `world_run_finished` either the
  Resume button or, with every chunk done, a mark in `new_world/done` and play: a second node with
  `play_directory` set to the folder, and the addon's walker at the first settlement.
- A resumed run gives the stages the saved history rather than a new one, since a chunk written
  from another history would not belong to the same world; the run skips every chunk the folder
  holds already.
- `check.gd` checks the history, the progress, stopping and resuming headless:
  `godot --headless --path . --script check.gd`. A whole run takes minutes, so finishing it and
  playing it is measured on a desktop instead
  ([desktop-measurements.md](../../docs/guides/desktop-measurements.md)).

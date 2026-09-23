# A history of a continent

An example of putting your own history into a Wave Forge world, written in GDScript only
([story N11](../../docs/product/user-stories.md)). Before play, a toy history of about a hundred
lines (`history.gd`) reads the continent's world map, sends rivers downhill to the sea, founds
villages beside them, lets them grow for a century, joins each to its nearest neighbour by a road,
and burns a few. It gives what it made to the stages as tables of facts, and then you walk through
the result: rivers carved into the ground, roads levelled across it, a town on every village and
open ruins where one burned.

## Running it

You need Godot 4.7 and Rust (<https://rustup.rs>), on Linux.

1. From this folder, run `./prepare.sh`. It builds the Wave Forge extension, which takes a few
   minutes the first time, and copies it here with the city's modules.
2. Open this folder in Godot and press play. Walk with W, A, S and D, look with the mouse, jump with
   space, and press Escape to free the mouse.
3. Press N for a new history. Only what it changed is generated again.

The first run takes a little longer: the towns' GPU kernels are compiled once and kept in Godot's
user folder. The history is saved there too, as `history.json`, and later runs restore the same
world from it; delete it to start over.

## How it fits together

- `continent.world.ron` is the pack: a world map one column per chunk (`land`, at scale 8), the
  ground read from it (`height`), and the stages that realise the history. `rivers` and `roads` make
  curves of the tables' rows, `carved` and `paved` draw them into the ground, `villages` puts a site
  on every village, `level` flattens it, and `towns` builds a town of the `city` rule set on it, or
  of `ruins` where the village's `fate` is `burned`.
- `history.gd` is the history. It reads the map with `atlas("land", ...)` and returns three tables,
  each an Array of Dictionaries with an `id` and a value for every column the pack declares.
  Positions are in cells, the unit every table uses; one map column is 8 cells.
- `main.gd` starts the `WaveForgeStages` node, runs or restores the history, hands each table over
  with `give_table`, and draws the towns as they arrive.
- `check.gd` checks all of this headless: `godot --headless --path . --script check.gd`.

The history is plain data, so it can grow without any Rust: a new kind of site is a new table in the
pack and the stages that read it. The pack format is in
[docs/reference/packs.md](../../docs/reference/packs.md), and the node's functions are in
[docs/reference/godot.md](../../docs/reference/godot.md).

## The game's history of the continent, simulated over its map before play (story M2): where its
## settlements were founded, how large each grew, when, of which culture, and what became of it,
## as the rows of the pack's `settlements` table. It reads the map with `sample`, which computes a
## stage at a point without generating anything, so it runs in about a second before any chunk
## exists.
##
## A candidate site lies in each square of 64 by 64 cells; it is kept where its biome is one a
## culture settles, on land below the high peaks and not too rough, at least 128 cells from every
## site kept before it, in an order the seed shuffles.
extends RefCounted

## The biomes each culture settles; the other biomes hold no settlement.
const CULTURES := {
	"coastfolk": ["beach", "shingle"],
	"steppe_riders": ["cold_steppe", "upland_steppe", "prairie", "savanna"],
	"woodlanders": ["broadleaf_forest", "mixed_woodland", "meadow", "oak_hills", "hill_pasture"],
	"sand_dwellers": ["desert", "dry_scrub", "shrubland", "mesa", "chaparral"],
	"highlanders": ["upland_heath", "pine_highland", "fir_highland", "alpine_meadow"],
	"marsh_folk": ["fen", "swamp", "muskeg", "upland_bog", "highland_marsh"],
	"jungle_folk": ["rainforest", "cloud_forest", "mangrove"],
	"frostfolk": ["tundra", "taiga", "spruce_forest", "bog_tundra", "lichen_highland"],
}
## The continent's squares of candidates along each axis, and their side in cells.
const SQUARES := 32
const SQUARE := 64
const APART := 128.0

## The settlements of the continent `stages` generates, for `seed`: an Array of Dictionaries, each
## with an `id` and a value for every column of the table, positions in cells.
func run(stages: Node, seed: int) -> Array:
	var biomes: PackedStringArray = stages.category_names("biome")
	var cell: Vector3 = stages.cell_size
	var random := RandomNumberGenerator.new()
	var candidates := []
	for y in SQUARES:
		for x in SQUARES:
			random.seed = hash([seed, x, y])
			var at := Vector2(x * SQUARE + 8 + random.randi_range(0, 47), y * SQUARE + 8 + random.randi_range(0, 47))
			var position := Vector3(at.x * cell.x, 0.0, at.y * cell.z)
			var culture := _culture_of(biomes[int(stages.sample("biome", position))])
			if culture.is_empty():
				continue
			var height: float = stages.sample("terrain", position)
			if height < 0.5 or height >= 45.0 or stages.sample("roughness", position) >= 3.0:
				continue
			candidates.append({"order": random.randi(), "at": at, "culture": culture, "seed": random.randi()})
	candidates.sort_custom(func(a: Dictionary, b: Dictionary) -> bool: return a["order"] < b["order"])
	var rows := []
	for candidate: Dictionary in candidates:
		var at: Vector2 = candidate["at"]
		if rows.any(func(row: Dictionary) -> bool: return Vector2(row["x"], row["y"]).distance_to(at) < APART):
			continue
		random.seed = candidate["seed"]
		var population := random.randi_range(150, 4149)
		var founded := random.randi_range(0, 499)
		rows.append({
			"id": rows.size(),
			"x": at.x,
			"y": at.y,
			"size": 2 + (2 if population > 1500 else 0) + (2 if population > 3000 else 0),
			"population": population,
			"founded": founded,
			"culture": candidate["culture"],
			"fate": _fate(founded, random.randi_range(0, 999)),
		})
	return rows

func _culture_of(biome: String) -> String:
	for culture: String in CULTURES:
		if CULTURES[culture].has(biome):
			return culture
	return ""

## What became of a settlement founded `founded` years in, by a roll of `fall` in 0 to 999: the old
## ones more often ruined or abandoned.
func _fate(founded: int, fall: int) -> String:
	if founded < 150 and fall < 350:
		return "ruined"
	if (founded < 150 and fall < 500) or (founded >= 150 and fall < 100):
		return "abandoned"
	if fall >= 500 and fall < 700:
		return "declining"
	return "thriving"

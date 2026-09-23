## Writes godot.json: Godot's own FastNoiseLite.get_noise_2d over a matrix of configurations, the
## golden values tests/fastnoise.rs holds the Rust port to.
##
## Run from any folder that holds a project.godot (an empty one will do), with the output path
## after `--`:
##
##     godot --headless --path <folder> --script <this file> -- tests/fixtures/fastnoise/godot.json
##
## Each case is one configuration: every property of the resource as Godot reports it after the
## case's settings, enums as their integers, and its values at the shared points in order.
extends SceneTree

const PROPERTIES := [
	"noise_type", "seed", "frequency", "offset",
	"fractal_type", "fractal_octaves", "fractal_lacunarity", "fractal_gain",
	"fractal_weighted_strength", "fractal_ping_pong_strength",
	"cellular_distance_function", "cellular_return_type", "cellular_jitter",
	"domain_warp_enabled", "domain_warp_type", "domain_warp_amplitude", "domain_warp_frequency",
	"domain_warp_fractal_type", "domain_warp_fractal_octaves", "domain_warp_fractal_lacunarity",
	"domain_warp_fractal_gain",
]

## Negative, fractional, on and near lattice lines, and far from the origin; `_points` adds a
## spread of irregular ones.
const CHOSEN_POINTS := [
	[0.0, 0.0], [0.5, 0.25], [-3.7, 11.2], [1.0, -1.0], [-2.0, -3.0], [100.0, 100.0],
	[-0.001, 0.001], [37.5, -1234.5], [-250.25, -250.75], [12345.6, -6543.21],
	[-12345.6, 789.01], [10000.3, 20000.7], [3.14159, 2.71828], [-99.99, 0.5],
]

const SEEDS := [0, 1337, -1, -987654, 2147483647, -2147483648, 42]
const FREQUENCIES := [0.01, 0.1, 0.37, 1.0]

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() != 1:
		push_error("generate.gd needs the output path after --")
		quit(1)
		return
	var lines := PackedStringArray()
	var points := _points()
	for settings in _cases():
		var noise := FastNoiseLite.new()
		for property in settings:
			noise.set(property, settings[property])
		var config := {}
		for property in PROPERTIES:
			var value = noise.get(property)
			config[property] = [value.x, value.y, value.z] if value is Vector3 else value
		var values := []
		for point in points:
			values.append(noise.get_noise_2d(point[0], point[1]))
		lines.append(JSON.stringify({"config": config, "values": values}, "", true, true))
	var file := FileAccess.open(args[0], FileAccess.WRITE)
	if file == null:
		push_error("cannot write %s: %s" % [args[0], error_string(FileAccess.get_open_error())])
		quit(1)
		return
	file.store_string("{\"godot\": %s,\n\"points\": %s,\n\"cases\": [\n%s\n]}\n" % [
		JSON.stringify(Engine.get_version_info()["string"]),
		JSON.stringify(points, "", true, true),
		",\n".join(lines)])
	file.close()
	print("generate.gd: %d cases at %d points" % [lines.size(), points.size()])
	quit(0)

func _points() -> Array:
	var points := CHOSEN_POINTS.duplicate()
	for k in 18:
		points.append([k * 37.77 - 301.3, 123.45 - k * k * 1.913])
	return points

## The settings of each case: properties it does not name keep Godot's defaults.
func _cases() -> Array:
	var cases: Array = [{}]
	var n := 0
	# Every noise type under every fractal type, over several seeds and frequencies.
	for noise_type in 6:
		for fractal_type in 4:
			cases.append({
				"noise_type": noise_type, "fractal_type": fractal_type,
				"seed": SEEDS[n % SEEDS.size()], "frequency": FREQUENCIES[n % FREQUENCIES.size()],
				"fractal_octaves": [1, 3, 5, 8][n % 4],
			})
			n += 1
	# Every cellular distance function with every return type.
	for distance in 4:
		for return_type in 7:
			cases.append({
				"noise_type": FastNoiseLite.TYPE_CELLULAR, "fractal_type": FastNoiseLite.FRACTAL_NONE,
				"cellular_distance_function": distance, "cellular_return_type": return_type,
				"seed": SEEDS[n % SEEDS.size()], "frequency": FREQUENCIES[n % FREQUENCIES.size()],
			})
			n += 1
	for jitter in [0.0, 0.45, 1.5]:
		cases.append({
			"noise_type": FastNoiseLite.TYPE_CELLULAR, "cellular_jitter": jitter,
			"cellular_return_type": FastNoiseLite.RETURN_DISTANCE2_SUB, "frequency": 0.1, "seed": 9,
		})
	# Weighted strength, lacunarity and gain (a negative gain too) under each fractal.
	for fractal_type in [FastNoiseLite.FRACTAL_FBM, FastNoiseLite.FRACTAL_RIDGED, FastNoiseLite.FRACTAL_PING_PONG]:
		for noise_type in [FastNoiseLite.TYPE_SIMPLEX, FastNoiseLite.TYPE_PERLIN, FastNoiseLite.TYPE_CELLULAR]:
			cases.append({
				"noise_type": noise_type, "fractal_type": fractal_type,
				"fractal_weighted_strength": 0.7, "fractal_octaves": 4,
				"fractal_lacunarity": 2.3, "fractal_gain": [0.6, -0.4, 0.35][n % 3],
				"frequency": 0.1, "seed": SEEDS[n % SEEDS.size()],
			})
			n += 1
	# Octaves above 1, where FBm's weighting clamps.
	cases.append({
		"noise_type": FastNoiseLite.TYPE_CELLULAR, "fractal_type": FastNoiseLite.FRACTAL_FBM,
		"cellular_distance_function": FastNoiseLite.DISTANCE_HYBRID,
		"cellular_return_type": FastNoiseLite.RETURN_DISTANCE2, "cellular_jitter": 1.5,
		"fractal_weighted_strength": 0.8, "frequency": 0.2,
	})
	for strength in [0.5, 3.3]:
		cases.append({
			"noise_type": FastNoiseLite.TYPE_SIMPLEX_SMOOTH,
			"fractal_type": FastNoiseLite.FRACTAL_PING_PONG, "fractal_ping_pong_strength": strength,
			"frequency": 0.1,
		})
	cases.append({"offset": Vector3(13.25, -7.5, 99.0), "frequency": 0.1})
	cases.append({"offset": Vector3(-1000.5, 2000.125, 0.0), "noise_type": FastNoiseLite.TYPE_VALUE})
	# Every warp type under every warp fractal type, over the noise types.
	for warp_type in 3:
		for warp_fractal_type in 3:
			cases.append({
				"domain_warp_enabled": true, "domain_warp_type": warp_type,
				"domain_warp_fractal_type": warp_fractal_type,
				"noise_type": n % 6, "fractal_type": n % 4,
				"seed": SEEDS[n % SEEDS.size()], "frequency": FREQUENCIES[n % FREQUENCIES.size()],
				"domain_warp_amplitude": [30.0, 4.5, 120.0][warp_type],
				"domain_warp_frequency": [0.05, 0.3, 0.012][warp_fractal_type],
				"domain_warp_fractal_octaves": [5, 3, 1][warp_type],
				"domain_warp_fractal_lacunarity": [6.0, 2.5][n % 2],
				"domain_warp_fractal_gain": [0.5, 0.8][n % 2],
				"offset": Vector3(0.0, 0.0, 0.0) if n % 2 == 0 else Vector3(5.5, -3.25, 0.0),
			})
			n += 1
	cases.append({"domain_warp_enabled": true})
	return cases

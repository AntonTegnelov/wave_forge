## Names a location table's sites through Godot's translations and checks what they say.
##
## Run by `../verify.sh` after `verify_sound.gd`. A site of kind `stone_circle` is named by the key
## `wf-place-stone-circle` with its `region_x`, `region_y` and `index`, never a finished string; a
## translation the game registers for the `wave_forge` context turns it into words with
## `tr(name_key, "wave_forge").format(name_args)`.
extends SceneTree

const CELLS := 8
const TIMEOUT_S := 30.0

var world: Node
var started_usec := 0

func _initialize() -> void:
	var english := Translation.new()
	english.locale = "en"
	english.add_message("wf-place-stone-circle", "Stone Circle {index} of {region_x}, {region_y}", "wave_forge")
	TranslationServer.add_translation(english)
	TranslationServer.set_locale("en")
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://names.world.ron"
	world.targets = PackedStringArray(["places"])
	world.seed = 5
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.view_radius = 3
	world.collider_radius = -1
	root.add_child(world)
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3.ZERO)
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("no site arrived")
		return true
	var sites := []
	for y in range(-3, 4):
		for x in range(-3, 4):
			sites.append_array(world.sites("places", Vector3i(x, y, 0)))
	if sites.is_empty():
		return false
	for site: Dictionary in sites:
		if site["name_key"] != "wf-place-stone-circle":
			_fail("a stone circle is named by %s" % site["name_key"])
			return true
		var args: Dictionary = site["name_args"]
		if args != {"region_x": site["region"].x, "region_y": site["region"].y, "index": site["index"]}:
			_fail("a stone circle's name has the arguments %s" % [args])
			return true
		var words := tr(site["name_key"], "wave_forge").format(args)
		var expected := "Stone Circle %d of %d, %d" % [site["index"], site["region"].x, site["region"].y]
		if words != expected:
			_fail("a stone circle is called %s, expected %s" % [words, expected])
			return true
	print("verify_names: %d sites named by wf-place-stone-circle, called \"%s\" through a translation" % [sites.size(), tr(sites[0]["name_key"], "wave_forge").format(sites[0]["name_args"])])
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("verify_names: " + message)
	quit(1)

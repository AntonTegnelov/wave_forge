## Names a location table's sites through Godot's translations and checks what they say.
##
## Run by `../verify.sh` after `verify_sound.gd`. A site of kind `stone_circle` is named by the key
## `wf-place-stone-circle` with its `region_x`, `region_y` and `index`, never a finished string; a
## translation the game registers for the `wave_forge` context turns it into words with
## `tr(name_key, "wave_forge").format(name_args)`. The translation is built from the entries the
## plugin lists in the editor's translation templates (`translation_keys.gd`), as a translator
## would fill a template in: every key the pack can give, in its context.
extends SceneTree

const CELLS := 8
const TIMEOUT_S := 30.0
const PACK := "res://names.world.ron"
const TranslationKeys := preload("res://addons/wave_forge/translation_keys.gd")
## What a translator writes for each key of the template.
const ENGLISH := {"wf-place-stone-circle": "Stone Circle {index} of {region_x}, {region_y}"}

var world: Node
var started_usec := 0

func _initialize() -> void:
	var entries := TranslationKeys.entries(PACK)
	if entries != [PackedStringArray(["wf-place-stone-circle", "wave_forge"])]:
		_fail("the template lists %s for the pack" % [entries])
		return
	if not TranslationKeys.entries("res://city.ron").is_empty():
		_fail("the template lists a rule set's strings")
		return
	var english := Translation.new()
	english.locale = "en"
	for entry in entries:
		english.add_message(entry[0], ENGLISH[entry[0]], entry[1])
	TranslationServer.add_translation(english)
	print("verify_names: the translation template lists the pack's one place name, wf-place-stone-circle, in the wave_forge context, and no rule set's strings")
	TranslationServer.set_locale("en")
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = PACK
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
	if not world.water().is_empty():
		_fail("a pack without water has %s" % world.water())
		return
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
	print("verify_names: %d sites named by wf-place-stone-circle, called \"%s\" through the translation built from the template" % [sites.size(), tr(sites[0]["name_key"], "wave_forge").format(sites[0]["name_args"])])
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("verify_names: " + message)
	quit(1)

## Starts a WaveForgeWorld beside a camera and never calls follow, and checks it generates around the
## camera, takes a chunk's tiles from elsewhere, keeps its compiled kernels where `kernel_cache`
## says, and stops following once a script calls follow.
##
## Run by `../verify.sh` after `verify.gd`. A world in a scene with a Camera3D generates the chunk the
## camera stands over with no code, as `follow_camera` promises; tiles imported into that chunk,
## another peer's say, replace its own and fire `chunk_updated`, while tiles of the wrong count or a
## tile the rule set lacks are refused; the kernels it compiled are in the cache directory
## afterwards; and a call to `follow` turns `follow_camera` off, so the script follows from then on.
extends SceneTree

const CELLS := 8
const CELL_SIZE := 2.0
const CACHE := "user://verify_follow_kernels"
## Building the device and compiling kernels is loading, not play.
const TIMEOUT_S := 180.0

var world: Node
var camera: Camera3D
var wanted: Vector3i
var arrived := false
## The tiles imported into the chunk under the camera, once they are, and its hash before.
var imported := PackedInt32Array()
var hash_before := ""
var started_usec := 0

func _initialize() -> void:
	_clear(CACHE)
	world = ClassDB.instantiate("WaveForgeWorld")
	if not world.follow_camera:
		_fail("follow_camera is off by default")
		return
	if world.kernel_cache != "user://wave_forge/kernels":
		_fail("kernel_cache defaults to %s" % world.kernel_cache)
		return
	world.seed = 7
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = Vector3(CELL_SIZE, CELL_SIZE, CELL_SIZE)
	world.view_radius = 1
	world.collider_radius = -1
	world.kernel_cache = CACHE
	world.chunk_updated.connect(_on_chunk_updated)
	world.generation_failed.connect(func(reason: String) -> void: _fail("generation failed: %s" % reason))
	root.add_child(world)
	camera = Camera3D.new()
	# Five chunks out along x and three along z, a few cells up; the root stands at the origin.
	camera.position = Vector3(5.5 * CELLS * CELL_SIZE, 6, 3.5 * CELLS * CELL_SIZE)
	camera.current = true
	root.add_child(camera)
	wanted = world.chunk_at(camera.position)
	if not world.load_rules(FileAccess.get_file_as_string("res://rules.ron")):
		_fail("the rule set was refused")
		return
	if not world.start():
		_fail("generation did not start")
		return
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("chunk %s under the camera never arrived" % wanted)
		return true
	if not arrived:
		return false
	if imported.is_empty():
		return _import()
	if world.tiles_at(wanted) != imported:
		_fail("chunk %s does not hold the imported tiles" % wanted)
		return true
	var hash_after: String = world.chunk_hash(wanted)
	if hash_after.length() != 16 or hash_after == hash_before:
		_fail("chunk %s hashed %s before its import and %s after" % [wanted, hash_before, hash_after])
		return true
	var cached := DirAccess.get_files_at(CACHE)
	if cached.is_empty():
		_fail("no kernels were kept in %s" % CACHE)
		return true
	world.follow(Vector3.ZERO)
	if world.follow_camera:
		_fail("follow did not turn follow_camera off")
		return true
	print("verify_follow: the world generated chunk %s under the camera with no follow, took that chunk's tiles from elsewhere and refused ill-formed ones, its hash changing with them, kept its kernels in %d file(s), and a script's follow took over" % [wanted, cached.size()])
	_clear(CACHE)
	quit(0)
	return true

## Imports into the chunk under the camera every one of its tiles turned into the next of the rule
## set, after two ill-formed imports that must be refused; returns whether the check is over.
func _import() -> bool:
	var own: PackedInt32Array = world.tiles_at(wanted)
	var count: int = world.tile_count()
	var next := PackedInt32Array()
	for tile in own:
		next.append((tile + 1) % count)
	var short := next.duplicate()
	short.remove_at(0)
	var unknown := next.duplicate()
	unknown[0] = count
	if world.import_tiles(wanted, short) or world.import_tiles(wanted, unknown):
		_fail("tiles of the wrong count or a tile the rule set lacks were taken")
		return true
	if world.tiles_at(wanted) != own:
		_fail("a refused import changed chunk %s" % wanted)
		return true
	hash_before = world.chunk_hash(wanted)
	if hash_before.length() != 16 or not world.chunk_hash(Vector3i(99, 99, 0)).is_empty():
		_fail("chunk %s hashed %s, and an ungenerated one %s" % [wanted, hash_before, world.chunk_hash(Vector3i(99, 99, 0))])
		return true
	arrived = false
	imported = next
	if not world.import_tiles(wanted, imported):
		_fail("the imported tiles were refused")
		return true
	return false

func _on_chunk_updated(chunk: Vector3i) -> void:
	if chunk == wanted:
		arrived = true

## Removes the cache directory `path` and what it holds, if it is there.
func _clear(path: String) -> void:
	if not DirAccess.dir_exists_absolute(path):
		return
	for file in DirAccess.get_files_at(path):
		DirAccess.remove_absolute(path.path_join(file))
	DirAccess.remove_absolute(path)

func _fail(message: String) -> void:
	printerr("verify_follow: FAILED: %s" % message)
	quit(1)

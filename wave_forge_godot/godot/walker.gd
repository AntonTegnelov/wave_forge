extends CharacterBody3D
## A first-person walker for the checks and examples to stand in a generated world (developer
## tooling, not part of the addon: a game brings its own player): drop `walker.tscn` into a scene
## with a WaveForgeStages node, whose `follow_camera` then follows its camera. It holds still until the ground and its body are there under it, so it never falls
## through a world still generating, and then stands on the ground. WASD or the arrow keys walk,
## Space jumps, a click captures the mouse to look around and Escape releases it.

## Walking speed, in units a second; 4.2 is a brisk walk (P1).
@export var speed := 4.2
## How high a jump starts, in units a second upward.
@export var jump := 4.5
## How far the view turns per pixel the mouse moves, in radians.
@export var look := 0.003

var _camera: Camera3D
var _world: Node
var _standing := false

func _ready() -> void:
	_camera = $Camera3D
	for action: String in ["forward", "back", "left", "right", "jump"]:
		_bind("wave_forge_" + action, {"forward": [KEY_W, KEY_UP], "back": [KEY_S, KEY_DOWN], "left": [KEY_A, KEY_LEFT], "right": [KEY_D, KEY_RIGHT], "jump": [KEY_SPACE]}[action])

## The WaveForgeStages node of the scene, found the first time it is there.
func world() -> Node:
	if _world == null or not is_instance_valid(_world):
		var found := get_tree().root.find_children("*", "WaveForgeStages", true, false)
		_world = null if found.is_empty() else found[0]
	return _world

## Whether the walker stands on the ground, which it does once the ground and its body are there.
func standing() -> bool:
	return _standing

func _physics_process(delta: float) -> void:
	if not _standing:
		_land()
		return
	var input := Input.get_vector("wave_forge_left", "wave_forge_right", "wave_forge_forward", "wave_forge_back")
	var along := (transform.basis * Vector3(input.x, 0.0, input.y)).normalized() * speed
	velocity.x = along.x
	velocity.z = along.z
	if is_on_floor():
		if Input.is_action_just_pressed("wave_forge_jump"):
			velocity.y = jump
	else:
		velocity.y -= ProjectSettings.get_setting("physics/3d/default_gravity") * delta
	move_and_slide()

## Stands the walker on the ground once the ground there has its height and its body.
func _land() -> void:
	var stages := world()
	if stages == null:
		return
	var ground: float = stages.ground_height(global_position)
	var size: Vector3 = Vector3(stages.chunk_cells.x, 0.0, stages.chunk_cells.y) * stages.cell_size
	var chunk := Vector3i(floori(global_position.x / size.x), floori(global_position.z / size.z), 0)
	if is_nan(ground) or not stages.collider_chunks().has(chunk):
		return
	global_position.y = ground + 0.05
	velocity = Vector3.ZERO
	_standing = true

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.pressed:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	elif event is InputEventKey and event.pressed and event.keycode == KEY_ESCAPE:
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	elif event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		rotate_y(-event.relative.x * look)
		_camera.rotate_x(-event.relative.y * look)
		_camera.rotation.x = clampf(_camera.rotation.x, -1.4, 1.4)

## Adds `action` with `keys` to the project's input map, if the project has not its own.
func _bind(action: String, keys: Array) -> void:
	if InputMap.has_action(action):
		return
	InputMap.add_action(action)
	for key: Key in keys:
		var event := InputEventKey.new()
		event.physical_keycode = key
		InputMap.action_add_event(action, event)

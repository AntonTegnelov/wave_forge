## A player who walks with W, A, S and D, looks with the mouse and jumps with space. It waits in the
## air until the ground under it has its collider, then falls onto it.
extends CharacterBody3D

const SPEED := 6.0
const JUMP := 6.0
const GRAVITY := 18.0

## Whether it may fall yet: the scene sets it once the ground under it can be stood on.
var landed_allowed := false
var camera := Camera3D.new()

func _ready() -> void:
	var shape := CollisionShape3D.new()
	shape.shape = CapsuleShape3D.new()
	add_child(shape)
	camera.position = Vector3(0, 0.7, 0)
	camera.far = 2000.0
	add_child(camera)
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		rotate_y(-event.relative.x * 0.003)
		camera.rotate_x(-event.relative.y * 0.003)
		camera.rotation.x = clampf(camera.rotation.x, -1.4, 1.4)
	if event.is_action_pressed("ui_cancel"):
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE

func _physics_process(delta: float) -> void:
	if not landed_allowed:
		return
	var input := Vector2(
		float(Input.is_physical_key_pressed(KEY_D)) - float(Input.is_physical_key_pressed(KEY_A)),
		float(Input.is_physical_key_pressed(KEY_S)) - float(Input.is_physical_key_pressed(KEY_W)))
	var direction := (transform.basis * Vector3(input.x, 0, input.y)).normalized()
	velocity.x = direction.x * SPEED
	velocity.z = direction.z * SPEED
	velocity.y -= GRAVITY * delta
	if is_on_floor() and Input.is_physical_key_pressed(KEY_SPACE):
		velocity.y = JUMP
	move_and_slide()

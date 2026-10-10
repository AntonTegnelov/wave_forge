## A camera to look around the demo with: hold the right mouse button to look, WASD to move, Q and
## E to sink and rise, Shift to go faster.
extends Camera3D

## Units a second.
@export var speed := 20.0
## Degrees a pixel of mouse movement turns the view.
@export var sensitivity := 0.2

var looking := false

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_RIGHT:
		looking = event.pressed
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED if looking else Input.MOUSE_MODE_VISIBLE
	elif event is InputEventMouseMotion and looking:
		rotation_degrees.y -= event.relative.x * sensitivity
		rotation_degrees.x = clampf(rotation_degrees.x - event.relative.y * sensitivity, -89.0, 89.0)

func _process(delta: float) -> void:
	var direction := Vector3(
		Input.get_axis(&"ui_left", &"ui_right") + float(Input.is_key_pressed(KEY_D)) - float(Input.is_key_pressed(KEY_A)),
		float(Input.is_key_pressed(KEY_E)) - float(Input.is_key_pressed(KEY_Q)),
		Input.get_axis(&"ui_up", &"ui_down") + float(Input.is_key_pressed(KEY_S)) - float(Input.is_key_pressed(KEY_W)),
	)
	var fast := 4.0 if Input.is_key_pressed(KEY_SHIFT) else 1.0
	translate(direction.limit_length(1.0) * speed * fast * delta)

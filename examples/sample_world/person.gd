extends Node3D
## One of the townsfolk: walks along the navigation map to a point `townsfolk.gd` picks, at a
## stroll, turning to face where it goes, and asks for the next point when it gets there. Swap the
## capsule in `person.tscn` for a character of your own; nothing else depends on it.

signal arrived(person: Node3D)

## Walking speed, in metres a second.
@export var speed := 1.4

@onready var _agent: NavigationAgent3D = $NavigationAgent3D

func _ready() -> void:
	_agent.navigation_finished.connect(func() -> void: arrived.emit(self))

## Sets off for `target`, a point on the navigation map.
func walk_to(target: Vector3) -> void:
	_agent.target_position = target

func _physics_process(delta: float) -> void:
	if _agent.is_navigation_finished():
		return
	var next := _agent.get_next_path_position()
	var step := next - global_position
	if Vector2(step.x, step.z).length() > 0.01:
		look_at(Vector3(next.x, global_position.y, next.z), Vector3.UP)
	global_position = global_position.move_toward(next, speed * delta)

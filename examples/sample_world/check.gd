## Checks the sample world headless: `godot --headless --path . --script check.gd`. It runs
## `main.tscn` until the townsfolk are out, then for `WALK_S` more seconds, and checks that the city
## stands in grass with trees around it, the day goes on, and every one of the townsfolk walks
## along the navigation map towards a point of the city's streets.
extends SceneTree

const TIMEOUT_S := 300.0
const WALK_S := 5.0

var main: Node
var started_usec := 0
var walking_usec := 0
## Where each of the townsfolk was last frame, and how far each has walked since the walk began.
var last := {}
var walked := {}
var time_before := 0.0

func _initialize() -> void:
	main = (load("res://main.tscn") as PackedScene).instantiate()
	root.add_child(main)
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	var townsfolk: Node3D = main.get_node("Townsfolk")
	if walking_usec == 0:
		if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
			return _fail("the townsfolk were not out after %d s" % TIMEOUT_S)
		if townsfolk.get_child_count() < townsfolk.count:
			return false
		for person: Node3D in townsfolk.get_children():
			last[person] = person.global_position
			walked[person] = 0.0
		time_before = main.get_node("DayNight").time
		walking_usec = Time.get_ticks_usec()
		return false
	for person: Node3D in townsfolk.get_children():
		walked[person] += person.global_position.distance_to(last[person])
		last[person] = person.global_position
	if (Time.get_ticks_usec() - walking_usec) / 1e6 < WALK_S:
		return false
	var city: Node = main.get_node("City")
	if city.grass_chunks().is_empty():
		return _fail("no chunk has grass")
	var stats: Dictionary = city.stats()
	if stats["placed_instances"] == 0:
		return _fail("nothing is placed: %s" % stats)
	var chunk_size: Vector3 = Vector3(city.chunk_cells) * city.cell_size
	var centre: Vector2 = townsfolk.streets.get_center()
	var town: Dictionary = city.town("city", Vector3i(floori(centre.x / chunk_size.x), floori(centre.y / chunk_size.z), 0))
	if town.is_empty() or town["tiles"].is_empty():
		return _fail("the city's middle chunk has no town")
	var day: Node = main.get_node("DayNight")
	if day.time <= time_before:
		return _fail("the time of day stayed at %.4f" % day.time)
	var map: RID = main.get_world_3d().navigation_map
	var streets: Rect2 = townsfolk.streets.grow(1.0)
	var distances := PackedFloat64Array()
	for person: Node3D in townsfolk.get_children():
		var at := person.global_position
		var moved: float = walked[person]
		if moved < 1.0:
			return _fail("a person walked %.2f m in %.0f s" % [moved, WALK_S])
		var target: Vector3 = person.get_node("NavigationAgent3D").target_position
		if not streets.has_point(Vector2(target.x, target.z)):
			return _fail("a person walks to %s, outside the city's streets" % target)
		var off := at.distance_to(NavigationServer3D.map_get_closest_point(map, at))
		if off > 0.5:
			return _fail("a person stands %.2f m off the navigation map, at %s" % [off, at])
		distances.append(moved)
	distances.sort()
	print("check: the city stands in grass, with %d trees and modules placed; the day went from %.3f to %.3f; %d townsfolk walked %.1f to %.1f m in %.0f s along the navigation map to points of the city's streets" % [
		stats["placed_instances"], time_before, day.time, distances.size(), distances[0], distances[distances.size() - 1], WALK_S])
	quit(0)
	return true

func _fail(message: String) -> bool:
	printerr("check: " + message)
	quit(1)
	return true

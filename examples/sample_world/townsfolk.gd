extends Node3D
## The townsfolk of the small city. Once the city's site is generated and every chunk of it has
## navigation, `count` people (`person.tscn`) stand at random points of its streets, and each walks
## to another whenever it arrives. The streets are the navigation map's points nearest the city's
## level ground, so the townsfolk keep to street level and the stairs and walkways between.

const PERSON := preload("res://person.tscn")

## How many townsfolk walk the city.
@export var count := 16
## The WaveForgeStages node the city stands in.
@export var stages: Node
## The Locations stage whose `city` site the townsfolk walk.
@export var sites_stage := "places"

## The city's footprint on the ground plane, in metres, and the height of its level ground.
var streets := Rect2()
var street_height := 0.0
var _random := RandomNumberGenerator.new()

func _process(_delta: float) -> void:
	if not _city_navigable():
		return
	for i in count:
		var person: Node3D = PERSON.instantiate()
		add_child(person)
		person.global_position = street_point()
		person.arrived.connect(func(who: Node3D) -> void: who.walk_to(street_point()))
		person.walk_to(street_point())
	set_process(false)

## Whether the city's site is known and every chunk of it is in the navigation map; finds the site
## among the chunks with navigation, the first time they hold it.
func _city_navigable() -> bool:
	var navigable: Array[Vector3i] = stages.navigation_chunks()
	if streets.size == Vector2.ZERO:
		for chunk in navigable:
			for site: Dictionary in stages.sites(sites_stage, chunk):
				if site["kind"] == "city":
					_take_site(site)
		if streets.size == Vector2.ZERO:
			return false
	var chunk_size: Vector3 = Vector3(stages.chunk_cells) * stages.cell_size
	var low := Vector2i(floori(streets.position.x / chunk_size.x), floori(streets.position.y / chunk_size.z))
	var high := Vector2i(floori(streets.end.x / chunk_size.x), floori(streets.end.y / chunk_size.z))
	for y in range(low.y, high.y):
		for x in range(low.x, high.x):
			if not navigable.has(Vector3i(x, y, 0)):
				return false
	return true

func _take_site(site: Dictionary) -> void:
	var chunk_size: Vector3 = Vector3(stages.chunk_cells) * stages.cell_size
	var low: Vector2i = site["min"]
	var high: Vector2i = site["max"]
	streets = Rect2(low.x * chunk_size.x, low.y * chunk_size.z, (high.x - low.x) * chunk_size.x, (high.y - low.y) * chunk_size.z)
	street_height = site["height"] * stages.cell_size.y

## A random point of the city's streets on the navigation map.
func street_point() -> Vector3:
	var at := Vector3(_random.randf_range(streets.position.x, streets.end.x), street_height, _random.randf_range(streets.position.y, streets.end.y))
	return NavigationServer3D.map_get_closest_point(get_world_3d().navigation_map, at)

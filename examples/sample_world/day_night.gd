extends Node
## A day and a night over `day_seconds`: the sun turns about the sky from east to west, its light
## warm and low at sunrise and sunset, and the moon lights the night. The sky's procedural material
## draws the sun's disc from the sun's light, so the sky follows it; the sky and the ambient light
## dim with the sun.

## How long a whole day and night lasts, in seconds.
@export var day_seconds := 180.0
## The time of day as a fraction of it: 0 at midnight, 0.25 at sunrise, 0.5 at noon, 0.75 at sunset.
@export_range(0.0, 1.0) var time := 0.3
@export var sun: DirectionalLight3D
@export var moon: DirectionalLight3D
@export var environment: WorldEnvironment

const SUNRISE_COLOUR := Color(1.0, 0.55, 0.3)
const NOON_COLOUR := Color(1.0, 0.96, 0.9)

func _process(delta: float) -> void:
	time = fposmod(time + delta / day_seconds, 1.0)
	# The sun lies on the horizon at sunrise, straight above at noon, on the far horizon at sunset
	# and below the ground at midnight; the moon stands opposite it.
	var turn := (time - 0.25) * TAU
	sun.rotation.x = -turn
	moon.rotation.x = PI - turn
	# How high the sun stands, from -1 at midnight to 1 at noon.
	var height := sin(turn)
	var day := clampf(height * 3.0, 0.0, 1.0)
	sun.light_energy = day * 1.2
	sun.light_color = SUNRISE_COLOUR.lerp(NOON_COLOUR, clampf(height * 2.0, 0.0, 1.0))
	sun.visible = day > 0.0
	moon.light_energy = clampf(-height * 3.0, 0.0, 1.0) * 0.2
	moon.visible = not sun.visible
	var sky := environment.environment.sky.sky_material as ProceduralSkyMaterial
	sky.energy_multiplier = lerpf(0.04, 1.0, clampf(height * 2.0 + 0.3, 0.0, 1.0))
	environment.environment.ambient_light_energy = lerpf(0.08, 0.35, clampf(height * 2.0 + 0.3, 0.0, 1.0))

## The light the look renders judge a terrain under (developer tool; docs/guides/testing.md,
## "Rendering tools"): the addon's `sun_and_sky.tscn`, the sun and sky a game gets from "Add sun and
## sky", with gentle depth fog added. A terrain judged under flat light and a flat background looks
## worse than it is, so the renders that are looked at share this one environment.

const SUN_AND_SKY := preload("res://addons/wave_forge/sun_and_sky.tscn")
## About 5 % of the light fogged at 100 m and 40 % at 1 km.
const FOG_DENSITY := 0.0005

## Adds the sun and the sky under `parent`, with an environment of its own, and returns the sun.
static func add_to(parent: Node) -> DirectionalLight3D:
	var sun_and_sky: Node = SUN_AND_SKY.instantiate()
	var sky: WorldEnvironment = sun_and_sky.get_node("WorldEnvironment")
	var environment: Environment = sky.environment.duplicate(true)
	environment.fog_enabled = true
	environment.fog_density = FOG_DENSITY
	environment.fog_aerial_perspective = 0.5
	# The sky stays the sky; the ground fades toward it.
	environment.fog_sky_affect = 0.0
	sky.environment = environment
	parent.add_child(sun_and_sky)
	return sun_and_sky.get_node("Sun")

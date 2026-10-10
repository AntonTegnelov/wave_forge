# The terrain's look against modern generators

How Wave Forge's generated terrain compares with three references the owner named on 2026-10-04,
what it lacks, and the order of work to close the gap. The work is issues #322 to #329; this page
is why they are what they are. Only the terrain is compared: a scene's atmosphere, its lighting and
fog, is the games' ([vision.md](../product/vision.md#non-goals)).

## The references

What was and was not read: the two Reddit posts through their RSS feeds (the full post text and
every comment), the first gallery image of the first post and the author's node graph from an
earlier post, and the second post's paper, project page and code. The posts' videos and the other
gallery images were not seen. Terrain3D was read from its README, its documentation and its shader
sources on GitHub's main branch.

### A node-graph terrain editor (r/proceduralgeneration, "Terrain Gen update!", 2026-09-14)

A standalone editor in three.js that generates square tiles which blend into each other, with a
continent height map across tiles and fault lines through several. The author's pipeline is "input
noise > warp > erosion > biomes and slopemapped colors > output", and the erosion is "just typical
slow droplet erosion". The colours come from elevation bands painted into biomes, masked by slope,
with procedural rock bands and textures on rock faces. Its nodes include Noise, Warp, Blend, Rock
Hardness, Terrace, Thermal, Climate Bands, Lakes and Height Painter.

Each tile is a 2 049² height field, rendered in 15 to 35 s; the mesh is coarser than the field
(780 000 vertices for three tiles), so its colour and normal detail is finer than its triangles
(inferred from those numbers). What it shows: gullies branching at many scales, channels coloured
from a flow map, strata on cliffs, and rock against vegetation by slope. It is an offline viewer:
no tile size in metres, no vegetation and no real-time levels of detail are shown.

### Stochastic geomorphological transport (McDonald and Cordonnier, SIGGRAPH 2026)

ACM TOG 45(4), a technical papers honourable mention. Elevation evolves by uplift, fluvial erosion
and deposition, debris flows and landslides above the angle of repose; fluvial erosion is the shear
stress form with velocity from momentum conservation rather than the stream power law's slope
assumption. It integrates the transport of water, sediment and debris stochastically, with particles
as flux packets, in parallel. It produces braided rivers, meanders, deltas, alluvial and debris fans
and valley networks at many scales, and its results hold across resolutions.

It runs in CUDA on an RTX 5070 Ti over 16 to 64 km² at 512² to 4 096² cells: 51 ms an iteration at
1 024², and whole runs of 1 to 10 minutes. It is global and iterative over a finite domain. Its
transport kernel (geotransport) is MIT; the full simulation (soillib) is LGPL-3.0. The author says
erosion suits finite maps generated once.

### Terrain3D (MIT, a Godot GDExtension)

- **Mesh:** a geometry clipmap with geomorphing, as in The Witcher 3: flat meshes recentred on the
  camera and displaced in the vertex shader, up to 10 levels, a pixel per vertex at 1 m by default,
  optional tessellation with height displacement, and collision built around the camera.
- **Normals** per fragment from the height map, so distant levels shade correctly.
- **Texturing** by an index map rather than a splat map: per vertex, a base and an overlay texture
  out of 32 PBR sets, a blend, a UV angle and scale, interpolated over the four vertices around a
  fragment and blended by height (`exp2(sharpness * log2(weight + id_w + albedo.a))`).
- **An autoshader** that picks base or overlay by slope and height.
- **Detiling, a steep-slope projection** (one projection snapped to 45°, not triplanar), dual
  scaling for distance, macro variation from two noises, a painted colour map and wetness.
- **Foliage** as a MultiMesh instancer with levels of detail.
- **Lighting** left to the scene's environment.

## What Wave Forge has

- **Scale.** Cells of 2 m; the near ground has a vertex per column, 2 m apart, with levels of
  detail at 1, 2, 4 and 8 and skirts (`src/ground.rs`). The continent's ground reaches three chunks,
  about 48 m, from the player; beyond, the far ground has a vertex per coarse column, 8 m, out to
  about 2 km (`src/far_ground.rs`).
- **Shape.** Field expressions over the built-in `Noise` (fractal value noise, no rotation between
  octaves) and `FastNoise` (Godot's FastNoiseLite exactly, with ridged fractals and domain warp);
  neighbourhood stages Blur, Delta, Erode and Area; per-biome formulas through Match
  ([packs.md](../reference/packs.md)). The continent's ranges, ridges, hills, plateaus and coast and
  the presets' landforms are warped `FastNoise`, ridged where they form crests, shaped with `Pow`
  (#324); the continent's far height leaves the ridges and hills out. `wfc-relief` draws a pack's
  height from above for judging a shape ([testing.md](../guides/testing.md#rendering-tools)).
- **Water and drainage.** Lakes by priority flood and rivers down steepest descent, both region
  jobs, carved by Apply. Gullies from the Erode filter (#325), and on finite worlds drainage by a
  Droplets stage a region at a time (#329), which the continent runs over its whole coarse map
  ([packs.md](../reference/packs.md#droplets)). Rivers follow a priority flood's flow through
  hollows and flats, and the continent's run on through its lakes to the sea (#347); nothing
  transports sediment (#348).
- **Ground shading.** The reference ground shader (#322) takes a palette colour per category per
  vertex and draws it procedurally: borders broken by noise rather than following the 2 m grid,
  rock by the fragment's slope in muted strata, gullies down steep ground finer than a column
  (#339), a broad colour variation, and a fine bump and grain near the camera ([godot.md](../reference/godot.md#ground-and-colliders)). Field stages can hand it
  a cavity, which darkens hollows, and a wetness, which darkens and smooths the ground (#326), and
  a cover, which tints the ground toward the grass where the grass fades out, so its edge leaves no
  ring (#342). It has no textures and no height blending. The far ground is coloured by coarse biomes (#314).
- **Water surfaces.** The sea, and since #328 lakes and rivers from a field of the water's level,
  drawn as a mesh per chunk that runs out under its banks ([packs.md](../reference/packs.md#water-surfaces)).
- **Vegetation.** Scatter as MultiMeshes in a global wind, grass from a cover field within one
  chunk by default, its blades shrinking away over the radius's last chunk.

## The gaps, by look gained per work

1. **Ground shading** (#322, then #323). The references' look is mostly per-pixel colour and
   normal detail. The procedural shader (#322) gives what needs no assets; an optional textured
   path after it (#323) needs a decision on assets. Both shaders, Godot's and Bevy's, have to keep
   in step, and on Compatibility too.
2. **Shape primitives** (#324). Warped, ridged gradient noise instead of axis-aligned value noise
   folded once, now in the continent and the presets.
3. **Local erosion** (#325). Branching gullies are what make the references read as terrain. A
   filter that evaluates each point from the input's smoothed gradient is deterministic and seamless
   in chunks with a bounded reach, which a simulation is not. It is the largest shape change that
   still fits infinite worlds. It does not give true drainage.
4. **Shading channels** (#326). Cavity, curvature and wetness from field stages, passed to the
   shader beside the material ids.
5. **Far ground fidelity** (#327). Ridges in the coarse height, finer far normals, and a grass
   hand-off, for P2's 2 km view.
6. **Water surfaces** (#328). Lakes and rivers drawn, so carved channels stop reading as dry pits;
   done in both engines.
7. **Erosion for finite worlds** (#329, then #348). A region job of droplets, then transport, for
   real drainage on worlds run ahead of time. The droplets are in (#329), about 0.58 s a region of
   512 by 512 columns ([measurements.md](measurements.md), E61), and rivers follow the drainage
   across flats and basins (#347); transport is #348. Its time competes with M1's 10 minutes,
   already exceeded on the desktop (#319).

Not on the list: a clipmap renderer of our own, which [engine-integration.md](../architecture/engine-integration.md)
decided against, with Terrain3D as the route for Godot users who want one; and a ground finer than
the 2 m cell, which is tied to the WFC cell, since detail comes from shading instead.

## A second look, 2026-10-10

With #322 to #329 done the terrain still read as amateurish to the owner, so the hills preset was
rendered four ways under Compatibility: as our render scripts light it (a flat background and
ambient, no shadows, a linear tonemap); under a procedural sky, a sun 28 degrees high with shadows,
a filmic tonemap and gentle fog; with its palette calibrated to real albedo; and at eye level.

- The lighting our tools use makes any terrain look like a toy: under the fair environment the same
  ground reads as natural light on hills. So we cannot judge the terrain until our renders are fair
  (#357).
- At eye level what gives it away is not the lighting: the stand-in cone trees, one size and evenly
  spread; the ground up close, one colour per material with blotchy noise, like smeared paint; a
  meadow with almost no visible grass; and the world ending against the sky about 70 m out, since no
  preset has a far ground.
- The palettes are about twice as bright as real ground (Far Cry 3 calibrated its materials against
  a colour chart for this reason), and the presets leave the cavity, wetness and cover channels
  unused.

The gaps, by look gained per work, each an issue:

1. **Fair reference renders** (#357): sky, low sun with shadows, filmic tonemap, fixed cameras
   including eye level. Tooling only.
2. **A far ground in every preset** (#358), so the world does not visibly end.
3. **Calibrated colours and ramps per material** (#359), and the presets using their channels.
4. **Grass that reads as a meadow** (#361): clumps, colour variation, bases tinted by the ground,
   density thinning with distance.
5. **Ground detail up close** (#360): per-material micro-height driving the near normal and
   height-based blending between materials, without textures; #323 remains the textured path.
6. **Vegetation that varies** (#364): Scatter density from a field, variation per instance and
   clump, several kinds per biome.
7. **Preset landforms with structure** (#366): ridgelines, valleys and flat floors under the noise.
8. **Erosion data maps** (#362): the Erode stage's creases and the Droplets stage's flow and
   deposition as fields, for colour and placement.
9. **Talus** (#363): debris at the angle of repose below steep ground.
10. **Water by depth** (#365): tint, shoreline and wet banks from the depth we already know.
11. **Large-scale occlusion** (#367): a horizon-angle field for ambient occlusion near and far.

Three things only the owner can decide are in #368: the vegetation and rock models (stand-in
cones are the loudest sign of a demo), whether preset scenes ship a light and an environment as a
starting point, and the textured ground (#323). A scene's atmosphere stays out of scope: the
references' look leans on it heavily (Ghost of Tsushima's tonemapping and atmosphere,
https://www.advances.realtimerendering.com/s2021/jpatry_advances2021.pdf), which is why our own
renders have to give the terrain a fair light before it is judged.

## Licences to watch

runevision's erosion filter is MPL-2.0, so Wave Forge writes its own from the published
description unless the owner decides otherwise. Terrain3D and geotransport are MIT. soillib is
LGPL-3.0.

## Sources

- r/proceduralgeneration posts 1wg6irm ("Terrain Gen update!"), 1vjjt4u and 1uo5wze, read through
  their RSS feeds.
- McDonald and Cordonnier, "Stochastic Geomorphological Transport for Terrain Erosion Simulation",
  ACM TOG 45(4), SIGGRAPH 2026: https://erosiv.studio/publications/stochastic-geomorphological-transport
  and https://github.com/erosiv/geotransport.
- runevision, "Fast and Gorgeous Erosion Filter", March 2026:
  https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html
- Terrain3D: https://github.com/TokisanGames/Terrain3D (README, `doc/docs/`, `src/shaders/`).

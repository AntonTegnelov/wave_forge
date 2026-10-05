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
  about 48 m, from the player; beyond, the far ground has a vertex per 16 m out to about 2 km
  (`src/far_ground.rs`).
- **Shape.** Field expressions over the built-in `Noise` (fractal value noise, no rotation between
  octaves) and `FastNoise` (Godot's FastNoiseLite exactly, with ridged fractals and domain warp);
  neighbourhood stages Blur, Delta and Area; per-biome formulas through Match
  ([packs.md](../reference/packs.md)). The continent uses only the built-in noise; its ridges are
  one fold of the summed noise, and its far height leaves the ridges and hills out.
- **Water and drainage.** Lakes by priority flood and rivers down steepest descent, both region
  jobs, carved by Apply. No erosion of any kind; [stages.md](../architecture/stages.md) records
  that no stage kind is global.
- **Ground shading.** The reference ground shader draws a flat palette colour per category per
  vertex, blended bilinearly, with constant roughness and per-vertex normals: no textures, no detail
  normal, no slope logic in the fragment, no height blending, no macro variation. Material borders
  follow the 2 m grid as blurred staircases. The far ground is coloured by coarse biomes (#314).
- **Water surfaces.** The sea only; lakes and rivers are not drawn.
- **Vegetation.** Scatter as MultiMeshes in a global wind, grass from a cover field within one
  chunk by default.

## The gaps, by look gained per work

1. **Ground shading** (#322, then #323). The references' look is mostly per-pixel colour and
   normal detail; ours is one colour per category with staircase borders. A procedural shader
   (rock by the fragment's slope, transitions broken by noise and height, macro variation, a detail
   normal, strata) needs no assets; an optional textured path after it needs a decision on assets.
   Both shaders, Godot's and Bevy's, have to keep in step, and on Compatibility too.
2. **Shape primitives** (#324). Warped, ridged gradient noise instead of axis-aligned value noise
   folded once. The noises exist already; the packs do not use them.
3. **Local erosion** (#325). Branching gullies are what make the references read as terrain. A
   filter that evaluates each point from the input's smoothed gradient is deterministic and seamless
   in chunks with a bounded reach, which a simulation is not. It is the largest shape change that
   still fits infinite worlds. It does not give true drainage.
4. **Shading channels** (#326). Cavity, curvature and wetness from field stages, passed to the
   shader beside the material ids.
5. **Far ground fidelity** (#327). Ridges in the coarse height, finer far normals, and a grass
   hand-off, for P2's 2 km view.
6. **Water surfaces** (#328). Lakes and rivers drawn, so carved channels stop reading as dry pits.
7. **Erosion for finite worlds** (#329). A region job of droplets, then transport, for real
   drainage on worlds run ahead of time. The largest item, and the one that pays off only on finite
   worlds; its time competes with M1's 10 minutes, already exceeded on the desktop (#319).

Not on the list: a clipmap renderer of our own, which [engine-integration.md](../architecture/engine-integration.md)
decided against, with Terrain3D as the route for Godot users who want one; and a ground finer than
the 2 m cell, which is tied to the WFC cell, since detail comes from shading instead.

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

## Whether a node's target stages have generated everything around the followed chunk, for the
## checks and renders that wait for a preset to settle (developer tool).

## Whether every target of `stages`, whose `stats()` are `stats`, has generated what it will. A
## target without a radius of its own has generated every chunk within the view's radius. One with
## a radius of its own has it in chunks of the lattice, which a coarse stage covers with an unknown
## number of its own, so it has to have stopped growing: `last` holds each target's products on the
## frame before and is updated to this frame's. The far ground has to be drawn too.
static func all_generated(stages: Node, stats: Dictionary, last: Dictionary) -> bool:
	var generated := true
	for target: String in stages.targets:
		# A stage that has not run yet has no entry.
		var products: int = stats["stages"][target]["products"] if stats["stages"].has(target) else 0
		if stages.target_radii.has(target):
			generated = generated and products > 0 and products == last.get(target, -1)
		else:
			generated = generated and products >= (2 * stages.view_radius + 1) * (2 * stages.view_radius + 1)
		last[target] = products
	return generated and stats["pending_far_grounds"] == 0

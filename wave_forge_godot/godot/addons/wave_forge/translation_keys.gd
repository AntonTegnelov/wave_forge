## The place names a pack file can give, as Godot's translation templates list strings: a message
## id and a context each (docs/reference/godot.md, "Place names in translation templates").
## `translation_parser.gd` hands them to the editor's template generation; a game or a check calls
## this directly. A rule set shares the `.ron` extension, so only `*.world.ron` files are packs.

## The context place names are translated in.
const CONTEXT := "wave_forge"

## One entry per name key the pack file at `path` can give, each its key and the context; none for
## a file that is not a pack file.
static func entries(path: String) -> Array[PackedStringArray]:
	var found: Array[PackedStringArray] = []
	if not path.ends_with(".world.ron"):
		return found
	for key in WaveForgeStages.pack_name_keys(FileAccess.get_file_as_string(path)):
		found.append(PackedStringArray([key, CONTEXT]))
	return found

@tool
extends EditorTranslationParserPlugin
## Lists the place names of the pack files in a project's template generation (Project Settings,
## Localization, Template Generation), each a name key in the `wave_forge` context, so a translator
## gets every place a pack can name (`translation_keys.gd`).

const TranslationKeys := preload("res://addons/wave_forge/translation_keys.gd")

func _get_recognized_extensions() -> PackedStringArray:
	return PackedStringArray(["ron"])

func _parse_file(path: String) -> Array[PackedStringArray]:
	return TranslationKeys.entries(path)

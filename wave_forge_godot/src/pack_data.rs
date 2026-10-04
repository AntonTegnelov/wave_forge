//! A pack as the plain data GDScript edits (docs/reference/godot.md, "Packs as data"): Dictionaries,
//! Arrays, Strings, numbers and booleans in the shape of the library's pack types, read from a pack
//! file's text and written back as one, so a stack of stages edited in the editor saves a pack the
//! library reads. The data goes through `serde_json`'s plain values, the library's serde types
//! deciding what is valid.

use godot::prelude::*;
use serde_json::{Map, Number, Value};
use wave_forge::stages::{Pack, PackFile};

/// The pack of `text` as plain data.
///
/// # Errors
/// The library's reason, if `text` is not a valid pack.
pub(crate) fn pack_data(text: &str) -> Result<VarDictionary, String> {
    let file = PackFile::read(text).map_err(|error| error.to_string())?;
    Pack::from_file(file.clone()).map_err(|error| error.to_string())?;
    let value = serde_json::to_value(&file).map_err(|error| error.to_string())?;
    match from_value(&value).try_to::<VarDictionary>() {
        Ok(data) => Ok(data),
        Err(_) => unreachable!("a pack file serializes as a map"),
    }
}

/// The text of the pack `data` holds, as a pack file is written.
///
/// # Errors
/// Where `data` holds something no pack holds, or the library's reason if it is no valid pack.
pub(crate) fn pack_text(data: &VarDictionary) -> Result<String, String> {
    let value = to_value(&data.to_variant(), "pack")?;
    let file: PackFile = serde_json::from_value(value).map_err(|error| error.to_string())?;
    Pack::from_file(file.clone()).map_err(|error| error.to_string())?;
    Ok(file.to_text())
}

/// `variant` as a plain value, at `at` in the data for an error. A whole floating-point number is
/// an integer, since a number typed in Godot's inspector or read from JSON may be either, and the
/// library's float fields take integers too.
fn to_value(variant: &Variant, at: &str) -> Result<Value, String> {
    let list = |items: Vec<Variant>| -> Result<Value, String> {
        items
            .iter()
            .enumerate()
            .map(|(index, item)| to_value(item, &format!("{at}[{index}]")))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array)
    };
    Ok(match variant.get_type() {
        VariantType::NIL => Value::Null,
        VariantType::BOOL => Value::Bool(variant.to::<bool>()),
        VariantType::INT => Value::from(variant.to::<i64>()),
        VariantType::FLOAT => {
            let float = variant.to::<f64>();
            if !float.is_finite() {
                return Err(format!("{at} is {float}, not a finite number"));
            }
            if float.fract() == 0.0 && float.abs() < 9.0e15 {
                Value::from(float as i64)
            } else {
                Value::Number(Number::from_f64(float).expect("a finite number"))
            }
        }
        VariantType::STRING | VariantType::STRING_NAME => Value::String(variant.to::<String>()),
        VariantType::ARRAY => list(variant.to::<VarArray>().iter_shared().collect())?,
        VariantType::PACKED_STRING_ARRAY => list(
            variant
                .to::<PackedStringArray>()
                .as_slice()
                .iter()
                .map(|item| item.to_variant())
                .collect(),
        )?,
        VariantType::PACKED_INT32_ARRAY => list(
            variant
                .to::<PackedInt32Array>()
                .as_slice()
                .iter()
                .map(|item| item.to_variant())
                .collect(),
        )?,
        VariantType::PACKED_INT64_ARRAY => list(
            variant
                .to::<PackedInt64Array>()
                .as_slice()
                .iter()
                .map(|item| item.to_variant())
                .collect(),
        )?,
        VariantType::PACKED_FLOAT32_ARRAY => list(
            variant
                .to::<PackedFloat32Array>()
                .as_slice()
                .iter()
                .map(|item| item.to_variant())
                .collect(),
        )?,
        VariantType::PACKED_FLOAT64_ARRAY => list(
            variant
                .to::<PackedFloat64Array>()
                .as_slice()
                .iter()
                .map(|item| item.to_variant())
                .collect(),
        )?,
        VariantType::DICTIONARY => {
            let mut fields = Map::new();
            for (key, value) in variant.to::<VarDictionary>().iter_shared() {
                if !matches!(
                    key.get_type(),
                    VariantType::STRING | VariantType::STRING_NAME
                ) {
                    return Err(format!("{at} has the key {key}, which is not a name"));
                }
                let key = key.to::<String>();
                let value = to_value(&value, &format!("{at}.{key}"))?;
                fields.insert(key, value);
            }
            Value::Object(fields)
        }
        other => return Err(format!("{at} is a {other:?}, which no pack holds")),
    })
}

/// `value` as GDScript holds it: an object a Dictionary, a list an Array, an integer an int.
fn from_value(value: &Value) -> Variant {
    match value {
        Value::Null => Variant::nil(),
        Value::Bool(flag) => flag.to_variant(),
        Value::Number(number) => match number.as_i64() {
            Some(whole) => whole.to_variant(),
            None => number.as_f64().expect("a JSON number").to_variant(),
        },
        Value::String(text) => GString::from(text.as_str()).to_variant(),
        Value::Array(items) => items
            .iter()
            .map(from_value)
            .collect::<VarArray>()
            .to_variant(),
        Value::Object(fields) => {
            let mut out = VarDictionary::new();
            for (key, value) in fields {
                out.set(
                    &GString::from(key.as_str()).to_variant(),
                    &from_value(value),
                );
            }
            out.to_variant()
        }
    }
}

//! How a product is kept in a store (docs/reference/packs.md, "Persistence and saves"): binary,
//! with a volume's values laid out byte by byte, and compressed. A whole continent's products came
//! to 5.7 GB as RON text; this keeps the same products, exactly, in about a seventh of that.

use super::runtime::{Product, Volume};
use flate2::Compression;
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use std::io::{Read, Write};

/// The format of the bytes [`encode`] makes, first among them, so bytes of another format are
/// refused rather than misread.
const FORMAT: u8 = 1;

/// `product` as the bytes a store keeps.
pub(crate) fn encode(product: &Product) -> Vec<u8> {
    let plain = match product {
        // A volume's values are smooth, so its floats' bytes repeat across neighbours far more than
        // within a float: laid out as four planes, the first byte of every value, then the second,
        // they compress to about two thirds of what the floats in order do.
        Product::Volume(volume) => {
            let head = Product::Volume(Volume {
                chunk: volume.chunk,
                size: volume.size,
                bottom: volume.bottom,
                values: Vec::new(),
                materials: volume.materials.clone(),
            });
            let mut plain = postcard::to_allocvec(&head).expect("a product is plain data");
            for byte in 0..4 {
                plain.extend(volume.values.iter().map(|value| value.to_le_bytes()[byte]));
            }
            plain
        }
        other => postcard::to_allocvec(other).expect("a product is plain data"),
    };
    let mut bytes = vec![FORMAT];
    let mut encoder = DeflateEncoder::new(&mut bytes, Compression::default());
    encoder
        .write_all(&plain)
        .expect("writing into memory does not fail");
    encoder.finish().expect("writing into memory does not fail");
    bytes
}

/// The product `bytes` hold, as [`encode`] made them.
///
/// # Errors
/// If the bytes are not a product [`encode`] made, with why.
pub(crate) fn decode(bytes: &[u8]) -> Result<Product, String> {
    let Some((&format, compressed)) = bytes.split_first() else {
        return Err("no bytes".to_owned());
    };
    if format != FORMAT {
        return Err(format!(
            "format {format}, where this version of Wave Forge keeps format {FORMAT}"
        ));
    }
    let mut plain = Vec::new();
    DeflateDecoder::new(compressed)
        .read_to_end(&mut plain)
        .map_err(|error| error.to_string())?;
    let (mut product, rest) =
        postcard::take_from_bytes::<Product>(&plain).map_err(|error| error.to_string())?;
    match &mut product {
        Product::Volume(volume) => {
            let count = volume
                .size
                .iter()
                .map(|&side| side as usize)
                .product::<usize>();
            if rest.len() != count * 4 {
                return Err(format!(
                    "{} bytes of values for a volume of {count}",
                    rest.len()
                ));
            }
            volume.values = (0..count)
                .map(|at| {
                    f32::from_le_bytes([
                        rest[at],
                        rest[count + at],
                        rest[2 * count + at],
                        rest[3 * count + at],
                    ])
                })
                .collect();
        }
        _ if !rest.is_empty() => return Err(format!("{} bytes after the product", rest.len())),
        _ => {}
    }
    Ok(product)
}

/// A chunk's products, each named and as [`encode`] made it, as the bytes of one store entry: a
/// world run keeps every target of a chunk together, so a world is a file per chunk rather than
/// one per chunk and target.
pub(crate) fn encode_chunk(products: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut bytes = vec![FORMAT];
    bytes.extend(postcard::to_allocvec(products).expect("names and bytes are plain data"));
    bytes
}

/// The named products `bytes` hold, as [`encode_chunk`] made them, each still to [`decode`].
///
/// # Errors
/// If the bytes are not an entry [`encode_chunk`] made, with why.
pub(crate) fn decode_chunk(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>, String> {
    let Some((&format, entry)) = bytes.split_first() else {
        return Err("no bytes".to_owned());
    };
    if format != FORMAT {
        return Err(format!(
            "format {format}, where this version of Wave Forge keeps format {FORMAT}"
        ));
    }
    postcard::from_bytes(entry).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::super::runtime::Field;
    use super::*;
    use wfc_core::ChunkCoord;

    #[test]
    fn a_volume_comes_back_with_every_value_bit_for_bit() {
        let values: Vec<f32> = (0..2 * 3 * 4)
            .map(|at| match at {
                0 => -0.0,
                1 => f32::MIN_POSITIVE / 2.0,
                2 => f32::MAX,
                _ => (at as f32 * 0.37).sin() * 40.0,
            })
            .collect();
        let volume = Product::Volume(Volume {
            chunk: ChunkCoord::new(-3, 7, 0),
            size: [2, 3, 4],
            bottom: -2,
            values: values.clone(),
            materials: (0..24).collect(),
        });

        let back = decode(&encode(&volume)).expect("bytes encode made");

        let Product::Volume(back) = &back else {
            panic!("a volume came back as {back:?}");
        };
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&back.values), bits(&values));
        assert_eq!(Product::Volume(back.clone()), volume);
    }

    #[test]
    fn a_field_comes_back_as_it_went() {
        let field = Product::Field(Field {
            chunk: ChunkCoord::new(1, 2, 0),
            size: [2, 2],
            values: vec![0.5, -1.25, 3.0, 8.0],
        });

        let back = decode(&encode(&field)).expect("bytes encode made");

        assert_eq!(back, field);
    }

    #[test]
    fn a_chunks_products_come_back_by_name() {
        let products = vec![
            ("trees".to_owned(), encode(&Product::Points(Vec::new()))),
            ("rock".to_owned(), vec![1, 2, 3]),
        ];

        let back = decode_chunk(&encode_chunk(&products)).expect("bytes encode_chunk made");

        assert_eq!(back, products);
    }

    #[test]
    fn bytes_of_another_format_are_refused() {
        let mut bytes = encode(&Product::Points(Vec::new()));
        bytes[0] = FORMAT + 1;

        let refused = decode(&bytes);

        assert!(refused.is_err_and(|reason| reason.contains("format")));
    }
}

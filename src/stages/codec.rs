//! How a product is kept in a store (docs/reference/packs.md, "Persistence and saves"): binary,
//! with a volume's values laid out byte by byte, and compressed, so it comes back exactly as it was
//! generated. What that saves is measured in docs/research/measurements.md (L47).

use super::runtime::{Categories, Field, Product, Volume};
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
        // they compress better than the floats in order (L47).
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

/// A 64-bit FNV-1a hash of `product`'s exact binary form, uncompressed, so it depends on the
/// product's every bit and on nothing else: the same on every machine for the same product.
pub(crate) fn hash(product: &Product) -> u64 {
    let plain = postcard::to_allocvec(product).expect("a product is plain data");
    fnv(FNV_START, &plain)
}

/// A 64-bit hash of a chunk's products of `stages`, in that order, from each one's `product_hash`,
/// once every one of them is there.
pub(crate) fn chunk_hash(
    stages: &[&str],
    product_hash: impl Fn(&str) -> Option<u64>,
) -> Option<u64> {
    stages.iter().try_fold(FNV_START, |hash, stage| {
        Some(fnv(hash, &product_hash(stage)?.to_le_bytes()))
    })
}

/// FNV-1a's offset basis, where a 64-bit hash starts.
pub(crate) const FNV_START: u64 = 0xCBF2_9CE4_8422_2325;

/// A 64-bit FNV-1a hash of `bytes`, continuing from `hash`.
pub(crate) fn fnv(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3)
    })
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
            // The size comes from the store, so it may be anything.
            let count = volume
                .size
                .iter()
                .try_fold(1_usize, |count, &side| count.checked_mul(side as usize))
                .filter(|count| count.checked_mul(4) == Some(rest.len()))
                .ok_or_else(|| {
                    format!(
                        "{} bytes of values for a volume of {:?}",
                        rest.len(),
                        volume.size
                    )
                })?;
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
            if !volume.materials.is_empty() && volume.materials.len() != count {
                return Err(format!(
                    "{} materials for a volume of {:?}",
                    volume.materials.len(),
                    volume.size
                ));
            }
        }
        _ if !rest.is_empty() => return Err(format!("{} bytes after the product", rest.len())),
        Product::Field(Field { size, values, .. })
            if values.len() as u64 != u64::from(size[0]) * u64::from(size[1]) =>
        {
            return Err(format!("{} values for a field of {size:?}", values.len()));
        }
        Product::Categories(Categories { size, values, .. })
            if values.len() as u64 != u64::from(size[0]) * u64::from(size[1]) =>
        {
            return Err(format!(
                "{} values for categories of {size:?}",
                values.len()
            ));
        }
        _ => {}
    }
    Ok(product)
}

/// The format of the entries [`encode_chunk`] makes, first among their bytes.
const ENTRY_FORMAT: u8 = 2;

/// What a world run keeps for one chunk ([`encode_chunk`]): the digest of what decided its products
/// (`Runtime::content_digest`), and each product named by its stage and as [`encode`] made it, in
/// the order of their names.
#[derive(Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub(crate) struct Entry {
    pub(crate) digest: u64,
    pub(crate) products: Vec<(String, Vec<u8>)>,
}

/// `entry` as the bytes of one store entry: a world run keeps every target of a chunk together, so
/// a world is a file per chunk rather than one per chunk and target.
pub(crate) fn encode_chunk(entry: &Entry) -> Vec<u8> {
    let mut bytes = vec![ENTRY_FORMAT];
    bytes.extend(postcard::to_allocvec(entry).expect("names and bytes are plain data"));
    bytes
}

/// The entry `bytes` hold, as [`encode_chunk`] made it, its products still to [`decode`].
///
/// # Errors
/// If the bytes are not an entry [`encode_chunk`] made, with why.
pub(crate) fn decode_chunk(bytes: &[u8]) -> Result<Entry, String> {
    let Some((&format, entry)) = bytes.split_first() else {
        return Err("no bytes".to_owned());
    };
    if format != ENTRY_FORMAT {
        return Err(format!(
            "format {format}, where this version of Wave Forge keeps format {ENTRY_FORMAT}"
        ));
    }
    postcard::from_bytes(entry).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wfc_core::ChunkCoord;

    #[test]
    fn a_volume_comes_back_with_every_value_bit_for_bit() {
        let values: Vec<f32> = (0..2 * 3 * 4)
            .map(|at| match at {
                0 => -0.0,
                1 => f32::MIN_POSITIVE / 2.0,
                2 => f32::MAX,
                _ => libm::sinf(at as f32 * 0.37) * 40.0,
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
        let entry = Entry {
            digest: 0x1234_5678_9abc_def0,
            products: vec![
                ("rock".to_owned(), vec![1, 2, 3]),
                ("trees".to_owned(), encode(&Product::Points(Vec::new()))),
            ],
        };

        let back = decode_chunk(&encode_chunk(&entry)).expect("bytes encode_chunk made");

        assert_eq!(back, entry);
    }

    // A store's bytes come from outside the program.
    #[test]
    fn a_volume_whose_size_overflows_is_refused() {
        let huge = Product::Volume(Volume {
            chunk: ChunkCoord::new(0, 0, 0),
            size: [u32::MAX, u32::MAX, u32::MAX],
            bottom: 0,
            values: Vec::new(),
            materials: Vec::new(),
        });

        let refused = decode(&encode(&huge));

        assert!(refused.is_err());
    }

    #[test]
    fn bytes_of_another_format_are_refused() {
        let mut bytes = encode(&Product::Points(Vec::new()));
        bytes[0] = FORMAT + 1;

        let refused = decode(&bytes);

        assert!(refused.is_err_and(|reason| reason.contains("format")));
    }

    #[test]
    fn a_field_of_fewer_values_than_its_size_is_refused() {
        let short = Product::Field(Field {
            chunk: ChunkCoord::new(0, 0, 0),
            size: [2, 2],
            values: vec![1.0; 3],
        });

        let refused = decode(&encode(&short));

        assert!(refused.is_err());
    }

    #[test]
    fn a_volume_of_fewer_materials_than_voxels_is_refused() {
        let short = Product::Volume(Volume {
            chunk: ChunkCoord::new(0, 0, 0),
            size: [2, 2, 2],
            bottom: 0,
            values: vec![1.0; 8],
            materials: vec![0; 7],
        });

        let refused = decode(&encode(&short));

        assert!(refused.is_err());
    }
}

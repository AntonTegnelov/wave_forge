//! Voxel models as meshes a game engine imports: binary glTF, in a Y-up engine's frame.
//!
//! A module's model is a cube of coloured voxels, `+z` up, as [`VoxelModel`] holds it. A game places
//! one model per module at the centre of each cell and turns it by the tile's rotation, so the mesh
//! here is centred on the origin, spans one unit, and is written in the frame `wave_forge::YUpSpace`
//! maps the lattice into: the engine's x is the lattice's x, its y the lattice's z, its z the
//! lattice's y. That mapping mirrors, so triangle winding is chosen from each face's outward normal
//! rather than copied from the lattice.
//!
//! Colours come from a palette texture, one texel per colour with nearest filtering, and a UV per
//! vertex, rather than from vertex colours. Every importer honours a base colour texture; Godot
//! 4.7's glTF importer reads a primitive's material before its `COLOR_0`, so a mesh of one primitive
//! never gets vertex colours as albedo (`modules/gltf/gltf_document.cpp`, 4.7-stable).
//!
//! These meshes are stand-ins, one quad per visible voxel face, good enough to walk a city; authored
//! models replace them without touching the library.

use crate::render::{Color, VoxelModel};

/// A triangle mesh coloured from a palette, in the engine's frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    /// Each vertex's colour, as an index into `palette`.
    pub colors: Vec<u32>,
    /// The colours the mesh uses, in the order it first used them; sRGB.
    pub palette: Vec<Color>,
    /// Three per triangle, counter-clockwise seen from outside.
    pub indices: Vec<u32>,
}

impl Mesh {
    /// How many triangles it has.
    #[must_use]
    pub fn triangles(&self) -> usize {
        self.indices.len() / 3
    }
}

/// The lattice's six directions, `+x, -x, +y, -y, +z, -z`, as offsets.
const DIRECTIONS: [[i32; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];

/// A point of the lattice's frame in the engine's.
const fn to_engine(lattice: [f32; 3]) -> [f32; 3] {
    [lattice[0], lattice[2], lattice[1]]
}

/// The visible faces of `model` as a mesh: a quad for every side of a voxel that does not touch
/// another voxel, including the model's outer surface.
#[must_use]
pub fn mesh(model: &VoxelModel) -> Mesh {
    let r = model.resolution;
    let size = 1.0 / r as f32;
    let filled = |x: i32, y: i32, z: i32| {
        let inside = |v: i32| v >= 0 && (v as usize) < r;
        inside(x)
            && inside(y)
            && inside(z)
            && model.get(x as usize, y as usize, z as usize).is_some()
    };
    let mut mesh = Mesh::default();
    for z in 0..r {
        for y in 0..r {
            for x in 0..r {
                let Some(color) = model.get(x, y, z) else {
                    continue;
                };
                for direction in DIRECTIONS {
                    let [dx, dy, dz] = direction;
                    if filled(x as i32 + dx, y as i32 + dy, z as i32 + dz) {
                        continue;
                    }
                    push_face(&mut mesh, [x, y, z], direction, size, color);
                }
            }
        }
    }
    mesh
}

/// One voxel face as two triangles facing `direction`.
fn push_face(mesh: &mut Mesh, voxel: [usize; 3], direction: [i32; 3], size: f32, color: Color) {
    let axis = direction.iter().position(|&d| d != 0).expect("a direction");
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
    // The face lies on the voxel's side along `axis`, spanning the other two axes.
    let corner = |du: usize, dv: usize| {
        let mut at = [voxel[0] as f32, voxel[1] as f32, voxel[2] as f32];
        if direction[axis] > 0 {
            at[axis] += 1.0;
        }
        at[u] += du as f32;
        at[v] += dv as f32;
        to_engine(at.map(|c| c * size - 0.5))
    };
    let normal = to_engine(direction.map(|d| d as f32));
    let corners = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)];
    let base = u32::try_from(mesh.positions.len()).expect("a model fits u32 indices");
    let entry = mesh
        .palette
        .iter()
        .position(|&c| c == color)
        .unwrap_or_else(|| {
            mesh.palette.push(color);
            mesh.palette.len() - 1
        });
    let entry = u32::try_from(entry).expect("a model has few colours");
    for position in corners {
        mesh.positions.push(position);
        mesh.normals.push(normal);
        mesh.colors.push(entry);
    }
    // The quad's corners go round one way in the lattice; whether that is counter-clockwise seen
    // from outside depends on the axis and on the mirroring into the engine's frame, so the normal
    // decides.
    let outward = dot(
        cross(sub(corners[1], corners[0]), sub(corners[2], corners[0])),
        normal,
    ) > 0.0;
    let order: [u32; 6] = if outward {
        [0, 1, 2, 0, 2, 3]
    } else {
        [0, 2, 1, 0, 3, 2]
    };
    mesh.indices.extend(order.map(|i| base + i));
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `mesh` as a binary glTF 2.0 file with one scene of one node called `name`. A mesh without
/// triangles gives a node without a mesh, which glTF allows and a mesh would not.
#[must_use]
pub fn glb(mesh: &Mesh, name: &str) -> Vec<u8> {
    let mut bin: Vec<u8> = Vec::new();
    let mut views = Vec::new();
    let mut push_view = |bytes: &[u8], target: u32| {
        views.push(serde_json::json!({
            "buffer": 0,
            "byteOffset": bin.len(),
            "byteLength": bytes.len(),
            "target": target,
        }));
        bin.extend_from_slice(bytes);
    };
    let floats = |values: &[[f32; 3]]| -> Vec<u8> {
        values
            .iter()
            .flatten()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    };
    // One texel per palette entry, sampled at its centre.
    let width = mesh.palette.len().max(1) as f32;
    let uvs: Vec<u8> = mesh
        .colors
        .iter()
        .flat_map(|&entry| [(entry as f32 + 0.5) / width, 0.5])
        .flat_map(f32::to_le_bytes)
        .collect();
    const VERTICES: u32 = 34_962;
    const INDICES: u32 = 34_963;
    let has_mesh = !mesh.indices.is_empty();
    let mut json = serde_json::json!({
        "asset": { "version": "2.0", "generator": "wave_forge wfc-export-models" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "name": name }],
    });
    if has_mesh {
        push_view(&floats(&mesh.positions), VERTICES);
        push_view(&floats(&mesh.normals), VERTICES);
        push_view(&uvs, VERTICES);
        let indices: Vec<u8> = mesh.indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        push_view(&indices, INDICES);
        let png = palette_png(&mesh.palette);
        let image_view = views.len();
        views.push(serde_json::json!({
            "buffer": 0,
            "byteOffset": bin.len(),
            "byteLength": png.len(),
        }));
        bin.extend_from_slice(&png);
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let (min, max) = bounds(&mesh.positions);
        let count = mesh.positions.len();
        json["nodes"][0]["mesh"] = serde_json::json!(0);
        json["meshes"] = serde_json::json!([{
            "name": name,
            "primitives": [{
                "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 },
                "indices": 3,
                "material": 0,
            }],
        }]);
        json["materials"] = serde_json::json!([{
            "name": "voxel",
            "pbrMetallicRoughness": {
                "baseColorTexture": { "index": 0 },
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
        }]);
        // Nearest filtering, so neighbouring palette entries never blend.
        json["samplers"] = serde_json::json!([{ "magFilter": 9728, "minFilter": 9728 }]);
        json["images"] = serde_json::json!([{ "bufferView": image_view, "mimeType": "image/png" }]);
        json["textures"] = serde_json::json!([{ "sampler": 0, "source": 0 }]);
        json["accessors"] = serde_json::json!([
            { "bufferView": 0, "componentType": 5126, "count": count, "type": "VEC3", "min": min, "max": max },
            { "bufferView": 1, "componentType": 5126, "count": count, "type": "VEC3" },
            { "bufferView": 2, "componentType": 5126, "count": count, "type": "VEC2" },
            { "bufferView": 3, "componentType": 5125, "count": mesh.indices.len(), "type": "SCALAR" },
        ]);
        json["bufferViews"] = serde_json::Value::Array(views);
        json["buffers"] = serde_json::json!([{ "byteLength": bin.len() }]);
    }
    let mut text = serde_json::to_vec(&json).expect("a JSON value serialises");
    // Chunks are 4-byte aligned: JSON pads with spaces, binary with zeros.
    while !text.len().is_multiple_of(4) {
        text.push(b' ');
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let chunk_header = 8;
    let mut total = 12 + chunk_header + text.len();
    if has_mesh {
        total += chunk_header + bin.len();
    }
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(
        &u32::try_from(total)
            .expect("a model under 4 GiB")
            .to_le_bytes(),
    );
    out.extend_from_slice(&u32::try_from(text.len()).expect("small").to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&text);
    if has_mesh {
        out.extend_from_slice(&u32::try_from(bin.len()).expect("small").to_le_bytes());
        out.extend_from_slice(b"BIN\0");
        out.extend_from_slice(&bin);
    }
    out
}

/// The palette as a PNG one texel tall.
fn palette_png(palette: &[Color]) -> Vec<u8> {
    let width = u32::try_from(palette.len()).expect("a model has few colours");
    let mut image = image::RgbImage::new(width, 1);
    for (x, &color) in palette.iter().enumerate() {
        image.put_pixel(x as u32, 0, image::Rgb(color));
    }
    let mut png = Vec::new();
    image
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("a PNG encodes into memory");
    png
}

fn bounds(points: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    points.iter().fold(
        ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]),
        |(min, max), p| {
            (
                [min[0].min(p[0]), min[1].min(p[1]), min[2].min(p[2])],
                [max[0].max(p[0]), max[1].max(p[1]), max[2].max(p[2])],
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::city;
    use std::collections::BTreeSet;

    const RED: Color = [255, 0, 0];

    #[test]
    fn a_lone_voxel_is_a_closed_box_of_twelve_outward_triangles() {
        let mut model = VoxelModel::empty(4);
        model.set(1, 2, 3, Some(RED));

        let mesh = mesh(&model);

        assert_eq!(mesh.triangles(), 12);
        // The voxel's centre in the engine's frame: lattice (1.5, 2.5, 3.5) quarters, centred.
        let centre = to_engine([1.5, 2.5, 3.5].map(|c| c * 0.25 - 0.5));
        for triangle in mesh.indices.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|i| mesh.positions[triangle[i] as usize]);
            let facing = cross(sub(b, a), sub(c, a));
            let away = sub(a, centre);
            assert!(dot(facing, away) > 0.0, "a triangle faces into the voxel");
            assert!(dot(facing, mesh.normals[triangle[0] as usize]) > 0.0);
        }
    }

    #[test]
    fn touching_voxels_hide_the_faces_between_them() {
        let mut model = VoxelModel::empty(2);
        model.set(0, 0, 0, Some(RED));
        model.set(1, 0, 0, Some(RED));

        assert_eq!(
            mesh(&model).triangles(),
            20,
            "two boxes less the two faces they share"
        );
    }

    #[test]
    fn the_lattice_up_is_the_engine_up() {
        let mut model = VoxelModel::empty(4);
        model.set(0, 0, 3, Some(RED));

        let mesh = mesh(&model);

        let highest = mesh.positions.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
        assert!(
            (highest - 0.5).abs() < 1e-6,
            "a voxel at the top reaches y = 0.5"
        );
    }

    /// A mesh as the set of its faces, each as its four corners rounded, in the order it winds and
    /// starting from the smallest. Where a quad is split into triangles is not part of the surface,
    /// so it is left out; which way a face winds is.
    fn face_set(mesh: &Mesh) -> BTreeSet<[[i32; 3]; 4]> {
        mesh.indices
            .chunks(6)
            .map(|face| {
                let base = face[0];
                // Faces are written as [0, 1, 2, 0, 2, 3] or its reverse [0, 2, 1, 0, 3, 2].
                let order = if face[1] == base + 1 {
                    [0, 1, 2, 3]
                } else {
                    [0, 3, 2, 1]
                };
                let corners = order.map(|i| {
                    mesh.positions[(base + i) as usize].map(|c| (c * 1000.0).round() as i32)
                });
                let start = (0..4).min_by_key(|&i| corners[i]).expect("four corners");
                [0, 1, 2, 3].map(|i| corners[(start + i) % 4])
            })
            .collect()
    }

    #[test]
    fn a_model_turned_by_the_tiles_yaw_is_the_rotated_variant() {
        // A game places the unrotated model and turns it by YUpSpace::yaw; that has to give the
        // mesh of the rotated voxel model, winding included, for every module of the city.
        let city = city::city();
        for (name, model) in city.prototype_models() {
            let base = mesh(model);
            for turns in 1..4u8 {
                let yaw = wave_forge::YUpSpace::yaw(turns);
                let (sin, cos) = yaw.sin_cos();
                let turned = Mesh {
                    positions: base
                        .positions
                        .iter()
                        .map(|p| [p[0] * cos + p[2] * sin, p[1], -p[0] * sin + p[2] * cos])
                        .collect(),
                    ..base.clone()
                };

                let expected = mesh(&model.rotated(turns));

                assert_eq!(
                    face_set(&turned),
                    face_set(&expected),
                    "{name} turned {turns} quarter turns"
                );
            }
        }
    }

    #[test]
    fn a_glb_holds_its_json_and_binary_chunks() {
        let mut model = VoxelModel::empty(4);
        model.set(0, 0, 0, Some(RED));
        let mesh = mesh(&model);

        let file = glb(&mesh, "red");

        assert_eq!(&file[0..4], b"glTF");
        let word = |at: usize| u32::from_le_bytes(file[at..at + 4].try_into().expect("4 bytes"));
        assert_eq!(word(4), 2, "glTF 2.0");
        assert_eq!(word(8) as usize, file.len());
        let json_len = word(12) as usize;
        assert_eq!(&file[16..20], b"JSON");
        let json: serde_json::Value =
            serde_json::from_slice(&file[20..20 + json_len]).expect("valid JSON");
        assert_eq!(json["accessors"][0]["count"], 24);
        assert_eq!(json["accessors"][3]["count"], 36);
        let bin_at = 20 + json_len;
        assert_eq!(&file[bin_at + 4..bin_at + 8], b"BIN\0");
        assert_eq!(
            word(bin_at) as usize,
            json["buffers"][0]["byteLength"].as_u64().expect("a length") as usize
        );
    }

    #[test]
    fn an_empty_model_is_a_node_without_a_mesh() {
        let file = glb(&mesh(&VoxelModel::empty(4)), "air");

        let json_len = u32::from_le_bytes(file[12..16].try_into().expect("4 bytes")) as usize;
        let json: serde_json::Value =
            serde_json::from_slice(&file[20..20 + json_len]).expect("valid JSON");
        assert_eq!(json["nodes"][0]["name"], "air");
        assert!(json.get("meshes").is_none());
        assert_eq!(file.len(), 20 + json_len, "no binary chunk");
    }
}

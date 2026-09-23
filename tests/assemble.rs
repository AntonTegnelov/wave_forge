//! Assemblies: a village of streets and houses and a dungeon of rooms and corridors, grown from
//! connectors inside their sites, the same all at once and chunk by chunk in either order, with
//! the ground levelled under the village's pieces and trees kept clear of them.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, PackError, Runtime, StageError, Stamp};
use wave_forge::{ChunkCoord, FocusPoint, InstanceId};

const VILLAGE: &str = r#"(
    version: 1,
    stages: [
        (name: "ground", kind: Field(Mul(Noise(frequency: 0.03, octaves: 2), Constant(12.0)))),
        (name: "villages", kind: Sites(height: "ground", region: 6, size: (2, 3), chance: 1.0)),
        (name: "village", kind: Assemble(sites: "villages", start: "square", max: 14, min: 5, rerolls: 4, pieces: [
            (name: "square", size: (4, 4, 1), doors: [
                (at: (3, 1, 0), facing: East, kind: "street"),
                (at: (0, 2, 0), facing: West, kind: "street"),
                (at: (1, 3, 0), facing: North, kind: "street"),
                (at: (2, 0, 0), facing: South, kind: "street"),
            ]),
            (name: "street", size: (1, 4, 1), weight: 3, doors: [
                (at: (0, 0, 0), facing: South, kind: "street"),
                (at: (0, 3, 0), facing: North, kind: "street"),
                (at: (0, 1, 0), facing: East, kind: "house"),
                (at: (0, 2, 0), facing: West, kind: "house"),
            ]),
            (name: "house", size: (3, 3, 1), weight: 2, doors: [(at: (1, 0, 0), facing: South, kind: "house")]),
            (name: "well", size: (2, 2, 1), doors: [(at: (0, 0, 0), facing: South, kind: "house")]),
            (name: "wall", size: (1, 1, 1), end: true, doors: [(at: (0, 0, 0), facing: South, kind: "street")]),
        ])),
        (name: "level", kind: Flatten(height: "ground", sites: "village", blend: 3)),
        (name: "trees", kind: Scatter(kind: "tree", height: "level", spacing: 3, avoid: Some(("village", 2)))),
    ],
)"#;

const DUNGEON: &str = r#"(
    version: 1,
    stages: [
        (name: "ground", kind: Field(Mul(Noise(frequency: 0.03, octaves: 2), Constant(12.0)))),
        (name: "entrances", kind: Locations(height: "ground", region: 6, kinds: [
            (name: "crypt", priority: 1, quota: 1, size: 3),
            (name: "cairn", quota: 2),
        ])),
        (name: "dungeon", kind: Assemble(sites: "entrances", kinds: ["crypt"], start: "entry", lift: 60.0,
            max: 20, min: 8, rerolls: 8, pieces: [
            (name: "entry", size: (3, 3, 2), doors: [(at: (1, 2, 0), facing: North, kind: "hall")]),
            (name: "room", size: (5, 5, 2), weight: 2, doors: [
                (at: (2, 0, 0), facing: South, kind: "hall"),
                (at: (2, 4, 0), facing: North, kind: "hall"),
                (at: (4, 2, 0), facing: East, kind: "hall"),
                (at: (0, 2, 0), facing: West, kind: "hall"),
            ]),
            (name: "corridor", size: (1, 4, 1), weight: 3, doors: [
                (at: (0, 0, 0), facing: South, kind: "hall"),
                (at: (0, 3, 0), facing: North, kind: "hall"),
            ]),
            (name: "turn", size: (2, 2, 1), doors: [
                (at: (0, 0, 0), facing: South, kind: "hall"),
                (at: (1, 1, 0), facing: East, kind: "hall"),
            ]),
            (name: "cap", size: (1, 1, 1), end: true, doors: [(at: (0, 0, 0), facing: South, kind: "hall")]),
        ])),
    ],
)"#;

const SIZE: [u32; 2] = [8, 8];

fn area() -> Vec<ChunkCoord> {
    (-8..8)
        .flat_map(|y| (-8..8).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

fn runtime(pack: &str) -> Runtime {
    Runtime::new(Arc::new(Pack::parse(pack).expect("a valid pack")), 21, SIZE)
}

/// What `targets` hold over the area, asked for in the order of `requests`: every stamp of the
/// first target by id, and every chunk's product of the others as text.
fn generate(
    pack: &str,
    targets: &[&str],
    requests: &[Vec<ChunkCoord>],
) -> (
    BTreeMap<InstanceId, Stamp>,
    BTreeMap<(String, ChunkCoord), String>,
) {
    let mut runtime = runtime(pack);
    let mut stamps = BTreeMap::new();
    let mut others = BTreeMap::new();
    for request in requests {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime.request(&focus, targets).expect("stages");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            for stamp in runtime.stamps(targets[0], chunk).expect("generated") {
                stamps.insert(stamp.id, stamp.clone());
            }
            for &other in &targets[1..] {
                let product = match runtime.field(other, chunk) {
                    Some(field) => format!("{:?}", field.values),
                    None => format!("{:?}", runtime.points(other, chunk).expect("generated")),
                };
                others.insert((other.to_owned(), chunk), product);
            }
        }
    }
    (stamps, others)
}

fn all_at_once() -> Vec<Vec<ChunkCoord>> {
    vec![area()]
}

fn one_at_a_time(backwards: bool) -> Vec<Vec<ChunkCoord>> {
    let mut chunks = area();
    if backwards {
        chunks.reverse();
    }
    chunks.into_iter().map(|chunk| vec![chunk]).collect()
}

fn overlap(a: &Stamp, b: &Stamp) -> bool {
    (0..2).all(|axis| a.min[axis] < b.max[axis] && b.min[axis] < a.max[axis])
}

#[test]
fn a_village_and_a_dungeon_are_the_same_all_at_once_and_chunk_by_chunk_in_either_order() {
    for (pack, targets) in [
        (VILLAGE, vec!["village", "level", "trees"]),
        (DUNGEON, vec!["dungeon"]),
    ] {
        let whole = generate(pack, &targets, &all_at_once());

        let forwards = generate(pack, &targets, &one_at_a_time(false));
        let backwards = generate(pack, &targets, &one_at_a_time(true));

        assert!(whole.0.len() > 20, "{} pieces", whole.0.len());
        assert_eq!(forwards, whole);
        assert_eq!(backwards, whole);
    }
}

#[test]
fn every_piece_stays_inside_its_site_and_no_two_overlap() {
    let mut runtime = runtime(VILLAGE);
    let focus: Vec<FocusPoint> = area().into_iter().map(|c| FocusPoint::new(c, 0)).collect();
    runtime
        .request(&focus, &["village", "villages"])
        .expect("stages");
    runtime.run_until_idle().expect("the stages run");
    let mut by_site: BTreeMap<String, Vec<Stamp>> = BTreeMap::new();
    let mut sites = BTreeMap::new();

    for chunk in area() {
        for site in runtime.sites("villages", chunk).expect("generated") {
            sites.insert(format!("{:?}", site.id), site.clone());
        }
        for stamp in runtime.stamps("village", chunk).expect("generated") {
            let (x0, y0) = (i64::from(chunk.x) * 8, i64::from(chunk.y) * 8);
            assert!(
                stamp.min[0] < x0 + 8 && x0 < stamp.max[0],
                "{stamp:?} in {chunk:?}"
            );
            assert!(
                stamp.min[1] < y0 + 8 && y0 < stamp.max[1],
                "{stamp:?} in {chunk:?}"
            );
            let site = by_site.entry(format!("{:?}", stamp.site)).or_default();
            if !site.contains(stamp) {
                site.push(stamp.clone());
            }
        }
    }

    assert!(!by_site.is_empty());
    for (id, stamps) in &by_site {
        let site = &sites[id];
        for stamp in stamps {
            assert!(
                stamp.min[0] >= i64::from(site.min.0) * 8
                    && stamp.max[0] <= i64::from(site.max.0) * 8
                    && stamp.min[1] >= i64::from(site.min.1) * 8
                    && stamp.max[1] <= i64::from(site.max.1) * 8,
                "{stamp:?} outside {site:?}"
            );
            for other in stamps {
                assert!(
                    other == stamp || !overlap(stamp, other),
                    "{stamp:?} and {other:?}"
                );
            }
        }
        let inside = (-8..=8).contains(&site.min.0)
            && (-8..=8).contains(&site.max.0)
            && (-8..=8).contains(&site.min.1)
            && (-8..=8).contains(&site.max.1);
        if inside {
            assert!(stamps.len() >= 5, "{id}: {} pieces", stamps.len());
        }
    }
}

#[test]
fn the_ground_under_each_piece_is_levelled_to_its_floor_and_trees_keep_clear() {
    let mut runtime = runtime(VILLAGE);
    let focus: Vec<FocusPoint> = area().into_iter().map(|c| FocusPoint::new(c, 0)).collect();
    runtime
        .request(&focus, &["village", "level", "trees"])
        .expect("stages");
    runtime.run_until_idle().expect("the stages run");
    let mut stamps: Vec<Stamp> = Vec::new();
    for chunk in area() {
        stamps.extend(
            runtime
                .stamps("village", chunk)
                .expect("generated")
                .iter()
                .cloned(),
        );
    }

    for stamp in &stamps {
        for x in stamp.min[0]..stamp.max[0] {
            for y in stamp.min[1]..stamp.max[1] {
                let chunk = ChunkCoord::new(x.div_euclid(8) as i32, y.div_euclid(8) as i32, 0);
                let Some(level) = runtime.field("level", chunk) else {
                    assert!(!area().contains(&chunk), "{chunk:?} not generated");
                    continue;
                };
                let value = level.get(x.rem_euclid(8) as u32, y.rem_euclid(8) as u32);
                assert_eq!(value, stamp.position[2], "({x}, {y}) under {stamp:?}");
            }
        }
    }
    for chunk in area() {
        for tree in runtime.points("trees", chunk).expect("generated") {
            let (x, y) = (
                tree.position[0].floor() as i64,
                tree.position[1].floor() as i64,
            );
            for stamp in &stamps {
                let dx = (stamp.min[0] - x).max(x - (stamp.max[0] - 1)).max(0) as f32;
                let dy = (stamp.min[1] - y).max(y - (stamp.max[1] - 1)).max(0) as f32;
                assert!((dx * dx + dy * dy).sqrt() >= 2.0, "{tree:?} by {stamp:?}");
            }
        }
    }
}

#[test]
fn a_dungeon_stands_above_its_entrance_on_its_own_kind_of_site() {
    let mut runtime = runtime(DUNGEON);
    let focus: Vec<FocusPoint> = area().into_iter().map(|c| FocusPoint::new(c, 0)).collect();
    runtime
        .request(&focus, &["dungeon", "entrances"])
        .expect("stages");
    runtime.run_until_idle().expect("the stages run");
    let mut sites = BTreeMap::new();
    let mut stamps = Vec::new();

    for chunk in area() {
        for site in runtime.sites("entrances", chunk).expect("generated") {
            sites.insert(format!("{:?}", site.id), site.clone());
        }
        stamps.extend(
            runtime
                .stamps("dungeon", chunk)
                .expect("generated")
                .iter()
                .cloned(),
        );
    }

    assert!(!stamps.is_empty());
    for stamp in &stamps {
        let site = &sites[&format!("{:?}", stamp.site)];
        assert_eq!(site.kind.as_deref(), Some("crypt"), "{stamp:?}");
        assert_eq!(stamp.position[2], site.height + 60.0, "{stamp:?}");
    }
    assert!(stamps.iter().any(|stamp| &*stamp.piece == "cap"));
}

#[test]
fn an_assembly_that_stays_under_its_least_count_is_refused_naming_its_stage() {
    let pack = VILLAGE.replace("max: 14, min: 5", "max: 400, min: 300");
    let mut runtime = runtime(&pack);

    let result = runtime
        .request(
            &[FocusPoint::new(ChunkCoord::new(0, 0, 0), 4)],
            &["village"],
        )
        .and_then(|_| runtime.run_until_idle());

    assert!(
        matches!(&result, Err(StageError::Assemble { stage, min: 300, .. }) if stage == "village"),
        "{result:?}"
    );
}

#[test]
fn pieces_that_cannot_grow_are_refused_by_stage() {
    let cases = [
        VILLAGE.replace(
            "(at: (3, 1, 0), facing: East",
            "(at: (2, 1, 0), facing: East",
        ),
        VILLAGE.replace("start: \"square\"", "start: \"plaza\""),
        VILLAGE.replace("start: \"square\"", "start: \"wall\""),
        VILLAGE.replace("max: 14, min: 5", "max: 4, min: 5"),
        VILLAGE.replace(
            "(name: \"well\", size: (2, 2, 1)",
            "(name: \"well\", size: (0, 2, 1)",
        ),
    ];

    for pack in cases {
        let result = Pack::parse(&pack);

        assert!(
            matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "village"),
            "{result:?}"
        );
    }
}

#[test]
fn a_piece_placed_by_its_turn_and_position_opens_its_door_onto_what_it_joined() {
    let mut runtime = runtime(VILLAGE);
    let focus: Vec<FocusPoint> = area().into_iter().map(|c| FocusPoint::new(c, 0)).collect();
    runtime.request(&focus, &["village"]).expect("stages");
    runtime.run_until_idle().expect("the stages run");
    let mut stamps: BTreeMap<InstanceId, Stamp> = BTreeMap::new();
    for chunk in area() {
        for stamp in runtime.stamps("village", chunk).expect("generated") {
            stamps.insert(stamp.id, stamp.clone());
        }
    }
    let inside = |stamp: &Stamp, (x, y): (f32, f32)| {
        (stamp.min[0] as f32..stamp.max[0] as f32).contains(&x)
            && (stamp.min[1] as f32..stamp.max[1] as f32).contains(&y)
    };
    // A house is 3 by 3 with its door in the middle of its south side, so as authored, centred on
    // its origin, the door's cell centre is at (0, -1) and it faces (0, -1).
    let turned = |stamp: &Stamp, (x, y): (f32, f32)| {
        let [row_x, _, row_y] = stamp.y_up_basis();
        (row_x[0] * x + row_x[2] * y, row_y[0] * x + row_y[2] * y)
    };

    let houses: Vec<&Stamp> = stamps.values().filter(|s| &*s.piece == "house").collect();

    assert!(houses.len() > 5, "{} houses", houses.len());
    for house in houses {
        let (dx, dy) = turned(house, (0.0, -1.0));
        let door = (house.position[0] + dx, house.position[1] + dy);
        let outside = (door.0 + dx, door.1 + dy);
        assert!(inside(house, door), "{house:?}: door at {door:?}");
        // Pieces outside the area were not asked for.
        if !(-64.0..64.0).contains(&outside.0) || !(-64.0..64.0).contains(&outside.1) {
            continue;
        }
        assert!(
            stamps
                .values()
                .any(|other| &*other.piece == "street" && inside(other, outside)),
            "{house:?}: its door at {door:?} opens onto no street"
        );
    }
}

#[test]
fn the_ring_world_grows_a_dungeon_above_every_crypt() {
    let text = std::fs::read_to_string(format!(
        "{}/examples/rings.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the ring world");
    let mut runtime = Runtime::new(Arc::new(Pack::parse(&text).expect("a valid pack")), 7, SIZE);
    runtime
        .request_bound(&["dungeon", "locations"])
        .expect("a bounded island");
    runtime.run_until_idle().expect("the stages run");
    let mut crypts = BTreeMap::new();
    let mut dungeons: BTreeMap<String, Vec<Stamp>> = BTreeMap::new();

    for y in -14..14 {
        for x in -14..14 {
            let chunk = ChunkCoord::new(x, y, 0);
            for site in runtime.sites("locations", chunk).into_iter().flatten() {
                if site.kind.as_deref() == Some("crypt") {
                    crypts.insert(format!("{:?}", site.id), site.clone());
                }
            }
            for stamp in runtime.stamps("dungeon", chunk).into_iter().flatten() {
                let pieces = dungeons.entry(format!("{:?}", stamp.site)).or_default();
                if !pieces.contains(stamp) {
                    pieces.push(stamp.clone());
                }
            }
        }
    }

    assert!(!crypts.is_empty());
    assert_eq!(
        dungeons.keys().collect::<Vec<_>>(),
        crypts.keys().collect::<Vec<_>>()
    );
    for (id, pieces) in &dungeons {
        let crypt = &crypts[id];
        let rooms = pieces.iter().filter(|piece| &*piece.piece != "cap").count();
        assert!(rooms >= 8, "{id}: {rooms} pieces");
        for piece in pieces {
            assert_eq!(piece.position[2], crypt.height + 40.0, "{piece:?}");
        }
    }
}

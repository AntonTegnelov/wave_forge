//! What a pack may say: a pack that loads is one the runtime can generate, so whatever would make
//! it panic, hang, allocate without bound or quietly misread a value is refused when it loads,
//! naming the stage and why. Misspelt fields, sizes and loops past their limits, numbers that would
//! overflow, names a store cannot hold and reads the runtime cannot honour are all refused.

use wave_forge::stages::{Pack, PackError};

/// A pack of `stages`, after a ground field the stages may read.
fn pack(stages: &str) -> Result<Pack, PackError> {
    Pack::parse(&format!(
        r#"(version: 1, water: Some((level: -100.0)), stages: [
            (name: "ground", kind: Field(Mul(Noise(frequency: 0.03, octaves: 2), Constant(12.0)))),
            {stages}
        ])"#
    ))
}

/// Why `stage` was refused, if it was refused as an invalid stage.
fn refusal(result: Result<Pack, PackError>, stage: &str) -> String {
    match result {
        Err(PackError::Invalid {
            stage: refused,
            message,
        }) if refused == stage => message,
        other => panic!("{stage} should be refused as invalid, not {other:?}"),
    }
}

#[test]
fn a_misspelt_field_inside_a_stage_kind_is_a_syntax_error() {
    let result = pack(
        r#"(name: "rivers", kind: Rivers(height: "ground", region: 4, sources: 2, through_lake: true)),"#,
    );

    assert!(
        matches!(&result, Err(PackError::Syntax(message)) if message.contains("through_lake")),
        "{result:?}"
    );
}

#[test]
fn a_misspelt_field_inside_an_expression_is_a_syntax_error() {
    let result = pack(
        r#"(name: "hills", kind: Field(Noise(frequency: 0.1, octaves: 2, nmae: Some("hills")))),"#,
    );

    assert!(
        matches!(&result, Err(PackError::Syntax(message)) if message.contains("nmae")),
        "{result:?}"
    );
}

#[test]
fn a_window_beyond_the_limit_is_refused() {
    let result = pack(r#"(name: "soft", kind: Blur(input: "ground", radius: 100000)),"#);

    assert!(refusal(result, "soft").contains("100000 cells"));
}

#[test]
fn a_size_that_would_overflow_is_refused_not_wrapped() {
    let result = pack(
        r#"(name: "towns", kind: Sites(height: "ground", region: 6, size: (1, 4294967295), chance: 1.0)),"#,
    );

    assert!(refusal(result, "towns").contains("4294967295"));
}

#[test]
fn a_reach_that_would_overflow_is_refused_not_wrapped() {
    let result = pack(
        r#"(name: "rivers", kind: Rivers(height: "ground", region: 4, sources: 2)),
           (name: "carved", kind: Apply(height: "ground", curves: "rivers", max_radius: 4294967295, blend: 1, profile: Level)),"#,
    );

    assert!(refusal(result, "carved").contains("cells beyond its chunk"));
}

#[test]
fn a_volume_of_too_many_levels_is_refused() {
    let result = pack(
        r#"(name: "rock", kind: Volume(density: Sub(Constant(0.0), Z), bottom: -1000000, top: 1000000)),"#,
    );

    assert!(refusal(result, "rock").contains("levels"));
}

#[test]
fn a_region_loop_beyond_the_limit_is_refused() {
    let result = pack(
        r#"(name: "shed", kind: Talus(height: "ground", region: 4, slope: 0.8, passes: 4000000000, fade: 8)),"#,
    );

    assert!(refusal(result, "shed").contains("passes"));
}

#[test]
fn a_painting_of_more_categories_than_a_byte_holds_is_refused() {
    let legend: Vec<String> = (0..257)
        .map(|i| format!("(({}, {}, 0), \"c{i}\")", i % 256, i / 256))
        .collect();
    let result = pack(&format!(
        r#"(name: "map", kind: Painted(painting: "map", origin: (0.0, 0.0), cell: 1.0, legend: [{}], otherwise: "none")),"#,
        legend.join(", ")
    ));

    assert!(refusal(result, "map").contains("categories"));
}

#[test]
fn a_locations_stage_with_no_region_or_no_kinds_is_refused() {
    let no_region = pack(
        r#"(name: "places", kind: Locations(height: "ground", region: 0, kinds: [(name: "cairn", quota: 1)])),"#,
    );
    let no_kinds =
        pack(r#"(name: "places", kind: Locations(height: "ground", region: 4, kinds: [])),"#);

    assert!(refusal(no_region, "places").contains("region"));
    assert!(refusal(no_kinds, "places").contains("kind"));
}

#[test]
fn an_assembly_of_a_kind_its_sites_never_have_is_refused() {
    let result = pack(
        r#"(name: "entrances", kind: Locations(height: "ground", region: 6, kinds: [(name: "crypt", quota: 1)])),
           (name: "dungeon", kind: Assemble(sites: "entrances", kinds: ["crpyt"], start: "entry", max: 4, pieces: [
               (name: "entry", size: (3, 3, 2), doors: []),
           ])),"#,
    );

    assert!(refusal(result, "dungeon").contains("crpyt"));
}

#[test]
fn a_noise_of_endless_octaves_is_refused() {
    let result = Pack::parse(
        r#"(version: 1, noises: {"hills": (fractal_octaves: 2000000000)}, stages: [
            (name: "ground", kind: Field(FastNoise("hills"))),
        ])"#,
    );

    assert!(
        matches!(&result, Err(PackError::Noise { name, .. }) if name == "hills"),
        "{result:?}"
    );
}

#[test]
fn a_stage_name_a_store_cannot_hold_is_refused() {
    for name in ["../escape", "Rock", "two words", ""] {
        let result = Pack::parse(&format!(
            r#"(version: 1, stages: [(name: "{name}", kind: Field(Constant(1.0)))])"#
        ));

        assert!(
            matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == name),
            "{name:?}: {result:?}"
        );
    }
}

#[test]
fn curves_read_at_another_scale_are_refused() {
    let result = pack(
        r#"(name: "coarse", scale: 4, kind: Field(Constant(0.0))),
           (name: "rivers", scale: 4, kind: Rivers(height: "coarse", region: 4, sources: 2)),
           (name: "carved", kind: Apply(height: "ground", curves: "rivers", max_radius: 3, profile: Level)),"#,
    );

    assert!(refusal(result, "carved").contains("curves"));
}

#[test]
fn a_cycle_of_tables_is_blamed_on_a_table_in_the_cycle() {
    let result = Pack::parse(
        r#"(version: 1, tables: [
            (name: "a", kind: Generated(parent: Some("b"), count: Constant(1.0), columns: [])),
            (name: "b", kind: Generated(parent: Some("c"), count: Constant(1.0), columns: [])),
            (name: "c", kind: Generated(parent: Some("b"), count: Constant(1.0), columns: [])),
        ], stages: [])"#,
    );

    assert!(
        matches!(&result, Err(PackError::InvalidTable { table, .. }) if table == "b" || table == "c"),
        "{result:?}"
    );
}

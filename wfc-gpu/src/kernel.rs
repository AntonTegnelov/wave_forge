//! The block kernel's source and parameters.
//!
//! One kernel is compiled per region shape and word count, because the shape decides the size of
//! the workgroup arrays and WGSL needs those at compile time. Substituting the constants into the
//! source is the whole of the specialisation, and it is the same mechanism for every backend: naga
//! bakes `override` values into a module per shape anyway, so nothing is gained by declaring them.

use crate::backend::BackendLimits;
use bytemuck::{Pod, Zeroable};
use wfc_core::rules::{AXES, MAX_WORDS};
use wfc_core::{RegionShape, Ruleset};

/// Words of statistics the kernel writes per region: status, sweeps, collapses, restarts,
/// contradiction cell, steps, backtracks, tries.
pub const STATS_WORDS: u32 = 8;
/// The kernel's entry point.
pub const ENTRY: &str = "solve_region";
/// Bytes of workgroup memory the control block and the shared flags take.
const CONTROL_BYTES: u32 = 14 * 4 + 4 * 4;

const SOURCE: &str = include_str!("kernel/block.wgsl");

/// How a solver runs a region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SolverConfig {
    /// Invocations per workgroup. Cost per step falls with cells per invocation, so this wants to
    /// be large; it cannot exceed the device's limit.
    pub invocations: u32,
    /// Chebyshev radius within which every local minimum collapses in the same round. Zero
    /// collapses one cell per round, which is the classic algorithm and slower.
    pub radius: u32,
    /// Attempts at a region before it is reported as exhausted.
    pub max_attempts: u32,
    /// Workgroup-wide steps a region may take. A hung shader resets the host's display driver, so
    /// this is a hard budget rather than a guess.
    pub max_steps: u32,
    /// Checkpoints kept per region for undo.
    pub ring: u32,
    /// Regions a batch may hold.
    pub max_batch: u32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            invocations: 256,
            radius: 1,
            max_attempts: 64,
            max_steps: 50_000,
            ring: 32,
            max_batch: 256,
        }
    }
}

/// What the kernel reads from its uniform buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Params {
    /// 0 propagates to a fixpoint and stops; 1 solves the region.
    pub mode: u32,
    pub max_steps: u32,
    pub max_attempts: u32,
    pub radius: u32,
    /// 0 restarts a region on a contradiction; 1 restores the checkpoint before the failing round.
    pub undo: u32,
    /// 1 when the batch is one problem tried with different seeds; see `RegionBatch::portfolio`.
    pub portfolio: u32,
    pub padding: [u32; 2],
}

impl Params {
    /// Parameters that solve a region.
    #[must_use]
    pub fn solve(config: &SolverConfig) -> Self {
        Self {
            mode: 1,
            max_steps: config.max_steps,
            max_attempts: config.max_attempts,
            radius: config.radius,
            undo: 1,
            portfolio: 0,
            padding: [0; 2],
        }
    }

    /// Parameters that only propagate the starting domains to their fixpoint, which is what the
    /// kernel's propagation is checked against.
    #[must_use]
    pub fn propagate_only(config: &SolverConfig) -> Self {
        Self {
            mode: 0,
            radius: 0,
            undo: 0,
            ..Self::solve(config)
        }
    }
}

/// One specialisation of the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelSpec {
    region: RegionShape,
    words_per_cell: u32,
    num_tiles: u32,
    invocations: u32,
    ring: u32,
}

impl KernelSpec {
    /// The kernel that solves regions of `region` under `ruleset`.
    #[must_use]
    pub fn new(region: RegionShape, ruleset: &Ruleset, config: &SolverConfig) -> Self {
        Self {
            region,
            words_per_cell: ruleset.words_per_cell(),
            num_tiles: ruleset.num_tiles(),
            invocations: config.invocations,
            ring: config.ring,
        }
    }

    /// The region shape it solves.
    #[must_use]
    pub const fn region(&self) -> RegionShape {
        self.region
    }

    /// Invocations per workgroup.
    #[must_use]
    pub const fn invocations(&self) -> u32 {
        self.invocations
    }

    /// Words the rule table holds.
    #[must_use]
    pub const fn rule_words(&self) -> u32 {
        AXES as u32 * self.num_tiles * self.words_per_cell
    }

    /// Words a region's domains hold.
    #[must_use]
    pub const fn domain_words(&self) -> u32 {
        self.region.cells() * self.words_per_cell
    }

    /// Words a region's checkpoint ring holds.
    #[must_use]
    pub const fn snapshot_words(&self) -> u32 {
        self.ring * self.domain_words()
    }

    /// Bytes of workgroup memory one region needs: its domains, a copy of the rule table, a change
    /// epoch and a selection key per cell, a key per lane, and the control block.
    #[must_use]
    pub const fn workgroup_bytes(&self) -> u32 {
        let cells = self.region.cells();
        (self.domain_words() + self.rule_words() + 2 * cells + self.invocations) * 4 + CONTROL_BYTES
    }

    /// Whether a device can run this kernel, and why not if it cannot.
    ///
    /// # Errors
    /// The needed and available workgroup memory, or invocation count.
    pub fn check(&self, limits: &BackendLimits) -> Result<(), (u32, u32)> {
        let needed = self.workgroup_bytes();
        if needed > limits.workgroup_storage {
            return Err((needed, limits.workgroup_storage));
        }
        if self.invocations > limits.invocations_per_workgroup {
            return Err((self.invocations, limits.invocations_per_workgroup));
        }
        Ok(())
    }

    /// The kernel's WGSL, with this specialisation's constants substituted.
    #[must_use]
    pub fn wgsl(&self) -> String {
        let cells = self.region.cells();
        SOURCE
            .replace("{CX}", &self.region.x.to_string())
            .replace("{CY}", &self.region.y.to_string())
            .replace("{CELLS}", &cells.to_string())
            .replace("{NT}", &self.num_tiles.to_string())
            .replace("{W}", &self.words_per_cell.to_string())
            .replace("{WG}", &self.invocations.to_string())
            .replace("{RULE_WORDS}", &self.rule_words().to_string())
            .replace("{DOM_WORDS}", &self.domain_words().to_string())
            .replace("{STATS}", &STATS_WORDS.to_string())
            .replace("{RING}", &self.ring.to_string())
            .replace("{MASK_PRELUDE}", &mask_prelude(self.words_per_cell))
    }
}

/// The mask type and its operations, with every word written out.
///
/// A cell's mask is a handful of words, and which word is which is known when the kernel is
/// specialised. Writing the words out keeps a mask in registers: indexing one dynamically, in a
/// loop over `words`, puts it in scratch memory instead, which measured twice the cost per step on
/// the city (docs/solver-fit.md). Masks up to four words are one `vec4`; wider ones are two, which
/// is why nothing here loops.
fn mask_prelude(words: u32) -> String {
    assert!(
        (1..=MAX_WORDS as u32).contains(&words),
        "{words} words is outside 1 to 8"
    );
    let vectors = words.div_ceil(4);
    let lanes = ["x", "y", "z", "w"];
    // Where word `index` lives in the mask value.
    let field = |value: &str, index: u32| -> String {
        let lane = lanes[(index % 4) as usize];
        if vectors == 1 {
            format!("{value}.{lane}")
        } else if index < 4 {
            format!("{value}.lo.{lane}")
        } else {
            format!("{value}.hi.{lane}")
        }
    };
    // A mask built from one expression per word, zero where the type is wider than the rule set.
    let build = |word: &dyn Fn(u32) -> String| -> String {
        let component = |vector: u32, lane: u32| {
            let index = vector * 4 + lane;
            if index < words {
                word(index)
            } else {
                "0u".to_owned()
            }
        };
        let vector = |vector: u32| {
            let lanes: Vec<String> = (0..4).map(|lane| component(vector, lane)).collect();
            format!("vec4<u32>({})", lanes.join(", "))
        };
        if vectors == 1 {
            vector(0)
        } else {
            format!("Mask({}, {})", vector(0), vector(1))
        }
    };
    let each = |separator: &str, line: &dyn Fn(u32) -> String| -> String {
        (0..words)
            .map(line)
            .collect::<Vec<String>>()
            .join(separator)
    };

    let mut prelude = String::new();
    if vectors == 1 {
        prelude.push_str("alias Mask = vec4<u32>;\n");
        prelude.push_str("fn mask_and(a: Mask, b: Mask) -> Mask { return a & b; }\n");
        prelude.push_str("fn mask_or(a: Mask, b: Mask) -> Mask { return a | b; }\n");
        prelude.push_str("fn mask_differs(a: Mask, b: Mask) -> bool { return any(a != b); }\n");
        prelude.push_str(
            "fn mask_empty(m: Mask) -> bool { return all(m == vec4<u32>(0u, 0u, 0u, 0u)); }\n",
        );
        prelude.push_str(
            "fn mask_count(m: Mask) -> u32 { let n = countOneBits(m); return n.x + n.y + n.z + n.w; }\n",
        );
    } else {
        prelude.push_str("struct Mask { lo: vec4<u32>, hi: vec4<u32> };\n");
        prelude.push_str(
            "fn mask_and(a: Mask, b: Mask) -> Mask { return Mask(a.lo & b.lo, a.hi & b.hi); }\n",
        );
        prelude.push_str(
            "fn mask_or(a: Mask, b: Mask) -> Mask { return Mask(a.lo | b.lo, a.hi | b.hi); }\n",
        );
        prelude.push_str(
            "fn mask_differs(a: Mask, b: Mask) -> bool { return any(a.lo != b.lo) || any(a.hi != b.hi); }\n",
        );
        prelude.push_str(
            "fn mask_empty(m: Mask) -> bool { let zero = vec4<u32>(0u, 0u, 0u, 0u); \
             return all(m.lo == zero) && all(m.hi == zero); }\n",
        );
        prelude.push_str(
            "fn mask_count(m: Mask) -> u32 { let lo = countOneBits(m.lo); let hi = countOneBits(m.hi); \
             return lo.x + lo.y + lo.z + lo.w + hi.x + hi.y + hi.z + hi.w; }\n",
        );
    }
    prelude.push_str(&format!(
        "fn mask_zero() -> Mask {{ return {}; }}\n",
        build(&|_| "0u".to_owned())
    ));
    // One tile, without writing a component the index chooses: every word is a select.
    prelude.push_str(&format!(
        "fn mask_one(tile: u32) -> Mask {{\n    let word = tile / 32u;\n    let bit = 1u << (tile % 32u);\n    return {};\n}}\n",
        build(&|index| format!("select(0u, bit, word == {index}u)"))
    ));
    prelude.push_str(&format!(
        "fn dom_load(c: u32) -> Mask {{ return {}; }}\n",
        build(&|index| format!("atomicLoad(&dom[c * W + {index}u])"))
    ));
    prelude.push_str(&format!(
        "fn dom_store(c: u32, m: Mask) {{ {} }}\n",
        each(" ", &|index| format!(
            "atomicStore(&dom[c * W + {index}u], {});",
            field("m", index)
        ))
    ));
    prelude.push_str(&format!(
        "fn init_load(at: u32) -> Mask {{ return {}; }}\n",
        build(&|index| format!("init[at + {index}u]"))
    ));
    prelude.push_str(&format!(
        "fn out_store(at: u32, m: Mask) {{ {} }}\n",
        each(" ", &|index| format!(
            "out[at + {index}u] = {};",
            field("m", index)
        ))
    ));
    prelude.push_str(&format!(
        "fn snap_load(at: u32) -> Mask {{ return {}; }}\n",
        build(&|index| format!("snaps[at + {index}u]"))
    ));
    prelude.push_str(&format!(
        "fn snap_store(at: u32, m: Mask) {{ {} }}\n",
        each(" ", &|index| format!(
            "snaps[at + {index}u] = {};",
            field("m", index)
        ))
    ));
    prelude.push_str(&format!(
        "fn rule_row(row: u32) -> Mask {{ return {}; }}\n",
        build(&|index| format!("rules_s[row + {index}u]"))
    ));
    // The tiles allowed along `axis` of a cell holding any tile of `m`, one word at a time.
    prelude.push_str(
        "fn allowed_by(m: Mask, axis: u32) -> Mask {\n    var acc = mask_zero();\n    var bits = 0u;\n",
    );
    prelude.push_str(&each("\n", &|index| {
        format!(
            "    bits = {};\n    while (bits != 0u) {{\n        let row_{index} = (axis * NT + {index}u * 32u + countTrailingZeros(bits)) * W;\n        bits &= bits - 1u;\n        acc = mask_or(acc, rule_row(row_{index}));\n    }}",
            field("m", index)
        )
    }));
    prelude.push_str("\n    return acc;\n}\n");
    // The tile `hash` picks out of `m`, in proportion to the weights.
    prelude.push_str(
        "fn weighted_tile(m: Mask, hash: u32) -> u32 {\n    var total = 0u;\n    var first = NONE;\n    var bits = 0u;\n",
    );
    prelude.push_str(&each("\n", &|index| {
        format!(
            "    bits = {};\n    while (bits != 0u) {{\n        let tile_{index} = {index}u * 32u + countTrailingZeros(bits);\n        bits &= bits - 1u;\n        total += weights[tile_{index}];\n        if (first == NONE) {{ first = tile_{index}; }}\n    }}",
            field("m", index)
        )
    }));
    prelude.push_str(
        "\n    if (total == 0u) {\n        return first;\n    }\n    var pick = hash % total;\n",
    );
    prelude.push_str(&each("\n", &|index| {
        format!(
            "    bits = {};\n    while (bits != 0u) {{\n        let tile_{index} = {index}u * 32u + countTrailingZeros(bits);\n        bits &= bits - 1u;\n        let weight_{index} = weights[tile_{index}];\n        if (pick < weight_{index}) {{ return tile_{index}; }}\n        pick -= weight_{index};\n    }}",
            field("m", index)
        )
    }));
    prelude.push_str("\n    return first;\n}\n");
    prelude
}

#[cfg(test)]
mod tests {
    use super::*;
    use wfc_core::ChunkShape;
    use wfc_rules::AdjacencyRules;

    fn ruleset(tiles: usize) -> Ruleset {
        let rules = AdjacencyRules::from_allowed_tuples(tiles, AXES, [(0, 0, 0)]);
        Ruleset::new(&rules, &vec![1.0; tiles]).expect("a rule set")
    }

    fn spec(tiles: usize) -> KernelSpec {
        let region = ChunkShape::cube(8).region([1, 1, 0]);
        KernelSpec::new(region, &ruleset(tiles), &SolverConfig::default())
    }

    /// The `{NAME}` placeholders a source declares. A brace followed by an upper-case letter is a
    /// placeholder; a brace followed by anything else opens a WGSL block.
    fn placeholders(source: &str) -> Vec<String> {
        source
            .split('{')
            .skip(1)
            .filter(|rest| rest.starts_with(|c: char| c.is_ascii_uppercase()))
            .filter_map(|rest| rest.split_once('}'))
            .map(|(name, _)| name.to_owned())
            .collect()
    }

    #[test]
    fn the_source_has_every_constant_substituted() {
        let wgsl = spec(81).wgsl();

        assert!(
            !placeholders(SOURCE).is_empty(),
            "the source declares the placeholders this substitutes"
        );
        assert_eq!(
            placeholders(&wgsl),
            Vec::<String>::new(),
            "a placeholder was left unsubstituted"
        );
        assert!(
            wgsl.contains("const CELLS: u32 = 800u;"),
            "10 x 10 x 8 cells"
        );
        assert!(
            wgsl.contains("const W: u32 = 3u;"),
            "81 tiles need three words"
        );
        assert!(
            wgsl.contains("alias Mask = vec4<u32>;"),
            "three words fit one vector"
        );
        assert!(
            wgsl.contains("fn dom_load(c: u32) -> Mask { return vec4<u32>(atomicLoad(&dom[c * W + 0u]), atomicLoad(&dom[c * W + 1u]), atomicLoad(&dom[c * W + 2u]), 0u); }"),
            "the words are written out, and the word the rule set does not use is zero"
        );
        assert!(wgsl.contains("@workgroup_size(256)"));
    }

    #[test]
    fn a_rule_set_wider_than_four_words_uses_two_vectors() {
        let wgsl = spec(200).wgsl();

        assert!(
            wgsl.contains("struct Mask { lo: vec4<u32>, hi: vec4<u32> };"),
            "seven words need two vectors"
        );
        assert!(
            wgsl.contains("m.hi.z"),
            "the seventh word is the third lane of the second vector"
        );
        assert!(
            !wgsl.contains("m.hi.w"),
            "the eighth word is not used by 200 tiles"
        );
    }

    #[test]
    fn the_workgroup_budget_counts_what_the_kernel_declares() {
        let spec = spec(81);

        // Domains 9600 B, rules 5832 B, epochs 3200 B, keys 1024 B, cell keys 3200 B, control 72 B.
        assert_eq!(spec.workgroup_bytes(), 22_928);
        assert_eq!(spec.rule_words(), 6 * 81 * 3);
        assert_eq!(spec.snapshot_words(), 32 * 800 * 3);
    }

    #[test]
    fn a_kernel_that_does_not_fit_says_so_in_numbers() {
        let tight = BackendLimits {
            workgroup_storage: 16_384,
            invocations_per_workgroup: 256,
            workgroups: 65_535,
        };

        let refused = spec(81).check(&tight);

        assert_eq!(refused, Err((22_928, 16_384)));
        assert!(spec(32).check(&tight).is_ok(), "a smaller rule set fits");
    }
}

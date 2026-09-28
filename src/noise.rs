// The noise here is a port of FastNoiseLite 1.1.0 (https://github.com/Auburn/FastNoiseLite), and
// its configuration and `NoiseConfig::sample` follow Godot's FastNoiseLite resource
// (modules/noise/fastnoise_lite.cpp). Both are MIT licensed:
//
// Copyright(c) 2023 Jordan Peck (jordan.me2@gmail.com)
// Copyright(c) 2023 Contributors
// Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md).
// Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Noise as Godot makes it: a port of FastNoiseLite 1.1.0's 2D noise and 2D domain warp.
//!
//! [`NoiseConfig`] has the properties of Godot's `FastNoiseLite` resource, with the same names
//! and defaults, so a noise tuned in Godot's inspector reads the same in a pack.
//! [`NoiseConfig::sample`] equals Godot 4.7's `FastNoiseLite.get_noise_2d` for the same
//! properties: it does the same `f32` operations in the same order, with the same wrapping integer
//! hashes, and `tests/fastnoise.rs` holds it to values Godot itself computed.
//!
//! 3D noise comes with density volumes.
//!
//! FastNoiseLite is Copyright (c) 2023 Jordan Peck and contributors, and Godot's resource is
//! Copyright (c) 2014-present Godot Engine contributors; both are MIT licensed, and the notice is
//! at the top of this file's source.

// Constants and tables keep FastNoiseLite's digits, so they can be checked against its source.
#![allow(clippy::excessive_precision)]

use serde::{Deserialize, Serialize};

/// The noise algorithm, in the order of Godot's `FastNoiseLite.NoiseType`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
pub enum NoiseType {
    /// OpenSimplex2 (`TYPE_SIMPLEX`).
    Simplex,
    /// OpenSimplex2S, smoother than [`Self::Simplex`] (`TYPE_SIMPLEX_SMOOTH`).
    #[default]
    SimplexSmooth,
    /// Distances to jittered points on a grid (`TYPE_CELLULAR`).
    Cellular,
    /// Gradient noise on a square grid (`TYPE_PERLIN`).
    Perlin,
    /// Value noise with cubic interpolation (`TYPE_VALUE_CUBIC`).
    ValueCubic,
    /// Value noise with Hermite interpolation (`TYPE_VALUE`).
    Value,
}

/// How octaves are combined, in the order of Godot's `FastNoiseLite.FractalType`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
pub enum FractalType {
    /// One octave (`FRACTAL_NONE`).
    None,
    /// Fractional Brownian motion: octaves summed (`FRACTAL_FBM`).
    #[default]
    Fbm,
    /// Ridges where each octave crosses zero (`FRACTAL_RIDGED`).
    Ridged,
    /// Each octave folded back and forth (`FRACTAL_PING_PONG`).
    PingPong,
}

/// How cellular noise measures distance, in the order of Godot's
/// `FastNoiseLite.CellularDistanceFunction`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
pub enum CellularDistanceFunction {
    /// `DISTANCE_EUCLIDEAN`.
    #[default]
    Euclidean,
    /// `DISTANCE_EUCLIDEAN_SQUARED`.
    EuclideanSquared,
    /// `DISTANCE_MANHATTAN`.
    Manhattan,
    /// Manhattan plus squared Euclidean (`DISTANCE_HYBRID`).
    Hybrid,
}

/// What cellular noise returns, in the order of Godot's `FastNoiseLite.CellularReturnType`.
/// "Distance" is to the nearest point and "Distance2" to the second nearest.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
pub enum CellularReturnType {
    /// A random value per cell (`RETURN_CELL_VALUE`).
    CellValue,
    /// `RETURN_DISTANCE`.
    #[default]
    Distance,
    /// `RETURN_DISTANCE2`.
    Distance2,
    /// `RETURN_DISTANCE2_ADD`.
    Distance2Add,
    /// `RETURN_DISTANCE2_SUB`.
    Distance2Sub,
    /// `RETURN_DISTANCE2_MUL`.
    Distance2Mul,
    /// `RETURN_DISTANCE2_DIV`.
    Distance2Div,
}

/// The domain warp algorithm, in the order of Godot's `FastNoiseLite.DomainWarpType`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
pub enum DomainWarpType {
    /// Along OpenSimplex2 gradients (`DOMAIN_WARP_SIMPLEX`).
    #[default]
    Simplex,
    /// Along OpenSimplex2 gradients with fewer lookups (`DOMAIN_WARP_SIMPLEX_REDUCED`).
    SimplexReduced,
    /// Along vectors interpolated on a square grid (`DOMAIN_WARP_BASIC_GRID`).
    BasicGrid,
}

/// How domain warp octaves are combined, in the order of Godot's
/// `FastNoiseLite.DomainWarpFractalType`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
pub enum DomainWarpFractalType {
    /// One octave (`DOMAIN_WARP_FRACTAL_NONE`).
    None,
    /// Each octave warps the position the previous one left (`DOMAIN_WARP_FRACTAL_PROGRESSIVE`).
    #[default]
    Progressive,
    /// Each octave reads the unwarped position (`DOMAIN_WARP_FRACTAL_INDEPENDENT`).
    Independent,
}

/// A noise configured as Godot's `FastNoiseLite` resource is: the same properties under the same
/// names, and the same defaults.
///
/// In a file every property may be left out, and takes Godot's default; written out, every
/// property is there.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::Reflect))]
#[serde(default, deny_unknown_fields)]
pub struct NoiseConfig {
    /// The algorithm. Default [`NoiseType::SimplexSmooth`].
    pub noise_type: NoiseType,
    /// Seeds the noise and the domain warp alike. Default 0.
    pub seed: i32,
    /// Lattice points per unit of the first octave. Default 0.01.
    pub frequency: f32,
    /// Added to every position before anything else; [`Self::sample`] reads `x` and `y`. Default
    /// zero.
    pub offset: [f32; 3],
    /// How octaves are combined. Default [`FractalType::Fbm`].
    pub fractal_type: FractalType,
    /// How many octaves. Default 5.
    pub fractal_octaves: i32,
    /// Frequency multiplier from one octave to the next. Default 2.
    pub fractal_lacunarity: f32,
    /// Amplitude multiplier from one octave to the next. Default 0.5.
    pub fractal_gain: f32,
    /// How much a low octave damps the next ones, from 0 to 1. Default 0.
    pub fractal_weighted_strength: f32,
    /// How often [`FractalType::PingPong`] folds. Default 2.
    pub fractal_ping_pong_strength: f32,
    /// How cellular noise measures distance. Default [`CellularDistanceFunction::Euclidean`].
    pub cellular_distance_function: CellularDistanceFunction,
    /// What cellular noise returns. Default [`CellularReturnType::Distance`].
    pub cellular_return_type: CellularReturnType,
    /// How far a cellular point may stray from its grid position; above 1 gives artifacts.
    /// Default 1.
    pub cellular_jitter: f32,
    /// Whether positions are warped before the noise is read. Default false.
    pub domain_warp_enabled: bool,
    /// The warp algorithm. Default [`DomainWarpType::Simplex`].
    pub domain_warp_type: DomainWarpType,
    /// How far the warp moves a position, at most. Default 30.
    pub domain_warp_amplitude: f32,
    /// Lattice points per unit of the warp's first octave. Default 0.05.
    pub domain_warp_frequency: f32,
    /// How warp octaves are combined. Default [`DomainWarpFractalType::Progressive`].
    pub domain_warp_fractal_type: DomainWarpFractalType,
    /// How many warp octaves. Default 5.
    pub domain_warp_fractal_octaves: i32,
    /// Warp frequency multiplier from one octave to the next. Default 6.
    pub domain_warp_fractal_lacunarity: f32,
    /// Warp amplitude multiplier from one octave to the next. Default 0.5.
    pub domain_warp_fractal_gain: f32,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            noise_type: NoiseType::SimplexSmooth,
            seed: 0,
            frequency: 0.01,
            offset: [0.0; 3],
            fractal_type: FractalType::Fbm,
            fractal_octaves: 5,
            fractal_lacunarity: 2.0,
            fractal_gain: 0.5,
            fractal_weighted_strength: 0.0,
            fractal_ping_pong_strength: 2.0,
            cellular_distance_function: CellularDistanceFunction::Euclidean,
            cellular_return_type: CellularReturnType::Distance,
            cellular_jitter: 1.0,
            domain_warp_enabled: false,
            domain_warp_type: DomainWarpType::Simplex,
            domain_warp_amplitude: 30.0,
            domain_warp_frequency: 0.05,
            domain_warp_fractal_type: DomainWarpFractalType::Progressive,
            domain_warp_fractal_octaves: 5,
            domain_warp_fractal_lacunarity: 6.0,
            domain_warp_fractal_gain: 0.5,
        }
    }
}

impl NoiseConfig {
    /// The noise at `(x, y)`, as Godot's `get_noise_2d` gives it: the offset added, the position
    /// warped if the warp is enabled, then the noise read there. Mostly within -1 to 1; weighted
    /// strengths, jitters and gains outside their usual ranges can leave it.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let mut x = x + self.offset[0];
        let mut y = y + self.offset[1];
        if self.domain_warp_enabled {
            (x, y) = self.warp(x, y);
        }
        self.noise(x, y)
    }

    /// FastNoiseLite's `GetNoise`: frequency and skew, then the fractal.
    fn noise(&self, x: f32, y: f32) -> f32 {
        let mut x = x * self.frequency;
        let mut y = y * self.frequency;
        if matches!(
            self.noise_type,
            NoiseType::Simplex | NoiseType::SimplexSmooth
        ) {
            let t = (x + y) * F2;
            x += t;
            y += t;
        }
        if self.fractal_type == FractalType::None {
            return self.single(self.seed, x, y);
        }
        let mut seed = self.seed;
        let mut sum = 0.0;
        let mut amp = fractal_bounding(self.fractal_octaves, self.fractal_gain);
        for _ in 0..self.fractal_octaves {
            let noise = self.single(seed, x, y);
            seed = seed.wrapping_add(1);
            // Each fractal's octave term and how much it weighs the next octaves.
            let (term, weight) = match self.fractal_type {
                FractalType::None => unreachable!("a single octave has returned above"),
                FractalType::Fbm => (noise, fast_min(noise + 1.0, 2.0) * 0.5),
                FractalType::Ridged => {
                    let ridge = fast_abs(noise);
                    (ridge * -2.0 + 1.0, 1.0 - ridge)
                }
                FractalType::PingPong => {
                    let folded = ping_pong((noise + 1.0) * self.fractal_ping_pong_strength);
                    ((folded - 0.5) * 2.0, folded)
                }
            };
            sum += term * amp;
            amp *= lerp(1.0, weight, self.fractal_weighted_strength);
            x *= self.fractal_lacunarity;
            y *= self.fractal_lacunarity;
            amp *= self.fractal_gain;
        }
        sum
    }

    /// FastNoiseLite's `GenNoiseSingle`: one octave at a transformed position.
    fn single(&self, seed: i32, x: f32, y: f32) -> f32 {
        match self.noise_type {
            NoiseType::Simplex => single_simplex(seed, x, y),
            NoiseType::SimplexSmooth => single_simplex_smooth(seed, x, y),
            NoiseType::Cellular => self.single_cellular(seed, x, y),
            NoiseType::Perlin => single_perlin(seed, x, y),
            NoiseType::ValueCubic => single_value_cubic(seed, x, y),
            NoiseType::Value => single_value(seed, x, y),
        }
    }

    fn single_cellular(&self, seed: i32, x: f32, y: f32) -> f32 {
        let xr = fast_round(x);
        let yr = fast_round(y);
        let mut distance0 = 1e10_f32;
        let mut distance1 = 1e10_f32;
        let mut closest_hash = 0;
        let jitter = 0.437_015_95 * self.cellular_jitter;

        let mut x_primed = (xr - 1).wrapping_mul(PRIME_X);
        let y_primed_base = (yr - 1).wrapping_mul(PRIME_Y);
        for xi in xr - 1..=xr + 1 {
            let mut y_primed = y_primed_base;
            for yi in yr - 1..=yr + 1 {
                let hash = hash(seed, x_primed, y_primed);
                let index = (hash & (255 << 1)) as usize;
                let vec_x = (xi as f32 - x) + RAND_VECS_2D[index] * jitter;
                let vec_y = (yi as f32 - y) + RAND_VECS_2D[index | 1] * jitter;
                let new_distance = match self.cellular_distance_function {
                    CellularDistanceFunction::Euclidean
                    | CellularDistanceFunction::EuclideanSquared => vec_x * vec_x + vec_y * vec_y,
                    CellularDistanceFunction::Manhattan => fast_abs(vec_x) + fast_abs(vec_y),
                    CellularDistanceFunction::Hybrid => {
                        (fast_abs(vec_x) + fast_abs(vec_y)) + (vec_x * vec_x + vec_y * vec_y)
                    }
                };
                distance1 = fast_max(fast_min(distance1, new_distance), distance0);
                if new_distance < distance0 {
                    distance0 = new_distance;
                    closest_hash = hash;
                }
                y_primed = y_primed.wrapping_add(PRIME_Y);
            }
            x_primed = x_primed.wrapping_add(PRIME_X);
        }

        if self.cellular_distance_function == CellularDistanceFunction::Euclidean
            && self.cellular_return_type != CellularReturnType::CellValue
        {
            distance0 = distance0.sqrt();
            if self.cellular_return_type != CellularReturnType::Distance {
                distance1 = distance1.sqrt();
            }
        }

        match self.cellular_return_type {
            CellularReturnType::CellValue => closest_hash as f32 * (1.0 / 2_147_483_648.0),
            CellularReturnType::Distance => distance0 - 1.0,
            CellularReturnType::Distance2 => distance1 - 1.0,
            CellularReturnType::Distance2Add => (distance1 + distance0) * 0.5 - 1.0,
            CellularReturnType::Distance2Sub => distance1 - distance0 - 1.0,
            CellularReturnType::Distance2Mul => distance1 * distance0 * 0.5 - 1.0,
            CellularReturnType::Distance2Div => distance0 / distance1 - 1.0,
        }
    }

    /// FastNoiseLite's `DomainWarp` on the separate warp object Godot keeps: it shares the seed and
    /// has its own frequency, amplitude, type and fractal settings.
    fn warp(&self, x: f32, y: f32) -> (f32, f32) {
        let mut seed = self.seed;
        // Godot's warp scales by the fractal bounding of its octaves and gain even when its fractal
        // type is none, as FastNoiseLite's `DomainWarpSingle` does.
        let mut amp = self.domain_warp_amplitude
            * fractal_bounding(
                self.domain_warp_fractal_octaves,
                self.domain_warp_fractal_gain,
            );
        let mut freq = self.domain_warp_frequency;
        let mut x = x;
        let mut y = y;
        match self.domain_warp_fractal_type {
            DomainWarpFractalType::None => {
                let (xs, ys) = self.warp_skew(x, y);
                let (dx, dy) = self.warp_single(seed, amp, freq, xs, ys);
                (x + dx, y + dy)
            }
            DomainWarpFractalType::Progressive => {
                for _ in 0..self.domain_warp_fractal_octaves {
                    let (xs, ys) = self.warp_skew(x, y);
                    let (dx, dy) = self.warp_single(seed, amp, freq, xs, ys);
                    x += dx;
                    y += dy;
                    seed = seed.wrapping_add(1);
                    amp *= self.domain_warp_fractal_gain;
                    freq *= self.domain_warp_fractal_lacunarity;
                }
                (x, y)
            }
            DomainWarpFractalType::Independent => {
                let (xs, ys) = self.warp_skew(x, y);
                for _ in 0..self.domain_warp_fractal_octaves {
                    let (dx, dy) = self.warp_single(seed, amp, freq, xs, ys);
                    x += dx;
                    y += dy;
                    seed = seed.wrapping_add(1);
                    amp *= self.domain_warp_fractal_gain;
                    freq *= self.domain_warp_fractal_lacunarity;
                }
                (x, y)
            }
        }
    }

    /// FastNoiseLite's `TransformDomainWarpCoordinate`: the simplex warps read a skewed position.
    fn warp_skew(&self, x: f32, y: f32) -> (f32, f32) {
        match self.domain_warp_type {
            DomainWarpType::Simplex | DomainWarpType::SimplexReduced => {
                let t = (x + y) * F2;
                (x + t, y + t)
            }
            DomainWarpType::BasicGrid => (x, y),
        }
    }

    /// FastNoiseLite's `DoSingleDomainWarp`: how far one warp octave moves a position.
    fn warp_single(&self, seed: i32, amp: f32, freq: f32, x: f32, y: f32) -> (f32, f32) {
        match self.domain_warp_type {
            DomainWarpType::Simplex => {
                warp_simplex_gradient(seed, amp * 38.283_687_591_552_734_375, freq, x, y, false)
            }
            DomainWarpType::SimplexReduced => {
                warp_simplex_gradient(seed, amp * 16.0, freq, x, y, true)
            }
            DomainWarpType::BasicGrid => warp_basic_grid(seed, amp, freq, x, y),
        }
    }
}

const PRIME_X: i32 = 501_125_321;
const PRIME_Y: i32 = 1_136_930_381;

const SQRT3: f32 = 1.7320508075688772935274463415059;
/// The skew from the square grid to the simplex grid.
const F2: f32 = 0.5 * (SQRT3 - 1.0);
/// The unskew back.
const G2: f32 = (3.0 - SQRT3) / 6.0;
/// The far corner's falloff in `SingleSimplex` and its relatives, derived from the first corner's
/// as `CORNER2_T * t + (CORNER2_A + a)`.
const CORNER2_T: f32 = 2.0 * (1.0 - 2.0 * G2) * (1.0 / G2 - 2.0);
const CORNER2_A: f32 = -2.0 * (1.0 - 2.0 * G2) * (1.0 - 2.0 * G2);

/// One over the largest sum `octaves` octaves of `gain` can reach, so a fractal stays in -1 to 1.
fn fractal_bounding(octaves: i32, gain: f32) -> f32 {
    let gain = fast_abs(gain);
    let mut amp = gain;
    let mut amp_fractal = 1.0;
    for _ in 1..octaves {
        amp_fractal += amp;
        amp *= gain;
    }
    1.0 / amp_fractal
}

// FastNoiseLite's helpers keep C++ semantics where they differ from Rust's: its comparisons treat
// NaN and signed zeros their own way, and `(int)` truncates toward zero.

fn fast_min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

fn fast_max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

fn fast_abs(f: f32) -> f32 {
    if f < 0.0 { -f } else { f }
}

/// FastNoiseLite's `FastFloor`, which gives one less than the floor at a negative integer.
fn fast_floor(f: f32) -> i32 {
    if f >= 0.0 { f as i32 } else { f as i32 - 1 }
}

fn fast_round(f: f32) -> i32 {
    if f >= 0.0 {
        (f + 0.5) as i32
    } else {
        (f - 0.5) as i32
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

fn interp_hermite(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn interp_quintic(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn cubic_lerp(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    let p = (d - c) - (a - b);
    t * t * t * p + t * t * ((a - b) - p) + t * (c - a) + b
}

fn ping_pong(t: f32) -> f32 {
    let t = t - ((t * 0.5) as i32).wrapping_mul(2) as f32;
    if t < 1.0 { t } else { 2.0 - t }
}

fn hash(seed: i32, x_primed: i32, y_primed: i32) -> i32 {
    (seed ^ x_primed ^ y_primed).wrapping_mul(0x27d4_eb2d)
}

fn val_coord(seed: i32, x_primed: i32, y_primed: i32) -> f32 {
    let mut hash = hash(seed, x_primed, y_primed);
    hash = hash.wrapping_mul(hash);
    hash ^= hash << 19;
    hash as f32 * (1.0 / 2_147_483_648.0)
}

fn grad_coord(seed: i32, x_primed: i32, y_primed: i32, xd: f32, yd: f32) -> f32 {
    let mut hash = hash(seed, x_primed, y_primed);
    hash ^= hash >> 15;
    let index = (hash & (127 << 1)) as usize;
    xd * GRADIENTS_2D[index] + yd * GRADIENTS_2D[index | 1]
}

/// FastNoiseLite's `GradCoordOut`: a random unit vector for a lattice point.
fn grad_coord_out(seed: i32, x_primed: i32, y_primed: i32) -> (f32, f32) {
    let index = (hash(seed, x_primed, y_primed) & (255 << 1)) as usize;
    (RAND_VECS_2D[index], RAND_VECS_2D[index | 1])
}

/// FastNoiseLite's `GradCoordDual`: a random unit vector scaled by a gradient's value.
fn grad_coord_dual(seed: i32, x_primed: i32, y_primed: i32, xd: f32, yd: f32) -> (f32, f32) {
    let hash = hash(seed, x_primed, y_primed);
    let index1 = (hash & (127 << 1)) as usize;
    let index2 = ((hash >> 7) & (255 << 1)) as usize;
    let value = xd * GRADIENTS_2D[index1] + yd * GRADIENTS_2D[index1 | 1];
    (
        value * RAND_VECS_2D[index2],
        value * RAND_VECS_2D[index2 | 1],
    )
}

/// FastNoiseLite's 2D `SingleSimplex`, which OpenSimplex2 is in 2D.
fn single_simplex(seed: i32, x: f32, y: f32) -> f32 {
    let i = fast_floor(x);
    let j = fast_floor(y);
    let xi = x - i as f32;
    let yi = y - j as f32;

    let t = (xi + yi) * G2;
    let x0 = xi - t;
    let y0 = yi - t;

    let i = i.wrapping_mul(PRIME_X);
    let j = j.wrapping_mul(PRIME_Y);

    let a = 0.5 - x0 * x0 - y0 * y0;
    let n0 = if a <= 0.0 {
        0.0
    } else {
        (a * a) * (a * a) * grad_coord(seed, i, j, x0, y0)
    };

    let c = CORNER2_T * t + (CORNER2_A + a);
    let n2 = if c <= 0.0 {
        0.0
    } else {
        let x2 = x0 + (2.0 * G2 - 1.0);
        let y2 = y0 + (2.0 * G2 - 1.0);
        (c * c)
            * (c * c)
            * grad_coord(
                seed,
                i.wrapping_add(PRIME_X),
                j.wrapping_add(PRIME_Y),
                x2,
                y2,
            )
    };

    let (x1, y1, i1, j1) = if y0 > x0 {
        (x0 + G2, y0 + (G2 - 1.0), i, j.wrapping_add(PRIME_Y))
    } else {
        (x0 + (G2 - 1.0), y0 + G2, i.wrapping_add(PRIME_X), j)
    };
    let b = 0.5 - x1 * x1 - y1 * y1;
    let n1 = if b <= 0.0 {
        0.0
    } else {
        (b * b) * (b * b) * grad_coord(seed, i1, j1, x1, y1)
    };

    (n0 + n1 + n2) * 99.836_854_463_036_47
}

/// FastNoiseLite's 2D `SingleOpenSimplex2S`.
fn single_simplex_smooth(seed: i32, x: f32, y: f32) -> f32 {
    let i = fast_floor(x);
    let j = fast_floor(y);
    let xi = x - i as f32;
    let yi = y - j as f32;

    let i = i.wrapping_mul(PRIME_X);
    let j = j.wrapping_mul(PRIME_Y);
    let i1 = i.wrapping_add(PRIME_X);
    let j1 = j.wrapping_add(PRIME_Y);

    let t = (xi + yi) * G2;
    let x0 = xi - t;
    let y0 = yi - t;

    let a0 = (2.0 / 3.0) - x0 * x0 - y0 * y0;
    let mut value = (a0 * a0) * (a0 * a0) * grad_coord(seed, i, j, x0, y0);

    let a1 = CORNER2_T * t + (CORNER2_A + a0);
    let x1 = x0 - (1.0 - 2.0 * G2);
    let y1 = y0 - (1.0 - 2.0 * G2);
    value += (a1 * a1) * (a1 * a1) * grad_coord(seed, i1, j1, x1, y1);

    // The two further corners, chosen by where the point lies in its cell.
    let xmyi = xi - yi;
    let (x2, y2, i2, j2, x3, y3, i3, j3);
    if t > G2 {
        (x2, y2, i2, j2) = if xi + xmyi > 1.0 {
            (
                x0 + (3.0 * G2 - 2.0),
                y0 + (3.0 * G2 - 1.0),
                i.wrapping_add(PRIME_X << 1),
                j.wrapping_add(PRIME_Y),
            )
        } else {
            (x0 + G2, y0 + (G2 - 1.0), i, j.wrapping_add(PRIME_Y))
        };
        (x3, y3, i3, j3) = if yi - xmyi > 1.0 {
            (
                x0 + (3.0 * G2 - 1.0),
                y0 + (3.0 * G2 - 2.0),
                i.wrapping_add(PRIME_X),
                j.wrapping_add(PRIME_Y << 1),
            )
        } else {
            (x0 + (G2 - 1.0), y0 + G2, i.wrapping_add(PRIME_X), j)
        };
    } else {
        (x2, y2, i2, j2) = if xi + xmyi < 0.0 {
            (x0 + (1.0 - G2), y0 - G2, i.wrapping_sub(PRIME_X), j)
        } else {
            (x0 + (G2 - 1.0), y0 + G2, i.wrapping_add(PRIME_X), j)
        };
        (x3, y3, i3, j3) = if yi < xmyi {
            (x0 - G2, y0 - (G2 - 1.0), i, j.wrapping_sub(PRIME_Y))
        } else {
            (x0 + G2, y0 + (G2 - 1.0), i, j.wrapping_add(PRIME_Y))
        };
    }
    let a2 = (2.0 / 3.0) - x2 * x2 - y2 * y2;
    if a2 > 0.0 {
        value += (a2 * a2) * (a2 * a2) * grad_coord(seed, i2, j2, x2, y2);
    }
    let a3 = (2.0 / 3.0) - x3 * x3 - y3 * y3;
    if a3 > 0.0 {
        value += (a3 * a3) * (a3 * a3) * grad_coord(seed, i3, j3, x3, y3);
    }

    value * 18.241_961_944_860_65
}

fn single_perlin(seed: i32, x: f32, y: f32) -> f32 {
    let x0 = fast_floor(x);
    let y0 = fast_floor(y);

    let xd0 = x - x0 as f32;
    let yd0 = y - y0 as f32;
    let xd1 = xd0 - 1.0;
    let yd1 = yd0 - 1.0;

    let xs = interp_quintic(xd0);
    let ys = interp_quintic(yd0);

    let x0 = x0.wrapping_mul(PRIME_X);
    let y0 = y0.wrapping_mul(PRIME_Y);
    let x1 = x0.wrapping_add(PRIME_X);
    let y1 = y0.wrapping_add(PRIME_Y);

    let xf0 = lerp(
        grad_coord(seed, x0, y0, xd0, yd0),
        grad_coord(seed, x1, y0, xd1, yd0),
        xs,
    );
    let xf1 = lerp(
        grad_coord(seed, x0, y1, xd0, yd1),
        grad_coord(seed, x1, y1, xd1, yd1),
        xs,
    );

    lerp(xf0, xf1, ys) * 1.424_769_110_467_781_3
}

fn single_value_cubic(seed: i32, x: f32, y: f32) -> f32 {
    let x1 = fast_floor(x);
    let y1 = fast_floor(y);

    let xs = x - x1 as f32;
    let ys = y - y1 as f32;

    let x1 = x1.wrapping_mul(PRIME_X);
    let y1 = y1.wrapping_mul(PRIME_Y);
    let x0 = x1.wrapping_sub(PRIME_X);
    let y0 = y1.wrapping_sub(PRIME_Y);
    let x2 = x1.wrapping_add(PRIME_X);
    let y2 = y1.wrapping_add(PRIME_Y);
    let x3 = x1.wrapping_add(PRIME_X << 1);
    let y3 = y1.wrapping_add(PRIME_Y << 1);

    let row = |y_primed: i32| {
        cubic_lerp(
            val_coord(seed, x0, y_primed),
            val_coord(seed, x1, y_primed),
            val_coord(seed, x2, y_primed),
            val_coord(seed, x3, y_primed),
            xs,
        )
    };
    cubic_lerp(row(y0), row(y1), row(y2), row(y3), ys) * (1.0 / (1.5 * 1.5))
}

fn single_value(seed: i32, x: f32, y: f32) -> f32 {
    let x0 = fast_floor(x);
    let y0 = fast_floor(y);

    let xs = interp_hermite(x - x0 as f32);
    let ys = interp_hermite(y - y0 as f32);

    let x0 = x0.wrapping_mul(PRIME_X);
    let y0 = y0.wrapping_mul(PRIME_Y);
    let x1 = x0.wrapping_add(PRIME_X);
    let y1 = y0.wrapping_add(PRIME_Y);

    let xf0 = lerp(val_coord(seed, x0, y0), val_coord(seed, x1, y0), xs);
    let xf1 = lerp(val_coord(seed, x0, y1), val_coord(seed, x1, y1), xs);

    lerp(xf0, xf1, ys)
}

/// FastNoiseLite's 2D `SingleDomainWarpBasicGrid`: how far it moves `(x, y)`.
fn warp_basic_grid(seed: i32, warp_amp: f32, frequency: f32, x: f32, y: f32) -> (f32, f32) {
    let xf = x * frequency;
    let yf = y * frequency;

    let x0 = fast_floor(xf);
    let y0 = fast_floor(yf);

    let xs = interp_hermite(xf - x0 as f32);
    let ys = interp_hermite(yf - y0 as f32);

    let x0 = x0.wrapping_mul(PRIME_X);
    let y0 = y0.wrapping_mul(PRIME_Y);
    let x1 = x0.wrapping_add(PRIME_X);
    let y1 = y0.wrapping_add(PRIME_Y);

    let hash0 = (hash(seed, x0, y0) & (255 << 1)) as usize;
    let hash1 = (hash(seed, x1, y0) & (255 << 1)) as usize;
    let lx0x = lerp(RAND_VECS_2D[hash0], RAND_VECS_2D[hash1], xs);
    let ly0x = lerp(RAND_VECS_2D[hash0 | 1], RAND_VECS_2D[hash1 | 1], xs);

    let hash0 = (hash(seed, x0, y1) & (255 << 1)) as usize;
    let hash1 = (hash(seed, x1, y1) & (255 << 1)) as usize;
    let lx1x = lerp(RAND_VECS_2D[hash0], RAND_VECS_2D[hash1], xs);
    let ly1x = lerp(RAND_VECS_2D[hash0 | 1], RAND_VECS_2D[hash1 | 1], xs);

    (
        lerp(lx0x, lx1x, ys) * warp_amp,
        lerp(ly0x, ly1x, ys) * warp_amp,
    )
}

/// FastNoiseLite's 2D `SingleDomainWarpSimplexGradient`: how far it moves `(x, y)`, read at a
/// skewed position. `reduced` takes each corner's random vector alone (`outGradOnly`).
fn warp_simplex_gradient(
    seed: i32,
    warp_amp: f32,
    frequency: f32,
    x: f32,
    y: f32,
    reduced: bool,
) -> (f32, f32) {
    let corner = |x_primed: i32, y_primed: i32, xd: f32, yd: f32| {
        if reduced {
            grad_coord_out(seed, x_primed, y_primed)
        } else {
            grad_coord_dual(seed, x_primed, y_primed, xd, yd)
        }
    };

    let x = x * frequency;
    let y = y * frequency;

    let i = fast_floor(x);
    let j = fast_floor(y);
    let xi = x - i as f32;
    let yi = y - j as f32;

    let t = (xi + yi) * G2;
    let x0 = xi - t;
    let y0 = yi - t;

    let i = i.wrapping_mul(PRIME_X);
    let j = j.wrapping_mul(PRIME_Y);

    let mut vx = 0.0;
    let mut vy = 0.0;

    let a = 0.5 - x0 * x0 - y0 * y0;
    if a > 0.0 {
        let aaaa = (a * a) * (a * a);
        let (xo, yo) = corner(i, j, x0, y0);
        vx += aaaa * xo;
        vy += aaaa * yo;
    }

    let c = CORNER2_T * t + (CORNER2_A + a);
    if c > 0.0 {
        let x2 = x0 + (2.0 * G2 - 1.0);
        let y2 = y0 + (2.0 * G2 - 1.0);
        let cccc = (c * c) * (c * c);
        let (xo, yo) = corner(i.wrapping_add(PRIME_X), j.wrapping_add(PRIME_Y), x2, y2);
        vx += cccc * xo;
        vy += cccc * yo;
    }

    let (x1, y1, i1, j1) = if y0 > x0 {
        (x0 + G2, y0 + (G2 - 1.0), i, j.wrapping_add(PRIME_Y))
    } else {
        (x0 + (G2 - 1.0), y0 + G2, i.wrapping_add(PRIME_X), j)
    };
    let b = 0.5 - x1 * x1 - y1 * y1;
    if b > 0.0 {
        let bbbb = (b * b) * (b * b);
        let (xo, yo) = corner(i1, j1, x1, y1);
        vx += bbbb * xo;
        vy += bbbb * yo;
    }

    (vx * warp_amp, vy * warp_amp)
}

// FastNoiseLite's 2D lookup tables, digit for digit.

const GRADIENTS_2D: [f32; 256] = [
    0.130526192220052,
    0.99144486137381,
    0.38268343236509,
    0.923879532511287,
    0.608761429008721,
    0.793353340291235,
    0.793353340291235,
    0.608761429008721,
    0.923879532511287,
    0.38268343236509,
    0.99144486137381,
    0.130526192220051,
    0.99144486137381,
    -0.130526192220051,
    0.923879532511287,
    -0.38268343236509,
    0.793353340291235,
    -0.60876142900872,
    0.608761429008721,
    -0.793353340291235,
    0.38268343236509,
    -0.923879532511287,
    0.130526192220052,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    -0.38268343236509,
    -0.923879532511287,
    -0.608761429008721,
    -0.793353340291235,
    -0.793353340291235,
    -0.608761429008721,
    -0.923879532511287,
    -0.38268343236509,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    0.130526192220051,
    -0.923879532511287,
    0.38268343236509,
    -0.793353340291235,
    0.608761429008721,
    -0.608761429008721,
    0.793353340291235,
    -0.38268343236509,
    0.923879532511287,
    -0.130526192220052,
    0.99144486137381,
    0.130526192220052,
    0.99144486137381,
    0.38268343236509,
    0.923879532511287,
    0.608761429008721,
    0.793353340291235,
    0.793353340291235,
    0.608761429008721,
    0.923879532511287,
    0.38268343236509,
    0.99144486137381,
    0.130526192220051,
    0.99144486137381,
    -0.130526192220051,
    0.923879532511287,
    -0.38268343236509,
    0.793353340291235,
    -0.60876142900872,
    0.608761429008721,
    -0.793353340291235,
    0.38268343236509,
    -0.923879532511287,
    0.130526192220052,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    -0.38268343236509,
    -0.923879532511287,
    -0.608761429008721,
    -0.793353340291235,
    -0.793353340291235,
    -0.608761429008721,
    -0.923879532511287,
    -0.38268343236509,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    0.130526192220051,
    -0.923879532511287,
    0.38268343236509,
    -0.793353340291235,
    0.608761429008721,
    -0.608761429008721,
    0.793353340291235,
    -0.38268343236509,
    0.923879532511287,
    -0.130526192220052,
    0.99144486137381,
    0.130526192220052,
    0.99144486137381,
    0.38268343236509,
    0.923879532511287,
    0.608761429008721,
    0.793353340291235,
    0.793353340291235,
    0.608761429008721,
    0.923879532511287,
    0.38268343236509,
    0.99144486137381,
    0.130526192220051,
    0.99144486137381,
    -0.130526192220051,
    0.923879532511287,
    -0.38268343236509,
    0.793353340291235,
    -0.60876142900872,
    0.608761429008721,
    -0.793353340291235,
    0.38268343236509,
    -0.923879532511287,
    0.130526192220052,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    -0.38268343236509,
    -0.923879532511287,
    -0.608761429008721,
    -0.793353340291235,
    -0.793353340291235,
    -0.608761429008721,
    -0.923879532511287,
    -0.38268343236509,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    0.130526192220051,
    -0.923879532511287,
    0.38268343236509,
    -0.793353340291235,
    0.608761429008721,
    -0.608761429008721,
    0.793353340291235,
    -0.38268343236509,
    0.923879532511287,
    -0.130526192220052,
    0.99144486137381,
    0.130526192220052,
    0.99144486137381,
    0.38268343236509,
    0.923879532511287,
    0.608761429008721,
    0.793353340291235,
    0.793353340291235,
    0.608761429008721,
    0.923879532511287,
    0.38268343236509,
    0.99144486137381,
    0.130526192220051,
    0.99144486137381,
    -0.130526192220051,
    0.923879532511287,
    -0.38268343236509,
    0.793353340291235,
    -0.60876142900872,
    0.608761429008721,
    -0.793353340291235,
    0.38268343236509,
    -0.923879532511287,
    0.130526192220052,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    -0.38268343236509,
    -0.923879532511287,
    -0.608761429008721,
    -0.793353340291235,
    -0.793353340291235,
    -0.608761429008721,
    -0.923879532511287,
    -0.38268343236509,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    0.130526192220051,
    -0.923879532511287,
    0.38268343236509,
    -0.793353340291235,
    0.608761429008721,
    -0.608761429008721,
    0.793353340291235,
    -0.38268343236509,
    0.923879532511287,
    -0.130526192220052,
    0.99144486137381,
    0.130526192220052,
    0.99144486137381,
    0.38268343236509,
    0.923879532511287,
    0.608761429008721,
    0.793353340291235,
    0.793353340291235,
    0.608761429008721,
    0.923879532511287,
    0.38268343236509,
    0.99144486137381,
    0.130526192220051,
    0.99144486137381,
    -0.130526192220051,
    0.923879532511287,
    -0.38268343236509,
    0.793353340291235,
    -0.60876142900872,
    0.608761429008721,
    -0.793353340291235,
    0.38268343236509,
    -0.923879532511287,
    0.130526192220052,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    -0.38268343236509,
    -0.923879532511287,
    -0.608761429008721,
    -0.793353340291235,
    -0.793353340291235,
    -0.608761429008721,
    -0.923879532511287,
    -0.38268343236509,
    -0.99144486137381,
    -0.130526192220052,
    -0.99144486137381,
    0.130526192220051,
    -0.923879532511287,
    0.38268343236509,
    -0.793353340291235,
    0.608761429008721,
    -0.608761429008721,
    0.793353340291235,
    -0.38268343236509,
    0.923879532511287,
    -0.130526192220052,
    0.99144486137381,
    0.38268343236509,
    0.923879532511287,
    0.923879532511287,
    0.38268343236509,
    0.923879532511287,
    -0.38268343236509,
    0.38268343236509,
    -0.923879532511287,
    -0.38268343236509,
    -0.923879532511287,
    -0.923879532511287,
    -0.38268343236509,
    -0.923879532511287,
    0.38268343236509,
    -0.38268343236509,
    0.923879532511287,
];

const RAND_VECS_2D: [f32; 512] = [
    -0.2700222198,
    -0.9628540911,
    0.3863092627,
    -0.9223693152,
    0.04444859006,
    -0.999011673,
    -0.5992523158,
    -0.8005602176,
    -0.7819280288,
    0.6233687174,
    0.9464672271,
    0.3227999196,
    -0.6514146797,
    -0.7587218957,
    0.9378472289,
    0.347048376,
    -0.8497875957,
    -0.5271252623,
    -0.879042592,
    0.4767432447,
    -0.892300288,
    -0.4514423508,
    -0.379844434,
    -0.9250503802,
    -0.9951650832,
    0.0982163789,
    0.7724397808,
    -0.6350880136,
    0.7573283322,
    -0.6530343002,
    -0.9928004525,
    -0.119780055,
    -0.0532665713,
    0.9985803285,
    0.9754253726,
    -0.2203300762,
    -0.7665018163,
    0.6422421394,
    0.991636706,
    0.1290606184,
    -0.994696838,
    0.1028503788,
    -0.5379205513,
    -0.84299554,
    0.5022815471,
    -0.8647041387,
    0.4559821461,
    -0.8899889226,
    -0.8659131224,
    -0.5001944266,
    0.0879458407,
    -0.9961252577,
    -0.5051684983,
    0.8630207346,
    0.7753185226,
    -0.6315704146,
    -0.6921944612,
    0.7217110418,
    -0.5191659449,
    -0.8546734591,
    0.8978622882,
    -0.4402764035,
    -0.1706774107,
    0.9853269617,
    -0.9353430106,
    -0.3537420705,
    -0.9992404798,
    0.03896746794,
    -0.2882064021,
    -0.9575683108,
    -0.9663811329,
    0.2571137995,
    -0.8759714238,
    -0.4823630009,
    -0.8303123018,
    -0.5572983775,
    0.05110133755,
    -0.9986934731,
    -0.8558373281,
    -0.5172450752,
    0.09887025282,
    0.9951003332,
    0.9189016087,
    0.3944867976,
    -0.2439375892,
    -0.9697909324,
    -0.8121409387,
    -0.5834613061,
    -0.9910431363,
    0.1335421355,
    0.8492423985,
    -0.5280031709,
    -0.9717838994,
    -0.2358729591,
    0.9949457207,
    0.1004142068,
    0.6241065508,
    -0.7813392434,
    0.662910307,
    0.7486988212,
    -0.7197418176,
    0.6942418282,
    -0.8143370775,
    -0.5803922158,
    0.104521054,
    -0.9945226741,
    -0.1065926113,
    -0.9943027784,
    0.445799684,
    -0.8951327509,
    0.105547406,
    0.9944142724,
    -0.992790267,
    0.1198644477,
    -0.8334366408,
    0.552615025,
    0.9115561563,
    -0.4111755999,
    0.8285544909,
    -0.5599084351,
    0.7217097654,
    -0.6921957921,
    0.4940492677,
    -0.8694339084,
    -0.3652321272,
    -0.9309164803,
    -0.9696606758,
    0.2444548501,
    0.08925509731,
    -0.996008799,
    0.5354071276,
    -0.8445941083,
    -0.1053576186,
    0.9944343981,
    -0.9890284586,
    0.1477251101,
    0.004856104961,
    0.9999882091,
    0.9885598478,
    0.1508291331,
    0.9286129562,
    -0.3710498316,
    -0.5832393863,
    -0.8123003252,
    0.3015207509,
    0.9534596146,
    -0.9575110528,
    0.2883965738,
    0.9715802154,
    -0.2367105511,
    0.229981792,
    0.9731949318,
    0.955763816,
    -0.2941352207,
    0.740956116,
    0.6715534485,
    -0.9971513787,
    -0.07542630764,
    0.6905710663,
    -0.7232645452,
    -0.290713703,
    -0.9568100872,
    0.5912777791,
    -0.8064679708,
    -0.9454592212,
    -0.325740481,
    0.6664455681,
    0.74555369,
    0.6236134912,
    0.7817328275,
    0.9126993851,
    -0.4086316587,
    -0.8191762011,
    0.5735419353,
    -0.8812745759,
    -0.4726046147,
    0.9953313627,
    0.09651672651,
    0.9855650846,
    -0.1692969699,
    -0.8495980887,
    0.5274306472,
    0.6174853946,
    -0.7865823463,
    0.8508156371,
    0.52546432,
    0.9985032451,
    -0.05469249926,
    0.1971371563,
    -0.9803759185,
    0.6607855748,
    -0.7505747292,
    -0.03097494063,
    0.9995201614,
    -0.6731660801,
    0.739491331,
    -0.7195018362,
    -0.6944905383,
    0.9727511689,
    0.2318515979,
    0.9997059088,
    -0.0242506907,
    0.4421787429,
    -0.8969269532,
    0.9981350961,
    -0.061043673,
    -0.9173660799,
    -0.3980445648,
    -0.8150056635,
    -0.5794529907,
    -0.8789331304,
    0.4769450202,
    0.0158605829,
    0.999874213,
    -0.8095464474,
    0.5870558317,
    -0.9165898907,
    -0.3998286786,
    -0.8023542565,
    0.5968480938,
    -0.5176737917,
    0.8555780767,
    -0.8154407307,
    -0.5788405779,
    0.4022010347,
    -0.9155513791,
    -0.9052556868,
    -0.4248672045,
    0.7317445619,
    0.6815789728,
    -0.5647632201,
    -0.8252529947,
    -0.8403276335,
    -0.5420788397,
    -0.9314281527,
    0.363925262,
    0.5238198472,
    0.8518290719,
    0.7432803869,
    -0.6689800195,
    -0.985371561,
    -0.1704197369,
    0.4601468731,
    0.88784281,
    0.825855404,
    0.5638819483,
    0.6182366099,
    0.7859920446,
    0.8331502863,
    -0.553046653,
    0.1500307506,
    0.9886813308,
    -0.662330369,
    -0.7492119075,
    -0.668598664,
    0.743623444,
    0.7025606278,
    0.7116238924,
    -0.5419389763,
    -0.8404178401,
    -0.3388616456,
    0.9408362159,
    0.8331530315,
    0.5530425174,
    -0.2989720662,
    -0.9542618632,
    0.2638522993,
    0.9645630949,
    0.124108739,
    -0.9922686234,
    -0.7282649308,
    -0.6852956957,
    0.6962500149,
    0.7177993569,
    -0.9183535368,
    0.3957610156,
    -0.6326102274,
    -0.7744703352,
    -0.9331891859,
    -0.359385508,
    -0.1153779357,
    -0.9933216659,
    0.9514974788,
    -0.3076565421,
    -0.08987977445,
    -0.9959526224,
    0.6678496916,
    0.7442961705,
    0.7952400393,
    -0.6062947138,
    -0.6462007402,
    -0.7631674805,
    -0.2733598753,
    0.9619118351,
    0.9669590226,
    -0.254931851,
    -0.9792894595,
    0.2024651934,
    -0.5369502995,
    -0.8436138784,
    -0.270036471,
    -0.9628500944,
    -0.6400277131,
    0.7683518247,
    -0.7854537493,
    -0.6189203566,
    0.06005905383,
    -0.9981948257,
    -0.02455770378,
    0.9996984141,
    -0.65983623,
    0.751409442,
    -0.6253894466,
    -0.7803127835,
    -0.6210408851,
    -0.7837781695,
    0.8348888491,
    0.5504185768,
    -0.1592275245,
    0.9872419133,
    0.8367622488,
    0.5475663786,
    -0.8675753916,
    -0.4973056806,
    -0.2022662628,
    -0.9793305667,
    0.9399189937,
    0.3413975472,
    0.9877404807,
    -0.1561049093,
    -0.9034455656,
    0.4287028224,
    0.1269804218,
    -0.9919052235,
    -0.3819600854,
    0.924178821,
    0.9754625894,
    0.2201652486,
    -0.3204015856,
    -0.9472818081,
    -0.9874760884,
    0.1577687387,
    0.02535348474,
    -0.9996785487,
    0.4835130794,
    -0.8753371362,
    -0.2850799925,
    -0.9585037287,
    -0.06805516006,
    -0.99768156,
    -0.7885244045,
    -0.6150034663,
    0.3185392127,
    -0.9479096845,
    0.8880043089,
    0.4598351306,
    0.6476921488,
    -0.7619021462,
    0.9820241299,
    0.1887554194,
    0.9357275128,
    -0.3527237187,
    -0.8894895414,
    0.4569555293,
    0.7922791302,
    0.6101588153,
    0.7483818261,
    0.6632681526,
    -0.7288929755,
    -0.6846276581,
    0.8729032783,
    -0.4878932944,
    0.8288345784,
    0.5594937369,
    0.08074567077,
    0.9967347374,
    0.9799148216,
    -0.1994165048,
    -0.580730673,
    -0.8140957471,
    -0.4700049791,
    -0.8826637636,
    0.2409492979,
    0.9705377045,
    0.9437816757,
    -0.3305694308,
    -0.8927998638,
    -0.4504535528,
    -0.8069622304,
    0.5906030467,
    0.06258973166,
    0.9980393407,
    -0.9312597469,
    0.3643559849,
    0.5777449785,
    0.8162173362,
    -0.3360095855,
    -0.941858566,
    0.697932075,
    -0.7161639607,
    -0.002008157227,
    -0.9999979837,
    -0.1827294312,
    -0.9831632392,
    -0.6523911722,
    0.7578824173,
    -0.4302626911,
    -0.9027037258,
    -0.9985126289,
    -0.05452091251,
    -0.01028102172,
    -0.9999471489,
    -0.4946071129,
    0.8691166802,
    -0.2999350194,
    0.9539596344,
    0.8165471961,
    0.5772786819,
    0.2697460475,
    0.962931498,
    -0.7306287391,
    -0.6827749597,
    -0.7590952064,
    -0.6509796216,
    -0.907053853,
    0.4210146171,
    -0.5104861064,
    -0.8598860013,
    0.8613350597,
    0.5080373165,
    0.5007881595,
    -0.8655698812,
    -0.654158152,
    0.7563577938,
    -0.8382755311,
    -0.545246856,
    0.6940070834,
    0.7199681717,
    0.06950936031,
    0.9975812994,
    0.1702942185,
    -0.9853932612,
    0.2695973274,
    0.9629731466,
    0.5519612192,
    -0.8338697815,
    0.225657487,
    -0.9742067022,
    0.4215262855,
    -0.9068161835,
    0.4881873305,
    -0.8727388672,
    -0.3683854996,
    -0.9296731273,
    -0.9825390578,
    0.1860564427,
    0.81256471,
    0.5828709909,
    0.3196460933,
    -0.9475370046,
    0.9570913859,
    0.2897862643,
    -0.6876655497,
    -0.7260276109,
    -0.9988770922,
    -0.047376731,
    -0.1250179027,
    0.992154486,
    -0.8280133617,
    0.560708367,
    0.9324863769,
    -0.3612051451,
    0.6394653183,
    0.7688199442,
    -0.01623847064,
    -0.9998681473,
    -0.9955014666,
    -0.09474613458,
    -0.81453315,
    0.580117012,
    0.4037327978,
    -0.9148769469,
    0.9944263371,
    0.1054336766,
    -0.1624711654,
    0.9867132919,
    -0.9949487814,
    -0.100383875,
    -0.6995302564,
    0.7146029809,
    0.5263414922,
    -0.85027327,
    -0.5395221479,
    0.841971408,
    0.6579370318,
    0.7530729462,
    0.01426758847,
    -0.9998982128,
    -0.6734383991,
    0.7392433447,
    0.639412098,
    -0.7688642071,
    0.9211571421,
    0.3891908523,
    -0.146637214,
    -0.9891903394,
    -0.782318098,
    0.6228791163,
    -0.5039610839,
    -0.8637263605,
    -0.7743120191,
    -0.6328039957,
];

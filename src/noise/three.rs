//! FastNoiseLite 1.1.0's 3D noise and 3D domain warp, as Godot's `get_noise_3d` computes them.

use super::{
    PRIME_X, PRIME_Y, cubic_lerp, fast_abs, fast_floor, fast_round, interp_hermite, interp_quintic,
    lerp,
};

pub(super) const PRIME_Z: i32 = 1_720_413_743;

/// FastNoiseLite's `TransformType3D_DefaultOpenSimplex2`, which the simplex noises and warps read
/// at when no 3D rotation is set, as in Godot: a rotation of the cube grid, not a skew.
pub(super) fn rotate(p: [f32; 3]) -> [f32; 3] {
    const R3: f32 = 2.0 / 3.0;
    let r = (p[0] + p[1] + p[2]) * R3;
    [r - p[0], r - p[1], r - p[2]]
}

fn hash(seed: i32, x_primed: i32, y_primed: i32, z_primed: i32) -> i32 {
    (seed ^ x_primed ^ y_primed ^ z_primed).wrapping_mul(0x27d4_eb2d)
}

fn val_coord(seed: i32, x_primed: i32, y_primed: i32, z_primed: i32) -> f32 {
    let mut hash = hash(seed, x_primed, y_primed, z_primed);
    hash = hash.wrapping_mul(hash);
    hash ^= hash << 19;
    hash as f32 * (1.0 / 2_147_483_648.0)
}

fn grad_coord(seed: i32, primed: [i32; 3], d: [f32; 3]) -> f32 {
    let mut hash = hash(seed, primed[0], primed[1], primed[2]);
    hash ^= hash >> 15;
    let index = (hash & (63 << 2)) as usize;
    d[0] * GRADIENTS_3D[index] + d[1] * GRADIENTS_3D[index | 1] + d[2] * GRADIENTS_3D[index | 2]
}

/// FastNoiseLite's `GradCoordOut`: a random unit vector for a lattice point.
fn grad_coord_out(seed: i32, primed: [i32; 3]) -> [f32; 3] {
    let index = (hash(seed, primed[0], primed[1], primed[2]) & (255 << 2)) as usize;
    [
        RAND_VECS_3D[index],
        RAND_VECS_3D[index | 1],
        RAND_VECS_3D[index | 2],
    ]
}

/// FastNoiseLite's `GradCoordDual`: a random unit vector scaled by a gradient's value.
fn grad_coord_dual(seed: i32, primed: [i32; 3], d: [f32; 3]) -> [f32; 3] {
    let hash = hash(seed, primed[0], primed[1], primed[2]);
    let index1 = (hash & (63 << 2)) as usize;
    let index2 = ((hash >> 6) & (255 << 2)) as usize;
    let value = d[0] * GRADIENTS_3D[index1]
        + d[1] * GRADIENTS_3D[index1 | 1]
        + d[2] * GRADIENTS_3D[index1 | 2];
    [
        value * RAND_VECS_3D[index2],
        value * RAND_VECS_3D[index2 | 1],
        value * RAND_VECS_3D[index2 | 2],
    ]
}

/// The walk over two offset cube grids that 3D OpenSimplex2 and its gradient warp share: the
/// lattice points around `(x, y, z)` with their offsets and falloffs, handed to `corner` in
/// FastNoiseLite's order. `next_seed` is how the second grid's seed follows from the first's.
fn simplex_corners(
    seed: i32,
    x: f32,
    y: f32,
    z: f32,
    next_seed: fn(i32) -> i32,
    mut corner: impl FnMut(i32, [i32; 3], [f32; 3], f32),
) {
    let mut seed = seed;
    let i = fast_round(x);
    let j = fast_round(y);
    let k = fast_round(z);
    let mut x0 = x - i as f32;
    let mut y0 = y - j as f32;
    let mut z0 = z - k as f32;

    let mut x_n_sign = (-1.0 - x0) as i32 | 1;
    let mut y_n_sign = (-1.0 - y0) as i32 | 1;
    let mut z_n_sign = (-1.0 - z0) as i32 | 1;

    let mut ax0 = x_n_sign as f32 * -x0;
    let mut ay0 = y_n_sign as f32 * -y0;
    let mut az0 = z_n_sign as f32 * -z0;

    let mut i = i.wrapping_mul(PRIME_X);
    let mut j = j.wrapping_mul(PRIME_Y);
    let mut k = k.wrapping_mul(PRIME_Z);

    let mut a = (0.6 - x0 * x0) - (y0 * y0 + z0 * z0);
    for l in 0..2 {
        if a > 0.0 {
            corner(seed, [i, j, k], [x0, y0, z0], a);
        }

        let mut b = a + 1.0;
        let (mut i1, mut j1, mut k1) = (i, j, k);
        let (mut x1, mut y1, mut z1) = (x0, y0, z0);
        if ax0 >= ay0 && ax0 >= az0 {
            x1 += x_n_sign as f32;
            b -= (x_n_sign * 2) as f32 * x1;
            i1 = i1.wrapping_sub(x_n_sign.wrapping_mul(PRIME_X));
        } else if ay0 > ax0 && ay0 >= az0 {
            y1 += y_n_sign as f32;
            b -= (y_n_sign * 2) as f32 * y1;
            j1 = j1.wrapping_sub(y_n_sign.wrapping_mul(PRIME_Y));
        } else {
            z1 += z_n_sign as f32;
            b -= (z_n_sign * 2) as f32 * z1;
            k1 = k1.wrapping_sub(z_n_sign.wrapping_mul(PRIME_Z));
        }
        if b > 0.0 {
            corner(seed, [i1, j1, k1], [x1, y1, z1], b);
        }

        if l == 1 {
            break;
        }

        ax0 = 0.5 - ax0;
        ay0 = 0.5 - ay0;
        az0 = 0.5 - az0;

        x0 = x_n_sign as f32 * ax0;
        y0 = y_n_sign as f32 * ay0;
        z0 = z_n_sign as f32 * az0;

        a += (0.75 - ax0) - (ay0 + az0);

        i = i.wrapping_add((x_n_sign >> 1) & PRIME_X);
        j = j.wrapping_add((y_n_sign >> 1) & PRIME_Y);
        k = k.wrapping_add((z_n_sign >> 1) & PRIME_Z);

        x_n_sign = -x_n_sign;
        y_n_sign = -y_n_sign;
        z_n_sign = -z_n_sign;

        seed = next_seed(seed);
    }
}

/// FastNoiseLite's 3D `SingleOpenSimplex2`, read at a rotated position.
pub(super) fn single_simplex(seed: i32, x: f32, y: f32, z: f32) -> f32 {
    let mut value = 0.0;
    simplex_corners(
        seed,
        x,
        y,
        z,
        |seed| !seed,
        |seed, primed, d, a| {
            value += (a * a) * (a * a) * grad_coord(seed, primed, d);
        },
    );
    value * 32.694_282_531_738_28
}

/// FastNoiseLite's 3D `SingleOpenSimplex2S`, read at a rotated position.
pub(super) fn single_simplex_smooth(seed: i32, x: f32, y: f32, z: f32) -> f32 {
    let i = fast_floor(x);
    let j = fast_floor(y);
    let k = fast_floor(z);
    let xi = x - i as f32;
    let yi = y - j as f32;
    let zi = z - k as f32;

    let i = i.wrapping_mul(PRIME_X);
    let j = j.wrapping_mul(PRIME_Y);
    let k = k.wrapping_mul(PRIME_Z);
    let seed2 = seed.wrapping_add(1_293_373);

    let x_n_mask = (-0.5 - xi) as i32;
    let y_n_mask = (-0.5 - yi) as i32;
    let z_n_mask = (-0.5 - zi) as i32;

    let x0 = xi + x_n_mask as f32;
    let y0 = yi + y_n_mask as f32;
    let z0 = zi + z_n_mask as f32;
    let a0 = 0.75 - x0 * x0 - y0 * y0 - z0 * z0;
    let mut value = (a0 * a0)
        * (a0 * a0)
        * grad_coord(
            seed,
            [
                i.wrapping_add(x_n_mask & PRIME_X),
                j.wrapping_add(y_n_mask & PRIME_Y),
                k.wrapping_add(z_n_mask & PRIME_Z),
            ],
            [x0, y0, z0],
        );

    let x1 = xi - 0.5;
    let y1 = yi - 0.5;
    let z1 = zi - 0.5;
    let a1 = 0.75 - x1 * x1 - y1 * y1 - z1 * z1;
    value += (a1 * a1)
        * (a1 * a1)
        * grad_coord(
            seed2,
            [
                i.wrapping_add(PRIME_X),
                j.wrapping_add(PRIME_Y),
                k.wrapping_add(PRIME_Z),
            ],
            [x1, y1, z1],
        );

    let x_a_flip_mask0 = ((x_n_mask | 1) << 1) as f32 * x1;
    let y_a_flip_mask0 = ((y_n_mask | 1) << 1) as f32 * y1;
    let z_a_flip_mask0 = ((z_n_mask | 1) << 1) as f32 * z1;
    let x_a_flip_mask1 = (-2 - (x_n_mask << 2)) as f32 * x1 - 1.0;
    let y_a_flip_mask1 = (-2 - (y_n_mask << 2)) as f32 * y1 - 1.0;
    let z_a_flip_mask1 = (-2 - (z_n_mask << 2)) as f32 * z1 - 1.0;

    let x_sign = (x_n_mask | 1) as f32;
    let y_sign = (y_n_mask | 1) as f32;
    let z_sign = (z_n_mask | 1) as f32;
    // The lattice point on the near side along an axis, and the one on the far side.
    let near = [
        i.wrapping_add(x_n_mask & PRIME_X),
        j.wrapping_add(y_n_mask & PRIME_Y),
        k.wrapping_add(z_n_mask & PRIME_Z),
    ];
    let far = [
        i.wrapping_add(!x_n_mask & PRIME_X),
        j.wrapping_add(!y_n_mask & PRIME_Y),
        k.wrapping_add(!z_n_mask & PRIME_Z),
    ];
    let outer = [
        i.wrapping_add(x_n_mask & (PRIME_X << 1)),
        j.wrapping_add(y_n_mask & (PRIME_Y << 1)),
        k.wrapping_add(z_n_mask & (PRIME_Z << 1)),
    ];
    let centre = [
        i.wrapping_add(PRIME_X),
        j.wrapping_add(PRIME_Y),
        k.wrapping_add(PRIME_Z),
    ];
    let mut add = |a: f32, seed: i32, primed: [i32; 3], d: [f32; 3]| {
        if a > 0.0 {
            value += (a * a) * (a * a) * grad_coord(seed, primed, d);
        }
    };

    let mut skip5 = false;
    let a2 = x_a_flip_mask0 + a0;
    if a2 > 0.0 {
        add(a2, seed, [far[0], near[1], near[2]], [x0 - x_sign, y0, z0]);
    } else {
        let a3 = y_a_flip_mask0 + z_a_flip_mask0 + a0;
        add(
            a3,
            seed,
            [near[0], far[1], far[2]],
            [x0, y0 - y_sign, z0 - z_sign],
        );

        let a4 = x_a_flip_mask1 + a1;
        if a4 > 0.0 {
            add(
                a4,
                seed2,
                [outer[0], centre[1], centre[2]],
                [x_sign + x1, y1, z1],
            );
            skip5 = true;
        }
    }

    let mut skip9 = false;
    let a6 = y_a_flip_mask0 + a0;
    if a6 > 0.0 {
        add(a6, seed, [near[0], far[1], near[2]], [x0, y0 - y_sign, z0]);
    } else {
        let a7 = x_a_flip_mask0 + z_a_flip_mask0 + a0;
        add(
            a7,
            seed,
            [far[0], near[1], far[2]],
            [x0 - x_sign, y0, z0 - z_sign],
        );

        let a8 = y_a_flip_mask1 + a1;
        if a8 > 0.0 {
            add(
                a8,
                seed2,
                [centre[0], outer[1], centre[2]],
                [x1, y_sign + y1, z1],
            );
            skip9 = true;
        }
    }

    let mut skip_d = false;
    let a_a = z_a_flip_mask0 + a0;
    if a_a > 0.0 {
        add(a_a, seed, [near[0], near[1], far[2]], [x0, y0, z0 - z_sign]);
    } else {
        let a_b = x_a_flip_mask0 + y_a_flip_mask0 + a0;
        add(
            a_b,
            seed,
            [far[0], far[1], near[2]],
            [x0 - x_sign, y0 - y_sign, z0],
        );

        let a_c = z_a_flip_mask1 + a1;
        if a_c > 0.0 {
            add(
                a_c,
                seed2,
                [centre[0], centre[1], outer[2]],
                [x1, y1, z_sign + z1],
            );
            skip_d = true;
        }
    }

    if !skip5 {
        let a5 = y_a_flip_mask1 + z_a_flip_mask1 + a1;
        add(
            a5,
            seed2,
            [centre[0], outer[1], outer[2]],
            [x1, y_sign + y1, z_sign + z1],
        );
    }
    if !skip9 {
        let a9 = x_a_flip_mask1 + z_a_flip_mask1 + a1;
        add(
            a9,
            seed2,
            [outer[0], centre[1], outer[2]],
            [x_sign + x1, y1, z_sign + z1],
        );
    }
    if !skip_d {
        let a_d = x_a_flip_mask1 + y_a_flip_mask1 + a1;
        add(
            a_d,
            seed2,
            [outer[0], outer[1], centre[2]],
            [x_sign + x1, y_sign + y1, z1],
        );
    }

    value * 9.046_026_385_208_288
}

/// FastNoiseLite's 3D `SingleCellular`, before its return type is applied: the two smallest
/// distances and the hash of the nearest point. `distance` measures an offset.
pub(super) fn cellular(
    seed: i32,
    x: f32,
    y: f32,
    z: f32,
    jitter: f32,
    distance: impl Fn([f32; 3]) -> f32,
) -> (f32, f32, i32) {
    let xr = fast_round(x);
    let yr = fast_round(y);
    let zr = fast_round(z);
    let mut distance0 = 1e10_f32;
    let mut distance1 = 1e10_f32;
    let mut closest_hash = 0;
    let jitter = 0.396_143_53 * jitter;

    let mut x_primed = (xr - 1).wrapping_mul(PRIME_X);
    let y_primed_base = (yr - 1).wrapping_mul(PRIME_Y);
    let z_primed_base = (zr - 1).wrapping_mul(PRIME_Z);
    for xi in xr - 1..=xr + 1 {
        let mut y_primed = y_primed_base;
        for yi in yr - 1..=yr + 1 {
            let mut z_primed = z_primed_base;
            for zi in zr - 1..=zr + 1 {
                let hash = hash(seed, x_primed, y_primed, z_primed);
                let index = (hash & (255 << 2)) as usize;
                let offset = [
                    (xi as f32 - x) + RAND_VECS_3D[index] * jitter,
                    (yi as f32 - y) + RAND_VECS_3D[index | 1] * jitter,
                    (zi as f32 - z) + RAND_VECS_3D[index | 2] * jitter,
                ];
                let new_distance = distance(offset);
                distance1 = super::fast_max(super::fast_min(distance1, new_distance), distance0);
                if new_distance < distance0 {
                    distance0 = new_distance;
                    closest_hash = hash;
                }
                z_primed = z_primed.wrapping_add(PRIME_Z);
            }
            y_primed = y_primed.wrapping_add(PRIME_Y);
        }
        x_primed = x_primed.wrapping_add(PRIME_X);
    }
    (distance0, distance1, closest_hash)
}

/// The cellular distances FastNoiseLite measures in 3D, by its distance function.
pub(super) fn euclidean_squared(v: [f32; 3]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

pub(super) fn manhattan(v: [f32; 3]) -> f32 {
    fast_abs(v[0]) + fast_abs(v[1]) + fast_abs(v[2])
}

pub(super) fn hybrid(v: [f32; 3]) -> f32 {
    manhattan(v) + euclidean_squared(v)
}

pub(super) fn single_perlin(seed: i32, x: f32, y: f32, z: f32) -> f32 {
    let x0 = fast_floor(x);
    let y0 = fast_floor(y);
    let z0 = fast_floor(z);

    let xd0 = x - x0 as f32;
    let yd0 = y - y0 as f32;
    let zd0 = z - z0 as f32;
    let xd1 = xd0 - 1.0;
    let yd1 = yd0 - 1.0;
    let zd1 = zd0 - 1.0;

    let xs = interp_quintic(xd0);
    let ys = interp_quintic(yd0);
    let zs = interp_quintic(zd0);

    let x0 = x0.wrapping_mul(PRIME_X);
    let y0 = y0.wrapping_mul(PRIME_Y);
    let z0 = z0.wrapping_mul(PRIME_Z);
    let x1 = x0.wrapping_add(PRIME_X);
    let y1 = y0.wrapping_add(PRIME_Y);
    let z1 = z0.wrapping_add(PRIME_Z);

    let along_x = |y: i32, z: i32, yd: f32, zd: f32| {
        lerp(
            grad_coord(seed, [x0, y, z], [xd0, yd, zd]),
            grad_coord(seed, [x1, y, z], [xd1, yd, zd]),
            xs,
        )
    };
    let xf00 = along_x(y0, z0, yd0, zd0);
    let xf10 = along_x(y1, z0, yd1, zd0);
    let xf01 = along_x(y0, z1, yd0, zd1);
    let xf11 = along_x(y1, z1, yd1, zd1);

    let yf0 = lerp(xf00, xf10, ys);
    let yf1 = lerp(xf01, xf11, ys);

    lerp(yf0, yf1, zs) * 0.964_921_414_852_142_333_984_375
}

pub(super) fn single_value_cubic(seed: i32, x: f32, y: f32, z: f32) -> f32 {
    let x1 = fast_floor(x);
    let y1 = fast_floor(y);
    let z1 = fast_floor(z);

    let xs = x - x1 as f32;
    let ys = y - y1 as f32;
    let zs = z - z1 as f32;

    let x1 = x1.wrapping_mul(PRIME_X);
    let y1 = y1.wrapping_mul(PRIME_Y);
    let z1 = z1.wrapping_mul(PRIME_Z);
    let xs_primed = [
        x1.wrapping_sub(PRIME_X),
        x1,
        x1.wrapping_add(PRIME_X),
        x1.wrapping_add(PRIME_X << 1),
    ];
    let ys_primed = [
        y1.wrapping_sub(PRIME_Y),
        y1,
        y1.wrapping_add(PRIME_Y),
        y1.wrapping_add(PRIME_Y << 1),
    ];
    let zs_primed = [
        z1.wrapping_sub(PRIME_Z),
        z1,
        z1.wrapping_add(PRIME_Z),
        z1.wrapping_add(PRIME_Z << 1),
    ];

    let row = |y: i32, z: i32| {
        cubic_lerp(
            val_coord(seed, xs_primed[0], y, z),
            val_coord(seed, xs_primed[1], y, z),
            val_coord(seed, xs_primed[2], y, z),
            val_coord(seed, xs_primed[3], y, z),
            xs,
        )
    };
    let plane = |z: i32| {
        cubic_lerp(
            row(ys_primed[0], z),
            row(ys_primed[1], z),
            row(ys_primed[2], z),
            row(ys_primed[3], z),
            ys,
        )
    };
    cubic_lerp(
        plane(zs_primed[0]),
        plane(zs_primed[1]),
        plane(zs_primed[2]),
        plane(zs_primed[3]),
        zs,
    ) * (1.0 / (1.5 * 1.5 * 1.5))
}

pub(super) fn single_value(seed: i32, x: f32, y: f32, z: f32) -> f32 {
    let x0 = fast_floor(x);
    let y0 = fast_floor(y);
    let z0 = fast_floor(z);

    let xs = interp_hermite(x - x0 as f32);
    let ys = interp_hermite(y - y0 as f32);
    let zs = interp_hermite(z - z0 as f32);

    let x0 = x0.wrapping_mul(PRIME_X);
    let y0 = y0.wrapping_mul(PRIME_Y);
    let z0 = z0.wrapping_mul(PRIME_Z);
    let x1 = x0.wrapping_add(PRIME_X);
    let y1 = y0.wrapping_add(PRIME_Y);
    let z1 = z0.wrapping_add(PRIME_Z);

    let xf00 = lerp(val_coord(seed, x0, y0, z0), val_coord(seed, x1, y0, z0), xs);
    let xf10 = lerp(val_coord(seed, x0, y1, z0), val_coord(seed, x1, y1, z0), xs);
    let xf01 = lerp(val_coord(seed, x0, y0, z1), val_coord(seed, x1, y0, z1), xs);
    let xf11 = lerp(val_coord(seed, x0, y1, z1), val_coord(seed, x1, y1, z1), xs);

    let yf0 = lerp(xf00, xf10, ys);
    let yf1 = lerp(xf01, xf11, ys);

    lerp(yf0, yf1, zs)
}

/// FastNoiseLite's 3D `SingleDomainWarpBasicGrid`: how far it moves `p`.
pub(super) fn warp_basic_grid(seed: i32, warp_amp: f32, frequency: f32, p: [f32; 3]) -> [f32; 3] {
    let xf = p[0] * frequency;
    let yf = p[1] * frequency;
    let zf = p[2] * frequency;

    let x0 = fast_floor(xf);
    let y0 = fast_floor(yf);
    let z0 = fast_floor(zf);

    let xs = interp_hermite(xf - x0 as f32);
    let ys = interp_hermite(yf - y0 as f32);
    let zs = interp_hermite(zf - z0 as f32);

    let x0 = x0.wrapping_mul(PRIME_X);
    let y0 = y0.wrapping_mul(PRIME_Y);
    let z0 = z0.wrapping_mul(PRIME_Z);
    let x1 = x0.wrapping_add(PRIME_X);
    let y1 = y0.wrapping_add(PRIME_Y);
    let z1 = z0.wrapping_add(PRIME_Z);

    // The random vectors at two lattice points along x, blended across it.
    let along_x = |y: i32, z: i32| {
        let hash0 = (hash(seed, x0, y, z) & (255 << 2)) as usize;
        let hash1 = (hash(seed, x1, y, z) & (255 << 2)) as usize;
        [
            lerp(RAND_VECS_3D[hash0], RAND_VECS_3D[hash1], xs),
            lerp(RAND_VECS_3D[hash0 | 1], RAND_VECS_3D[hash1 | 1], xs),
            lerp(RAND_VECS_3D[hash0 | 2], RAND_VECS_3D[hash1 | 2], xs),
        ]
    };
    let l0x = along_x(y0, z0);
    let l1x = along_x(y1, z0);
    let l0y: [f32; 3] = std::array::from_fn(|axis| lerp(l0x[axis], l1x[axis], ys));
    let l0x = along_x(y0, z1);
    let l1x = along_x(y1, z1);
    std::array::from_fn(|axis| lerp(l0y[axis], lerp(l0x[axis], l1x[axis], ys), zs) * warp_amp)
}

/// FastNoiseLite's 3D `SingleDomainWarpOpenSimplex2Gradient`: how far it moves `p`, read at a
/// rotated position. `reduced` takes each corner's random vector alone (`outGradOnly`).
pub(super) fn warp_simplex_gradient(
    seed: i32,
    warp_amp: f32,
    frequency: f32,
    p: [f32; 3],
    reduced: bool,
) -> [f32; 3] {
    let mut v = [0.0_f32; 3];
    simplex_corners(
        seed,
        p[0] * frequency,
        p[1] * frequency,
        p[2] * frequency,
        |seed| seed.wrapping_add(1_293_373),
        |seed, primed, d, a| {
            let aaaa = (a * a) * (a * a);
            let o = if reduced {
                grad_coord_out(seed, primed)
            } else {
                grad_coord_dual(seed, primed, d)
            };
            for axis in 0..3 {
                v[axis] += aaaa * o[axis];
            }
        },
    );
    v.map(|value| value * warp_amp)
}
// FastNoiseLite's 3D lookup tables, digit for digit.

pub(super) const GRADIENTS_3D: [f32; 256] = [
    0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
    -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0,
    1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0,
    0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0,
    0.0, -1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0,
    -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0,
    0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
    -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0,
    1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0,
    1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0,
    0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0,
    0.0, -1.0, -1.0, 0.0,
];

pub(super) const RAND_VECS_3D: [f32; 1024] = [
    -0.7292736885,
    -0.6618439697,
    0.1735581948,
    0.0,
    0.790292081,
    -0.5480887466,
    -0.2739291014,
    0.0,
    0.7217578935,
    0.6226212466,
    -0.3023380997,
    0.0,
    0.565683137,
    -0.8208298145,
    -0.0790000257,
    0.0,
    0.760049034,
    -0.5555979497,
    -0.3370999617,
    0.0,
    0.3713945616,
    0.5011264475,
    0.7816254623,
    0.0,
    -0.1277062463,
    -0.4254438999,
    -0.8959289049,
    0.0,
    -0.2881560924,
    -0.5815838982,
    0.7607405838,
    0.0,
    0.5849561111,
    -0.662820239,
    -0.4674352136,
    0.0,
    0.3307171178,
    0.0391653737,
    0.94291689,
    0.0,
    0.8712121778,
    -0.4113374369,
    -0.2679381538,
    0.0,
    0.580981015,
    0.7021915846,
    0.4115677815,
    0.0,
    0.503756873,
    0.6330056931,
    -0.5878203852,
    0.0,
    0.4493712205,
    0.601390195,
    0.6606022552,
    0.0,
    -0.6878403724,
    0.09018890807,
    -0.7202371714,
    0.0,
    -0.5958956522,
    -0.6469350577,
    0.475797649,
    0.0,
    -0.5127052122,
    0.1946921978,
    -0.8361987284,
    0.0,
    -0.9911507142,
    -0.05410276466,
    -0.1212153153,
    0.0,
    -0.2149721042,
    0.9720882117,
    -0.09397607749,
    0.0,
    -0.7518650936,
    -0.5428057603,
    0.3742469607,
    0.0,
    0.5237068895,
    0.8516377189,
    -0.02107817834,
    0.0,
    0.6333504779,
    0.1926167129,
    -0.7495104896,
    0.0,
    -0.06788241606,
    0.3998305789,
    0.9140719259,
    0.0,
    -0.5538628599,
    -0.4729896695,
    -0.6852128902,
    0.0,
    -0.7261455366,
    -0.5911990757,
    0.3509933228,
    0.0,
    -0.9229274737,
    -0.1782808786,
    0.3412049336,
    0.0,
    -0.6968815002,
    0.6511274338,
    0.3006480328,
    0.0,
    0.9608044783,
    -0.2098363234,
    -0.1811724921,
    0.0,
    0.06817146062,
    -0.9743405129,
    0.2145069156,
    0.0,
    -0.3577285196,
    -0.6697087264,
    -0.6507845481,
    0.0,
    -0.1868621131,
    0.7648617052,
    -0.6164974636,
    0.0,
    -0.6541697588,
    0.3967914832,
    0.6439087246,
    0.0,
    0.6993340405,
    -0.6164538506,
    0.3618239211,
    0.0,
    -0.1546665739,
    0.6291283928,
    0.7617583057,
    0.0,
    -0.6841612949,
    -0.2580482182,
    -0.6821542638,
    0.0,
    0.5383980957,
    0.4258654885,
    0.7271630328,
    0.0,
    -0.5026987823,
    -0.7939832935,
    -0.3418836993,
    0.0,
    0.3202971715,
    0.2834415347,
    0.9039195862,
    0.0,
    0.8683227101,
    -0.0003762656404,
    -0.4959995258,
    0.0,
    0.791120031,
    -0.08511045745,
    0.6057105799,
    0.0,
    -0.04011016052,
    -0.4397248749,
    0.8972364289,
    0.0,
    0.9145119872,
    0.3579346169,
    -0.1885487608,
    0.0,
    -0.9612039066,
    -0.2756484276,
    0.01024666929,
    0.0,
    0.6510361721,
    -0.2877799159,
    -0.7023778346,
    0.0,
    -0.2041786351,
    0.7365237271,
    0.644859585,
    0.0,
    -0.7718263711,
    0.3790626912,
    0.5104855816,
    0.0,
    -0.3060082741,
    -0.7692987727,
    0.5608371729,
    0.0,
    0.454007341,
    -0.5024843065,
    0.7357899537,
    0.0,
    0.4816795475,
    0.6021208291,
    -0.6367380315,
    0.0,
    0.6961980369,
    -0.3222197429,
    0.641469197,
    0.0,
    -0.6532160499,
    -0.6781148932,
    0.3368515753,
    0.0,
    0.5089301236,
    -0.6154662304,
    -0.6018234363,
    0.0,
    -0.1635919754,
    -0.9133604627,
    -0.372840892,
    0.0,
    0.52408019,
    -0.8437664109,
    0.1157505864,
    0.0,
    0.5902587356,
    0.4983817807,
    -0.6349883666,
    0.0,
    0.5863227872,
    0.494764745,
    0.6414307729,
    0.0,
    0.6779335087,
    0.2341345225,
    0.6968408593,
    0.0,
    0.7177054546,
    -0.6858979348,
    0.120178631,
    0.0,
    -0.5328819713,
    -0.5205125012,
    0.6671608058,
    0.0,
    -0.8654874251,
    -0.0700727088,
    -0.4960053754,
    0.0,
    -0.2861810166,
    0.7952089234,
    0.5345495242,
    0.0,
    -0.04849529634,
    0.9810836427,
    -0.1874115585,
    0.0,
    -0.6358521667,
    0.6058348682,
    0.4781800233,
    0.0,
    0.6254794696,
    -0.2861619734,
    0.7258696564,
    0.0,
    -0.2585259868,
    0.5061949264,
    -0.8227581726,
    0.0,
    0.02136306781,
    0.5064016808,
    -0.8620330371,
    0.0,
    0.200111773,
    0.8599263484,
    0.4695550591,
    0.0,
    0.4743561372,
    0.6014985084,
    -0.6427953014,
    0.0,
    0.6622993731,
    -0.5202474575,
    -0.5391679918,
    0.0,
    0.08084972818,
    -0.6532720452,
    0.7527940996,
    0.0,
    -0.6893687501,
    0.0592860349,
    0.7219805347,
    0.0,
    -0.1121887082,
    -0.9673185067,
    0.2273952515,
    0.0,
    0.7344116094,
    0.5979668656,
    -0.3210532909,
    0.0,
    0.5789393465,
    -0.2488849713,
    0.7764570201,
    0.0,
    0.6988182827,
    0.3557169806,
    -0.6205791146,
    0.0,
    -0.8636845529,
    -0.2748771249,
    -0.4224826141,
    0.0,
    -0.4247027957,
    -0.4640880967,
    0.777335046,
    0.0,
    0.5257722489,
    -0.8427017621,
    0.1158329937,
    0.0,
    0.9343830603,
    0.316302472,
    -0.1639543925,
    0.0,
    -0.1016836419,
    -0.8057303073,
    -0.5834887393,
    0.0,
    -0.6529238969,
    0.50602126,
    -0.5635892736,
    0.0,
    -0.2465286165,
    -0.9668205684,
    -0.06694497494,
    0.0,
    -0.9776897119,
    -0.2099250524,
    -0.007368825344,
    0.0,
    0.7736893337,
    0.5734244712,
    0.2694238123,
    0.0,
    -0.6095087895,
    0.4995678998,
    0.6155736747,
    0.0,
    0.5794535482,
    0.7434546771,
    0.3339292269,
    0.0,
    -0.8226211154,
    0.08142581855,
    0.5627293636,
    0.0,
    -0.510385483,
    0.4703667658,
    0.7199039967,
    0.0,
    -0.5764971849,
    -0.07231656274,
    -0.8138926898,
    0.0,
    0.7250628871,
    0.3949971505,
    -0.5641463116,
    0.0,
    -0.1525424005,
    0.4860840828,
    -0.8604958341,
    0.0,
    -0.5550976208,
    -0.4957820792,
    0.667882296,
    0.0,
    -0.1883614327,
    0.9145869398,
    0.357841725,
    0.0,
    0.7625556724,
    -0.5414408243,
    -0.3540489801,
    0.0,
    -0.5870231946,
    -0.3226498013,
    -0.7424963803,
    0.0,
    0.3051124198,
    0.2262544068,
    -0.9250488391,
    0.0,
    0.6379576059,
    0.577242424,
    -0.5097070502,
    0.0,
    -0.5966775796,
    0.1454852398,
    -0.7891830656,
    0.0,
    -0.658330573,
    0.6555487542,
    -0.3699414651,
    0.0,
    0.7434892426,
    0.2351084581,
    0.6260573129,
    0.0,
    0.5562114096,
    0.8264360377,
    -0.0873632843,
    0.0,
    -0.3028940016,
    -0.8251527185,
    0.4768419182,
    0.0,
    0.1129343818,
    -0.985888439,
    -0.1235710781,
    0.0,
    0.5937652891,
    -0.5896813806,
    0.5474656618,
    0.0,
    0.6757964092,
    -0.5835758614,
    -0.4502648413,
    0.0,
    0.7242302609,
    -0.1152719764,
    0.6798550586,
    0.0,
    -0.9511914166,
    0.0753623979,
    -0.2992580792,
    0.0,
    0.2539470961,
    -0.1886339355,
    0.9486454084,
    0.0,
    0.571433621,
    -0.1679450851,
    -0.8032795685,
    0.0,
    -0.06778234979,
    0.3978269256,
    0.9149531629,
    0.0,
    0.6074972649,
    0.733060024,
    -0.3058922593,
    0.0,
    -0.5435478392,
    0.1675822484,
    0.8224791405,
    0.0,
    -0.5876678086,
    -0.3380045064,
    -0.7351186982,
    0.0,
    -0.7967562402,
    0.04097822706,
    -0.6029098428,
    0.0,
    -0.1996350917,
    0.8706294745,
    0.4496111079,
    0.0,
    -0.02787660336,
    -0.9106232682,
    -0.4122962022,
    0.0,
    -0.7797625996,
    -0.6257634692,
    0.01975775581,
    0.0,
    -0.5211232846,
    0.7401644346,
    -0.4249554471,
    0.0,
    0.8575424857,
    0.4053272873,
    -0.3167501783,
    0.0,
    0.1045223322,
    0.8390195772,
    -0.5339674439,
    0.0,
    0.3501822831,
    0.9242524096,
    -0.1520850155,
    0.0,
    0.1987849858,
    0.07647613266,
    0.9770547224,
    0.0,
    0.7845996363,
    0.6066256811,
    -0.1280964233,
    0.0,
    0.09006737436,
    -0.9750989929,
    -0.2026569073,
    0.0,
    -0.8274343547,
    -0.542299559,
    0.1458203587,
    0.0,
    -0.3485797732,
    -0.415802277,
    0.840000362,
    0.0,
    -0.2471778936,
    -0.7304819962,
    -0.6366310879,
    0.0,
    -0.3700154943,
    0.8577948156,
    0.3567584454,
    0.0,
    0.5913394901,
    -0.548311967,
    -0.5913303597,
    0.0,
    0.1204873514,
    -0.7626472379,
    -0.6354935001,
    0.0,
    0.616959265,
    0.03079647928,
    0.7863922953,
    0.0,
    0.1258156836,
    -0.6640829889,
    -0.7369967419,
    0.0,
    -0.6477565124,
    -0.1740147258,
    -0.7417077429,
    0.0,
    0.6217889313,
    -0.7804430448,
    -0.06547655076,
    0.0,
    0.6589943422,
    -0.6096987708,
    0.4404473475,
    0.0,
    -0.2689837504,
    -0.6732403169,
    -0.6887635427,
    0.0,
    -0.3849775103,
    0.5676542638,
    0.7277093879,
    0.0,
    0.5754444408,
    0.8110471154,
    -0.1051963504,
    0.0,
    0.9141593684,
    0.3832947817,
    0.131900567,
    0.0,
    -0.107925319,
    0.9245493968,
    0.3654593525,
    0.0,
    0.377977089,
    0.3043148782,
    0.8743716458,
    0.0,
    -0.2142885215,
    -0.8259286236,
    0.5214617324,
    0.0,
    0.5802544474,
    0.4148098596,
    -0.7008834116,
    0.0,
    -0.1982660881,
    0.8567161266,
    -0.4761596756,
    0.0,
    -0.03381553704,
    0.3773180787,
    -0.9254661404,
    0.0,
    -0.6867922841,
    -0.6656597827,
    0.2919133642,
    0.0,
    0.7731742607,
    -0.2875793547,
    -0.5652430251,
    0.0,
    -0.09655941928,
    0.9193708367,
    -0.3813575004,
    0.0,
    0.2715702457,
    -0.9577909544,
    -0.09426605581,
    0.0,
    0.2451015704,
    -0.6917998565,
    -0.6792188003,
    0.0,
    0.977700782,
    -0.1753855374,
    0.1155036542,
    0.0,
    -0.5224739938,
    0.8521606816,
    0.02903615945,
    0.0,
    -0.7734880599,
    -0.5261292347,
    0.3534179531,
    0.0,
    -0.7134492443,
    -0.269547243,
    0.6467878011,
    0.0,
    0.1644037271,
    0.5105846203,
    -0.8439637196,
    0.0,
    0.6494635788,
    0.05585611296,
    0.7583384168,
    0.0,
    -0.4711970882,
    0.5017280509,
    -0.7254255765,
    0.0,
    -0.6335764307,
    -0.2381686273,
    -0.7361091029,
    0.0,
    -0.9021533097,
    -0.270947803,
    -0.3357181763,
    0.0,
    -0.3793711033,
    0.872258117,
    0.3086152025,
    0.0,
    -0.6855598966,
    -0.3250143309,
    0.6514394162,
    0.0,
    0.2900942212,
    -0.7799057743,
    -0.5546100667,
    0.0,
    -0.2098319339,
    0.85037073,
    0.4825351604,
    0.0,
    -0.4592603758,
    0.6598504336,
    -0.5947077538,
    0.0,
    0.8715945488,
    0.09616365406,
    -0.4807031248,
    0.0,
    -0.6776666319,
    0.7118504878,
    -0.1844907016,
    0.0,
    0.7044377633,
    0.312427597,
    0.637304036,
    0.0,
    -0.7052318886,
    -0.2401093292,
    -0.6670798253,
    0.0,
    0.081921007,
    -0.7207336136,
    -0.6883545647,
    0.0,
    -0.6993680906,
    -0.5875763221,
    -0.4069869034,
    0.0,
    -0.1281454481,
    0.6419895885,
    0.7559286424,
    0.0,
    -0.6337388239,
    -0.6785471501,
    -0.3714146849,
    0.0,
    0.5565051903,
    -0.2168887573,
    -0.8020356851,
    0.0,
    -0.5791554484,
    0.7244372011,
    -0.3738578718,
    0.0,
    0.1175779076,
    -0.7096451073,
    0.6946792478,
    0.0,
    -0.6134619607,
    0.1323631078,
    0.7785527795,
    0.0,
    0.6984635305,
    -0.02980516237,
    -0.715024719,
    0.0,
    0.8318082963,
    -0.3930171956,
    0.3919597455,
    0.0,
    0.1469576422,
    0.05541651717,
    -0.9875892167,
    0.0,
    0.708868575,
    -0.2690503865,
    0.6520101478,
    0.0,
    0.2726053183,
    0.67369766,
    -0.68688995,
    0.0,
    -0.6591295371,
    0.3035458599,
    -0.6880466294,
    0.0,
    0.4815131379,
    -0.7528270071,
    0.4487723203,
    0.0,
    0.9430009463,
    0.1675647412,
    -0.2875261255,
    0.0,
    0.434802957,
    0.7695304522,
    -0.4677277752,
    0.0,
    0.3931996188,
    0.594473625,
    0.7014236729,
    0.0,
    0.7254336655,
    -0.603925654,
    0.3301814672,
    0.0,
    0.7590235227,
    -0.6506083235,
    0.02433313207,
    0.0,
    -0.8552768592,
    -0.3430042733,
    0.3883935666,
    0.0,
    -0.6139746835,
    0.6981725247,
    0.3682257648,
    0.0,
    -0.7465905486,
    -0.5752009504,
    0.3342849376,
    0.0,
    0.5730065677,
    0.810555537,
    -0.1210916791,
    0.0,
    -0.9225877367,
    -0.3475211012,
    -0.167514036,
    0.0,
    -0.7105816789,
    -0.4719692027,
    -0.5218416899,
    0.0,
    -0.08564609717,
    0.3583001386,
    0.929669703,
    0.0,
    -0.8279697606,
    -0.2043157126,
    0.5222271202,
    0.0,
    0.427944023,
    0.278165994,
    0.8599346446,
    0.0,
    0.5399079671,
    -0.7857120652,
    -0.3019204161,
    0.0,
    0.5678404253,
    -0.5495413974,
    -0.6128307303,
    0.0,
    -0.9896071041,
    0.1365639107,
    -0.04503418428,
    0.0,
    -0.6154342638,
    -0.6440875597,
    0.4543037336,
    0.0,
    0.1074204368,
    -0.7946340692,
    0.5975094525,
    0.0,
    -0.3595449969,
    -0.8885529948,
    0.28495784,
    0.0,
    -0.2180405296,
    0.1529888965,
    0.9638738118,
    0.0,
    -0.7277432317,
    -0.6164050508,
    -0.3007234646,
    0.0,
    0.7249729114,
    -0.00669719484,
    0.6887448187,
    0.0,
    -0.5553659455,
    -0.5336586252,
    0.6377908264,
    0.0,
    0.5137558015,
    0.7976208196,
    -0.3160000073,
    0.0,
    -0.3794024848,
    0.9245608561,
    -0.03522751494,
    0.0,
    0.8229248658,
    0.2745365933,
    -0.4974176556,
    0.0,
    -0.5404114394,
    0.6091141441,
    0.5804613989,
    0.0,
    0.8036581901,
    -0.2703029469,
    0.5301601931,
    0.0,
    0.6044318879,
    0.6832968393,
    0.4095943388,
    0.0,
    0.06389988817,
    0.9658208605,
    -0.2512108074,
    0.0,
    0.1087113286,
    0.7402471173,
    -0.6634877936,
    0.0,
    -0.713427712,
    -0.6926784018,
    0.1059128479,
    0.0,
    0.6458897819,
    -0.5724548511,
    -0.5050958653,
    0.0,
    -0.6553931414,
    0.7381471625,
    0.159995615,
    0.0,
    0.3910961323,
    0.9188871375,
    -0.05186755998,
    0.0,
    -0.4879022471,
    -0.5904376907,
    0.6429111375,
    0.0,
    0.6014790094,
    0.7707441366,
    -0.2101820095,
    0.0,
    -0.5677173047,
    0.7511360995,
    0.3368851762,
    0.0,
    0.7858573506,
    0.226674665,
    0.5753666838,
    0.0,
    -0.4520345543,
    -0.604222686,
    -0.6561857263,
    0.0,
    0.002272116345,
    0.4132844051,
    -0.9105991643,
    0.0,
    -0.5815751419,
    -0.5162925989,
    0.6286591339,
    0.0,
    -0.03703704785,
    0.8273785755,
    0.5604221175,
    0.0,
    -0.5119692504,
    0.7953543429,
    -0.3244980058,
    0.0,
    -0.2682417366,
    -0.9572290247,
    -0.1084387619,
    0.0,
    -0.2322482736,
    -0.9679131102,
    -0.09594243324,
    0.0,
    0.3554328906,
    -0.8881505545,
    0.2913006227,
    0.0,
    0.7346520519,
    -0.4371373164,
    0.5188422971,
    0.0,
    0.9985120116,
    0.04659011161,
    -0.02833944577,
    0.0,
    -0.3727687496,
    -0.9082481361,
    0.1900757285,
    0.0,
    0.91737377,
    -0.3483642108,
    0.1925298489,
    0.0,
    0.2714911074,
    0.4147529736,
    -0.8684886582,
    0.0,
    0.5131763485,
    -0.7116334161,
    0.4798207128,
    0.0,
    -0.8737353606,
    0.18886992,
    -0.4482350644,
    0.0,
    0.8460043821,
    -0.3725217914,
    0.3814499973,
    0.0,
    0.8978727456,
    -0.1780209141,
    -0.4026575304,
    0.0,
    0.2178065647,
    -0.9698322841,
    -0.1094789531,
    0.0,
    -0.1518031304,
    -0.7788918132,
    -0.6085091231,
    0.0,
    -0.2600384876,
    -0.4755398075,
    -0.8403819825,
    0.0,
    0.572313509,
    -0.7474340931,
    -0.3373418503,
    0.0,
    -0.7174141009,
    0.1699017182,
    -0.6756111411,
    0.0,
    -0.684180784,
    0.02145707593,
    -0.7289967412,
    0.0,
    -0.2007447902,
    0.06555605789,
    -0.9774476623,
    0.0,
    -0.1148803697,
    -0.8044887315,
    0.5827524187,
    0.0,
    -0.7870349638,
    0.03447489231,
    0.6159443543,
    0.0,
    -0.2015596421,
    0.6859872284,
    0.6991389226,
    0.0,
    -0.08581082512,
    -0.10920836,
    -0.9903080513,
    0.0,
    0.5532693395,
    0.7325250401,
    -0.396610771,
    0.0,
    -0.1842489331,
    -0.9777375055,
    -0.1004076743,
    0.0,
    0.0775473789,
    -0.9111505856,
    0.4047110257,
    0.0,
    0.1399838409,
    0.7601631212,
    -0.6344734459,
    0.0,
    0.4484419361,
    -0.845289248,
    0.2904925424,
    0.0,
];

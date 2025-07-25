#![allow(clippy::needless_range_loop, clippy::missing_safety_doc)]

use crate::{variance_avx2, variance_avx512, variance_naive, variance_sse4};

#[inline(always)]
pub fn standardize(data: &mut [f64], df: usize) {
    if is_x86_feature_detected!("avx512f") {
        unsafe { standardize_avx512(data, df) }
    } else if is_x86_feature_detected!("avx2") {
        unsafe { standardize_avx2(data, df) }
    } else if is_x86_feature_detected!("sse4.1") {
        unsafe { standardize_sse4(data, df) }
    } else {
        standardize_naive(data, df)
    }
}

#[inline(always)]
pub fn standardize_recip(data: &mut [f64], df: usize) {
    if is_x86_feature_detected!("avx512f") {
        unsafe { standardize_avx512_recip(data, df) }
    } else if is_x86_feature_detected!("avx2") {
        unsafe { standardize_avx2_recip(data, df) }
    } else if is_x86_feature_detected!("sse4.1") {
        unsafe { standardize_sse4_recip(data, df) }
    } else {
        standardize_naive_recip(data, df)
    }
}

#[inline(always)]
pub fn standardize_naive(data: &mut [f64], df: usize) {
    let (mean, var) = variance_naive(data, df);
    let std = var.sqrt();
    for x in data.iter_mut() {
        *x = (*x - mean) / std;
    }
}

#[inline(always)]
pub fn standardize_naive_recip(data: &mut [f64], df: usize) {
    let (mean, var) = variance_naive(data, df);
    let std = var.sqrt();
    let std_recip = 1.0 / std;
    for x in data.iter_mut() {
        *x = (*x - mean) * std_recip;
    }
}

#[inline(always)]
pub unsafe fn standardize_sse4(data: &mut [f64], df: usize) {
    let (mean, var) = variance_sse4(data, df);
    let std = var.sqrt();
    core::arch::asm!(
        // xmm0 = mean
        "movddup xmm0, xmm0",
        // xmm1 = std
        "movddup xmm1, xmm1",
        "test rax, rax",
        "jz 3f",
            "2:",
            "movupd xmm2, [rsi]",
            "subpd xmm2, xmm0",
            "divpd xmm2, xmm1",
            "movupd [rsi], xmm2",

            "add rsi, 16",
            "dec rax",
            "jnz 2b",
        "3:",
        inout("xmm0") mean => _,
        inout("xmm1") std => _,
        inout("rsi") data.as_ptr() => _,
        inout("rax") data.len() / 2 => _,
    );
}

#[inline(always)]
pub unsafe fn standardize_sse4_recip(data: &mut [f64], df: usize) {
    let (mean, var) = variance_sse4(data, df);
    let std = var.sqrt();
    let std_recip = 1.0 / std;
    core::arch::asm!(
        // xmm0 = mean
        "movddup xmm0, xmm0",
        // xmm1 = std_recip
        "movddup xmm1, xmm1",
        "test rax, rax",
        "jz 3f",
            "2:",
            "movupd xmm2, [rsi]",
            "subpd xmm2, xmm0",
            "mulpd xmm2, xmm1",
            "movupd [rsi], xmm2",

            "add rsi, 16",
            "dec rax",
            "jnz 2b",
        "3:",
        inout("xmm0") mean => _,
        inout("xmm1") std_recip => _,
        inout("rsi") data.as_ptr() => _,
        inout("rax") data.len() / 2 => _,
    );
    if data.len() % 2 == 1 {
        let last = data.len() - 1;
        data[last] = (data[last] - mean) * std_recip;
    }
}

#[inline(always)]
pub unsafe fn standardize_avx2(data: &mut [f64], df: usize) {
    let (mean, var) = variance_avx2(data, df);
    let std = var.sqrt();
    core::arch::asm!(
        // ymm0 = mean
        "vbroadcastsd ymm0, xmm0",
        // ymm1 = std
        "vbroadcastsd ymm1, xmm1",
        "test rax, rax",
        "jz 3f",
            "2:",
            "vmovupd ymm2, [rsi]",
            "vsubpd ymm2, ymm2, ymm0",
            "vdivpd ymm2, ymm2, ymm1",
            "vmovupd [rsi], ymm2",

            "add rsi, 32",
            "dec rax",
            "jnz 2b",
        "3:",
        inout("xmm0") mean => _,
        inout("xmm1") std => _,
        inout("rsi") data.as_ptr() => _,
        inout("rax") data.len() / 4 => _,
    );
    if data.len() % 4 != 0 {
        let last = data.len() - data.len() % 4;
        for i in last..data.len() {
            data[i] = (data[i] - mean) / std;
        }
    }
}

#[inline(always)]
pub unsafe fn standardize_avx2_recip(data: &mut [f64], df: usize) {
    let (mean, var) = variance_avx2(data, df);
    let std = var.sqrt();
    let std_recip = 1.0 / std;
    core::arch::asm!(
        // ymm0 = mean
        "vbroadcastsd ymm0, xmm0",
        // ymm1 = std_recip
        "vbroadcastsd ymm1, xmm1",
        "test rax, rax",
        "jz 3f",
            "2:",
            "vmovupd ymm2, [rsi]",
            "vsubpd ymm2, ymm2, ymm0",
            "vmulpd ymm2, ymm2, ymm1",
            "vmovupd [rsi], ymm2",

            "add rsi, 32",
            "dec rax",
            "jnz 2b",
        "3:",
        inout("xmm0") mean => _,
        inout("xmm1") std_recip => _,
        inout("rsi") data.as_ptr() => _,
        inout("rax") data.len() / 4 => _,
    );
    if data.len() % 4 != 0 {
        let last = data.len() - data.len() % 4;
        for i in last..data.len() {
            data[i] = (data[i] - mean) * std_recip;
        }
    }
}

#[inline(always)]
pub unsafe fn standardize_avx512(data: &mut [f64], df: usize) {
    let (mean, var) = variance_avx512(data, df);
    let std = var.sqrt();
    core::arch::asm!(
        // zmm0 = mean
        "vbroadcastsd zmm0, xmm0",
        // zmm1 = std
        "vbroadcastsd zmm1, xmm1",
        "test rax, rax",
        "jz 3f",
            "2:",
            "vmovupd zmm2, [rsi]",
            "vsubpd zmm2, zmm2, zmm0",
            "vdivpd zmm2, zmm2, zmm1",
            "vmovupd [rsi], zmm2",

            "add rsi, 64",
            "dec rax",
            "jnz 2b",
        "3:",
        inout("xmm0") mean => _,
        inout("xmm1") std => _,
        inout("rsi") data.as_ptr() => _,
        inout("rax") data.len() / 8 => _,
    );
    if data.len() % 8 != 0 {
        let last = data.len() - data.len() % 8;
        for i in last..data.len() {
            data[i] = (data[i] - mean) / std;
        }
    }
}

#[inline(always)]
pub unsafe fn standardize_avx512_recip(data: &mut [f64], df: usize) {
    let (mean, var) = variance_avx512(data, df);
    let std = var.sqrt();
    let std_recip = 1.0 / std;
    core::arch::asm!(
        // zmm0 = mean
        "vbroadcastsd zmm0, xmm0",
        // zmm1 = std_recip
        "vbroadcastsd zmm1, xmm1",
        "test rax, rax",
        "jz 3f",
            "2:",
            "vmovupd zmm2, [rsi]",
            "vsubpd zmm2, zmm2, zmm0",
            "vmulpd zmm2, zmm2, zmm1",
            "vmovupd [rsi], zmm2",

            "add rsi, 64",
            "dec rax",
            "jnz 2b",
        "3:",
        inout("xmm0") mean => _,
        inout("xmm1") std_recip => _,
        inout("rsi") data.as_ptr() => _,
        inout("rax") data.len() / 8 => _,
    );
    if data.len() % 8 != 0 {
        let last = data.len() - data.len() % 8;
        for i in last..data.len() {
            data[i] = (data[i] - mean) * std_recip;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mean::mean_naive, variance::variance_naive};

    macro_rules! assert_float_eq {
        ($a:expr, $b:expr, $tol:expr) => {
            assert!(($a - $b).abs() < $tol, "{:.22} != {:.22}", $a, $b);
        };
    }

    macro_rules! float_eq {
        ($a:expr, $b:expr) => {
            assert_float_eq!($a, $b, 1e-12);
        };
    }

    fn data() -> Vec<f64> {
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    }

    const EXPECTED: &[f64] = &[
        -1.4288690166235207,
        -1.0206207261596576,
        -0.6123724356957946,
        -0.20412414523193154,
        0.20412414523193154,
        0.6123724356957946,
        1.0206207261596576,
        1.4288690166235207,
    ];

    fn data_nan() -> Vec<f64> {
        vec![1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0, 7.0, 8.0]
    }

    const EXPECTED_NAN: &[f64] = &[
        -1.3544880821826935,
        -0.9752314191715392,
        -0.595974756160385,
        f64::NAN,
        0.1625385698619233,
        0.5417952328730775,
        0.9210518958842316,
        1.3003085588953858,
    ];

    #[test]
    fn test_standardize_naive() {
        let mut data = data();
        standardize_naive(&mut data, 1);
        for i in 0..data.len() {
            float_eq!(data[i], EXPECTED[i]);
        }
    }

    #[test]
    fn test_standardize_naive_nan() {
        let mut data = data_nan();
        standardize_naive(&mut data, 1);
        for i in 0..data.len() {
            if data[i].is_nan() {
                assert!(EXPECTED_NAN[i].is_nan());
            } else {
                float_eq!(data[i], EXPECTED_NAN[i]);
            }
        }
    }

    #[test]
    fn test_standardize_naive_recip() {
        let mut data = data();
        standardize_naive_recip(&mut data, 1);
        for i in 0..data.len() {
            float_eq!(data[i], EXPECTED[i]);
        }
    }

    #[test]
    fn test_standardize_naive_recip_nan() {
        let mut data = data_nan();
        standardize_naive_recip(&mut data, 1);
        for i in 0..data.len() {
            if data[i].is_nan() {
                assert!(EXPECTED_NAN[i].is_nan());
            } else {
                float_eq!(data[i], EXPECTED_NAN[i]);
            }
        }
    }

    #[test]
    fn test_standardize_sse4() {
        if is_x86_feature_detected!("sse4.1") {
            let mut data = data();
            unsafe { standardize_sse4(&mut data, 1) };
            for i in 0..data.len() {
                float_eq!(data[i], EXPECTED[i]);
            }
        }
    }

    #[test]
    fn test_standardize_sse4_nan() {
        if is_x86_feature_detected!("sse4.1") {
            let mut data = data_nan();
            unsafe { standardize_sse4(&mut data, 1) };
            for i in 0..data.len() {
                if data[i].is_nan() {
                    assert!(EXPECTED_NAN[i].is_nan());
                } else {
                    float_eq!(data[i], EXPECTED_NAN[i]);
                }
            }
        }
    }

    #[test]
    fn test_standardize_sse4_recip() {
        if is_x86_feature_detected!("sse4.1") {
            let mut data = data();
            unsafe { standardize_sse4_recip(&mut data, 1) };
            for i in 0..data.len() {
                float_eq!(data[i], EXPECTED[i]);
            }
        }
    }

    #[test]
    fn test_standardize_sse4_recip_nan() {
        if is_x86_feature_detected!("sse4.1") {
            let mut data = data_nan();
            unsafe { standardize_sse4_recip(&mut data, 1) };
            for i in 0..data.len() {
                if data[i].is_nan() {
                    assert!(EXPECTED_NAN[i].is_nan());
                } else {
                    float_eq!(data[i], EXPECTED_NAN[i]);
                }
            }
        }
    }

    #[test]
    fn test_standardize_avx2() {
        if is_x86_feature_detected!("avx2") {
            let mut data = data();
            unsafe { standardize_avx2(&mut data, 1) };
            for i in 0..data.len() {
                float_eq!(data[i], EXPECTED[i]);
            }
        }
    }

    #[test]
    fn test_standardize_avx2_nan() {
        if is_x86_feature_detected!("avx2") {
            let mut data = data_nan();
            unsafe { standardize_avx2(&mut data, 1) };
            for i in 0..data.len() {
                if data[i].is_nan() {
                    assert!(EXPECTED_NAN[i].is_nan());
                } else {
                    float_eq!(data[i], EXPECTED_NAN[i]);
                }
            }
        }
    }

    #[test]
    fn test_standardize_avx2_recip() {
        if is_x86_feature_detected!("avx2") {
            let mut data = data();
            unsafe { standardize_avx2_recip(&mut data, 1) };
            for i in 0..data.len() {
                float_eq!(data[i], EXPECTED[i]);
            }
        }
    }

    #[test]
    fn test_standardize_avx2_recip_nan() {
        if is_x86_feature_detected!("avx2") {
            let mut data = data_nan();
            unsafe { standardize_avx2_recip(&mut data, 1) };
            for i in 0..data.len() {
                if data[i].is_nan() {
                    assert!(EXPECTED_NAN[i].is_nan());
                } else {
                    float_eq!(data[i], EXPECTED_NAN[i]);
                }
            }
        }
    }

    #[test]
    fn test_standardize_avx512() {
        if is_x86_feature_detected!("avx512f") {
            let mut data = data();
            unsafe { standardize_avx512(&mut data, 1) };
            for i in 0..data.len() {
                float_eq!(data[i], EXPECTED[i]);
            }
        }
    }

    #[test]
    fn test_standardize_avx512_nan() {
        if is_x86_feature_detected!("avx512f") {
            let mut data = data_nan();
            unsafe { standardize_avx512(&mut data, 1) };
            for i in 0..data.len() {
                if data[i].is_nan() {
                    assert!(EXPECTED_NAN[i].is_nan());
                } else {
                    float_eq!(data[i], EXPECTED_NAN[i]);
                }
            }
        }
    }

    #[test]
    fn test_standardize_avx512_recip() {
        if is_x86_feature_detected!("avx512f") {
            let mut data = data();
            unsafe { standardize_avx512_recip(&mut data, 1) };
            for i in 0..data.len() {
                float_eq!(data[i], EXPECTED[i]);
            }
        }
    }

    #[test]
    fn test_standardize_avx512_recip_nan() {
        if is_x86_feature_detected!("avx512f") {
            let mut data = data_nan();
            unsafe { standardize_avx512_recip(&mut data, 1) };
            for i in 0..data.len() {
                if data[i].is_nan() {
                    assert!(EXPECTED_NAN[i].is_nan());
                } else {
                    float_eq!(data[i], EXPECTED_NAN[i]);
                }
            }
        }
    }
}

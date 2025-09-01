#![allow(clippy::needless_range_loop, clippy::missing_safety_doc)]

use crate::{mean_avx2, mean_avx512, mean_naive, mean_sse4};

#[inline(always)]
pub fn r2(actual: &[f64], predicted: &[f64]) -> f64 {
    assert_eq!(
        actual.len(),
        predicted.len(),
        "actual and predicted must have the same length"
    );
    if is_x86_feature_detected!("avx512f") {
        unsafe { r2_avx512(actual, predicted) }
    } else if is_x86_feature_detected!("avx2") {
        unsafe { r2_avx2(actual, predicted) }
    } else if is_x86_feature_detected!("sse4.1") {
        unsafe { r2_sse4(actual, predicted) }
    } else {
        r2_naive(actual, predicted)
    }
}

#[inline(always)]
pub fn r2_naive(actual: &[f64], predicted: &[f64]) -> f64 {
    let (mean, _) = mean_naive(actual);
    let mut rss = 0.0;
    let mut tss = 0.0;
    for i in 0..actual.len() {
        let a = actual[i];
        let p = predicted[i];
        rss += (a - p).powi(2);
        tss += (a - mean).powi(2);
    }
    1.0 - rss / tss
}

#[inline(always)]
pub unsafe fn r2_sse4(actual: &[f64], predicted: &[f64]) -> f64 {
    let (mean, _) = mean_sse4(actual);
    let mut rss: f64;
    let mut tss: f64;
    core::arch::asm! {
        // xmm0 = mean
        "movddup xmm0, xmm0",
        // xmm1 = rss
        "xorpd xmm1, xmm1",
        // xmm2 = tss
        "xorpd xmm2, xmm2",
        "test rax, rax",
        "jz 3f",
            "2:",

            // load the actual values
            "movupd xmm3, [rsi]",
            "movupd xmm4, [rsi]",
            "movupd xmm5, [rdx]",
            // actual - predicted
            "subpd xmm3, xmm5",
            // actual - mean
            "subpd xmm4, xmm0",
            // square the differences
            "mulpd xmm3, xmm3",
            "mulpd xmm4, xmm4",
            // add to the sums
            "addpd xmm1, xmm3",
            "addpd xmm2, xmm4",

            "add rsi, 16",
            "add rdx, 16",
            "dec rax",
            "jnz 2b",
        "3:",
        "haddpd xmm1, xmm1",
        "haddpd xmm2, xmm2",

        inout("xmm0") mean => _,
        out("xmm1") rss,
        out("xmm2") tss,
        out("xmm3") _,
        out("xmm4") _,
        out("xmm5") _,
        inout("rsi") actual.as_ptr() => _,
        inout("rdx") predicted.as_ptr() => _,
        inout("rax") actual.len() / 2 => _,
        options(readonly, nostack),
    }
    if actual.len() % 2 == 1 {
        let a = actual[actual.len() - 1];
        let p = predicted[predicted.len() - 1];
        rss += (a - p).powi(2);
        tss += (a - mean).powi(2);
    }
    1.0 - rss / tss
}

#[inline(always)]
pub unsafe fn r2_avx2(actual: &[f64], predicted: &[f64]) -> f64 {
    let (mean, _) = mean_avx2(actual);
    let mut rss: f64;
    let mut tss: f64;
    core::arch::asm! {
        // xmm0 = mean
        "vbroadcastsd ymm0, xmm0",
        // xmm1 = rss
        "vxorpd ymm1, ymm1, ymm1",
        // xmm2 = tss
        "vxorpd ymm2, ymm2, ymm2",
        "test rax, rax",
        "jz 3f",
            "2:",

            // load the actual values
            "vmovupd ymm4, [rsi]",
            // actual - predicted
            "vsubpd ymm3, ymm4, [rdx]",
            // actual - mean
            "vsubpd ymm4, ymm4, ymm0",
            // square the differences
            "vmulpd ymm3, ymm3, ymm3",
            "vmulpd ymm4, ymm4, ymm4",
            // add to the sums
            "vaddpd ymm1, ymm1, ymm3",
            "vaddpd ymm2, ymm2, ymm4",

            "add rsi, 32",
            "add rdx, 32",
            "dec rax",
            "jnz 2b",
        "3:",
        "vextractf128 xmm3, ymm1, 1",
        "vextractf128 xmm4, ymm2, 1",
        "vaddpd xmm1, xmm1, xmm3",
        "vaddpd xmm2, xmm2, xmm4",
        "vhaddpd xmm1, xmm1, xmm1",
        "vhaddpd xmm2, xmm2, xmm2",
        inout("xmm0") mean => _,
        out("xmm1") rss,
        out("xmm2") tss,
        out("ymm3") _,
        out("ymm4") _,
        inout("rsi") actual.as_ptr() => _,
        inout("rdx") predicted.as_ptr() => _,
        inout("rax") actual.len() / 4 => _,
        options(readonly, nostack),
    }
    if actual.len() % 4 != 0 {
        for i in (actual.len() - actual.len() % 4)..actual.len() {
            let a = actual[i];
            let p = predicted[i];
            rss += (a - p).powi(2);
            tss += (a - mean).powi(2);
        }
    }
    1.0 - rss / tss
}

#[inline(always)]
pub unsafe fn r2_avx512(actual: &[f64], predicted: &[f64]) -> f64 {
    let (mean, _) = mean_avx512(actual);
    let mut rss: f64;
    let mut tss: f64;
    core::arch::asm! {
        // xmm0 = mean
        "vbroadcastsd zmm0, xmm0",
        // xmm1 = rss
        "vxorpd zmm1, zmm1, zmm1",
        // xmm2 = tss
        "vxorpd zmm2, zmm2, zmm2",
        "test rax, rax",
        "jz 3f",
            "2:",

            // load the actual values
            "vmovupd zmm4, [rsi]",
            // actual - predicted
            "vsubpd zmm3, zmm4, [rdx]",
            // actual - mean
            "vsubpd zmm4, zmm4, zmm0",
            // square the differences
            "vfmadd231pd zmm1, zmm3, zmm3",
            "vfmadd231pd zmm2, zmm4, zmm4",

            "add rsi, 64",
            "add rdx, 64",
            "dec rax",
            "jnz 2b",
        "3:",
        "vextractf64x4 ymm3, zmm1, 1",
        "vextractf64x4 ymm4, zmm2, 1",
        "vaddpd ymm1, ymm1, ymm3",
        "vaddpd ymm2, ymm2, ymm4",
        "vextractf64x2 xmm3, ymm1, 1",
        "vextractf64x2 xmm4, ymm2, 1",
        "vaddpd xmm1, xmm1, xmm3",
        "vaddpd xmm2, xmm2, xmm4",
        "vhaddpd xmm1, xmm1, xmm1",
        "vhaddpd xmm2, xmm2, xmm2",
        inout("xmm0") mean => _,
        out("xmm1") rss,
        out("xmm2") tss,
        out("ymm3") _,
        out("ymm4") _,
        inout("rsi") actual.as_ptr() => _,
        inout("rdx") predicted.as_ptr() => _,
        inout("rax") actual.len() / 8 => _,
        options(readonly, nostack),
    }
    if actual.len() % 8 != 0 {
        for i in (actual.len() - actual.len() % 8)..actual.len() {
            let a = actual[i];
            let p = predicted[i];
            rss += (a - p).powi(2);
            tss += (a - mean).powi(2);
        }
    }
    1.0 - rss / tss
}

#[cfg(test)]
mod tests {
    use super::*;

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

    const DATA_1: &[f64] = &[
        1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0,
        4.0, 5.0,
    ];
    const R2_1: f64 = 1.0;
    const DATA_2: &[f64] = &[
        1.0, 2.0, 3.0, 5.0, 6.0, 1.0, 2.0, 3.0, 5.0, 6.0, 1.0, 2.0, 3.0, 5.0, 6.0, 1.0, 2.0, 3.0,
        5.0, 6.0,
    ];
    const R2_2: f64 = 0.8837209302325582;

    #[test]
    fn test_r2_naive() {
        float_eq!(r2_naive(DATA_1, DATA_1), R2_1);
        float_eq!(r2_naive(DATA_2, DATA_1), R2_2);
    }

    #[test]
    #[cfg_attr(not(target_feature = "sse4.1"), ignore)]
    fn test_r2_sse4() {
        unsafe {
            float_eq!(r2_sse4(DATA_1, DATA_1), R2_1);
            float_eq!(r2_sse4(DATA_2, DATA_1), R2_2);
        }
        unsafe {
            float_eq!(r2_sse4(DATA_1, DATA_1), R2_1);
            float_eq!(r2_sse4(DATA_2, DATA_1), R2_2);
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_r2_avx2() {
        unsafe {
            float_eq!(r2_avx2(DATA_1, DATA_1), R2_1);
            float_eq!(r2_avx2(DATA_2, DATA_1), R2_2);
        }
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_r2_avx512() {
        unsafe {
            float_eq!(r2_avx512(DATA_1, DATA_1), R2_1);
            float_eq!(r2_avx512(DATA_2, DATA_1), R2_2);
        }
    }
}

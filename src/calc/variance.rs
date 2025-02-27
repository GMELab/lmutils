#![allow(clippy::needless_range_loop, clippy::missing_safety_doc)]

use super::mean::{mean_avx2, mean_avx512, mean_naive};

pub fn variance(data: &[f64]) -> f64 {
    if is_x86_feature_detected!("avx512f") {
        unsafe { variance_avx512(data) }
    } else if is_x86_feature_detected!("avx") {
        unsafe { variance_avx2(data) }
    } else {
        variance_naive(data)
    }
}

pub fn variance_naive(data: &[f64]) -> f64 {
    let m = mean_naive(data);
    let mut sum = 0.0;
    for i in 0..data.len() {
        sum += (data[i] - m).powi(2);
    }
    sum / data.len() as f64
}

#[target_feature(enable = "avx")]
pub unsafe fn variance_avx2(data: &[f64]) -> f64 {
    let m = mean_avx2(data);
    let mut sum = 0.0;
    core::arch::asm! {
        // this is the accumulation of the squared differences
        "vbroadcastsd ymm0, xmm0",
        // this is the mean
        "vbroadcastsd ymm1, xmm1",
        "test rax, rax",
        "jz 3f",
            "2:",
            "vmovupd ymm2, ymmword ptr [rsi]",
            "vsubpd ymm2, ymm2, ymm1",
            "vmulpd ymm2, ymm2, ymm2",
            "vaddpd ymm0, ymm0, ymm2",

            "add rsi, 32",
            "dec rax",
            "jnz 2b",
        "3:",
        // extract the two parts ymm0 into xmm1 and xmm0
        "vextractf128 xmm1, ymm0, 1",
        "vaddpd xmm2, xmm1, xmm0",
        "vhaddpd xmm2, xmm2, xmm2",
        "vzeroupper",

        in("xmm0") 0.0,
        in("xmm1") m,
        inout("xmm2") sum => sum,
        in("rax") data.len() / 4,
        in("rsi") data.as_ptr(),
    }
    if data.len() % 4 != 0 {
        for i in (data.len() - data.len() % 4)..data.len() {
            sum += (data[i] - m).powi(2);
        }
    }
    sum / data.len() as f64
}

#[target_feature(enable = "avx")]
pub unsafe fn variance_avx512(data: &[f64]) -> f64 {
    let m = mean_avx512(data);
    let mut sum = 0.0;
    core::arch::asm! {
        // this is the accumulation of the squared differences
        "vbroadcastsd zmm0, xmm0",
        // this is the mean
        "vbroadcastsd zmm1, xmm1",
        "test rax, rax",
        "jz 3f",
            "2:",
            "vmovupd zmm2, zmmword ptr [rsi]",
            "vsubpd zmm2, zmm2, zmm1",
            "vfmadd231pd zmm0, zmm2, zmm2",

            "add rsi, 64",
            "dec rax",
            "jnz 2b",
        "3:",
        // extract the two parts zmm0 into ymm2 and ymm3
        "vextractf64x4 ymm2, zmm0, 0",
        "vextractf64x4 ymm3, zmm0, 1",
        // add the two parts
        "vaddpd ymm1, ymm2, ymm3",
        // extract the two parts ymm3 into xmm1 and xmm3
        "vextractf64x2 xmm2, ymm1, 0",
        "vextractf64x2 xmm3, ymm1, 1",
        "vaddpd xmm2, xmm2, xmm3",
        "vhaddpd xmm2, xmm2, xmm2",
        "vzeroupper",

        in("xmm0") 0.0,
        in("xmm1") m,
        inout("xmm2") sum => sum,
        out("ymm3") _,
        in("rax") data.len() / 8,
        in("rsi") data.as_ptr(),
    }
    if data.len() % 8 != 0 {
        for i in (data.len() - data.len() % 8)..data.len() {
            sum += (data[i] - m).powi(2);
        }
    }
    sum / data.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_float_eq {
        ($a:expr, $b:expr, $tol:expr) => {
            assert!(($a - $b).abs() < $tol);
        };
    }

    macro_rules! float_eq {
        ($a:expr, $b:expr) => {
            assert_float_eq!($a, $b, 1e-13);
        };
    }

    fn data() -> Vec<f64> {
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .cycle()
            .take(8 * 1000 - 1)
            .copied()
            .collect::<Vec<f64>>()
    }
    const VARIANCE: f64 = 5.249124699157198;

    #[test]
    fn test_variance_naive() {
        assert_eq!(variance_naive(&data()), VARIANCE);
    }

    #[test]
    fn test_variance_avx2() {
        float_eq!(unsafe { variance_avx2(&data()) }, VARIANCE);
    }

    #[test]
    fn test_variance_avx512() {
        float_eq!(unsafe { variance_avx512(&data()) }, VARIANCE);
    }
}

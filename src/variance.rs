#![allow(clippy::needless_range_loop, clippy::missing_safety_doc)]

use super::mean::{mean_avx2, mean_avx512, mean_naive, mean_sse4};

#[inline(always)]
pub fn variance(data: &[f64], df: usize) -> f64 {
    if is_x86_feature_detected!("avx512f") {
        unsafe { variance_avx512(data, df).1 }
    } else if is_x86_feature_detected!("avx2") {
        unsafe { variance_avx2(data, df).1 }
    } else if is_x86_feature_detected!("sse4.1") {
        unsafe { variance_sse4(data, df).1 }
    } else {
        variance_naive(data, df).1
    }
}

#[inline(always)]
pub fn variance_naive(data: &[f64], df: usize) -> (f64, f64) {
    let (m, count) = mean_naive(data);
    let mut sum = 0.0;
    for i in 0..data.len() {
        let d = data[i];
        if !d.is_nan() {
            sum += (d - m).powi(2);
        }
    }
    let df = df as u64;
    if count <= df {
        return (m, 0.0);
    }
    (m, sum / (count - df) as f64)
}

#[inline(always)]
pub unsafe fn variance_sse4(data: &[f64], df: usize) -> (f64, f64) {
    let (m, count) = mean_sse4(data);
    let mut sum = 0.0;
    core::arch::asm! {
        // this is the accumulation of the squared differences
        "xorpd xmm1, xmm1",
        // this is the mean
        "movddup xmm2, xmm2",
        // zero vector
        "xorpd xmm4, xmm4",
        "test rax, rax",
        "jz 3f",
            "2:",
            "movupd xmm3, [rsi]",
            "subpd xmm3, xmm2",
            "mulpd xmm3, xmm3",
            "movupd xmm0, xmm3",
            "cmppd xmm0, xmm0, 3",
            "blendvpd xmm3, xmm4",
            "addpd xmm1, xmm3",

            "add rsi, 16",
            "dec rax",
            "jnz 2b",
        "3:",
        "haddpd xmm1, xmm1",

        // mask register
        out("xmm0") _,
        out("xmm1") sum,
        inout("xmm2") m => _,
        out("xmm3") _,
        out("xmm4") _,
        inout("rsi") data.as_ptr() => _,
        inout("rax") data.len() / 2 => _,
        options(readonly, nostack),
    }
    if data.len() % 2 != 0 {
        for i in (data.len() - data.len() % 2)..data.len() {
            let d = data[i];
            if !d.is_nan() {
                sum += (d - m).powi(2);
            }
        }
    }
    let df = df as u64;
    if count <= df {
        return (m, 0.0);
    }
    (m, sum / (count - df) as f64)
}

#[inline(always)]
pub unsafe fn variance_avx2(data: &[f64], df: usize) -> (f64, f64) {
    let (m, count) = mean_avx2(data);
    let mut sum = 0.0;
    core::arch::asm! {
        // this is the accumulation of the squared differences
        "vpxor ymm0, ymm0, ymm0",
        // this is the mean
        "vbroadcastsd ymm1, xmm1",
        // zero vector
        "vpxor ymm3, ymm3, ymm3",
        "test rax, rax",
        "jz 3f",
            "2:",
            "vmovupd ymm2, ymmword ptr [rsi]",
            "vsubpd ymm2, ymm2, ymm1",
            "vmulpd ymm2, ymm2, ymm2",
            "vcmppd ymm4, ymm2, ymm2, 0",
            "vblendvpd ymm2, ymm3, ymm2, ymm4",
            "vaddpd ymm0, ymm0, ymm2",

            "add rsi, 32",
            "dec rax",
            "jnz 2b",
        "3:",
        // extract the two parts ymm0 into xmm1 and xmm0
        "vextractf128 xmm1, ymm0, 1",
        "vaddpd xmm0, xmm1, xmm0",
        "vhaddpd xmm0, xmm0, xmm0",
        "vzeroupper",

        out("xmm0") sum,
        inout("xmm1") m => _,
        out("xmm2") _,
        out("ymm3") _,
        out("ymm4") _,
        in("rax") data.len() / 4,
        in("rsi") data.as_ptr(),
        options(readonly, nostack),
    }
    if data.len() % 4 != 0 {
        for i in (data.len() - data.len() % 4)..data.len() {
            let d = data[i];
            if !d.is_nan() {
                sum += (d - m).powi(2);
            }
        }
    }
    let df = df as u64;
    if count <= df {
        return (m, 0.0);
    }
    (m, sum / (count - df) as f64)
}

#[inline(always)]
pub unsafe fn variance_avx512(data: &[f64], df: usize) -> (f64, f64) {
    let (m, count) = mean_avx512(data);
    let mut sum = 0.0;
    core::arch::asm! {
        // this is the accumulation of the squared differences
        "vpxorq zmm0, zmm0, zmm0",
        // this is the mean
        "vbroadcastsd zmm1, xmm1",
        "test rax, rax",
        "jz 3f",
            "2:",
            "vmovupd zmm2, zmmword ptr [rsi]",
            "vsubpd zmm2, zmm2, zmm1",
            "vcmppd k1, zmm2, zmm2, 0",
            "vfmadd231pd zmm0{{k1}}, zmm2, zmm2",

            "add rsi, 64",
            "dec rax",
            "jnz 2b",
        "3:",
        // extract the two parts zmm0 into ymm2 and ymm3
        "vextractf64x4 ymm1, zmm0, 1",
        // add the two parts
        "vaddpd ymm0, ymm0, ymm1",
        // extract the two parts ymm3 into xmm1 and xmm3
        "vextractf64x2 xmm1, ymm0, 1",
        "vaddpd xmm0, xmm0, xmm1",
        "vhaddpd xmm0, xmm0, xmm0",
        "vzeroupper",

        out("xmm0") sum,
        inout("xmm1") m => _,
        out("zmm2") _,
        inout("rax") data.len() / 8 => _,
        inout("rsi") data.as_ptr() => _,
        options(readonly, nostack),
    }
    if data.len() % 8 != 0 {
        for i in (data.len() - data.len() % 8)..data.len() {
            let d = data[i];
            if !d.is_nan() {
                sum += (d - m).powi(2);
            }
        }
    }
    let df = df as u64;
    if count <= df {
        return (m, 0.0);
    }
    (m, sum / (count - df) as f64)
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
    fn data_nan() -> Vec<f64> {
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, f64::NAN, f64::NAN]
            .iter()
            .cycle()
            .take(8 * 1000 - 1)
            .copied()
            .collect::<Vec<f64>>()
    }

    const VARIANCE: f64 = 5.249124699157198;
    const VARIANCE_NAN: f64 = 2.9166666666666665;

    #[test]
    fn test_variance_naive() {
        assert_eq!(variance_naive(&data(), 0).1, VARIANCE);
    }

    #[test]
    fn test_variance_naive_nan() {
        assert_eq!(variance_naive(&data_nan(), 0).1, VARIANCE_NAN);
    }

    #[test]
    fn test_variance_sse4() {
        if is_x86_feature_detected!("sse4.1") {
            float_eq!(unsafe { variance_sse4(&data(), 0).1 }, VARIANCE);
        }
    }

    #[test]
    fn test_variance_sse4_nan() {
        if is_x86_feature_detected!("sse4.1") {
            float_eq!(unsafe { variance_sse4(&data_nan(), 0).1 }, VARIANCE_NAN);
        }
    }

    #[test]
    fn test_variance_avx2() {
        if is_x86_feature_detected!("avx") {
            float_eq!(unsafe { variance_avx2(&data(), 0).1 }, VARIANCE);
        }
    }

    #[test]
    fn test_variance_avx2_nan() {
        if is_x86_feature_detected!("avx") {
            float_eq!(unsafe { variance_avx2(&data_nan(), 0).1 }, VARIANCE_NAN);
        }
    }

    #[test]
    fn test_variance_avx512() {
        if is_x86_feature_detected!("avx512f") {
            float_eq!(unsafe { variance_avx512(&data(), 0).1 }, VARIANCE);
        }
    }

    #[test]
    fn test_variance_avx512_nan() {
        if is_x86_feature_detected!("avx512f") {
            float_eq!(unsafe { variance_avx512(&data_nan(), 0).1 }, VARIANCE_NAN);
        }
    }
}

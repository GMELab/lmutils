#![allow(clippy::needless_range_loop, clippy::missing_safety_doc)]

pub fn mean(data: &[f64]) -> f64 {
    if is_x86_feature_detected!("avx512f") {
        unsafe { mean_avx512(data) }
    } else if is_x86_feature_detected!("avx2") {
        unsafe { mean_avx2(data) }
    } else if is_x86_feature_detected!("sse4.1") {
        unsafe { mean_sse(data) }
    } else {
        mean_naive(data)
    }
}

pub fn mean_naive(data: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0.0;
    for i in 0..data.len() {
        let d = data[i];
        if !d.is_nan() {
            count += 1.0;
            sum += d;
        }
    }
    if count == 0.0 {
        0.0
    } else {
        sum / count
    }
}

pub unsafe fn mean_sse(data: &[f64]) -> f64 {
    let mut sum: f64;
    let mut count: u64;
    core::arch::asm! {
        // xmm1 = sum
        "xorpd xmm1, xmm1",
        // xmm2 = count
        "pxor xmm2, xmm2",
        // xmm3 = 1
        "movddup xmm3, xmm3",
        // xmm4 = 0
        "xorpd xmm4, xmm4",
        "test rax, rax",
        "jz 3f",
            "2:",
            // load from memory
            "movupd xmm0, [rsi]",
            "movupd xmm5, xmm0",
            // check for NaNs
            "cmppd xmm0, xmm0, 3",
            // replace NaNs with 0.0
            "blendvpd xmm5, xmm4",
            // replace NaNs with 1 for counting
            "pand xmm0, xmm3",
            // count NaNs
            "paddq xmm2, xmm0",
            // sum
            "addpd xmm1, xmm5",

            "add rsi, 16",
            "dec rax",
            "jnz 2b",
        "3:",
        "vhaddpd xmm1, xmm1, xmm1",
        // sum xmm2
        "movupd xmm3, xmm2",
        "psrldq xmm2, 8",
        "paddq xmm2, xmm3",

        out("xmm0") _,
        out("xmm1") sum,
        out("xmm2") count,
        inout("xmm3") 1 => _,
        out("xmm4") _,
        out("xmm5") _,
        inout("rsi") data.as_ptr() => _,
        inout("rax") data.len() / 2 => _,
        options(readonly, nostack),
    }
    if data.len() % 2 != 0 {
        let d = data[data.len() - 1];
        if d.is_nan() {
            count += 1;
        } else {
            sum += d;
        }
    }
    count = data.len() as u64 - count;
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

// #[target_feature(enable = "avx")]
#[inline(always)]
pub unsafe fn mean_avx2(data: &[f64]) -> f64 {
    let mut sum: f64;
    let mut count: u64;
    core::arch::asm! {
        // ymm0 = sum
        "vxorpd ymm0, ymm0, ymm0",
        // ymm1 = count
        "vpxor ymm1, ymm1, ymm1",
        // ymm2 = 1
        "vbroadcastsd ymm2, xmm2",
        // ymm3 = 0
        "vxorpd ymm3, ymm3, ymm3",
        "test rax, rax",
        "jz 3f",
            "2:",
            // load from memory
            "vmovupd ymm5, [rsi]",
            // check for NaNs
            "vcmppd ymm4, ymm5, ymm5, 3",
            // replace NaNs with 0.0
            "vblendvpd ymm5, ymm5, ymm3, ymm4",
            // replace NaNs with 1 for counting
            "vpand ymm4, ymm4, ymm2",
            // count NaNs
            "vpaddq ymm1, ymm1, ymm4",
            // sum
            "vaddpd ymm0, ymm0, ymm5",

            "add rsi, 32",
            "dec rax",
            "jnz 2b",
        "3:",
        // extract sum
        "vextractf128 xmm2, ymm0, 1",
        "vaddpd xmm0, xmm0, xmm2",
        "vhaddpd xmm0, xmm0, xmm0",
        // extract count
        "vextracti128 xmm2, ymm1, 1",
        "vpaddq xmm1, xmm1, xmm2",
        "movupd xmm2, xmm1",
        "psrldq xmm1, 8",
        "paddq xmm1, xmm2",
        "vzeroupper",

        out("xmm0") sum,
        out("xmm1") count,
        inout("xmm2") 1 => _,
        out("ymm3") _,
        out("ymm4") _,
        out("ymm5") _,
        inout("rax") data.len() / 4 => _,
        inout("rsi") data.as_ptr() => _,
        options(readonly, nostack),
    }
    if data.len() % 4 != 0 {
        for i in (data.len() - data.len() % 4)..data.len() {
            let d = data[i];
            if d.is_nan() {
                count += 1;
            } else {
                sum += d;
            }
        }
    }
    count = data.len() as u64 - count;
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

// #[target_feature(enable = "avx")]
#[inline(always)]
pub unsafe fn mean_avx512(data: &[f64]) -> f64 {
    let mut sum: f64;
    let mut non_nan: u64;
    core::arch::asm! {
        // zmm0 = sum
        "vxorpd zmm0, zmm0, zmm0",
        // zmm1 = count
        "vpxorq zmm1, zmm1, zmm1",
        // zmm2 = 1
        "vbroadcastsd zmm2, xmm2",
        // zmm3 = 0
        "vxorpd zmm3, zmm3, zmm3",
        "test rax, rax",
        "jz 3f",
            "2:",
            // load from memory
            "vmovupd zmm4, [rsi]",
            // check for NaNs
            "vcmppd k1, zmm4, zmm4, 0",
            // count non-NaNs
            "vpaddq zmm1{{k1}}, zmm1, zmm2",
            // sum
            "vaddpd zmm0{{k1}}, zmm0, zmm4",

            "add rsi, 64",
            "dec rax",
            "jnz 2b",
        "3:",
        // extract sum
        "vextractf64x4 ymm2, zmm0, 1",
        "vaddpd ymm0, ymm0, ymm2",
        "vextractf64x2 xmm2, ymm0, 1",
        "vaddpd xmm0, xmm0, xmm2",
        "vhaddpd xmm0, xmm0, xmm0",
        // extract count
        "vextracti64x4 ymm2, zmm1, 1",
        "vpaddq ymm1, ymm1, ymm2",
        "vextracti64x2 xmm2, ymm1, 1",
        "vpaddq xmm1, xmm1, xmm2",
        "movupd xmm2, xmm1",
        "psrldq xmm1, 8",
        "paddq xmm1, xmm2",
        "vzeroupper",

        out("xmm0") sum,
        out("xmm1") non_nan,
        inout("xmm2") 1 => _,
        out("zmm3") _,
        out("zmm4") _,
        inout("rax") data.len() / 8 => _,
        inout("rsi") data.as_ptr() => _,
        options(readonly, nostack),
    }
    if data.len() % 8 != 0 {
        for i in (data.len() - data.len() % 8)..data.len() {
            let d = data[i];
            if !d.is_nan() {
                non_nan += 1;
                sum += d;
            }
        }
    }
    if non_nan == 0 {
        0.0
    } else {
        sum / non_nan as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    const MEAN: f64 = (36000.0 - 8.0) / 7999.0;
    const MEAN_NAN: f64 = 21000.0 / 6000.0;

    #[test]
    fn test_mean_naive() {
        assert_eq!(mean_naive(&data()), MEAN);
    }

    #[test]
    fn test_mean_naive_nan() {
        assert_eq!(mean_naive(&data_nan()), MEAN_NAN);
    }

    #[test]
    fn test_mean_sse2() {
        assert_eq!(unsafe { mean_sse(&data()) }, MEAN);
    }

    #[test]
    fn test_mean_sse2_nan() {
        assert_eq!(unsafe { mean_sse(&data_nan()) }, MEAN_NAN);
    }

    #[test]
    fn test_mean_avx2() {
        assert_eq!(unsafe { mean_avx2(&data()) }, MEAN);
    }

    #[test]
    fn test_mean_avx2_nan() {
        assert_eq!(unsafe { mean_avx2(&data_nan()) }, MEAN_NAN);
    }

    #[test]
    fn test_mean_avx512() {
        assert_eq!(unsafe { mean_avx512(&data()) }, MEAN);
    }

    #[test]
    fn test_mean_avx512_nan() {
        assert_eq!(unsafe { mean_avx512(&data_nan()) }, MEAN_NAN);
    }
}

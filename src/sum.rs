#![allow(clippy::needless_range_loop, clippy::missing_safety_doc)]

#[inline(always)]
pub fn sum(data: &[f64]) -> f64 {
    if is_x86_feature_detected!("avx512f") {
        unsafe { sum_avx512(data) }
    } else if is_x86_feature_detected!("avx2") {
        unsafe { sum_avx2(data) }
    } else if is_x86_feature_detected!("sse4.1") {
        unsafe { sum_sse4(data) }
    } else {
        sum_naive(data)
    }
}

#[inline(always)]
pub fn sum_naive(data: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..data.len() {
        let d = data[i];
        if !d.is_nan() {
            sum += d;
        }
    }
    sum
}

#[inline(always)]
pub unsafe fn sum_sse4(data: &[f64]) -> f64 {
    let mut sum: f64;
    core::arch::asm! {
        // xmm1 = sum
        "xorpd xmm1, xmm1",
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
            // sum
            "addpd xmm1, xmm5",

            "add rsi, 16",
            "dec rax",
            "jnz 2b",
        "3:",
        "haddpd xmm1, xmm1",
        "psrldq xmm2, 8",
        "paddq xmm2, xmm3",

        out("xmm0") _,
        out("xmm1") sum,
        inout("xmm3") 1 => _,
        out("xmm4") _,
        out("xmm5") _,
        inout("rsi") data.as_ptr() => _,
        inout("rax") data.len() / 2 => _,
        options(readonly, nostack),
    }
    if data.len() % 2 != 0 {
        let d = data[data.len() - 1];
        if !d.is_nan() {
            sum += d;
        }
    }
    sum
}

#[inline(always)]
pub unsafe fn sum_avx2(data: &[f64]) -> f64 {
    let mut sum: f64;
    core::arch::asm! {
        // ymm0 = sum
        "vxorpd ymm0, ymm0, ymm0",
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

        out("xmm0") sum,
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
            if !d.is_nan() {
                sum += d;
            }
        }
    }
    sum
}

#[inline(always)]
pub unsafe fn sum_avx512(data: &[f64]) -> f64 {
    let mut sum: f64;
    core::arch::asm! {
        // zmm0 = sum
        "vxorpd zmm0, zmm0, zmm0",
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

        out("xmm0") sum,
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
                sum += d;
            }
        }
    }
    sum
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

    const SUM: f64 = 36000.0 - 8.0;
    const SUM_NAN: f64 = 21000.0;

    #[test]
    fn test_sum_naive() {
        assert_eq!(sum_naive(&data()), SUM);
    }

    #[test]
    fn test_sum_naive_nan() {
        assert_eq!(sum_naive(&data_nan()), SUM_NAN);
    }

    #[test]
    #[cfg_attr(not(target_feature = "sse4.1"), ignore)]
    fn test_sum_sse4() {
        assert_eq!(unsafe { sum_sse4(&data()) }, SUM);
    }

    #[test]
    #[cfg_attr(not(target_feature = "sse4.1"), ignore)]
    fn test_sum_sse4_nan() {
        assert_eq!(unsafe { sum_sse4(&data_nan()) }, SUM_NAN);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_sum_avx2() {
        assert_eq!(unsafe { sum_avx2(&data()) }, SUM);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_sum_avx2_nan() {
        assert_eq!(unsafe { sum_avx2(&data_nan()) }, SUM_NAN);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_sum_avx512() {
        assert_eq!(unsafe { sum_avx512(&data()) }, SUM);
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_sum_avx512_nan() {
        assert_eq!(unsafe { sum_avx512(&data_nan()) }, SUM_NAN);
    }
}

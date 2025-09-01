use crate::ld::{LD_BLOCK_SIZE, MASK_LOW_ORDER_U64};
use aligned_vec::AVec;

#[inline(always)]
pub fn encode_and_missing_naive(data: &mut AVec<u8>) -> (AVec<u8>, u64) {
    let mut missing = AVec::<u8>::with_capacity(LD_BLOCK_SIZE, data.len());
    unsafe { missing.set_len(data.len()) };
    let mut count_non_missing = 0u64;
    for i in 0..data.len() {
        let d = data[i];
        let shifted = d >> 1 & 0b01010101;
        let m = ((!d) & 0b01010101) | shifted;
        let m = m + m + m;
        count_non_missing += (m as u64).count_ones() as u64;
        data[i] = d - shifted;
        missing[i] = m;
    }
    (missing, count_non_missing >> 1)
}

#[inline(always)]
pub fn encode_and_missing_avx512(data: &mut AVec<u8>) -> (AVec<u8>, u64) {
    unsafe {
        let mut missing = AVec::<u8>::with_capacity(LD_BLOCK_SIZE, data.len());
        missing.set_len(data.len());
        let bytes = data.len();
        let data_ptr = data.as_mut_ptr() as *mut i64;
        let missing_ptr = missing.as_mut_ptr() as *mut i64;
        let mut count_non_missing: u64;
        std::arch::asm! {
            "vpbroadcastq zmm0, {mask_low_order}",
            "vpxorq zmm1, zmm1, zmm1",
            "test rax, rax",
            "jz 3f",
            "2:",
            "vmovdqa64 zmm2, [{data}]",
            // shifted = data >> 1
            "vpsrlq zmm3, zmm2, 1",
            // shifted = shifted & mask_low_order
            "vpandq zmm3, zmm3, zmm0",
            // missing = ~data & mask_low_order
            "vpandnq zmm4, zmm2, zmm0",
            // missing = missing | shifted
            "vporq zmm4, zmm3, zmm4",
            // missing = missing + missing + missing
            "vpaddq zmm5, zmm4, zmm4",
            "vpaddq zmm5, zmm5, zmm4",
            // count_non_missing += missing.count_ones()
            "vpopcntq zmm6, zmm5",
            "vpaddq zmm1, zmm1, zmm6",
            // data -= shifted
            "vpsubq zmm2, zmm2, zmm3",
            "vmovdqa64 [{missing}], zmm5",
            "vmovdqa64 [{data}], zmm2",

            "add {data}, 64",
            "add {missing}, 64",

            "dec rax",
            "jnz 2b",
            "3:",

            "vextracti64x4 ymm2, zmm1, 1",
            "vpaddq ymm1, ymm1, ymm2",
            "vextracti64x2 xmm2, zmm1, 1",
            "vpaddq xmm1, xmm1, xmm2",
            "movq {count_non_missing}, xmm1",
            "pextrq rax, xmm1, 1",
            "add {count_non_missing}, rax",

            in("rax") bytes / 64,
            out("xmm0") _,
            out("xmm1") _,
            out("xmm2") _,
            out("xmm3") _,
            out("xmm4") _,
            out("xmm5") _,
            out("xmm6") _,
            mask_low_order = inout(reg) MASK_LOW_ORDER_U64 => _,
            data = inout(reg) data_ptr => _,
            missing = inout(reg) missing_ptr => _,
            count_non_missing = out(reg) count_non_missing,
            options(nostack),
        }
        // divide count_non_missing by 2, just don't spend a billion cycles in a div instruction
        (missing, count_non_missing >> 1)
    }
}

use crate::ld::{
    MASK_LOW_EIGHT_BITS_U64, MASK_LOW_FOUR_BITS_U64, MASK_LOW_ORDER_U64, MASK_LOW_TWO_BITS_U64,
};
use aligned_vec::AVec;

#[inline(always)]
pub fn get_maf_naive(data: &AVec<u8>, num_samples: u64, num_non_missing: u64) -> f64 {
    let mut acc: u64 = 0;
    let num_missing = num_samples - num_non_missing;

    for byte in data.iter() {
        let four_bit = (byte & 0b00110011) + ((byte >> 2) & 0b00110011);
        let eight_bit = (four_bit & 0b00001111) + (four_bit >> 4);
        acc += eight_bit as u64;
    }

    acc -= num_missing;
    // Calculate MAF
    let maf = acc as f64 / (num_non_missing * 2) as f64;
    if maf > 0.5 {
        1.0 - maf
    } else {
        maf
    }
}

// TODO: in these, i convert from -1/0/1 to 0/1/2, can't i just use the original data and this
// once i know the number of missing values? is that faster?
// i could actually just have super fast pipelined popcnt functions, then when i do the encode and
// missing step i have a follow up function that can perform the remainder of the computation
// TODO: if i do the above, explore maybe just a pipelined popcnt approach rather than this
// complicated mess of shifting and masking (could be faster then SSE4.1 and AVX2 even, if
// the loads are pipelined at least)
#[inline(always)]
pub fn get_maf_sse4(data: &AVec<u8>, num_samples: u64, num_non_missing: u64) -> f64 {
    unsafe {
        let mut acc: u32;
        let num_missing = num_samples - num_non_missing;
        std::arch::asm! {
            "movq xmm0, {mask_low_order_two_bits}",
            "movq xmm4, {mask_low_order_four_bits}",
            "movq xmm5, {mask_low_order_eight_bits}",
            "pshufd xmm0, xmm0, 0", // broadcast to all lanes
            "pshufd xmm4, xmm4, 0", // broadcast to all lanes
            "pshufd xmm5, xmm5, 0", // broadcast to all lanes

            "pxor xmm1, xmm1", // xmm1 will hold the accumulator
            "test rax, rax",
            "jz 3f",
            "2:",

            "movdqa xmm2, [{data}]",
            // we don't have a vector popcnt, so we need to do this manually
            // xmm2 is already 2-bit sums conveniently, we need to convert them to 32-bit sums
            "movdqa xmm3, xmm2",
            "psrlq xmm3, 2", // shifted = data >> 2
            "pand xmm2, xmm0", // data = data & mask_low_order_two_bits
            "pand xmm3, xmm0", // masked = shifted & mask_low_order_two_bits
            "paddq xmm2, xmm3", // data = data + masked
            // now we have 4-bit sums
            "movdqa xmm3, xmm2",
            "psrlq xmm3, 4", // shifted = data
            // we can do this because the highest our 4-bit sums can be is 4, so the highest this
            // can be is 8 (0b1000)
            "paddq xmm2, xmm3", // data = data + shifted
            "pand xmm2, xmm4", // data = data & mask_low_order_four_bits
            "movdqa xmm3, xmm2",
            "psrlq xmm3, 8", // shifted = data >> 8
            "paddq xmm2, xmm3", // data = data + shifted
            "movdqa xmm3, xmm2",
            "psrlq xmm3, 16", // shifted = data >> 16
            "paddq xmm2, xmm3", // data = data + shifted
            "pand xmm2, xmm5", // data = data & mask_low_order_eight_bits
            // now we have packed 32-bit sums

            "paddd xmm1, xmm2", // acc += count
            "add {data}, 16",
            "dec rax",
            "jnz 2b",
            "3:",
            // horizontal sum of xmm1, which contains 4 32-bit sums
            // we sum two 64-bit halves
            "pshufd xmm2, xmm1, 0x1",
            "paddd xmm1, xmm2",
            // we sum two 32-bit halves
            "movd {acc:e}, xmm1",
            "pextrd eax, xmm1, 1",
            "add {acc:e}, eax",
            mask_low_order_two_bits = inout(reg) MASK_LOW_TWO_BITS_U64 => _,
            mask_low_order_four_bits = inout(reg) MASK_LOW_FOUR_BITS_U64 => _,
            mask_low_order_eight_bits = inout(reg) MASK_LOW_EIGHT_BITS_U64 => _,
            data = inout(reg) data.as_ptr() => _,
            acc = out(reg) acc,
            inout("rax") data.len() / 16 => _,
            out("xmm0") _,
            out("xmm1") _,
            out("xmm2") _,
            out("xmm3") _,
        }
        let mut acc = acc as u64;
        // Handle any remaining bytes that were not processed in the loop
        let remaining_bytes = data.len() % 16;
        let start = data.len() - remaining_bytes;
        for i in 0..remaining_bytes {
            let four_bit = (data[start + i] & 0b00110011) + ((data[start + i] >> 2) & 0b00110011);
            let eight_bit = (four_bit & 0b00001111) + (four_bit >> 4);
            acc += eight_bit as u64;
        }
        acc -= num_missing;
        // Calculate MAF
        let maf = acc as f64 / (num_non_missing * 2) as f64;
        if maf > 0.5 {
            1.0 - maf
        } else {
            maf
        }
    }
}

#[inline(always)]
pub fn get_maf_avx2(data: &AVec<u8>, num_samples: u64, num_non_missing: u64) -> f64 {
    unsafe {
        let mut acc: u32;
        let num_missing = num_samples - num_non_missing;
        std::arch::asm! {
            "vpbroadcastq ymm0, {mask_low_order_two_bits}",
            "vpbroadcastq ymm4, {mask_low_order_four_bits}",
            "vpbroadcastq ymm5, {mask_low_order_eight_bits}",

            "vpxorq ymm1, ymm1, ymm1", // ymm1 will hold the accumulator
            "test rax, rax",
            "jz 3f",
            "2:",

            "vmovdqa ymm2, [{data}]",
            // we don't have a vector popcnt, so we need to do this manually
            // ymm2 is already 2-bit sums conveniently, we need to convert them to 32-bit sums
            "vpsrlq ymm3, ymm2, 2", // shifted = data >> 2
            "vpandq ymm2, ymm2, ymm0", // data = data & mask_low_order_two_bits
            "vpandq ymm3, ymm3, ymm0", // masked = shifted & mask_low_order_two_bits
            "vpaddq ymm2, ymm2, ymm3", // data = data + masked
            // now we have 4-bit sums
            "vpsrlq ymm3, ymm2, 4", // shifted = data
            // we can do this because the highest our 4-bit sums can be is 4, so the highest this
            // can be is 8 (0b1000)
            "vpaddq ymm2, ymm2, ymm3", // data = data + shifted
            "vpandq ymm2, ymm2, ymm4", // data = data & mask_low_order_four_bits
            "vpsrlq ymm3, ymm2, 8", // shifted = data >> 8
            "vpaddq ymm2, ymm2, ymm3", // data = data + shifted
            "vpsrlq ymm3, ymm2, 16", // shifted = data >> 16
            "vpaddq ymm2, ymm2, ymm3", // data = data + shifted
            "vpandd ymm2, ymm2, ymm5", // data = data & mask_low_order_eight_bits
            // now we have packed 32-bit sums

            "vpaddd ymm1, ymm1, ymm2", // acc += count
            "add {data}, 32",
            "dec rax",
            "jnz 2b",
            "3:",
            // horizontal sum of ymm1, which contains 8 32-bit sums
            // we sum of two 128-bit halves
            "vextracti128 xmm2, ymm1, 1",
            "vzeroupper",
            "vpaddd xmm1, xmm1, xmm2",
            // we sum two 64-bit halves
            "vpshufd xmm2, xmm1, 0x1",
            "vpaddd xmm1, xmm1, xmm2",
            // we sum two 32-bit halves
            "movd {acc:e}, xmm1",
            "pextrd eax, xmm1, 1",
            "add {acc:e}, eax",
            mask_low_order_two_bits = inout(reg) MASK_LOW_TWO_BITS_U64 => _,
            mask_low_order_four_bits = inout(reg) MASK_LOW_FOUR_BITS_U64 => _,
            mask_low_order_eight_bits = inout(reg) MASK_LOW_EIGHT_BITS_U64 => _,
            data = inout(reg) data.as_ptr() => _,
            acc = out(reg) acc,
            inout("rax") data.len() / 32 => _,
            out("xmm0") _,
            out("xmm1") _,
            out("xmm2") _,
            out("xmm3") _,
            out("xmm4") _,
            out("xmm5") _,
        }
        let mut acc = acc as u64;
        // Handle any remaining bytes that were not processed in the loop
        let remaining_bytes = data.len() % 32;
        let start = data.len() - remaining_bytes;
        for i in 0..remaining_bytes {
            let four_bit = (data[start + i] & 0b00110011) + ((data[start + i] >> 2) & 0b00110011);
            let eight_bit = (four_bit & 0b00001111) + (four_bit >> 4);
            acc += eight_bit as u64;
        }
        acc -= num_missing;
        // Calculate MAF
        let maf = acc as f64 / (num_non_missing * 2) as f64;
        if maf > 0.5 {
            1.0 - maf
        } else {
            maf
        }
    }
}

pub fn get_maf_avx512(data: &AVec<u8>, num_samples: u64, num_non_missing: u64) -> f64 {
    unsafe {
        let mut acc: u64;
        let num_missing = num_samples - num_non_missing;
        std::arch::asm! {
            "vpbroadcastq zmm0, {mask_low_order}",
            "vpxorq zmm1, zmm1, zmm1", // zmm1 will hold the accumulator
            "test rax, rax",
            "jz 3f",
            "2:",

            "vmovdqa64 zmm2, [{data}]",
            "vpsrlq zmm3, zmm2, 1", // shifted = data >> 1
            "vpandq zmm3, zmm3, zmm0", // masked = shifted & mask_low_order
            "vporq zmm2, zmm2, zmm3", // data = data | masked
            "vpopcntq zmm3, zmm2", // count = popcnt(data)
            "vpaddq zmm1, zmm1, zmm3", // acc += count

            "add {data}, 64",
            "dec rax",
            "jnz 2b",
            "3:",

            "vextracti64x4 ymm2, zmm1, 1",
            "vpaddq ymm1, ymm1, ymm2",
            "vextracti64x2 xmm2, ymm1, 1",
            "vzeroupper",
            "vpaddq xmm1, xmm1, xmm2",
            "movq {acc}, xmm1",
            "pextrq rax, xmm1, 1",
            "add {acc}, rax",
            mask_low_order = inout(reg) MASK_LOW_ORDER_U64 => _,
            data = inout(reg) data.as_ptr() => _,
            acc = out(reg) acc,
            in("rax") data.len() / 64,
            out("xmm0") _,
            out("xmm1") _,
            out("xmm2") _,
            out("xmm3") _,
        }
        // Handle any remaining bytes that were not processed in the loop
        let remaining_bytes = data.len() % 64;
        let start = data.len() - remaining_bytes;
        for i in 0..remaining_bytes {
            let four_bit = (data[start + i] & 0b00110011) + ((data[start + i] >> 2) & 0b00110011);
            let eight_bit = (four_bit & 0b00001111) + (four_bit >> 4);
            acc += eight_bit as u64;
        }

        acc -= num_missing;
        // Calculate MAF
        let maf = acc as f64 / (num_non_missing * 2) as f64;
        if maf > 0.5 {
            1.0 - maf
        } else {
            maf
        }
    }
}

use crate::ld::{Values, LD_BLOCK_SIZE, MASK_LOW_ORDER_U64};
use aligned_vec::AVec;

#[inline(always)]
pub fn values_naive(
    left_data: &AVec<u8>,
    right_data: &AVec<u8>,
    left_missing: &AVec<u8>,
    right_missing: &AVec<u8>,
    num_samples: u64,
    num_left_non_missing: u64,
    num_right_non_missing: u64,
) -> Values {
    let mut left_sum: u32 = 0;
    let mut right_sum: u32 = 0;
    let mut left_squared_sum: u32 = 0;
    let mut right_squared_sum: u32 = 0;
    let mut left_right_sum: u32 = 0;
    let mut non_missing: u32 = 0;
    let low_order_mask = MASK_LOW_ORDER_U64;
    for i in 0..(left_data.len() / 8) {
        let (mut left, mut right, left_miss, right_miss);
        unsafe {
            left = left_data.as_ptr().cast::<u64>().wrapping_add(i).read();
            right = right_data.as_ptr().cast::<u64>().wrapping_add(i).read();
            left_miss = left_missing.as_ptr().cast::<u64>().wrapping_add(i).read();
            right_miss = right_missing.as_ptr().cast::<u64>().wrapping_add(i).read();
        }
        let data_or = left | right;
        let is_either_missing_or_zero = data_or & low_order_mask;

        let left_sum_i = left & right_miss;
        let right_sum_i = right & left_miss;

        let left_squared_sum_i = left_sum_i & low_order_mask;
        let right_squared_sum_i = right_sum_i & low_order_mask;

        let either_add = low_order_mask + is_either_missing_or_zero;
        let data_xor = left ^ right;

        let left_right_sum_i = ((!either_add) & data_xor) | is_either_missing_or_zero;

        left_sum += (((left_sum_i >> 1) & low_order_mask) | left_sum_i).count_ones();
        right_sum += (((right_sum_i >> 1) & low_order_mask) | right_sum_i).count_ones();
        left_squared_sum +=
            (((left_squared_sum_i >> 1) & low_order_mask) | left_squared_sum_i).count_ones();
        right_squared_sum +=
            (((right_squared_sum_i >> 1) & low_order_mask) | right_squared_sum_i).count_ones();
        left_right_sum +=
            (((left_right_sum_i >> 1) & low_order_mask) | left_right_sum_i).count_ones();

        let missing = left_miss & right_miss;
        non_missing += missing.count_ones();
    }
    let left_sum = left_sum as i32 - num_right_non_missing as i32;
    let right_sum = right_sum as i32 - num_left_non_missing as i32;
    let left_squared_sum = num_right_non_missing as i32 - left_squared_sum as i32;
    let right_squared_sum = num_left_non_missing as i32 - right_squared_sum as i32;
    let left_right_sum = num_samples as i32 - left_right_sum as i32;
    non_missing >>= 1;
    Values {
        left_sum,
        right_sum,
        left_squared_sum,
        right_squared_sum,
        left_right_sum,
        non_missing,
    }
}

#[inline(always)]
pub fn values_avx512(
    left_data: &AVec<u8>,
    right_data: &AVec<u8>,
    left_missing: &AVec<u8>,
    right_missing: &AVec<u8>,
    num_samples: u64,
    num_left_non_missing: u64,
    num_right_non_missing: u64,
) -> Values {
    unsafe {
        let left_data_ptr = left_data.as_ptr();
        let right_data_ptr = right_data.as_ptr();
        let left_missing_ptr = left_missing.as_ptr();
        let right_missing_ptr = right_missing.as_ptr();
        let iters = left_data.len() / LD_BLOCK_SIZE;
        let mut left_sum: i64;
        let mut right_sum: i64;
        let mut left_squared_sum: i64;
        let mut right_squared_sum: i64;
        let mut left_right_sum: i64;
        let mut non_missing: u64;
        std::arch::asm! {
            // zmm0 is the mask for low order bits
            "vpbroadcastq zmm0, {mask_low_order}",
            "vpxorq zmm1, zmm1, zmm1", // zmm1 will hold acc_left_sum
            "vpxorq zmm2, zmm2, zmm2", // zmm2 will hold acc_right_sum
            "vpxorq zmm3, zmm3, zmm3", // zmm3 will hold acc_left_squared_sum
            "vpxorq zmm4, zmm4, zmm4", // zmm4 will hold acc_right_squared_sum
            "vpxorq zmm5, zmm5, zmm5", // zmm5 will hold acc_left_right_sum
            "vpxorq zmm6, zmm6, zmm6", // zmm6 will hold acc_non_missing
            // now we want to loop for len / BLOCK_SIZE times
            "test rax, rax",
            "jz 3f",
            "2:",
            // load from memory
            "vmovdqa64 zmm7, [{left_data}]",
            "vmovdqa64 zmm8, [{right_data}]",
            "vmovdqa64 zmm9, [{left_missing}]",
            "vmovdqa64 zmm10, [{right_missing}]",

            // data_or = left_data | right_data
            "vporq zmm11, zmm7, zmm8",
            // is_either_missing_or_zero = data_or & mask_low_order
            "vpandq zmm11, zmm11, zmm0",

            // when summing we don't need to do anything but make sure it's properly masked
            // at the very end we subtract the number of non-missing right values from this value
            // we don't need to mask it with the left missing mask since we know that all places where we
            // would have missing values are 01, so will be cancelled out when we subtract

            // left_sum = left_data & right_missing
            "vpandq zmm12, zmm7, zmm10",
            // right_sum = right_data & left_missing
            "vpandq zmm13, zmm8, zmm9",

            // when finding our squared values we find the number of zeroes (will either be 00 or 01) and
            // in the end we subtract that from the total number of non-missing right values for the same
            // reason as above, ending up with the sum of the squares

            // left_squared_sum = left_sum & mask_low_order
            "vpandq zmm14, zmm12, zmm0",
            // right_squared_sum = right_sum & mask_low_order
            "vpandq zmm15, zmm13, zmm0",

            // either_add = mask_low_order + is_either_missing_or_zero
            "vpaddq zmm16, zmm0, zmm11",
            // data_xor = left_data ^ right_data
            "vpxorq zmm17, zmm7, zmm8",

            // now we solve for our multiplication
            // we end up with 1 - left_right_sum being our expected value
            // so at the end we can do num - sum12 and we have our sum!

            // left_right_sum = ~either_add & data_xor
            "vpandnq zmm18, zmm16, zmm17",
            // left_right_sum |= is_either_missing_or_zero
            "vporq zmm18, zmm18, zmm11",

            // sum left_sum
            "vpsrlq zmm19, zmm12, 1",
            "vpandq zmm19, zmm19, zmm0",
            "vporq zmm19, zmm12, zmm19",
            "vpopcntq zmm19, zmm19",
            "vpaddq zmm1, zmm1, zmm19", // acc_left_sum += left_sum

            // sum right_sum
            "vpsrlq zmm19, zmm13, 1",
            "vpandq zmm19, zmm19, zmm0",
            "vporq zmm19, zmm13, zmm19",
            "vpopcntq zmm19, zmm19",
            "vpaddq zmm2, zmm2, zmm19", // acc_right_sum += right_sum

            // sum left_squared_sum
            "vpsrlq zmm19, zmm14, 1",
            "vpandq zmm19, zmm19, zmm0",
            "vporq zmm19, zmm14, zmm19",
            "vpopcntq zmm19, zmm19",
            "vpaddq zmm3, zmm3, zmm19", // acc_left_squared_sum += left_squared_sum

            // sum right_squared_sum
            "vpsrlq zmm19, zmm15, 1",
            "vpandq zmm19, zmm19, zmm0",
            "vporq zmm19, zmm15, zmm19",
            "vpopcntq zmm19, zmm19",
            "vpaddq zmm4, zmm4, zmm19", // acc_right_squared_sum += right_squared_sum

            // sum left_right_sum
            "vpsrlq zmm19, zmm18, 1",
            "vpandq zmm19, zmm19, zmm0",
            "vporq zmm19, zmm18, zmm19",
            "vpopcntq zmm19, zmm19",
            "vpaddq zmm5, zmm5, zmm19", // acc_left_right_sum += left_right_sum

            "vpandq zmm20, zmm9, zmm10", // missing = left_missing & right_missing
            "vpopcntq zmm20, zmm20", // count non-missing
            "vpaddq zmm6, zmm6, zmm20", // acc_non_missing += count_non_missing

            "add {left_data}, 64",
            "add {right_data}, 64",
            "add {left_missing}, 64",
            "add {right_missing}, 64",

            "dec rax",
            "jnz 2b",
            "3:",

            // now we need to reduce the results
            "vextracti64x4 ymm21, zmm1, 1", // extract high part of acc_left_sum
            "vextracti64x4 ymm22, zmm2, 1", // extract high part of acc_right_sum
            "vextracti64x4 ymm23, zmm3, 1", // extract high part of acc_left_squared_sum
            "vextracti64x4 ymm24, zmm4, 1", // extract high part of acc_right_squared_sum
            "vextracti64x4 ymm25, zmm5, 1", // extract high part of acc_left_right_sum
            "vextracti64x4 ymm26, zmm6, 1", // extract high part of acc_non_missing
            "vpaddq ymm1, ymm1, ymm21", // acc_left_sum += high part
            "vpaddq ymm2, ymm2, ymm22", // acc_right_sum += high part
            "vpaddq ymm3, ymm3, ymm23", // acc_left_squared_sum += high part
            "vpaddq ymm4, ymm4, ymm24", // acc_right_squared_sum += high part
            "vpaddq ymm5, ymm5, ymm25", // acc_left_right_sum += high part
            "vpaddq ymm6, ymm6, ymm26", // acc_non_missing += high part
            // now we have 4 64-bit integers in the low half
            "vextracti64x2 xmm21, ymm1, 1", // extract low part of acc_left_sum
            "vextracti64x2 xmm22, ymm2, 1", // extract low part of acc_right_sum
            "vextracti64x2 xmm23, ymm3, 1", // extract low part of acc_left_squared_sum
            "vextracti64x2 xmm24, ymm4, 1", // extract low part of acc_right_squared_sum
            "vextracti64x2 xmm25, ymm5, 1", // extract low part of acc_left_right_sum
            "vextracti64x2 xmm26, ymm6, 1", // extract low part of acc_non_missing
            "vpaddq xmm1, xmm1, xmm21", // acc_left_sum += low part
            "vpaddq xmm2, xmm2, xmm22", // acc_right_sum += low part
            "vpaddq xmm3, xmm3, xmm23", // acc_left_squared_sum += low part
            "vpaddq xmm4, xmm4, xmm24", // acc_right_squared_sum += low part
            "vpaddq xmm5, xmm5, xmm25", // acc_left_right_sum += low part
            "vpaddq xmm6, xmm6, xmm26", // acc_non_missing += low part
            "vzeroupper",
            // now we have 2 64-bit integers in the low half
            "movq {left_sum}, xmm1", // move acc_left_sum to left_sum
            "pextrq rax, xmm1, 1", // extract high part of acc_left_sum
            "add {left_sum}, rax", // acc_left_sum += high part

            "movq {right_sum}, xmm2", // move acc_right_sum to right_sum
            "pextrq rax, xmm2, 1", // extract high part of acc_right_sum
            "add {right_sum}, rax", // acc_right_sum += high part

            "movq {left_squared_sum}, xmm3", // move acc_left_squared_sum to left_squared_sum
            "pextrq rax, xmm3, 1", // extract high part of acc_left_squared_sum
            "add {left_squared_sum}, rax", // acc_left_squared_sum += high part

            "movq {right_squared_sum}, xmm4", // move acc_right_squared_sum to right_squared_sum
            "pextrq rax, xmm4, 1", // extract high part of acc_right_squared_sum
            "add {right_squared_sum}, rax", // acc_right_squared_sum += high part

            "movq {left_right_sum}, xmm5", // move acc_left_right_sum to left_right_sum
            "pextrq rax, xmm5, 1", // extract high part of acc_left_right_sum
            "add {left_right_sum}, rax", // acc_left_right_sum += high part

            "movq {non_missing}, xmm6", // move acc_non_missing to non_missing
            "pextrq rax, xmm6, 1", // extract high part of acc_non_missing
            "add {non_missing}, rax", // acc_non_missing += high part

            mask_low_order = inout(reg) MASK_LOW_ORDER_U64 as i64 => _,
            inout("rax") iters => _,
            out("xmm0") _,
            out("xmm1") _,
            out("xmm2") _,
            out("xmm3") _,
            out("xmm4") _,
            out("xmm5") _,
            out("xmm6") _,
            out("xmm7") _,
            out("xmm8") _,
            out("xmm9") _,
            out("xmm10") _,
            out("xmm11") _,
            out("xmm12") _,
            out("xmm13") _,
            out("xmm14") _,
            out("xmm15") _,
            out("xmm16") _,
            out("xmm17") _,
            out("xmm18") _,
            out("xmm19") _,
            out("xmm20") _,
            out("xmm21") _,
            out("xmm22") _,
            out("xmm23") _,
            out("xmm24") _,
            out("xmm25") _,
            out("xmm26") _,
            left_data = inout(reg) left_data_ptr => _,
            right_data = inout(reg) right_data_ptr => _,
            left_missing = inout(reg) left_missing_ptr => _,
            right_missing = inout(reg) right_missing_ptr => _,
            left_sum = out(reg) left_sum,
            right_sum = out(reg) right_sum,
            left_squared_sum = out(reg) left_squared_sum,
            right_squared_sum = out(reg) right_squared_sum,
            left_right_sum = out(reg) left_right_sum,
            non_missing = out(reg) non_missing,
            options(readonly, nostack),
        };
        left_sum -= num_right_non_missing as i64;
        right_sum -= num_left_non_missing as i64;
        left_squared_sum = num_right_non_missing as i64 - left_squared_sum;
        right_squared_sum = num_left_non_missing as i64 - right_squared_sum;
        left_right_sum = num_samples as i64 - left_right_sum;
        non_missing >>= 1;
        Values {
            left_sum: left_sum as i32,
            right_sum: right_sum as i32,
            left_squared_sum: left_squared_sum as i32,
            right_squared_sum: right_squared_sum as i32,
            left_right_sum: left_right_sum as i32,
            non_missing: non_missing as u32,
        }
    }
}

#[inline(always)]
pub fn values_avx512_2(
    left_data: &AVec<u8>,
    right_data: &AVec<u8>,
    left_missing: &AVec<u8>,
    right_missing: &AVec<u8>,
    num_samples: u64,
    num_left_non_missing: u64,
    num_right_non_missing: u64,
) -> Values {
    unsafe {
        let left_data_ptr = left_data.as_ptr();
        let right_data_ptr = right_data.as_ptr();
        let left_missing_ptr = left_missing.as_ptr();
        let right_missing_ptr = right_missing.as_ptr();
        let iters = left_data.len() / LD_BLOCK_SIZE;
        let mut left_sum: i64;
        let mut right_sum: i64;
        let mut left_squared_sum: i64;
        let mut right_squared_sum: i64;
        let mut left_right_sum: i64;
        let mut non_missing: u64;
        std::arch::asm! {
            // zmm0 is the mask for low order bits
            "vpbroadcastq zmm0, {mask_low_order}",
            "vpxorq zmm1, zmm1, zmm1", // zmm1 will hold acc_left_sum
            "vpxorq zmm2, zmm2, zmm2", // zmm2 will hold acc_right_sum
            "vpxorq zmm3, zmm3, zmm3", // zmm3 will hold acc_left_squared_sum
            "vpxorq zmm4, zmm4, zmm4", // zmm4 will hold acc_right_squared_sum
            "vpxorq zmm5, zmm5, zmm5", // zmm5 will hold acc_left_right_sum
            "vpxorq zmm6, zmm6, zmm6", // zmm6 will hold acc_non_missing
            // now we want to loop for len / BLOCK_SIZE times
            "test rax, rax",
            "jz 3f",
            "2:",
            // load from memory
            "vmovdqa64 zmm7, [{left_data}]",
            "vmovdqa64 zmm8, [{right_data}]",
            "vmovdqa64 zmm9, [{left_missing}]",
            "vmovdqa64 zmm10, [{right_missing}]",

            // data_or = left_data | right_data
            "vporq zmm11, zmm7, zmm8",
            // is_either_missing_or_zero = data_or & mask_low_order
            "vpandq zmm11, zmm11, zmm0",

            // when summing we don't need to do anything but make sure it's properly masked
            // at the very end we subtract the number of non-missing right values from this value
            // we don't need to mask it with the left missing mask since we know that all places where we
            // would have missing values are 01, so will be cancelled out when we subtract

            // left_sum = left_data & right_missing
            "vpandq zmm12, zmm7, zmm10",
            // right_sum = right_data & left_missing
            "vpandq zmm13, zmm8, zmm9",

            // when finding our squared values we find the number of zeroes (will either be 00 or 01) and
            // in the end we subtract that from the total number of non-missing right values for the same
            // reason as above, ending up with the sum of the squares

            // left_squared_sum = left_sum & mask_low_order
            "vpandq zmm14, zmm12, zmm0",
            // right_squared_sum = right_sum & mask_low_order
            "vpandq zmm15, zmm13, zmm0",

            // either_add = mask_low_order + is_either_missing_or_zero
            "vpaddq zmm16, zmm0, zmm11",
            // data_xor = left_data ^ right_data
            "vpxorq zmm17, zmm7, zmm8",

            // now we solve for our multiplication
            // we end up with 1 - left_right_sum being our expected value
            // so at the end we can do num - sum12 and we have our sum!

            // left_right_sum = ~either_add & data_xor
            "vpandnq zmm18, zmm16, zmm17",
            // left_right_sum |= is_either_missing_or_zero
            "vporq zmm18, zmm18, zmm11",

            // sums
            "vpsrlq zmm19, zmm12, 1",
            "vpsrlq zmm20, zmm13, 1",
            "vpsrlq zmm21, zmm14, 1",
            "vpsrlq zmm22, zmm15, 1",
            "vpsrlq zmm23, zmm18, 1",

            "vpandq zmm19, zmm19, zmm0",
            "vpandq zmm20, zmm20, zmm0",
            "vpandq zmm21, zmm21, zmm0",
            "vpandq zmm22, zmm22, zmm0",
            "vpandq zmm23, zmm23, zmm0",
            "vpandq zmm24, zmm9, zmm10", // missing = left_missing & right_missing

            "vporq zmm19, zmm12, zmm19",
            "vporq zmm20, zmm13, zmm20",
            "vporq zmm21, zmm14, zmm21",
            "vporq zmm22, zmm15, zmm22",
            "vporq zmm23, zmm18, zmm23",

            "vpopcntq zmm19, zmm19",
            "vpopcntq zmm20, zmm20",
            "vpopcntq zmm21, zmm21",
            "vpopcntq zmm22, zmm22",
            "vpopcntq zmm23, zmm23",
            "vpopcntq zmm24, zmm24", // count non-missing

            "vpaddq zmm1, zmm1, zmm19", // acc_left_sum += left_sum
            "vpaddq zmm2, zmm2, zmm20", // acc_right_sum += right_sum
            "vpaddq zmm3, zmm3, zmm21", // acc_left_squared_sum += left_squared_sum
            "vpaddq zmm4, zmm4, zmm22", // acc_right_squared_sum += right_squared_sum
            "vpaddq zmm5, zmm5, zmm23", // acc_left_right_sum += left_right_sum
            "vpaddq zmm6, zmm6, zmm24", // acc_non_missing += count_non_missing

            "add {left_data}, 64",
            "add {right_data}, 64",
            "add {left_missing}, 64",
            "add {right_missing}, 64",

            "dec rax",
            "jnz 2b",
            "3:",

            // now we need to reduce the results
            "vextracti64x4 ymm21, zmm1, 1", // extract high part of acc_left_sum
            "vextracti64x4 ymm22, zmm2, 1", // extract high part of acc_right_sum
            "vextracti64x4 ymm23, zmm3, 1", // extract high part of acc_left_squared_sum
            "vextracti64x4 ymm24, zmm4, 1", // extract high part of acc_right_squared_sum
            "vextracti64x4 ymm25, zmm5, 1", // extract high part of acc_left_right_sum
            "vextracti64x4 ymm26, zmm6, 1", // extract high part of acc_non_missing
            "vpaddq ymm1, ymm1, ymm21", // acc_left_sum += high part
            "vpaddq ymm2, ymm2, ymm22", // acc_right_sum += high part
            "vpaddq ymm3, ymm3, ymm23", // acc_left_squared_sum += high part
            "vpaddq ymm4, ymm4, ymm24", // acc_right_squared_sum += high part
            "vpaddq ymm5, ymm5, ymm25", // acc_left_right_sum += high part
            "vpaddq ymm6, ymm6, ymm26", // acc_non_missing += high part
            // now we have 4 64-bit integers in the low half
            "vextracti64x2 xmm21, ymm1, 1", // extract low part of acc_left_sum
            "vextracti64x2 xmm22, ymm2, 1", // extract low part of acc_right_sum
            "vextracti64x2 xmm23, ymm3, 1", // extract low part of acc_left_squared_sum
            "vextracti64x2 xmm24, ymm4, 1", // extract low part of acc_right_squared_sum
            "vextracti64x2 xmm25, ymm5, 1", // extract low part of acc_left_right_sum
            "vextracti64x2 xmm26, ymm6, 1", // extract low part of acc_non_missing
            "vpaddq xmm1, xmm1, xmm21", // acc_left_sum += low part
            "vpaddq xmm2, xmm2, xmm22", // acc_right_sum += low part
            "vpaddq xmm3, xmm3, xmm23", // acc_left_squared_sum += low part
            "vpaddq xmm4, xmm4, xmm24", // acc_right_squared_sum += low part
            "vpaddq xmm5, xmm5, xmm25", // acc_left_right_sum += low part
            "vpaddq xmm6, xmm6, xmm26", // acc_non_missing += low part
            "vzeroupper",
            // now we have 2 64-bit integers in the low half
            "movq {left_sum}, xmm1", // move acc_left_sum to left_sum
            "pextrq rax, xmm1, 1", // extract high part of acc_left_sum
            "add {left_sum}, rax", // acc_left_sum += high part

            "movq {right_sum}, xmm2", // move acc_right_sum to right_sum
            "pextrq rax, xmm2, 1", // extract high part of acc_right_sum
            "add {right_sum}, rax", // acc_right_sum += high part

            "movq {left_squared_sum}, xmm3", // move acc_left_squared_sum to left_squared_sum
            "pextrq rax, xmm3, 1", // extract high part of acc_left_squared_sum
            "add {left_squared_sum}, rax", // acc_left_squared_sum += high part

            "movq {right_squared_sum}, xmm4", // move acc_right_squared_sum to right_squared_sum
            "pextrq rax, xmm4, 1", // extract high part of acc_right_squared_sum
            "add {right_squared_sum}, rax", // acc_right_squared_sum += high part

            "movq {left_right_sum}, xmm5", // move acc_left_right_sum to left_right_sum
            "pextrq rax, xmm5, 1", // extract high part of acc_left_right_sum
            "add {left_right_sum}, rax", // acc_left_right_sum += high part

            "movq {non_missing}, xmm6", // move acc_non_missing to non_missing
            "pextrq rax, xmm6, 1", // extract high part of acc_non_missing
            "add {non_missing}, rax", // acc_non_missing += high part

            mask_low_order = inout(reg) MASK_LOW_ORDER_U64 as i64 => _,
            inout("rax") iters => _,
            out("xmm0") _,
            out("xmm1") _,
            out("xmm2") _,
            out("xmm3") _,
            out("xmm4") _,
            out("xmm5") _,
            out("xmm6") _,
            out("xmm7") _,
            out("xmm8") _,
            out("xmm9") _,
            out("xmm10") _,
            out("xmm11") _,
            out("xmm12") _,
            out("xmm13") _,
            out("xmm14") _,
            out("xmm15") _,
            out("xmm16") _,
            out("xmm17") _,
            out("xmm18") _,
            out("xmm19") _,
            out("xmm20") _,
            out("xmm21") _,
            out("xmm22") _,
            out("xmm23") _,
            out("xmm24") _,
            out("xmm25") _,
            out("xmm26") _,
            left_data = inout(reg) left_data_ptr => _,
            right_data = inout(reg) right_data_ptr => _,
            left_missing = inout(reg) left_missing_ptr => _,
            right_missing = inout(reg) right_missing_ptr => _,
            left_sum = out(reg) left_sum,
            right_sum = out(reg) right_sum,
            left_squared_sum = out(reg) left_squared_sum,
            right_squared_sum = out(reg) right_squared_sum,
            left_right_sum = out(reg) left_right_sum,
            non_missing = out(reg) non_missing,
            options(readonly, nostack),
        };
        left_sum -= num_right_non_missing as i64;
        right_sum -= num_left_non_missing as i64;
        left_squared_sum = num_right_non_missing as i64 - left_squared_sum;
        right_squared_sum = num_left_non_missing as i64 - right_squared_sum;
        left_right_sum = num_samples as i64 - left_right_sum;
        non_missing >>= 1;
        Values {
            left_sum: left_sum as i32,
            right_sum: right_sum as i32,
            left_squared_sum: left_squared_sum as i32,
            right_squared_sum: right_squared_sum as i32,
            left_right_sum: left_right_sum as i32,
            non_missing: non_missing as u32,
        }
    }
}

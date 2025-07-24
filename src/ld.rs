#![allow(
    clippy::uninit_vec,
    clippy::needless_range_loop,
    clippy::uninlined_format_args
)]
use crate::Error;
use std::{
    collections::HashSet,
    io::{Read, Write as WriteIo},
};

use aligned_vec::AVec;

pub struct Variant {
    chromosome: String,
    identifier: String,
    coordinate: usize,
}

pub struct LdPruneResult {
    pub pruned: usize,
    pub prune_in: Vec<String>,
    pub prune_out: Vec<String>,
}

pub struct PlinkDataset {
    pub bed_path: std::path::PathBuf,
    pub real_num_samples: usize,
    pub num_samples: usize,
    pub variants: Vec<(usize, Variant)>,
    // aligned to a multiple of 64-bits padded with missing values (01)
    pub data: Vec<AVec<u8>>,
}

impl PlinkDataset {
    pub fn read(bed_path: std::path::PathBuf) -> Result<Self, crate::Error> {
        let fam_file = bed_path.with_extension("fam");
        let bim_file = bed_path.with_extension("bim");
        let fam_content = std::fs::read_to_string(fam_file)?;
        let bim_content = std::fs::read_to_string(bim_file)?;

        let fam_lines = fam_content.lines().collect::<Vec<_>>();
        let bim_lines = bim_content.lines().collect::<Vec<_>>();

        let mut real_num_samples = fam_lines.len();
        let mut variants = Vec::with_capacity(bim_lines.len());

        for (i, line) in bim_lines.iter().enumerate() {
            let mut split = line.split_whitespace();
            let chromosome = split
                .next()
                .ok_or(Error::PlinkMissingChromosome(i + 1))?
                .to_string();
            let identifier = split
                .next()
                .ok_or(Error::PlinkMissingIdentifier(i + 1))?
                .to_string();
            let _position = split.next().ok_or(Error::PlinkMissingPosition(i + 1))?;
            let coordinate = split
                .next()
                .ok_or(Error::PlinkMissingCoordinate(i + 1))?
                .parse::<usize>()
                .map_err(|_| Error::PlinkInvalidCoordinate(i + 1))?;
            variants.push((
                i,
                Variant {
                    chromosome,
                    identifier,
                    coordinate,
                },
            ));
        }

        let mut bed_file = std::fs::File::open(&bed_path)?;
        let magic_number = [0x6c, 0x1b, 0x01];
        let mut buf = [0; 3];
        bed_file.read_exact(&mut buf)?;
        if buf != magic_number {
            panic!("invalid .bed file format: magic number does not match");
        }

        let num_blocks = variants.len();
        let bytes_per_block = real_num_samples.div_ceil(4);
        let data = read(bed_file, num_blocks, bytes_per_block, real_num_samples)?;
        let num_samples = (bytes_per_block.next_multiple_of(BLOCK_SIZE) * 4);
        Ok(PlinkDataset {
            bed_path,
            real_num_samples,
            num_samples,
            variants,
            data,
        })
    }

    pub fn ld_prune(
        mut self,
        window_size: usize,
        threshold: f64,
        step_size: usize,
    ) -> LdPruneResult {
        // since we process in blocks of 64 bytes, we need to update our data to be in blocks of
        // 64 bytes, we don't want to do this earlier because not all functions need this
        let len = self.num_samples.div_ceil(4);
        let bytes_per_block = self.real_num_samples.div_ceil(4);
        for block in &mut self.data {
            if !self.real_num_samples.is_multiple_of(4) {
                // If the number of samples is not a multiple of 4, we need to pad our final byte with
                // missing values
                let mut last_byte = block[bytes_per_block - 1];
                for i in 0..(4 - (self.real_num_samples % 4)) {
                    last_byte |= 0b01 << (2 * (3 - i));
                }
                block[bytes_per_block - 1] = last_byte;
            }
            for _ in block.len()..len {
                block.push(0b01010101);
            }
        }

        let num_blocks = self.variants.len();
        let mut missing = Vec::with_capacity(num_blocks);
        let mut non_missing = Vec::with_capacity(num_blocks);
        let mut mafs: Vec<f64> = Vec::with_capacity(num_blocks);
        // since we pretend that the 00 data at the end of the last byte is missing, we need those
        // to be included
        for block in &mut self.data {
            // we want the REAL number samples here though since we need to set those bits to be
            // missing
            let (block_missing, num_non_missing) = encode_and_missing(block);
            mafs.push(get_maf(block, self.num_samples as u64, num_non_missing));
            missing.push(block_missing);
            non_missing.push(num_non_missing);
        }
        let mut pruning = vec![false; self.variants.len()];
        // we want to enumerate so that we can restore the order at the end
        // now we want to sort by coordinate, then we don't need to iterate through the ENTIRE
        // variants vector and find the ones in range every time, we can just find the start and
        // end iterate through
        self.variants.sort_by_key(|v| v.1.coordinate);
        let mut start_variant = 0;
        let mut starts = (1..=self.variants.len()).collect::<Vec<_>>();
        while start_variant < self.variants.len() {
            let end_coordinate = self.variants[start_variant].1.coordinate + window_size;
            let mut end_variant = start_variant + 1;
            while end_variant < self.variants.len()
                && self.variants[end_variant].1.coordinate <= end_coordinate
            {
                end_variant += 1;
            }
            loop {
                let mut any_pruned = false;
                'next_i: for var_i in start_variant..end_variant {
                    let range = starts[var_i]..end_variant;
                    let i = self.variants[var_i].0;
                    if pruning[i] {
                        continue;
                    }
                    for var_j in range {
                        let j = self.variants[var_j].0;
                        if pruning[j] {
                            continue;
                        }
                        let r2_value = bit_r2(values(
                            &self.data[i],
                            &self.data[j],
                            &missing[i],
                            &missing[j],
                            self.num_samples as u64,
                            non_missing[i],
                            non_missing[j],
                        ));
                        if !r2_value.is_nan() && r2_value > threshold {
                            if mafs[i] < (1.0 - SMALL_EPSILON) * mafs[j] {
                                pruning[i] = true;
                            } else {
                                pruning[j] = true;
                            }
                            starts[var_i] = var_j + 1;
                            any_pruned = true;
                            // for whatever reason plink only prunes one pair per i before looping
                            // back to the start again
                            continue 'next_i;
                        }
                    }
                    starts[var_i] = end_variant;
                }
                if !any_pruned {
                    break;
                }
            }
            start_variant += step_size;
        }
        self.variants.sort_by_key(|v| v.0);
        let start = std::time::Instant::now();
        let mut pruned = 0;
        let mut prune_out = Vec::with_capacity(self.variants.len());
        let mut prune_in = Vec::with_capacity(self.variants.len());
        for (i, variant) in self.variants.iter() {
            if pruning[*i] {
                pruned += 1;
                prune_out.push(variant.identifier.clone());
            } else {
                prune_in.push(variant.identifier.clone());
            }
        }
        LdPruneResult {
            pruned,
            prune_in,
            prune_out,
        }
    }
}

fn read(
    mut reader: impl Read,
    num_blocks: usize,
    bytes_per_block: usize,
    num_samples: usize,
) -> Result<Vec<AVec<u8>>, std::io::Error> {
    let mut data = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        let len = bytes_per_block.next_multiple_of(BLOCK_SIZE);
        let mut block = AVec::<u8>::with_capacity(BLOCK_SIZE, len);
        unsafe {
            block.set_len(bytes_per_block);
        }
        reader.read_exact(&mut block)?;
        data.push(block);
    }
    Ok(data)
}

#[inline(always)]
fn encode_and_missing(data: &mut AVec<u8>) -> (AVec<u8>, u64) {
    unsafe {
        let mut missing = AVec::<u8>::with_capacity(BLOCK_SIZE, data.len());
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

#[inline(always)]
fn values(
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
        let iters = left_data.len() / BLOCK_SIZE;
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

const BLOCK_SIZE: usize = 64;
const MASK_LOW_ORDER: u8 = 0b01010101;
const MASK_LOW_ORDER_U64: u64 = 0x5555555555555555;
// 2^-44
#[allow(clippy::excessive_precision)]
const SMALL_EPSILON: f64 = 0.00000000000005684341886080801486968994140625;

fn preprocess(data: u8) -> (u8, u8) {
    // the mask should be 11 when the data is 00, 10, or 11 but 00 when 01
    // to do so, we assemble this state in the low order bit (flag bit) then multiply by 3
    // if the high order bit is set, then we obvisously set the flag bit to 1
    // if the low order bit is 1, then we set the flag bit to 1
    // multiplication by three takes 01 and makes it 11
    // the ANDNOT can be encoded in a single instruction (ANDN / PANDN / VPANDN)
    let shifted = (data >> 1) & MASK_LOW_ORDER;
    let missing = (shifted | (!data & MASK_LOW_ORDER)) * 3;
    // now we want to keep 00 as 00, 01 as 01, but make 10 into 01 (1 in binary) and turn 11 into
    // 10 (2 in binary)
    let data = data - shifted;
    (data, missing)
}

#[derive(Debug, Clone, Copy)]
struct Values {
    left_sum: i32,
    left_squared_sum: i32,
    right_sum: i32,
    right_squared_sum: i32,
    left_right_sum: i32,
    non_missing: u32,
}

impl std::ops::Add for Values {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Values {
            left_sum: self.left_sum + other.left_sum,
            left_squared_sum: self.left_squared_sum + other.left_squared_sum,
            right_sum: self.right_sum + other.right_sum,
            right_squared_sum: self.right_squared_sum + other.right_squared_sum,
            left_right_sum: self.left_right_sum + other.left_right_sum,
            non_missing: self.non_missing + other.non_missing,
        }
    }
}

fn bit_covariance(
    Values {
        left_sum,
        right_sum,
        left_right_sum,
        non_missing,
        ..
    }: Values,
) -> f64 {
    let left_mean = left_sum as f64 / non_missing as f64;
    let right_mean = right_sum as f64 / non_missing as f64;
    (left_right_sum as f64 - left_mean * right_sum as f64 - right_mean * left_sum as f64
        + non_missing as f64 * left_mean * right_mean)
        / (non_missing - 1) as f64
}

fn bit_sd(
    Values {
        left_sum,
        left_squared_sum,
        right_sum,
        right_squared_sum,
        non_missing,
        ..
    }: Values,
) -> (f64, f64) {
    let n = non_missing as f64;
    let mean_x = left_sum as f64 / n;
    let mean_y = right_sum as f64 / n;

    let var_x = (left_squared_sum as f64 - 2.0 * mean_x * left_sum as f64 + n * mean_x * mean_x)
        / (n - 1.0);
    let var_y = (right_squared_sum as f64 - 2.0 * mean_y * right_sum as f64 + n * mean_y * mean_y)
        / (n - 1.0);

    (var_x.sqrt(), var_y.sqrt())
}

fn bit_correlation(values: Values) -> f64 {
    let (sd_x, sd_y) = bit_sd(values);
    if sd_x == 0.0 || sd_y == 0.0 {
        return f64::NAN; // Avoid division by zero
    }
    bit_covariance(values) / (sd_x * sd_y)
}

fn bit_r2(values: Values) -> f64 {
    let r = bit_correlation(values);
    r * r
}

fn get_maf(data: &AVec<u8>, num_samples: u64, num_non_missing: u64) -> f64 {
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

    const VALUES: Values = Values {
        left_sum: 0,
        left_squared_sum: 2,
        right_sum: -1,
        right_squared_sum: 1,
        left_right_sum: 1,
        non_missing: 3,
    };
    const VALUES_960: Values = Values {
        left_sum: 0,
        left_squared_sum: 960 * 2,
        right_sum: -960,
        right_squared_sum: 960,
        left_right_sum: 960,
        non_missing: 960 * 3,
    };
    const LEFT_PLINK: u8 = 0b00011011; // 0, NA, 1, 2
    const LEFT_DATA: u8 = 0b00010110; // -1, 0, 0, 1
    const LEFT_MISSING: u8 = 0b11001111;
    // const RIGHT_PLINK: u8 = 0b00111010; // 0, 2, 1, 1
    const RIGHT_DATA: u8 = 0b00100101; // -1, 1, 0, 0
    const RIGHT_MISSING: u8 = 0b11111111;
    // const NUM_SAMPLES: u64 = 4;
    // const NUM_LEFT_NON_MISSING: u64 = 3;
    // const NUM_RIGHT_NON_MISSING: u64 = 4;
    const LEFT_VALUES: [f64; 3] = [-1.0, 0.0, 1.0];
    const RIGHT_VALUES: [f64; 3] = [-1.0, 0.0, 0.0];

    fn avec_repeat<T: Default + Clone>(value: T, count: usize) -> AVec<T> {
        let mut vec = AVec::with_capacity(BLOCK_SIZE, count);
        vec.resize(count, value);
        vec
    }

    #[test]
    fn test_preprocess() {
        let data = 0b00011011;
        let (processed_data, missing) = super::preprocess(data);
        assert_eq!(processed_data, 0b00010110);
        assert_eq!(missing, 0b11001111);
    }

    fn covariance(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() {
            panic!("Covariance requires vectors of the same length");
        }
        let n = x.len();
        let mean_x = x.iter().sum::<f64>() / n as f64;
        let mean_y = y.iter().sum::<f64>() / n as f64;
        x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / (n - 1) as f64
    }

    #[test]
    fn test_bit_covariance() {
        let result = bit_covariance(VALUES);
        let normal = covariance(&LEFT_VALUES, &RIGHT_VALUES);
        println!("Bit covariance result: {}", result);
        println!("Normal covariance result: {}", normal);
        assert!(
            (result - normal).abs() < 1e-10,
            "Expected bit covariance to match normal covariance"
        );
    }

    fn sd(x: &[f64]) -> f64 {
        let n = x.len();
        let mean = x.iter().sum::<f64>() / n as f64;
        (x.iter().map(|xi| (xi - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
    }

    #[test]
    fn test_bit_sd() {
        let (sd_x, sd_y) = bit_sd(VALUES);
        let normal_x = sd(&LEFT_VALUES);
        let normal_y = sd(&RIGHT_VALUES);
        println!("Bit SD result: ({}, {})", sd_x, sd_y);
        println!("Normal SD result: ({}, {})", normal_x, normal_y);
        assert!(
            (sd_x - normal_x).abs() < 1e-10,
            "Expected bit SD to match normal SD"
        );
        assert!(
            (sd_y - normal_y).abs() < 1e-10,
            "Expected bit SD to match normal SD"
        );
    }

    fn correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() {
            panic!("Correlation requires vectors of the same length");
        }
        covariance(x, y) / (sd(x) * sd(y))
    }

    #[test]
    fn test_bit_correlation() {
        let result = bit_correlation(VALUES);
        let normal = correlation(&LEFT_VALUES, &RIGHT_VALUES);
        println!("Bit correlation result: {}", result);
        println!("Normal correlation result: {}", normal);
        assert!(
            (result - normal).abs() < 1e-10,
            "Expected bit correlation to match normal correlation"
        );
    }

    fn r2(x: &[f64], y: &[f64]) -> f64 {
        let r = correlation(x, y);
        r * r
    }

    #[test]
    fn test_bit_r2() {
        let result = bit_r2(VALUES);
        let normal = r2(&LEFT_VALUES, &RIGHT_VALUES);
        println!("Bit R2 result: {}", result);
        println!("Normal R2 result: {}", normal);
        assert!(
            (result - normal).abs() < 1e-10,
            "Expected bit R2 to match normal R2"
        );
    }

    #[test]
    fn test_encode_and_missing() {
        let mut data = avec_repeat(LEFT_PLINK, 960);
        let (missing, count_non_missing) = encode_and_missing(&mut data);
        for d in data.iter() {
            assert_eq!(*d, LEFT_DATA);
        }
        for m in missing.iter() {
            assert_eq!(*m, LEFT_MISSING);
        }
        assert_eq!(missing.len(), 960);
        assert_eq!(count_non_missing, 2880);
    }

    #[test]
    fn test_values() {
        let left_data = avec_repeat(LEFT_DATA, 960);
        let right_data = avec_repeat(RIGHT_DATA, 960);
        let left_missing = avec_repeat(LEFT_MISSING, 960);
        let right_missing = avec_repeat(RIGHT_MISSING, 960);
        let num_samples = 960 * 4;
        let num_left_non_missing = 960 * 3;
        let num_right_non_missing = 960 * 4;

        let values = values(
            &left_data,
            &right_data,
            &left_missing,
            &right_missing,
            num_samples,
            num_left_non_missing,
            num_right_non_missing,
        );

        assert_eq!(values.left_sum, VALUES_960.left_sum);
        assert_eq!(values.left_squared_sum, VALUES_960.left_squared_sum);
        assert_eq!(values.right_sum, VALUES_960.right_sum);
        assert_eq!(values.right_squared_sum, VALUES_960.right_squared_sum);
        assert_eq!(values.left_right_sum, VALUES_960.left_right_sum);
        assert_eq!(values.non_missing, VALUES_960.non_missing);
    }

    #[test]
    fn test_get_maf() {
        let data = avec_repeat(LEFT_DATA, 1000);
        let num_samples = 4000;
        let num_non_missing = 3000;

        let maf = get_maf(&data, num_samples, num_non_missing);
        assert!((maf - 0.5).abs() < 1e-10, "Expected MAF to be close to 0.5");
    }

    fn get_test_dataset() -> PlinkDataset {
        PlinkDataset::read("tests/test.bed".into()).unwrap()
    }

    fn get_small_test_dataset() -> PlinkDataset {
        PlinkDataset::read("tests/small_test.bed".into()).unwrap()
    }

    #[test]
    fn test_read_plink_dataset() {
        let dataset = get_small_test_dataset();

        assert_eq!(dataset.real_num_samples, 6);
        assert_eq!(dataset.num_samples, 256);
        assert_eq!(dataset.variants[0].0, 0);
        assert_eq!(dataset.variants[1].0, 1);
        assert_eq!(dataset.variants[2].0, 2);
        assert_eq!(dataset.variants[0].1.chromosome, "1");
        assert_eq!(dataset.variants[1].1.chromosome, "1");
        assert_eq!(dataset.variants[2].1.chromosome, "1");
        assert_eq!(dataset.variants[0].1.identifier, "snp1");
        assert_eq!(dataset.variants[1].1.identifier, "snp2");
        assert_eq!(dataset.variants[2].1.identifier, "snp3");
        assert_eq!(dataset.variants[0].1.coordinate, 1);
        assert_eq!(dataset.variants[1].1.coordinate, 2);
        assert_eq!(dataset.variants[2].1.coordinate, 3);

        let snp1 = &dataset.data[0];
        assert_eq!(snp1.len(), 2);
        assert_eq!(snp1[0], 0b11011100);
        assert_eq!(snp1[1], 0b00001111);
        let snp2 = &dataset.data[1];
        assert_eq!(snp2.len(), 2);
        assert_eq!(snp2[0], 0b11100111);
        assert_eq!(snp2[1], 0b00001111);
        let snp3 = &dataset.data[2];
        assert_eq!(snp3.len(), 2);
        assert_eq!(snp3[0], 0b01101011);
        assert_eq!(snp3[1], 0b00000001);
    }

    #[test]
    fn test_ld_prune() {
        let dataset = get_test_dataset();
        let result = dataset.ld_prune(2000, 0.001, 1);
        let prune_out = std::fs::read_to_string("tests/test.prune.out").unwrap();
        let prune_in = std::fs::read_to_string("tests/test.prune.in").unwrap();
        let prune_out = prune_out.lines().collect::<Vec<_>>();
        let prune_in = prune_in.lines().collect::<Vec<_>>();
        assert_eq!(result.pruned, prune_out.len());
        assert_eq!(result.prune_out, prune_out);
        assert_eq!(result.prune_in, prune_in);
    }
}

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

pub mod encode_and_missing;
pub mod maf;
pub mod values;

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
        let num_samples = (bytes_per_block.next_multiple_of(LD_BLOCK_SIZE) * 4);
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
        step_size: usize,
        threshold: f64,
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
        if is_x86_feature_detected!("avx512f") {
            for block in &mut self.data {
                // we want the REAL number samples here though since we need to set those bits to be
                // missing
                let (block_missing, num_non_missing) =
                    encode_and_missing::encode_and_missing_avx512(block);
                mafs.push(maf::get_maf_avx512(
                    block,
                    self.num_samples as u64,
                    num_non_missing,
                ));
                missing.push(block_missing);
                non_missing.push(num_non_missing);
            }
        } else if is_x86_feature_detected!("avx2") {
            for block in &mut self.data {
                // we want the REAL number samples here though since we need to set those bits to be
                // missing
                let (block_missing, num_non_missing) =
                    encode_and_missing::encode_and_missing_naive(block);
                mafs.push(maf::get_maf_avx2(
                    block,
                    self.num_samples as u64,
                    num_non_missing,
                ));
                missing.push(block_missing);
                non_missing.push(num_non_missing);
            }
        } else if is_x86_feature_detected!("sse4.1") {
            for block in &mut self.data {
                // we want the REAL number samples here though since we need to set those bits to be
                // missing
                let (block_missing, num_non_missing) =
                    encode_and_missing::encode_and_missing_naive(block);
                mafs.push(maf::get_maf_sse4(
                    block,
                    self.num_samples as u64,
                    num_non_missing,
                ));
                missing.push(block_missing);
                non_missing.push(num_non_missing);
            }
        } else {
            for block in &mut self.data {
                // we want the REAL number samples here though since we need to set those bits to be
                // missing
                let (block_missing, num_non_missing) =
                    encode_and_missing::encode_and_missing_naive(block);
                mafs.push(maf::get_maf_naive(
                    block,
                    self.num_samples as u64,
                    num_non_missing,
                ));
                missing.push(block_missing);
                non_missing.push(num_non_missing);
            }
        }
        let mut pruning = vec![false; self.variants.len()];
        // we want to enumerate so that we can restore the order at the end
        // now we want to sort by coordinate, then we don't need to iterate through the ENTIRE
        // variants vector and find the ones in range every time, we can just find the start and
        // end iterate through
        self.variants.sort_by_key(|v| v.1.coordinate);
        let mut start_variant = 0;
        let mut starts = (1..=self.variants.len()).collect::<Vec<_>>();
        // TODO: looks like there's a bottleneck in here somewhere, but not in the values call, 30%
        // or so of the time is taken NOT in the actual values call
        if is_x86_feature_detected!("avx512f") {
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
                            let r2_value = bit_r2(values::values_avx512(
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
        } else {
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
                            let r2_value = bit_r2(values::values_naive(
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
        let len = bytes_per_block.next_multiple_of(LD_BLOCK_SIZE);
        let mut block = AVec::<u8>::with_capacity(LD_BLOCK_SIZE, len);
        unsafe {
            block.set_len(bytes_per_block);
        }
        reader.read_exact(&mut block)?;
        data.push(block);
    }
    Ok(data)
}

pub const LD_BLOCK_SIZE: usize = 64;
const MASK_LOW_ORDER: u8 = 0b01010101;
const MASK_LOW_ORDER_U64: u64 = 0x5555555555555555;
const MASK_LOW_TWO_BITS_U64: u64 = 0x3333333333333333;
const MASK_LOW_FOUR_BITS_U64: u64 = 0x0f0f0f0f0f0f0f0f;
const MASK_LOW_EIGHT_BITS_U64: u64 = 0x000000ff000000ff;
// 2^-44
#[allow(clippy::excessive_precision)]
const SMALL_EPSILON: f64 = 0.00000000000005684341886080801486968994140625;

#[derive(Debug, Clone, Copy)]
pub struct Values {
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

fn bit_r2(values: Values) -> f64 {
    let left_right_sum = values.left_right_sum as f64;
    let left_sum = values.left_sum as f64;
    let right_sum = values.right_sum as f64;
    let left_squared_sum = -values.left_squared_sum as f64;
    let right_squared_sum = -values.right_squared_sum as f64;
    let non_missing = values.non_missing as f64;
    let dxx = left_sum;
    let dyy = right_sum;
    let cov12 = left_right_sum * non_missing - dxx * dyy;
    let dxx = 1.0
        / ((left_squared_sum * non_missing + dxx * dxx)
            * (right_squared_sum * non_missing + dyy * dyy));
    cov12 * cov12 * dxx
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
        let mut vec = AVec::with_capacity(LD_BLOCK_SIZE, count);
        vec.resize(count, value);
        vec
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

    fn sd(x: &[f64]) -> f64 {
        let n = x.len();
        let mean = x.iter().sum::<f64>() / n as f64;
        (x.iter().map(|xi| (xi - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
    }

    fn correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() {
            panic!("Correlation requires vectors of the same length");
        }
        covariance(x, y) / (sd(x) * sd(y))
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
    fn test_encode_and_missing_naive() {
        let mut data = avec_repeat(LEFT_PLINK, 960);
        let (missing, count_non_missing) = encode_and_missing::encode_and_missing_naive(&mut data);
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
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_encode_and_missing_avx512() {
        let mut data = avec_repeat(LEFT_PLINK, 960);
        let (missing, count_non_missing) = encode_and_missing::encode_and_missing_avx512(&mut data);
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
    fn test_values_naive() {
        let left_data = avec_repeat(LEFT_DATA, 960);
        let right_data = avec_repeat(RIGHT_DATA, 960);
        let left_missing = avec_repeat(LEFT_MISSING, 960);
        let right_missing = avec_repeat(RIGHT_MISSING, 960);
        let num_samples = 960 * 4;
        let num_left_non_missing = 960 * 3;
        let num_right_non_missing = 960 * 4;

        let values = values::values_naive(
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
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_values_avx512() {
        let left_data = avec_repeat(LEFT_DATA, 960);
        let right_data = avec_repeat(RIGHT_DATA, 960);
        let left_missing = avec_repeat(LEFT_MISSING, 960);
        let right_missing = avec_repeat(RIGHT_MISSING, 960);
        let num_samples = 960 * 4;
        let num_left_non_missing = 960 * 3;
        let num_right_non_missing = 960 * 4;

        let values = values::values_avx512(
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
    fn test_get_maf_naive() {
        let data = avec_repeat(LEFT_DATA, 1000);
        let num_samples = 4000;
        let num_non_missing = 3000;

        let maf = maf::get_maf_naive(&data, num_samples, num_non_missing);
        assert!((maf - 0.5).abs() < 1e-10, "Expected MAF to be close to 0.5");
    }

    #[test]
    #[cfg_attr(not(target_feature = "sse4.1"), ignore)]
    fn test_get_maf_sse41() {
        let data = avec_repeat(LEFT_DATA, 1000);
        let num_samples = 4000;
        let num_non_missing = 3000;

        let maf = maf::get_maf_sse4(&data, num_samples, num_non_missing);
        assert!((maf - 0.5).abs() < 1e-10, "Expected MAF to be close to 0.5");
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx2"), ignore)]
    fn test_get_maf_avx2() {
        let data = avec_repeat(LEFT_DATA, 1000);
        let num_samples = 4000;
        let num_non_missing = 3000;

        let maf = maf::get_maf_avx2(&data, num_samples, num_non_missing);
        assert!((maf - 0.5).abs() < 1e-10, "Expected MAF to be close to 0.5");
    }

    #[test]
    #[cfg_attr(not(target_feature = "avx512f"), ignore)]
    fn test_get_maf_avx512() {
        let data = avec_repeat(LEFT_DATA, 1000);
        let num_samples = 4000;
        let num_non_missing = 3000;

        let maf = maf::get_maf_avx512(&data, num_samples, num_non_missing);
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
        let result = dataset.ld_prune(2000, 1, 0.001);
        let prune_out = std::fs::read_to_string("tests/test.prune.out").unwrap();
        let prune_in = std::fs::read_to_string("tests/test.prune.in").unwrap();
        let prune_out = prune_out.lines().collect::<Vec<_>>();
        let prune_in = prune_in.lines().collect::<Vec<_>>();
        assert_eq!(result.pruned, prune_out.len());
        assert_eq!(result.prune_out, prune_out);
        assert_eq!(result.prune_in, prune_in);
    }
}

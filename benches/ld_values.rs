use aligned_vec::AVec;
use diol::prelude::*;
use lmutils::ld::{
    values::{values_avx512, values_naive},
    LD_BLOCK_SIZE,
};

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        list![naive, avx512],
        [64, 640, 6400, 64000, 640000, 6400000, 64000000, 640000000],
    );
    bench.run()?;
    Ok(())
}

const LEFT_DATA: u8 = 0b00010110;
const LEFT_MISSING: u8 = 0b11001111;
const RIGHT_DATA: u8 = 0b00100101;
const RIGHT_MISSING: u8 = 0b11111111;

fn avec_repeat<T: Default + Clone>(value: T, count: usize) -> AVec<T> {
    let mut vec = AVec::with_capacity(LD_BLOCK_SIZE, count);
    vec.resize(count, value);
    vec
}

#[allow(clippy::type_complexity)]
fn data(len: usize) -> (AVec<u8>, AVec<u8>, AVec<u8>, AVec<u8>, u64, u64, u64) {
    let left_data = avec_repeat(LEFT_DATA, len);
    let right_data = avec_repeat(RIGHT_DATA, len);
    let left_missing = avec_repeat(LEFT_MISSING, len);
    let right_missing = avec_repeat(RIGHT_MISSING, len);
    let num_samples = len * 4;
    let num_left_non_missing = len * 3;
    let num_right_non_missing = len * 4;
    (
        left_data,
        right_data,
        left_missing,
        right_missing,
        num_samples as u64,
        num_left_non_missing as u64,
        num_right_non_missing as u64,
    )
}

fn naive(bencher: Bencher, len: usize) {
    let (
        left_data,
        right_data,
        left_missing,
        right_missing,
        num_samples,
        num_left_non_missing,
        num_right_non_missing,
    ) = data(len);
    bencher.bench(|| {
        std::hint::black_box(values_naive(
            &left_data,
            &right_data,
            &left_missing,
            &right_missing,
            num_samples,
            num_left_non_missing,
            num_right_non_missing,
        ));
    });
}

fn avx512(bencher: Bencher, len: usize) {
    let (
        left_data,
        right_data,
        left_missing,
        right_missing,
        num_samples,
        num_left_non_missing,
        num_right_non_missing,
    ) = data(len);
    bencher.bench(|| {
        if is_x86_feature_detected!("avx512f") {
            std::hint::black_box(values_avx512(
                &left_data,
                &right_data,
                &left_missing,
                &right_missing,
                num_samples,
                num_left_non_missing,
                num_right_non_missing,
            ));
        }
    });
}

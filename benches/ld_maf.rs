use aligned_vec::AVec;
use diol::prelude::*;
use lmutils::ld::{
    maf::{get_maf_avx2, get_maf_avx512, get_maf_naive, get_maf_sse4},
    LD_BLOCK_SIZE,
};

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        list![naive, sse, avx2, avx512],
        [8, 80, 800, 8000, 80000, 800000, 8000000],
    );
    bench.run()?;
    Ok(())
}

fn data(len: usize) -> AVec<u8> {
    let mut vec = AVec::with_capacity(LD_BLOCK_SIZE, len);
    vec.resize(len, 0b00010110);
    vec
}

fn naive(bencher: Bencher, len: usize) {
    let data = data(len);
    bencher.bench(|| {
        std::hint::black_box(get_maf_naive(&data, len as u64, ((len / 4) * 3) as u64));
    });
}

fn sse(bencher: Bencher, len: usize) {
    let data = data(len);
    bencher.bench(std::hint::black_box(|| {
        if is_x86_feature_detected!("sse4.1") {
            get_maf_sse4(&data, len as u64, ((len / 4) * 3) as u64);
        }
    }));
}

fn avx2(bencher: Bencher, len: usize) {
    let data = data(len);
    bencher.bench(|| {
        if is_x86_feature_detected!("avx2") {
            get_maf_avx2(&data, len as u64, ((len / 4) * 3) as u64);
        }
    });
}

fn avx512(bencher: Bencher, len: usize) {
    let data = data(len);
    bencher.bench(|| {
        if is_x86_feature_detected!("avx512f") {
            get_maf_avx512(&data, len as u64, ((len / 4) * 3) as u64);
        }
    });
}

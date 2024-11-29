use diol::prelude::*;
use lmutils::{
    unpack_avx2_par,
    unpack_avx2_sync,
    unpack_avx512_par,
    unpack_avx512_sync,
    unpack_naive_par,
    unpack_naive_sync,
};

fn out(len: usize) -> Vec<f64> {
    vec![0.0; bits(len) as usize]
}

fn bytes(len: usize) -> Vec<u8> {
    vec![0b10101010, 0b01010101]
        .into_iter()
        .cycle()
        .take(len)
        .collect()
}

fn bits(len: usize) -> u64 {
    len as u64 * 8
}

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        list![
            naive_sync,
            naive_par,
            avx2_sync,
            avx2_par,
            avx512_sync,
            avx512_par
        ],
        [100, 1000, 10000, 100000, 1000000, 10000000],
    );
    bench.run()?;
    Ok(())
}

fn naive_sync(bencher: Bencher, len: usize) {
    let mut out = out(len);
    bencher.bench(|| {
        unpack_naive_sync(&mut out, &bytes(len), bits(len), 0.0, 1.0);
    });
}

fn naive_par(bencher: Bencher, len: usize) {
    let mut out = out(len);
    bencher.bench(|| {
        unpack_naive_par(128, &mut out, &bytes(len), bits(len), 0.0, 1.0);
    });
}

fn avx2_sync(bencher: Bencher, len: usize) {
    if let Some(simd) = pulp::x86::V3::try_new() {
        let mut out = out(len);
        bencher.bench(|| {
            unpack_avx2_sync(simd, &mut out, &bytes(len), bits(len), 0.0, 1.0);
        });
    }
}

fn avx2_par(bencher: Bencher, len: usize) {
    if let Some(simd) = pulp::x86::V3::try_new() {
        let mut out = out(len);
        bencher.bench(|| {
            unpack_avx2_par(128, simd, &mut out, &bytes(len), bits(len), 0.0, 1.0);
        });
    }
}

fn avx512_sync(bencher: Bencher, len: usize) {
    if is_x86_feature_detected!("avx512f") {
        let mut out = out(len);
        bencher.bench(|| {
            unpack_avx512_sync(&mut out, &bytes(len), bits(len), 0.0, 1.0);
        });
    }
}

fn avx512_par(bencher: Bencher, len: usize) {
    if is_x86_feature_detected!("avx512f") {
        let mut out = out(len);
        bencher.bench(|| {
            unpack_avx512_par(128, &mut out, &bytes(len), bits(len), 0.0, 1.0);
        });
    }
}

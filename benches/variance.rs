use diol::prelude::*;
use faer::{MatRef, RowMut, RowRef};
use lmutils::{mean, variance_avx2, variance_avx512, variance_naive};

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        list![naive, avx2, avx512, faer_],
        [10, 100, 1000, 10000, 100000, 1000000, 10000000],
    );
    bench.run()?;
    Ok(())
}

fn data(len: usize) -> Vec<f64> {
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        .iter()
        .cycle()
        .take(len)
        .copied()
        .collect::<Vec<f64>>()
}

fn naive(bencher: Bencher, len: usize) {
    let data = data(len);
    bencher.bench(|| {
        variance_naive(&data, 0);
    });
}

fn avx2(bencher: Bencher, len: usize) {
    let data = data(len);
    bencher.bench(|| unsafe {
        if is_x86_feature_detected!("avx2") {
            variance_avx2(&data, 0);
        }
    });
}

fn avx512(bencher: Bencher, len: usize) {
    let data = data(len);
    bencher.bench(|| unsafe {
        if is_x86_feature_detected!("avx512f") {
            variance_avx512(&data, 0);
        }
    });
}

fn faer_(bencher: Bencher, len: usize) {
    let data = data(len);
    bencher.bench(|| {
        let mean = mean(&data);
        let mut variance = 0.0;
        faer::stats::row_varm(
            RowMut::from_mut(&mut variance),
            MatRef::from_column_major_slice(data.as_slice(), data.len(), 1),
            RowRef::from_ref(&mean),
            faer::stats::NanHandling::Ignore,
        );
    });
}

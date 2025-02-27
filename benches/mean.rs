use diol::prelude::*;
use lmutils::mean::{mean_avx2, mean_avx512, mean_naive, mean_sse};

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        list![naive, sse, avx2, avx512, faer_],
        [8, 80, 800, 8000, 80000, 800000, 8000000],
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
        mean_naive(&data);
    });
}

fn sse(bencher: Bencher, len: usize) {
    let data = data(len);
    bencher.bench(std::hint::black_box(|| unsafe {
        if is_x86_feature_detected!("sse4.1") {
            mean_sse(&data);
        }
    }));
}

fn avx2(bencher: Bencher, len: usize) {
    let data = data(len);
    bencher.bench(|| unsafe {
        if is_x86_feature_detected!("avx2") {
            mean_avx2(&data);
        }
    });
}

fn avx512(bencher: Bencher, len: usize) {
    let data = data(len);
    bencher.bench(|| unsafe {
        if is_x86_feature_detected!("avx512f") {
            mean_avx512(&data);
        }
    });
}

fn faer_(bencher: Bencher, len: usize) {
    let data = data(len);
    bencher.bench(|| {
        let mut mean = 0.0;
        faer::stats::row_mean(
            faer::row::from_mut(&mut mean),
            faer::mat::from_column_major_slice(data.as_slice(), data.len(), 1),
            faer::stats::NanHandling::Ignore,
        );
    });
}

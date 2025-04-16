use diol::prelude::*;

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        list![naive, sse4, avx2, avx512,],
        [8, 80, 800, 8000, 80000, 800000, 8000000],
    );
    bench.run()?;
    Ok(())
}

fn naive(bencher: Bencher, len: usize) {
    let actual = (0..len).map(|x| x as f64).collect::<Vec<_>>();
    let predicted = (0..len).map(|x| (x as f64) * 2.0).collect::<Vec<_>>();
    bencher.bench(|| {
        std::hint::black_box(lmutils::r2_naive(&actual, &predicted));
    });
}

fn sse4(bencher: Bencher, len: usize) {
    if is_x86_feature_detected!("sse4.1") {
        let actual = (0..len).map(|x| x as f64).collect::<Vec<_>>();
        let predicted = (0..len).map(|x| (x as f64) * 2.0).collect::<Vec<_>>();
        bencher.bench(|| unsafe {
            lmutils::r2_sse4(&actual, &predicted);
        });
    }
}

fn avx2(bencher: Bencher, len: usize) {
    if is_x86_feature_detected!("avx2") {
        let actual = (0..len).map(|x| x as f64).collect::<Vec<_>>();
        let predicted = (0..len).map(|x| (x as f64) * 2.0).collect::<Vec<_>>();
        bencher.bench(|| unsafe {
            lmutils::r2_avx2(&actual, &predicted);
        });
    }
}

fn avx512(bencher: Bencher, len: usize) {
    if is_x86_feature_detected!("avx512f") {
        let actual = (0..len).map(|x| x as f64).collect::<Vec<_>>();
        let predicted = (0..len).map(|x| (x as f64) * 2.0).collect::<Vec<_>>();
        bencher.bench(|| unsafe {
            lmutils::r2_avx512(&actual, &predicted);
        });
    }
}

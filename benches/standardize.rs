use diol::prelude::*;

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        list![
            naive,
            naive_recip,
            sse4,
            sse4_recip,
            avx2,
            avx2_recip,
            avx512,
            avx512_recip,
        ],
        [8, 80, 800, 8000, 80000, 800000, 8000000],
    );
    bench.run()?;
    Ok(())
}

fn naive(bencher: Bencher, len: usize) {
    let mut x = (0..len).map(|x| x as f64).collect::<Vec<_>>();
    bencher.bench(|| {
        lmutils::standardize_naive(&mut x, 1);
    });
}

fn naive_recip(bencher: Bencher, len: usize) {
    let mut x = (0..len).map(|x| x as f64).collect::<Vec<_>>();
    bencher.bench(|| {
        lmutils::standardize_naive_recip(&mut x, 1);
    });
}

fn sse4(bencher: Bencher, len: usize) {
    let mut x = (0..len).map(|x| x as f64).collect::<Vec<_>>();
    bencher.bench(|| unsafe {
        if is_x86_feature_detected!("sse4.1") {
            lmutils::standardize_sse4(&mut x, 1);
        }
    });
}

fn sse4_recip(bencher: Bencher, len: usize) {
    let mut x = (0..len).map(|x| x as f64).collect::<Vec<_>>();
    bencher.bench(|| unsafe {
        if is_x86_feature_detected!("sse4.1") {
            lmutils::standardize_sse4_recip(&mut x, 1);
        }
    });
}

fn avx2(bencher: Bencher, len: usize) {
    let mut x = (0..len).map(|x| x as f64).collect::<Vec<_>>();
    bencher.bench(|| unsafe {
        if is_x86_feature_detected!("avx2") {
            lmutils::standardize_avx2(&mut x, 1);
        }
    });
}

fn avx2_recip(bencher: Bencher, len: usize) {
    let mut x = (0..len).map(|x| x as f64).collect::<Vec<_>>();
    bencher.bench(|| unsafe {
        if is_x86_feature_detected!("avx2") {
            lmutils::standardize_avx2_recip(&mut x, 1);
        }
    });
}

fn avx512(bencher: Bencher, len: usize) {
    let mut x = (0..len).map(|x| x as f64).collect::<Vec<_>>();
    bencher.bench(|| unsafe {
        if is_x86_feature_detected!("avx512f") {
            lmutils::standardize_avx512(&mut x, 1);
        }
    });
}

fn avx512_recip(bencher: Bencher, len: usize) {
    let mut x = (0..len).map(|x| x as f64).collect::<Vec<_>>();
    bencher.bench(|| unsafe {
        if is_x86_feature_detected!("avx512f") {
            lmutils::standardize_avx512_recip(&mut x, 1);
        }
    });
}

use diol::prelude::*;
use lmutils::R2Simd;

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        list![naive, simd],
        [100, 1000, 10000, 100000, 1000000, 10000000],
    );
    bench.run()?;
    Ok(())
}

fn naive(bencher: Bencher, len: usize) {
    let v = (0..len).map(|x| x as f64).collect::<Vec<_>>();
    let r2 = R2Simd::new(&v, &v);
    bencher.bench(|| {
        r2.clone().calculate_no_simd();
    });
}

fn simd(bencher: Bencher, len: usize) {
    let v = (0..len).map(|x| x as f64).collect::<Vec<_>>();
    let r2 = R2Simd::new(&v, &v);
    bencher.bench(|| {
        r2.clone().calculate();
    });
}

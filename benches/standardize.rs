use diol::prelude::*;
use lmutils::standardize_row;

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args());
    bench.register(standardize, [100, 1000, 10000, 100000, 1000000, 10000000]);
    bench.run()?;
    Ok(())
}

fn standardize(bencher: Bencher, len: usize) {
    let v = (0..len).map(|x| x as f64).collect::<Vec<_>>();
    let mut r = faer::row::Row::from_fn(len, |i| v[i]);
    bencher.bench(|| {
        standardize_row(r.as_mut());
    });
}

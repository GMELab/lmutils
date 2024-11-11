use diol::prelude::*;
use lmutils::{logistic_regression_irls, logistic_regression_newton_raphson};
use rand_distr::Distribution;

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args());
    bench.register_many(list![irls, newton_raphson], [5, 50, 500, 5000]);
    bench.run()?;
    Ok(())
}

fn irls(bencher: Bencher, len: usize) {
    let xs = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
    let ys = statrs::distribution::Bernoulli::new(0.5).unwrap();
    let xs = xs
        .sample_iter(rand::thread_rng())
        .take(len)
        .collect::<Vec<_>>();
    let ys = ys
        .sample_iter(rand::thread_rng())
        .take(len)
        .collect::<Vec<_>>();
    let xs = faer::mat::from_column_major_slice(xs.as_slice(), len, 1);
    bencher.bench(|| {
        logistic_regression_irls(xs, &ys);
    });
}

fn newton_raphson(bencher: Bencher, len: usize) {
    let xs = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
    let ys = statrs::distribution::Bernoulli::new(0.5).unwrap();
    let xs = xs
        .sample_iter(rand::thread_rng())
        .take(len)
        .collect::<Vec<_>>();
    let ys = ys
        .sample_iter(rand::thread_rng())
        .take(len)
        .collect::<Vec<_>>();
    let xs = faer::mat::from_column_major_slice(xs.as_slice(), len, 1);
    bencher.bench(|| {
        logistic_regression_newton_raphson(xs, &ys);
    });
}

use std::fmt::Debug;

use diol::prelude::*;
use lmutils::{logistic_regression_irls, logistic_regression_newton_raphson};
use rand::SeedableRng;
use rand_distr::Distribution;

#[derive(Clone)]
struct Arg {
    nrow: usize,
    ncol: usize,
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl Debug for Arg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Arg")
            .field("nrow", &self.nrow)
            .field("ncol", &self.ncol)
            .finish()
    }
}

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let args = [1, 2, 3, 4].iter().map(|len| {
        let nrow = 10_usize.pow(*len);
        let ncol = 5_usize.pow(*len);
        let xs = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
        let ys = statrs::distribution::Bernoulli::new(0.5).unwrap();
        let xs = xs
            .sample_iter(&mut rng)
            .take(nrow * ncol)
            .collect::<Vec<_>>();
        let ys = ys.sample_iter(&mut rng).take(nrow).collect::<Vec<_>>();
        Arg { nrow, ncol, xs, ys }
    });
    bench.register_many(list![irls, newton_raphson], args);
    bench.run()?;
    Ok(())
}

fn irls(bencher: Bencher, Arg { nrow, ncol, xs, ys }: Arg) {
    let xs = faer::mat::from_column_major_slice(xs.as_slice(), nrow, ncol);
    bencher.bench(|| {
        logistic_regression_irls(xs, &ys);
    });
}

fn newton_raphson(bencher: Bencher, Arg { nrow, ncol, xs, ys }: Arg) {
    let xs = faer::mat::from_column_major_slice(xs.as_slice(), nrow, ncol);
    bencher.bench(|| {
        logistic_regression_newton_raphson(xs, &ys);
    });
}

use std::fmt::Debug;

use diol::prelude::*;
use lmutils::{logistic_regression_irls, logistic_regression_newton_raphson};
use rand_distr::Distribution;

#[derive(Clone)]
struct Arg {
    len: usize,
    xs:  Vec<f64>,
    ys:  Vec<f64>,
}

impl Debug for Arg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Arg").field("len", &self.len).finish()
    }
}

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args());
    let xs = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
    let ys = statrs::distribution::Bernoulli::new(0.5).unwrap();
    let args = [5, 50, 500, 5000].iter().map(|len| {
        let xs = xs
            .sample_iter(rand::thread_rng())
            .take(*len)
            .collect::<Vec<_>>();
        let ys = ys
            .sample_iter(rand::thread_rng())
            .take(*len)
            .collect::<Vec<_>>();
        Arg { len: *len, xs, ys }
    });
    bench.register_many(list![irls, newton_raphson], args);
    bench.run()?;
    Ok(())
}

fn irls(bencher: Bencher, Arg { len, xs, ys }: Arg) {
    let xs = faer::mat::from_column_major_slice(xs.as_slice(), len, 1);
    bencher.bench(|| {
        logistic_regression_irls(xs, &ys);
    });
}

fn newton_raphson(bencher: Bencher, Arg { len, xs, ys }: Arg) {
    let xs = faer::mat::from_column_major_slice(xs.as_slice(), len, 1);
    bencher.bench(|| {
        logistic_regression_newton_raphson(xs, &ys);
    });
}

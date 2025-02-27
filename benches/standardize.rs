use diol::prelude::*;
use faer::Col;
use pulp::Arch;

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        list![
            standardize,
            standardize_recip,
            standardize_auto_vectorize,
            standardize_auto_vectorize_recip
        ],
        [100, 1000, 10000, 100000, 1000000, 10000000],
    );
    bench.run()?;
    Ok(())
}

fn bench_standardize<F>(bencher: Bencher, len: usize, f: impl Fn(Col<f64>, f64, f64) -> F)
where
    F: FnMut(),
{
    let v = (0..len).map(|x| x as f64).collect::<Vec<_>>();
    let x = faer::col::Col::from_fn(len, |i| v[i]);
    let mut mean = 0.0;
    let mut std: f64 = 0.0;
    faer::stats::row_mean(
        faer::row::from_mut(&mut mean),
        x.as_ref().as_2d(),
        faer::stats::NanHandling::Ignore,
    );
    faer::stats::row_varm(
        faer::row::from_mut(&mut std),
        x.as_ref().as_2d(),
        faer::row::from_ref(&mean),
        faer::stats::NanHandling::Ignore,
    );
    let std = std.sqrt();
    bencher.bench(f(x, mean, std));
}

fn standardize(bencher: Bencher, len: usize) {
    bench_standardize(bencher, len, |mut x, mean, std| {
        move || {
            for x in x.iter_mut() {
                *x = (*x - mean) / std;
            }
        }
    });
}
fn standardize_recip(bencher: Bencher, len: usize) {
    bench_standardize(bencher, len, |mut x, mean, std| {
        move || {
            let std_recip = 1.0 / std;
            for x in x.iter_mut() {
                *x = (*x - mean) * std_recip;
            }
        }
    });
}

fn standardize_auto_vectorize(bencher: Bencher, len: usize) {
    bench_standardize(bencher, len, |mut x, mean, std| {
        move || {
            let xx = x.as_mut();
            if let Some(x) = xx.try_as_slice_mut() {
                Arch::new().dispatch(|| {
                    for x in x.iter_mut() {
                        *x = (*x - mean) / std;
                    }
                });
            } else {
                for x in x.iter_mut() {
                    *x = (*x - mean) / std;
                }
            }
        }
    });
}

fn standardize_auto_vectorize_recip(bencher: Bencher, len: usize) {
    bench_standardize(bencher, len, |mut x, mean, std| {
        move || {
            let xx = x.as_mut();
            let std_recip = 1.0 / std;
            if let Some(x) = xx.try_as_slice_mut() {
                Arch::new().dispatch(|| {
                    for x in x.iter_mut() {
                        *x = (*x - mean) * std_recip;
                    }
                });
            } else {
                for x in x.iter_mut() {
                    *x = (*x - mean) * std_recip;
                }
            }
        }
    });
}

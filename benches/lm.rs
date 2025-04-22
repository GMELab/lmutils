use std::fmt::Debug;

use diol::prelude::*;
use faer::MatRef;
use lmutils::{File, IntoMatrix, Lm, OwnedMatrix};
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
    let args = [1, 2, 3, 4, 5].iter().map(|len| {
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
    bench.register_many(list![lm, r], args);
    bench.run()?;
    Ok(())
}

fn lm(bencher: Bencher, Arg { nrow, ncol, xs, ys }: Arg) {
    let xs = MatRef::from_column_major_slice(xs.as_slice(), nrow, ncol);
    bencher.bench(|| {
        Lm::fit(xs, &ys);
    });
}

fn r(bencher: Bencher, Arg { nrow, ncol, xs, ys }: Arg) {
    let mut mat = OwnedMatrix::new(nrow, ncol, xs, None).into_matrix();
    File::new("lm.mat", lmutils::FileType::Mat, false)
        .write(&mut mat)
        .unwrap();
    std::fs::write(
        "lm.r",
        format!(
            r#"
            xs <- lmutils::load("lm.mat")[[1]]
            ys <- c({})
            start <- Sys.time()
            m <- lm(ys ~ xs)
            elapsed <- Sys.time() - start
            cat(as.character(as.numeric(elapsed), digits = 22))
            "#,
            ys.iter()
                .map(|y| y.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    )
    .unwrap();
    let timings = (0..5)
        .map(|_| {
            let output = std::process::Command::new("Rscript")
                .arg("lm.r")
                .output()
                .unwrap();
            if !output.status.success() {
                panic!("{}", String::from_utf8(output.stderr).unwrap());
            }
            let output = String::from_utf8(output.stdout).unwrap();
            output.trim().parse::<f64>().unwrap()
        })
        .collect::<Vec<_>>();
    let duration =
        std::time::Duration::from_secs_f64(timings.iter().sum::<f64>() / timings.len() as f64);

    bencher.bench(|| std::thread::sleep(duration));
    std::fs::remove_file("lm.r").unwrap();
    std::fs::remove_file("lm.mat").unwrap();
}

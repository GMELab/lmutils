use std::{fmt::Debug, num::NonZero};

use diol::prelude::*;
use faer::{set_global_parallelism, MatRef};
use lmutils::{family, File, Glm, IntoMatrix, OwnedMatrix};
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
    println!("Threads: {}", rayon::current_num_threads());
    let mut bench = Bench::new(BenchConfig::from_args()?);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let args = [1, 2, 3, 4];
    // let args = [4];
    let args = args.iter().flat_map(|len| {
        let nrow = 10_usize.pow(len + 2);
        args.iter()
            .map(|len| {
                let ncol = 5_usize.pow(*len);
                let xs = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
                let ys = statrs::distribution::Bernoulli::new(0.5).unwrap();
                let xs = xs
                    .sample_iter(&mut rng)
                    .take(nrow * ncol)
                    .collect::<Vec<_>>();
                let ys = ys.sample_iter(&mut rng).take(nrow).collect::<Vec<_>>();

                Arg { nrow, ncol, xs, ys }
            })
            .collect::<Vec<_>>()
    });
    bench.register_many(
        list![
            irls, // _newton_raphson,
            irls_seq,
            irls_firth,
            irls_firth_seq,
            r,
            r_firth,
        ],
        args,
    );
    bench.run()?;
    Ok(())
}

fn irls(bencher: Bencher, Arg { nrow, ncol, xs, ys }: Arg) {
    let xs = MatRef::from_column_major_slice(xs.as_slice(), nrow, ncol);
    set_global_parallelism(faer::Par::Rayon(
        NonZero::new(rayon::current_num_threads()).unwrap(),
    ));
    bencher.bench(|| {
        Glm::irls::<family::BinomialLogit>(xs, &ys, 1e-8, 25, false);
    });
}

fn irls_seq(bencher: Bencher, Arg { nrow, ncol, xs, ys }: Arg) {
    let xs = MatRef::from_column_major_slice(xs.as_slice(), nrow, ncol);
    set_global_parallelism(faer::Par::Seq);
    bencher.bench(|| {
        Glm::irls::<family::BinomialLogit>(xs, &ys, 1e-8, 25, false);
    });
}

fn irls_firth(bencher: Bencher, Arg { nrow, ncol, xs, ys }: Arg) {
    let xs = MatRef::from_column_major_slice(xs.as_slice(), nrow, ncol);
    set_global_parallelism(faer::Par::Rayon(
        NonZero::new(rayon::current_num_threads()).unwrap(),
    ));
    bencher.bench(|| {
        Glm::irls::<family::BinomialLogit>(xs, &ys, 1e-8, 25, true);
    });
}

fn irls_firth_seq(bencher: Bencher, Arg { nrow, ncol, xs, ys }: Arg) {
    let xs = MatRef::from_column_major_slice(xs.as_slice(), nrow, ncol);
    set_global_parallelism(faer::Par::Seq);
    bencher.bench(|| {
        Glm::irls::<family::BinomialLogit>(xs, &ys, 1e-8, 25, true);
    });
}

fn _newton_raphson(bencher: Bencher, Arg { nrow, ncol, xs, ys }: Arg) {
    let xs = MatRef::from_column_major_slice(xs.as_slice(), nrow, ncol);
    bencher.bench(|| {
        Glm::newton_raphson::<family::BinomialLogit>(xs, &ys, 1e-8, 25);
    });
}

fn r(bencher: Bencher, Arg { nrow, ncol, xs, ys }: Arg) {
    let mut mat = OwnedMatrix::new(nrow, ncol, xs, None).into_matrix();
    File::new("glm.mat", lmutils::FileType::Mat, false)
        .write(&mut mat)
        .unwrap();
    let mut mat = OwnedMatrix::new(nrow, 1, ys, None).into_matrix();
    File::new("glm_ys.mat", lmutils::FileType::Mat, false)
        .write(&mut mat)
        .unwrap();
    std::fs::write(
        "glm.r",
        r#"
            xs <- lmutils::load("glm.mat")[[1]]
            ys <- lmutils::load("glm_ys.mat")[[1]][,1]
            start <- Sys.time()
            m <- glm(ys ~ xs, family = binomial(link="logit"))
            elapsed <- Sys.time() - start
            cat(as.character(as.numeric(elapsed, units = "secs"), digits = 22))
            "#,
    )
    .unwrap();
    let timings = (0..5)
        .map(|_| {
            let output = std::process::Command::new("Rscript")
                .arg("glm.r")
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
    std::fs::remove_file("glm.r").unwrap();
    std::fs::remove_file("glm.mat").unwrap();
    std::fs::remove_file("glm_ys.mat").unwrap();
}

fn r_firth(bencher: Bencher, Arg { nrow, ncol, xs, ys }: Arg) {
    let mut mat = OwnedMatrix::new(nrow, ncol, xs, None).into_matrix();
    File::new("glm.mat", lmutils::FileType::Mat, false)
        .write(&mut mat)
        .unwrap();
    let mut mat = OwnedMatrix::new(nrow, 1, ys, None).into_matrix();
    File::new("glm_ys.mat", lmutils::FileType::Mat, false)
        .write(&mut mat)
        .unwrap();
    std::fs::write(
        "glm.r",
        r#"
            xs <- lmutils::load("glm.mat")[[1]]
            ys <- lmutils::load("glm_ys.mat")[[1]][,1]
            start <- Sys.time()
            m <- logistf::logistf(ys ~ xs)
            elapsed <- Sys.time() - start
            cat(as.character(as.numeric(elapsed, units = "secs"), digits = 22))
            "#,
    )
    .unwrap();
    let timings = (0..5)
        .map(|_| {
            let output = std::process::Command::new("Rscript")
                .arg("glm.r")
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
    std::fs::remove_file("glm.r").unwrap();
    std::fs::remove_file("glm.mat").unwrap();
    std::fs::remove_file("glm_ys.mat").unwrap();
}

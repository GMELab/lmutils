use faer::{solvers::SpSolver, Mat, MatRef};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;

pub fn variance(data: &[f64]) -> f64 {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64
}

#[derive(Debug, Clone, Copy)]
pub struct R2 {
    r2: f64,
    adj_r2: f64,
}

impl R2 {
    #[inline]
    pub fn r2(&self) -> f64 {
        self.r2
    }

    #[inline]
    pub fn adj_r2(&self) -> f64 {
        self.adj_r2
    }
}

pub fn get_r2s(data: MatRef<f64>, outcomes: MatRef<f64>) -> Vec<R2> {
    let n = data.nrows();
    let m = data.ncols();

    // could potentially use QR decomposition here, but in testing it seems to be slower
    // if memory becomes a serious issue it could be a huge help
    let c_all = data.transpose() * outcomes;
    let c_matrix = data.transpose() * data;
    let inv_matrix = c_matrix.partial_piv_lu().solve(Mat::<f64>::identity(m, m));
    let betas = inv_matrix * c_all;
    (0..outcomes.ncols())
        .into_par_iter()
        .map(|i| {
            let predicted = data * betas.get(.., i);
            let r2 = variance(predicted.as_slice());
            let adj_r2 = 1.0 - (1.0 - r2) * (n as f64 - 1.0) / (n as f64 - m as f64 - 1.0);
            R2 { r2, adj_r2 }
        })
        .collect()
}

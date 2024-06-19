use faer::{get_global_parallelism, linalg, mat::As2D, solvers::SpSolver, Mat, MatRef, Side};
use log::debug;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, StudentsT};

pub fn variance(data: &[f64]) -> f64 {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64
}

#[derive(Debug, Clone)]
pub struct R2 {
    r2: f64,
    adj_r2: f64,
    predicted: Vec<f64>,
    pub(crate) data: Option<String>,
    pub(crate) outcome: Option<String>,
    n: u32,
    m: u32,
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

    #[inline]
    pub fn predicted(&self) -> &[f64] {
        &self.predicted
    }

    #[inline]
    pub fn data(&self) -> Option<&str> {
        self.data.as_deref()
    }

    #[inline]
    pub fn outcome(&self) -> Option<&str> {
        self.outcome.as_deref()
    }

    #[inline]
    pub fn n(&self) -> u32 {
        self.n
    }

    #[inline]
    pub fn m(&self) -> u32 {
        self.m
    }
}

pub fn cross_product(data: MatRef<f64>) -> Mat<f64> {
    let mut mat = Mat::zeros(data.ncols(), data.ncols());
    faer::linalg::matmul::triangular::matmul(
        mat.as_mut(),
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        data.transpose(),
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        data,
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        None,
        1.0,
        get_global_parallelism(),
    );
    mat
}

pub fn get_r2s(data: MatRef<f64>, outcomes: MatRef<f64>) -> Vec<R2> {
    debug!("Calculating R2s");
    let n = data.nrows();
    let m = data.ncols();

    let c_all = data.transpose() * outcomes;
    // let c_matrix = data.transpose() * data;
    let mut c_matrix = Mat::zeros(data.ncols(), data.ncols());
    faer::linalg::matmul::triangular::matmul(
        c_matrix.as_mut(),
        faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
        data.transpose(),
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        data,
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        None,
        1.0,
        get_global_parallelism(),
    );
    // let inv_matrix = c_matrix.partial_piv_lu().solve(Mat::<f64>::identity(m, m));
    // let betas = inv_matrix * c_all;
    let betas = c_matrix.cholesky(Side::Lower).unwrap().solve(c_all);

    debug!("Calculated betas");

    // having experimented with QR decomposition like below, it increased runtime by close to
    // an order of magnitude before i gave up and also more than 2xed memory usage
    // let qr = data.qr();
    // let betas = qr.solve_lstsq(outcomes);
    // let q = qr.compute_thin_q();
    // let r = qr.compute_thin_r();
    // let betas = r.partial_piv_lu().solve(q.transpose() * outcomes);

    let r2s = (0..outcomes.ncols())
        .into_par_iter()
        .map(|i| {
            let predicted = data * betas.get(.., i);
            let r2 = variance(predicted.as_slice());
            let adj_r2 = 1.0 - (1.0 - r2) * (n as f64 - 1.0) / (n as f64 - m as f64 - 1.0);
            R2 {
                r2,
                adj_r2,
                predicted: predicted.as_slice().to_vec(),
                outcome: None,
                data: None,
                n: n as u32,
                m: m as u32,
            }
        })
        .collect();
    debug!("Calculated R2s");
    r2s
}

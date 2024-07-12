use faer::{
    get_global_parallelism,
    mat::AsMatRef,
    solvers::{SpSolver, Svd},
    ComplexField, Mat, MatRef, Side, SimpleEntity,
};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, StudentsT};
use tracing::{debug, warn};

pub fn variance(data: &[f64]) -> f64 {
    let mut mean = 0.0;
    let mut var = 0.0;
    faer::stats::row_mean(
        faer::row::from_mut(&mut mean),
        faer::mat::from_column_major_slice(data, data.len(), 1),
        faer::stats::NanHandling::Ignore,
    );
    faer::stats::row_varm(
        faer::row::from_mut(&mut var),
        faer::mat::from_column_major_slice(data, data.len(), 1),
        faer::row::from_ref(&mean),
        faer::stats::NanHandling::Ignore,
    );
    var
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
    pub fn set_data(&mut self, data: String) {
        self.data = Some(data);
    }

    #[inline]
    pub fn outcome(&self) -> Option<&str> {
        self.outcome.as_deref()
    }

    #[inline]
    pub fn set_outcome(&mut self, outcome: String) {
        self.outcome = Some(outcome);
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

#[inline]
pub fn cross_product(data: MatRef<f64>) -> Mat<f64> {
    data.transpose() * data
}

#[tracing::instrument(skip(data, outcomes))]
pub fn get_r2s(data: MatRef<f64>, outcomes: MatRef<f64>) -> Vec<R2> {
    debug!("Calculating R2s");
    let n = data.nrows();
    let m = data.ncols();

    let c_all = data.transpose() * outcomes;
    let c_matrix = data.transpose() * data;
    let betas = match c_matrix.cholesky(Side::Lower) {
        Ok(chol) => chol.solve(c_all),
        Err(_) => {
            warn!("Using pseudo inverse");
            Svd::new(c_matrix.as_mat_ref()).pseudoinverse() * &c_all
        },
    };

    debug!("Calculated betas");

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

#[derive(Debug, Clone)]
pub struct PValue {
    p_value: f64,
    pub(crate) data: Option<String>,
    pub(crate) data_column: Option<u32>,
    pub(crate) outcome: Option<String>,
}

impl PValue {
    #[inline]
    pub fn p_value(&self) -> f64 {
        self.p_value
    }

    #[inline]
    pub fn data(&self) -> Option<&str> {
        self.data.as_deref()
    }

    #[inline]
    pub fn set_data(&mut self, data: String) {
        self.data = Some(data);
    }

    #[inline]
    pub fn data_column(&self) -> Option<u32> {
        self.data_column
    }

    #[inline]
    pub fn set_data_column(&mut self, data_column: u32) {
        self.data_column = Some(data_column);
    }

    #[inline]
    pub fn outcome(&self) -> Option<&str> {
        self.outcome.as_deref()
    }

    #[inline]
    pub fn set_outcome(&mut self, outcome: String) {
        self.outcome = Some(outcome);
    }
}

#[tracing::instrument(skip(xs, ys))]
pub fn p_value(xs: &[f64], ys: &[f64]) -> PValue {
    debug!("Calculating p-values");
    let mut x = Mat::new();
    x.resize_with(
        xs.len(),
        2,
        #[inline(always)]
        |i, j| if j == 0 { xs[i] } else { 1.0 },
    );
    let y: MatRef<'_, f64> = faer::mat::from_column_major_slice(ys, ys.len(), 1);
    let c_all = x.transpose() * y;
    let mut c_matrix = faer::Mat::zeros(2, 2);
    faer::linalg::matmul::triangular::matmul(
        c_matrix.as_mut(),
        faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
        x.transpose(),
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        &x,
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        None,
        1.0,
        get_global_parallelism(),
    );
    let chol = c_matrix.cholesky(Side::Lower).unwrap();
    let inv_matrix = chol.solve(Mat::<f64>::identity(2, 2));
    let betas = chol.solve(c_all);
    let m = betas.get(0, 0);
    let intercept = betas.get(1, 0);
    let df = xs.len() as f64 - 2.0;
    let residuals = ys
        .iter()
        .zip(xs.iter())
        .map(|(y, x)| (y - (intercept + m * x)))
        .collect::<Vec<_>>();

    let se =
        (inv_matrix[(0, 0)] * ((residuals.iter().map(|x| x.powi(2)).sum::<f64>()) / df)).sqrt();
    let t = m / se;
    let t_distr = StudentsT::new(0.0, 1.0, (xs.len() - 2) as f64).unwrap();
    PValue {
        p_value: 2.0 * (1.0 - t_distr.cdf(t.abs())),
        data: None,
        data_column: None,
        outcome: None,
    }
}

pub fn mean<E: ComplexField + SimpleEntity>(x: &[E]) -> E {
    let mut mean = E::faer_zero();
    faer::stats::row_mean(
        faer::row::from_mut(&mut mean),
        faer::mat::from_column_major_slice(x, 1, x.len()).as_ref(),
        faer::stats::NanHandling::Ignore,
    );
    mean
}

pub fn standardization(x: &mut [f64]) {
    let mut mean = 0.0;
    let mut std: f64 = 0.0;
    faer::stats::row_mean(
        faer::row::from_mut(&mut mean),
        faer::mat::from_column_major_slice(&*x, 1, x.len()).as_ref(),
        faer::stats::NanHandling::Ignore,
    );
    faer::stats::row_varm(
        faer::row::from_mut(&mut std),
        faer::mat::from_column_major_slice(&*x, 1, x.len()).as_ref(),
        faer::row::from_ref(&mean),
        faer::stats::NanHandling::Ignore,
    );
    let std = std.sqrt();
    if std == 0.0 {
        x.fill(0.0);
        return;
    }
    for x in x.iter_mut() {
        *x = (*x - mean) / std;
    }
}

#[non_exhaustive]
pub struct LinearModel {
    pub slopes: Vec<f64>,
    pub intercept: f64,
    pub residuals: Vec<f64>,
    pub r2: f64,
    pub adj_r2: f64,
}

pub fn linear_regression(xs: MatRef<'_, f64>, ys: &[f64]) -> LinearModel {
    let ncols = xs.ncols();
    let mut x = xs.to_owned();
    x.resize_with(
        xs.nrows(),
        xs.ncols() + 1,
        #[inline(always)]
        |_, _| 1.0,
    );
    let y: MatRef<'_, f64> = faer::mat::from_column_major_slice(ys, ys.len(), 1);
    let c_all = x.transpose() * y;
    let mut c_matrix = faer::Mat::zeros(ncols + 1, ncols + 1);
    faer::linalg::matmul::triangular::matmul(
        c_matrix.as_mut(),
        faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
        x.transpose(),
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        &x,
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        None,
        1.0,
        get_global_parallelism(),
    );
    let betas = c_matrix.cholesky(Side::Lower).unwrap().solve(c_all);
    let betas = betas.col(0).try_as_slice().unwrap();
    let intercept = betas[ncols];
    let residuals = ys
        .iter()
        .enumerate()
        .map(|(i, y)| y - (intercept + (0..ncols).map(|j| betas[j] * x[(i, j)]).sum::<f64>()))
        .collect::<Vec<_>>();
    let mean_y = mean(ys);
    let r2 = 1.0
        - residuals.iter().map(|x| x.powi(2)).sum::<f64>()
            / ys.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();
    let adj_r2 =
        1.0 - (1.0 - r2) * (ys.len() as f64 - 1.0) / (ys.len() as f64 - ncols as f64 - 1.0);
    LinearModel {
        slopes: betas[..ncols].to_vec(),
        intercept,
        residuals,
        r2,
        adj_r2,
    }
}

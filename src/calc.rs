use faer::{
    get_global_parallelism,
    mat::AsMatRef,
    solvers::{SpSolver, Svd},
    ColMut, ComplexField, Mat, MatRef, RowMut, Side, SimpleEntity,
};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, StudentsT};
use tracing::{debug, warn};

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
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn r2(&self) -> f64 {
        self.r2
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn adj_r2(&self) -> f64 {
        self.adj_r2
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn predicted(&self) -> &[f64] {
        &self.predicted
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn data(&self) -> Option<&str> {
        self.data.as_deref()
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn set_data(&mut self, data: String) {
        self.data = Some(data);
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn outcome(&self) -> Option<&str> {
        self.outcome.as_deref()
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn set_outcome(&mut self, outcome: String) {
        self.outcome = Some(outcome);
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn n(&self) -> u32 {
        self.n
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn m(&self) -> u32 {
        self.m
    }
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
            let mut predicted = (data * betas.get(.., i)).as_slice().to_vec();
            let r2 = variance(predicted.as_slice());
            let adj_r2 = 1.0 - (1.0 - r2) * (n as f64 - 1.0) / (n as f64 - m as f64 - 1.0);
            if std::env::var("LMUTILS_DISABLE_PREDICTED").is_ok() {
                predicted = Vec::new();
            }
            R2 {
                r2,
                adj_r2,
                predicted,
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
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn p_value(&self) -> f64 {
        self.p_value
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn data(&self) -> Option<&str> {
        self.data.as_deref()
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn set_data(&mut self, data: String) {
        self.data = Some(data);
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn data_column(&self) -> Option<u32> {
        self.data_column
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn set_data_column(&mut self, data_column: u32) {
        self.data_column = Some(data_column);
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn outcome(&self) -> Option<&str> {
        self.outcome.as_deref()
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
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

pub fn mean<E: ComplexField + SimpleEntity>(x: &[E]) -> E {
    let mut mean = E::faer_zero();
    faer::stats::row_mean(
        faer::row::from_mut(&mut mean),
        faer::mat::from_column_major_slice(x, x.len(), 1).as_ref(),
        faer::stats::NanHandling::Ignore,
    );
    mean
}

pub fn standardize_column(mut x: ColMut<f64>) {
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
    if std == 0.0 {
        x.fill(0.0);
        return;
    }
    for x in x.iter_mut() {
        *x = (*x - mean) / std;
    }
}

pub fn standardize_row(mut x: RowMut<f64>) {
    let mut mean = 0.0;
    let mut std: f64 = 0.0;
    faer::stats::col_mean(
        faer::col::from_mut(&mut mean),
        x.as_ref().as_2d(),
        faer::stats::NanHandling::Ignore,
    );
    faer::stats::col_varm(
        faer::col::from_mut(&mut std),
        x.as_ref().as_2d(),
        faer::col::from_ref(&mean),
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

pub struct LinearModel {
    slopes: Vec<f64>,
    intercept: f64,
    residuals: Vec<f64>,
    r2: f64,
    adj_r2: f64,
}

impl LinearModel {
    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn slopes(&self) -> &[f64] {
        &self.slopes
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn r2(&self) -> f64 {
        self.r2
    }

    #[inline]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn adj_r2(&self) -> f64 {
        self.adj_r2
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use assert_float_eq::*;
    use test_log::test;

    macro_rules! float_eq {
        ($a:expr, $b:expr) => {
            assert_float_absolute_eq!($a, $b, 0.0000001);
        };
    }

    #[test]
    fn test_variance() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(variance(&data), 2.5);
    }

    #[test]
    fn test_mean() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&data), 3.0);
    }

    #[test]
    fn test_standardize_column() {
        let mut data = [1.0, 2.0, 3.0];
        standardize_column(faer::col::from_slice_mut(&mut data));
        assert_eq!(data, [-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_standardize_row() {
        let mut data = [1.0, 2.0, 3.0];
        standardize_row(faer::row::from_slice_mut(&mut data));
        assert_eq!(data, [-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_get_r2s() {
        let data = faer::mat::from_column_major_slice::<f64>(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0],
            4,
            2,
        );
        let outcomes = faer::mat::from_column_major_slice::<f64>(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            4,
            2,
        );
        let r2s = get_r2s(data.as_mat_ref(), outcomes.as_mat_ref());
        assert_eq!(r2s.len(), 2);
        float_eq!(r2s[0].r2(), 1.6666666666666);
        float_eq!(r2s[0].adj_r2(), 2.9999999999999999);
        let pred: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(r2s[0].predicted().len(), 4);
        for (a, b) in r2s[0].predicted().iter().copied().zip(pred.iter().copied()) {
            float_eq!(a, b);
        }
        assert_eq!(r2s[0].n(), 4);
        assert_eq!(r2s[0].m(), 2);
        assert!(r2s[0].data().is_none());
        assert!(r2s[0].outcome().is_none());
        assert_eq!(r2s[1].r2(), 1.8575631074638974);
        float_eq!(r2s[1].adj_r2(), 3.5726893223916925);
        let pred: [f64; 4] = [
            5.0478087649402426,
            5.6334661354581685,
            7.334661354581674,
            7.9203187250996,
        ];
        assert_eq!(r2s[1].predicted().len(), 4);
        for (a, b) in r2s[1].predicted().iter().copied().zip(pred.iter().copied()) {
            float_eq!(a, b);
        }
        assert_eq!(r2s[1].n(), 4);
        assert_eq!(r2s[1].m(), 2);
        assert!(r2s[1].data().is_none());
        assert!(r2s[1].outcome().is_none());
    }

    #[test]
    fn test_p_value() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [1.0, 2.0, 3.0, 4.0, 5.0];
        let p_value = p_value(&xs, &ys);
        float_eq!(p_value.p_value(), 0.0);
        assert!(p_value.data().is_none());
        assert!(p_value.outcome().is_none());
    }

    #[test]
    fn test_linear_regression() {
        let xs = faer::mat::from_column_major_slice::<f64>(&[1.0, 2.0, 3.0, 4.0, 5.0], 5, 1);
        let ys = [1.0, 2.0, 3.0, 4.0, 5.0];
        let model = linear_regression(xs.as_mat_ref(), &ys);
        assert_eq!(model.slopes().len(), 1);
        float_eq!(model.slopes()[0], 1.0);
        float_eq!(model.intercept(), 0.0);
        assert_eq!(model.residuals().len(), 5);
        let residuals = [0.0, 0.0, 0.0, 0.0, 0.0];
        for (a, b) in model
            .residuals()
            .iter()
            .copied()
            .zip(residuals.iter().copied())
        {
            float_eq!(a, b);
        }
        float_eq!(model.r2(), 1.0);
        float_eq!(model.adj_r2(), 1.0);
    }
}

use faer::{
    linalg::solvers::{DenseSolveCore, Solve},
    mat::AsMatRef,
    Mat, MatRef,
};
use statrs::distribution::{ContinuousCDF, StudentsT};
use tracing::warn;

use crate::{calculate_adj_r2, coef::Coef, should_disable_predicted, R2Simd};

#[derive(Debug, Clone)]
pub struct Lm {
    // the last element is the intercept
    coefs: Vec<Coef>,
    predicted: Vec<f64>,
    r2: f64,
    adj_r2: f64,
    n: u64,
    m: u64,
}

impl Lm {
    pub fn fit(xs: MatRef<'_, f64>, ys: &[f64]) -> Self {
        Self::fit_many(xs, MatRef::from_column_major_slice(ys, ys.len(), 1))
            .into_iter()
            .next()
            .unwrap()
    }

    pub fn fit_many(xs: MatRef<'_, f64>, ys: MatRef<'_, f64>) -> Vec<Self> {
        let ncols = xs.ncols();
        let mut x = xs.to_owned();
        x.resize_with(
            xs.nrows(),
            xs.ncols() + 1,
            #[inline(always)]
            |_, _| 1.0,
        );
        let mut models = Vec::with_capacity(ys.ncols());
        let df = (xs.nrows() - xs.ncols() - 1) as f64;
        let t_distr = StudentsT::new(0.0, 1.0, df).unwrap();
        let xtx = x.transpose() * &x;
        let chol = xtx
            .llt(faer::Side::Lower)
            .expect("could not compute Cholesky decomposition");
        let xtx_inv = chol.inverse();
        for i in 0..ys.ncols() {
            let y = ys.col(i);
            let c_all = x.transpose() * y;
            let betas = chol.solve(c_all);
            let mut predicted = (&x * &betas)
                .try_as_col_major()
                .expect("could not get slice")
                .as_slice()
                .to_vec();
            let r2 = crate::r2(
                y.try_as_col_major()
                    .expect("could not get slice")
                    .as_slice(),
                &predicted,
            );
            let theta_hat = y
                .try_as_col_major()
                .expect("could not get slice")
                .as_slice()
                .iter()
                .zip(predicted.iter())
                .map(|(yi, pi)| (yi - pi).powi(2))
                .sum::<f64>()
                / df;
            let intercept = betas[ncols];
            let betas = betas
                .try_as_col_major()
                .expect("could not get slice")
                .as_slice();
            let adj_r2 = calculate_adj_r2(r2, y.nrows(), ncols);
            if should_disable_predicted() {
                predicted = Vec::new();
            }
            models.push(Self {
                coefs: (0..ncols)
                    .map(|i| {
                        let coef = betas[i];
                        let std_err = (theta_hat * xtx_inv[(i, i)] / df).sqrt();
                        let t = coef / std_err;
                        let p = 2.0 * (1.0 - t_distr.cdf(t.abs()));
                        Coef::new(format!("x[{}]", i), coef, std_err, t, p)
                    })
                    .chain(std::iter::once({
                        let std_err = (theta_hat * xtx_inv[(ncols, ncols)] / df).sqrt();
                        let t = intercept / std_err;
                        let p = 2.0 * (1.0 - t_distr.cdf(t.abs()));
                        Coef::new_intercept(intercept, std_err, t, p)
                    }))
                    .collect(),
                predicted,
                r2,
                adj_r2,
                n: y.nrows() as u64,
                m: ncols as u64,
            });
        }
        models
    }

    pub fn slopes(&self) -> &[Coef] {
        &self.coefs[..self.coefs.len() - 1]
    }

    pub fn intercept(&self) -> &Coef {
        &self.coefs[self.coefs.len() - 1]
    }

    pub fn predicted(&self) -> &[f64] {
        &self.predicted
    }

    pub fn r2(&self) -> f64 {
        self.r2
    }

    pub fn adj_r2(&self) -> f64 {
        self.adj_r2
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        let mut v = self.intercept().coef();
        let slopes = self.slopes();
        for i in 0..self.slopes().len() {
            v += slopes[i].coef() * x[i];
        }
        v
    }

    pub fn coefs(&self) -> &[Coef] {
        &self.coefs
    }

    pub fn n(&self) -> u64 {
        self.n
    }

    pub fn m(&self) -> u64 {
        self.m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_float_eq {
        ($a:expr, $b:expr, $tol:expr) => {
            assert!(($a - $b).abs() < $tol, "{:.22} != {:.22}", $a, $b);
        };
    }

    macro_rules! float_eq {
        ($a:expr, $b:expr) => {
            assert_float_eq!($a, $b, 1e-12);
        };
    }

    #[test]
    fn test_lm_simple() {
        let xs = MatRef::from_column_major_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], 5, 1);
        let ys = [1.0, 2.0, 3.0, 4.0, 5.0];
        let model = Lm::fit(xs.as_mat_ref(), &ys);
        assert_eq!(model.slopes().len(), 1);
        float_eq!(model.slopes()[0].coef(), 1.0);
        float_eq!(model.intercept().coef(), 0.0);
        assert_eq!(model.predicted().len(), 5);
        let predicted = [1.0, 2.0, 3.0, 4.0, 5.0];
        for (a, b) in model
            .predicted()
            .iter()
            .copied()
            .zip(predicted.iter().copied())
        {
            float_eq!(a, b);
        }
        float_eq!(model.r2(), 1.0);
        float_eq!(model.adj_r2(), 1.0);
    }

    #[test]
    fn test_lm_less_simple() {
        let xs = MatRef::from_column_major_slice(&[1.0, 2.0, 3.0, 5.0, 1.0, 3.0, 2.0, 4.0], 4, 2);
        let ys = [2.0, 4.0, 5.0, 8.0];
        let model = Lm::fit(xs.as_mat_ref(), &ys);
        assert_eq!(model.slopes().len(), 2);
        float_eq!(model.slopes()[0].coef(), 1.259259259259255);
        float_eq!(model.slopes()[0].std_err(), 0.08281733249999221);
        float_eq!(model.slopes()[0].t(), 15.20526224699852);
        float_eq!(model.slopes()[0].p(), 0.041808177201457575);
        float_eq!(model.slopes()[1].coef(), 0.31481481481482115);
        float_eq!(model.slopes()[1].std_err(), 0.10955703302036324);
        float_eq!(model.slopes()[1].t(), 2.873524466077015);
        float_eq!(model.slopes()[1].p(), 0.21320151615245053);
        float_eq!(model.intercept().coef(), 0.5);
        float_eq!(model.intercept().std_err(), 0.16666666666666669);
        float_eq!(model.intercept().t(), 2.999999999999973);
        float_eq!(model.intercept().p(), 0.2048327646991348);
        assert_eq!(model.predicted().len(), 4);
        let expected = [
            2.0740740740740717,
            3.962962962962969,
            4.907407407407403,
            8.055555555555555,
        ];
        for (a, b) in model
            .predicted()
            .iter()
            .copied()
            .zip(expected.iter().copied())
        {
            float_eq!(a, b);
        }
        float_eq!(model.r2(), 0.9990123456790123);
        float_eq!(model.adj_r2(), 0.9970370370370369);
        assert_eq!(model.n(), 4);
        assert_eq!(model.m(), 2);
    }

    #[test]
    fn test_lm_predict() {
        let xs = MatRef::from_column_major_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], 5, 1);
        let ys = [1.0, 2.0, 3.0, 4.0, 5.0];
        let model = Lm::fit(xs.as_mat_ref(), &ys);
        float_eq!(model.predict(&[6.0]), 6.0);
        let xs = MatRef::from_column_major_slice(&[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], 4, 2);
        let ys = [2.0, 4.0, 6.0, 8.0];
        let model = Lm::fit(xs.as_mat_ref(), &ys);
        float_eq!(model.predict(&[1.0, 1.0]), 2.0);
    }
}

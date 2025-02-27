use faer::{
    mat::AsMatRef,
    solvers::{SpSolver, Svd},
    MatRef,
};
use tracing::warn;

use crate::{calculate_adj_r2, should_disable_predicted, R2Simd};

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
}

#[derive(Debug, Clone)]
pub struct Coef {
    coef: f64,
    std_err: f64,
    t: f64,
    p: f64,
}

impl Coef {
    pub fn coef(&self) -> f64 {
        self.coef
    }

    pub fn std_err(&self) -> f64 {
        self.std_err
    }

    pub fn t(&self) -> f64 {
        self.t
    }

    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Lm {
    pub fn new(xs: MatRef<'_, f64>, ys: &[f64]) -> Self {
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
        let c_matrix = x.transpose() * &x;
        let betas = match c_matrix.cholesky(faer::Side::Lower) {
            Ok(chol) => chol.solve(c_all),
            Err(_) => {
                warn!("Using pseudo inverse");
                Svd::new(c_matrix.as_mat_ref()).pseudoinverse() * &c_all
            },
        };
        let betas = betas.col(0).try_as_slice().unwrap();
        let intercept = betas[ncols];
        let mut predicted = (0..ys.len())
            .map(|i| intercept + (0..ncols).map(|j| betas[j] * x[(i, j)]).sum::<f64>())
            .collect::<Vec<_>>();
        let r2 = R2Simd::new(ys, &predicted).calculate();
        let adj_r2 = calculate_adj_r2(r2, ys.len(), ncols);
        if should_disable_predicted() {
            predicted = Vec::new();
        }
        Lm {
            coefs: (0..ncols)
                .map(|i| {
                    let coef = betas[i];
                    let std_err = 0.0;
                    let t = 0.0;
                    let p = 0.0;
                    Coef {
                        coef,
                        std_err,
                        t,
                        p,
                    }
                })
                .chain(std::iter::once(Coef {
                    coef: intercept,
                    std_err: 0.0,
                    t: 0.0,
                    p: 0.0,
                }))
                .collect(),
            predicted,
            r2,
            adj_r2,
            n: ys.len() as u64,
            m: ncols as u64,
        }
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        let mut v = self.intercept().coef();
        let slopes = self.slopes();
        for i in 0..self.slopes().len() {
            v += slopes[i].coef() * x[i];
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_float_eq {
        ($a:expr, $b:expr, $tol:expr) => {
            assert!(($a - $b).abs() < $tol);
        };
    }

    macro_rules! float_eq {
        ($a:expr, $b:expr) => {
            assert_float_eq!($a, $b, 1e-14);
        };
    }

    #[test]
    fn test_lm() {
        let xs = faer::mat::from_column_major_slice::<f64>(&[1.0, 2.0, 3.0, 4.0, 5.0], 5, 1);
        let ys = [1.0, 2.0, 3.0, 4.0, 5.0];
        let model = Lm::new(xs.as_mat_ref(), &ys);
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
    fn test_lm_predict() {
        let xs = faer::mat::from_column_major_slice::<f64>(&[1.0, 2.0, 3.0, 4.0, 5.0], 5, 1);
        let ys = [1.0, 2.0, 3.0, 4.0, 5.0];
        let model = Lm::new(xs.as_mat_ref(), &ys);
        float_eq!(model.predict(&[6.0]), 6.0);
        let xs = faer::mat::from_column_major_slice::<f64>(
            &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            4,
            2,
        );
        let ys = [2.0, 4.0, 6.0, 8.0];
        let model = Lm::new(xs.as_mat_ref(), &ys);
        float_eq!(model.predict(&[1.0, 1.0]), 2.0);
    }
}

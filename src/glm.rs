use std::mem::MaybeUninit;

use faer::{
    get_global_parallelism,
    mat::AsMatRef,
    solvers::{SpSolver, SpSolverLstsq, Svd, ThinSvd},
    Col, Mat, MatRef,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use tracing::{debug, warn};

use crate::{calculate_adj_r2, coef::Coef, should_disable_predicted, R2Simd};

#[derive(Debug, Clone)]
pub struct Glm {
    // the last element is the intercept
    coefs: Vec<Coef>,
    predicted: Vec<f64>,
    r2: f64,
    adj_r2: f64,
    n: u64,
    m: u64,
}

impl Glm {
    #[tracing::instrument(skip(xs, ys))]
    pub fn irls<F: Family>(
        xs: MatRef<'_, f64>,
        ys: &[f64],
        epsilon: f64,
        max_iterations: usize,
    ) -> Self {
        let ncols = xs.ncols();
        let mut mu = ys.iter().map(|y| F::mu_start(*y)).collect::<Vec<_>>();
        let mut delta = 1.0;
        // let mut l = 0.0;
        let mut x = xs.to_owned();
        x.resize_with(
            xs.nrows(),
            xs.ncols() + 1,
            #[inline(always)]
            |_, _| 1.0,
        );
        let mut slopes = vec![0.0; xs.ncols()];
        let mut intercept = 0.0;
        let mut z = vec![0.0; ys.len()];
        // let mut w = Mat::zeros(ys.len(), ys.len());
        let mut w = vec![0.0; ys.len()];
        let xt = x.transpose();
        let mut xtw = Mat::<f64>::zeros(x.ncols(), ys.len());
        let mut xtwx = Mat::zeros(x.ncols(), x.ncols());
        let mut xtwz = Col::zeros(x.ncols());
        let mut i = 0;
        let mut converged = true;
        let mut dev = 0.0;
        for i in 0..ys.len() {
            dev += F::dev_resids(ys[i], mu[i]);
        }
        while delta > epsilon {
            for ((z, mu), y) in z.iter_mut().zip(mu.iter()).zip(ys) {
                *z = F::linkfun(*mu) + (y - mu) * F::mu_eta(*mu);
            }
            for (i, mu) in mu.iter().enumerate() {
                w[i] = 1.0 / (F::mu_eta(*mu).powi(2) * F::variance(*mu));
            }

            xtw.par_col_chunks_mut(1)
                .zip(w.par_iter())
                .enumerate()
                .for_each(|(j, (mut xtw, w))| {
                    xtw.col_iter_mut().for_each(|x| {
                        x.iter_mut()
                            .enumerate()
                            .for_each(|(i, x)| *x = xt[(i, j)] * w)
                    })
                });
            faer::linalg::matmul::matmul(
                xtwx.as_mut(),
                &xtw,
                &x,
                None,
                1.0,
                get_global_parallelism(),
            );
            faer::linalg::matmul::matmul(
                xtwz.as_mut(),
                &xtw,
                faer::col::from_slice(z.as_slice()),
                None,
                1.0,
                get_global_parallelism(),
            );

            let beta = match xtwx.cholesky(faer::Side::Lower) {
                Ok(chol) => chol.solve(&xtwz),
                Err(_) => {
                    warn!("Using pseudo inverse");
                    ThinSvd::new(xtwx.as_mat_ref()).pseudoinverse() * &xtwz
                },
            };
            let b = beta.try_as_slice().unwrap();
            slopes.as_mut_slice().copy_from_slice(&b[..xs.ncols()]);
            intercept = b[xs.ncols()];
            let eta = &x * beta;
            let eta = eta.try_as_slice().unwrap();
            for (mu, eta) in mu.iter_mut().zip(eta) {
                *mu = F::linkinv(*eta);
            }
            // let old_ll = l;
            // l = ll(mu.as_slice(), ys);
            let mut new_dev = 0.0;
            for i in 0..ys.len() {
                new_dev += F::dev_resids(ys[i], mu[i]);
            }
            delta = (new_dev - dev);
            dev = new_dev;
            if i >= max_iterations {
                warn!("Did not converge after {} iterations", max_iterations);
                converged = false;
                break;
            }
            i += 1;
        }
        if converged {
            debug!("Converged after {} iterations", i);
        }

        let r2 = R2Simd::new(ys, &mu).calculate();
        let adj_r2 = calculate_adj_r2(r2, ys.len(), xs.ncols());

        if should_disable_predicted() {
            mu = Vec::new();
        }

        Self {
            coefs: slopes
                .into_iter()
                .enumerate()
                .map(|(i, coef)| {
                    let std_err = 0.0;
                    let t = 0.0;
                    let p = 0.0;
                    Coef::new(format!("x[{}]", i), coef, std_err, t, p)
                })
                .chain(std::iter::once(Coef::new(
                    "(Intercept)",
                    intercept,
                    0.0,
                    0.0,
                    0.0,
                )))
                .collect(),
            predicted: mu,
            r2,
            adj_r2,
            n: ys.len() as u64,
            m: ncols as u64,
        }
    }

    // UNFINISHED AND NOT FUNCTIONAL, MEANT TO BE THE IRLS IMPLEMENTATION THAT R USES WHICH SEEMS
    // TO USE SIGNIFICANTLY LESS MATMULS
    #[tracing::instrument(skip(xs, ys))]
    pub fn irls_r<F: Family>(
        xs: MatRef<'_, f64>,
        ys: &[f64],
        epsilon: f64,
        max_iterations: usize,
    ) -> Self {
        let ncols = xs.ncols();
        let mut mu = ys.iter().map(|y| F::mu_start(*y)).collect::<Vec<_>>();
        let mut delta = 1.0;
        let mut l = 0.0;
        let mut x = xs.to_owned();
        x.resize_with(
            xs.nrows(),
            xs.ncols() + 1,
            #[inline(always)]
            |_, _| 1.0,
        );
        let mut slopes = vec![0.0; xs.ncols()];
        let mut intercept = 0.0;
        // let mut z = vec![0.0; ys.len()];
        // let mut w = Mat::zeros(ys.len(), ys.len());
        let mut w = vec![0.0; ys.len()];
        // let xt = x.transpose();
        // let mut xtw = Mat::<f64>::zeros(x.ncols(), ys.len());
        // let mut xtwx = Mat::zeros(x.ncols(), x.ncols());
        // let mut xtwz = Col::zeros(x.ncols());
        let mut i = 0;
        let mut converged = true;
        let eta = mu.iter().map(|m| F::linkfun(*m)).collect::<Vec<_>>();
        for i in &mut mu {
            *i = F::linkinv(*i);
        }
        let mut xw_vec = vec![0.0; xs.nrows() * x.ncols()];
        let mut xw: MatRef<f64> =
            faer::mat::from_column_major_slice(xw_vec.as_slice(), xs.nrows(), x.ncols());
        let mut zw_vec = vec![0.0; ys.len()];
        let mut zw_mut =
            unsafe { std::slice::from_raw_parts_mut(zw_vec.as_mut_ptr(), zw_vec.len()) };
        let mut zw = faer::col::from_slice(&zw_vec);
        let mut dev = 0.0;
        for i in 0..ys.len() {
            dev += F::dev_resids(ys[i], mu[i]);
        }
        while delta > epsilon {
            for i in 0..ys.len() {
                let mu_eta = F::mu_eta(eta[i]);
                w[i] = (mu_eta.powi(2) / F::variance(mu[i])).sqrt();
                let z = (eta[i] + (ys[i] - mu[i])) / mu_eta;
                zw_mut[i] = z * w[i];
            }
            // w multiplies each row
            let ncols = x.ncols();
            let nrows = x.nrows();
            (0..ys.len()).into_par_iter().for_each(|r| {
                let xw_mut = unsafe {
                    std::slice::from_raw_parts_mut(xw_vec.as_ptr().cast_mut(), xw_vec.len())
                };
                for c in 0..ncols {
                    xw_mut[c * nrows + r] = x[(r, c)] * w[r];
                }
            });

            let beta = xw.qr().solve_lstsq(&zw);
            let b = beta.try_as_slice().unwrap();
            slopes.as_mut_slice().copy_from_slice(&b[..xs.ncols()]);
            intercept = b[xs.ncols()];
            let eta = &x * beta;
            let eta = eta.try_as_slice().unwrap();
            for (mu, eta) in mu.iter_mut().zip(eta) {
                *mu = F::linkinv(*eta);
            }
            let mut new_dev = 0.0;
            for i in 0..ys.len() {
                new_dev += F::dev_resids(ys[i], mu[i]);
            }
            delta = (new_dev - dev);
            dev = new_dev;
            // let old_ll = l;
            // l = ll(mu.as_slice(), ys);
            // delta = (l - old_ll).abs();
            if i >= max_iterations {
                warn!("Did not converge after {} iterations", max_iterations);
                converged = false;
                break;
            }
            i += 1;
        }
        if converged {
            debug!("Converged after {} iterations", i);
        }

        let r2 = R2Simd::new(ys, &mu).calculate();
        let adj_r2 = calculate_adj_r2(r2, ys.len(), xs.ncols());

        if should_disable_predicted() {
            mu = Vec::new();
        }

        Self {
            coefs: slopes
                .into_iter()
                .enumerate()
                .map(|(i, coef)| {
                    let std_err = 0.0;
                    let t = 0.0;
                    let p = 0.0;
                    Coef::new(format!("x[{}]", i), coef, std_err, t, p)
                })
                .chain(std::iter::once(Coef::new(
                    "(Intercept)",
                    intercept,
                    0.0,
                    0.0,
                    0.0,
                )))
                .collect(),
            predicted: mu,
            r2,
            adj_r2,
            n: ys.len() as u64,
            m: ncols as u64,
        }
    }

    #[tracing::instrument(skip(xs, ys))]
    pub fn newton_raphson<F: Family>(
        xs: MatRef<'_, f64>,
        ys: &[f64],
        epsilon: f64,
        max_iterations: usize,
    ) -> Self {
        let ncols = xs.ncols();
        let mut x = xs.to_owned();
        x.resize_with(
            xs.nrows(),
            xs.ncols() + 1,
            #[inline(always)]
            |_, _| 1.0,
        );
        let mut beta = vec![0.0; x.ncols()];
        let mut mu = ys.iter().map(|y| F::mu_start(*y)).collect::<Vec<_>>();
        // let mut w = Mat::zeros(ys.len(), ys.len());
        let mut w = vec![0.0; ys.len()];
        let mut linear_predictor = Col::zeros(ys.len());
        let mut ys_sub_mu = vec![0.0; ys.len()];
        let xt = x.transpose();
        let mut jacobian = Col::zeros(x.ncols());
        let mut xtw = Mat::<f64>::zeros(x.ncols(), ys.len());
        let mut hessian = Mat::zeros(x.ncols(), x.ncols());
        let mut converged = false;
        for i in 0..max_iterations {
            faer::linalg::matmul::matmul(
                linear_predictor.as_mut(),
                &x,
                faer::col::from_slice(beta.as_slice()),
                None,
                1.0,
                get_global_parallelism(),
            );
            for (mu, l) in mu.iter_mut().zip(linear_predictor.try_as_slice().unwrap()) {
                *mu = F::linkinv(*l);
            }
            for (i, mu) in mu.iter().enumerate() {
                w[i] = mu * (1.0 - mu);
            }
            for (i, (mu, y)) in mu.iter().zip(ys).enumerate() {
                ys_sub_mu[i] = *y - mu;
            }

            faer::linalg::matmul::matmul(
                jacobian.as_mut(),
                xt,
                faer::col::from_slice(ys_sub_mu.as_slice()),
                None,
                1.0,
                get_global_parallelism(),
            );
            xtw.par_col_chunks_mut(1)
                .zip(w.par_iter())
                .enumerate()
                .for_each(|(j, (mut xtw, w))| {
                    xtw.col_iter_mut().for_each(|x| {
                        x.iter_mut()
                            .enumerate()
                            .for_each(|(i, x)| *x = xt[(i, j)] * w)
                    })
                });
            faer::linalg::matmul::matmul(
                hessian.as_mut(),
                &xtw,
                &x,
                None,
                1.0,
                get_global_parallelism(),
            );

            let beta_new = faer::col::from_slice(beta.as_slice())
                + match hessian.cholesky(faer::Side::Lower) {
                    Ok(chol) => chol.solve(&jacobian),
                    Err(_) => {
                        warn!("Using pseudo inverse");
                        ThinSvd::new(hessian.as_mat_ref()).pseudoinverse() * &jacobian
                    },
                };

            if (&beta_new - faer::col::from_slice(beta.as_slice())).norm_l1() < epsilon {
                debug!("Converged after {} iterations", i);
                converged = true;
                beta.copy_from_slice(beta_new.try_as_slice().unwrap());
                break;
            }
            beta.copy_from_slice(beta_new.try_as_slice().unwrap());
        }
        if !converged {
            warn!("Did not converge after {} iterations", max_iterations);
        }
        let r2 = R2Simd::new(ys, &mu).calculate();
        let adj_r2 = calculate_adj_r2(r2, ys.len(), xs.ncols());

        let predicted = if should_disable_predicted() {
            Vec::new()
        } else {
            (&x * faer::col::from_slice(beta.as_slice()))
                .try_as_slice()
                .unwrap()
                .iter()
                .map(|x| F::linkinv(*x))
                .collect()
        };

        Self {
            coefs: (0..ncols)
                .map(|i| {
                    let coef = beta[i];
                    let std_err = 0.0;
                    let t = 0.0;
                    let p = 0.0;
                    Coef::new(format!("x[{}]", i), coef, std_err, t, p)
                })
                .chain(std::iter::once(Coef::new(
                    "(Intercept)",
                    beta[ncols],
                    0.0,
                    0.0,
                    0.0,
                )))
                .collect(),
            predicted,
            r2,
            adj_r2,
            n: ys.len() as u64,
            m: x.ncols() as u64,
        }
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

    pub fn predict<F: Family>(&self, x: &[f64]) -> f64 {
        let mut v = self.intercept().coef();
        let slopes = self.slopes();
        for i in 0..self.slopes().len() {
            v += slopes[i].coef() * x[i];
        }
        F::linkinv(v)
    }
}

#[inline(always)]
fn ll(p: &[f64], y: &[f64]) -> f64 {
    p.iter()
        .zip(y)
        .map(|(p, y)| y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        .sum()
}

pub trait Link {
    fn fun(mu: f64) -> f64;
    fn inv(eta: f64) -> f64;
}

pub trait Family {
    fn linkfun(mu: f64) -> f64;
    fn linkinv(eta: f64) -> f64;
    fn variance(mu: f64) -> f64;
    fn mu_eta(eta: f64) -> f64;
    fn dev_resids(y: f64, mu: f64) -> f64;
    fn mu_start(y: f64) -> f64;
}

#[inline(always)]
fn y_log_y(y: f64, mu: f64) -> f64 {
    if y == 0.0 {
        0.0
    } else {
        y * (y / mu).ln()
    }
}

pub mod family {
    use crate::{dcauchy, dnorm, pnorm, qcauchy, qnorm};

    use super::*;

    /// Gaussian family with identity link function
    pub struct GaussianIdentity;
    impl Family for GaussianIdentity {
        fn linkfun(mu: f64) -> f64 {
            mu
        }

        fn linkinv(eta: f64) -> f64 {
            eta
        }

        fn variance(_mu: f64) -> f64 {
            1.0
        }

        fn mu_eta(_eta: f64) -> f64 {
            1.0
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            (y - mu).powi(2)
        }

        fn mu_start(y: f64) -> f64 {
            y
        }
    }

    /// Gaussian family with log link function
    pub struct GaussianLog;
    impl Family for GaussianLog {
        fn linkfun(mu: f64) -> f64 {
            mu.ln()
        }

        fn linkinv(eta: f64) -> f64 {
            eta.exp().max(f64::EPSILON)
        }

        fn variance(_mu: f64) -> f64 {
            1.0
        }

        fn mu_eta(eta: f64) -> f64 {
            eta.exp().max(f64::EPSILON)
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            (y - mu).powi(2)
        }

        fn mu_start(y: f64) -> f64 {
            y
        }
    }

    /// Gaussian family with inverse link function
    pub struct GaussianInverse;
    impl Family for GaussianInverse {
        fn linkfun(mu: f64) -> f64 {
            1.0 / mu
        }

        fn linkinv(eta: f64) -> f64 {
            1.0 / eta
        }

        fn variance(mu: f64) -> f64 {
            1.0
        }

        fn mu_eta(eta: f64) -> f64 {
            -1.0 / eta.powi(2)
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            (y - mu).powi(2)
        }

        fn mu_start(y: f64) -> f64 {
            y
        }
    }

    /// Binomial family with logit link function
    pub struct BinomialLogit;
    impl Family for BinomialLogit {
        fn linkfun(mu: f64) -> f64 {
            (mu / (1.0 - mu)).ln()
        }

        fn linkinv(eta: f64) -> f64 {
            1.0 / (1.0 + (-eta).exp())
        }

        fn variance(mu: f64) -> f64 {
            mu * (1.0 - mu)
        }

        fn mu_eta(eta: f64) -> f64 {
            1.0 / (eta * (1.0 - eta))
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            2.0 * (y_log_y(y, mu) + y_log_y(1.0 - y, 1.0 - mu))
        }

        fn mu_start(y: f64) -> f64 {
            (y + 0.5) / 2.0
        }
    }

    /// Binomial family with probit link function
    pub struct BinomialProbit;
    impl Family for BinomialProbit {
        fn linkfun(mu: f64) -> f64 {
            qnorm(mu)
        }

        fn linkinv(eta: f64) -> f64 {
            let thresh = qnorm(f64::EPSILON);
            pnorm(eta.clamp(-thresh, thresh))
        }

        fn variance(mu: f64) -> f64 {
            mu * (1.0 - mu)
        }

        fn mu_eta(eta: f64) -> f64 {
            dnorm(eta).max(f64::EPSILON)
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            2.0 * (y_log_y(y, mu) + y_log_y(1.0 - y, 1.0 - mu))
        }

        fn mu_start(y: f64) -> f64 {
            (y + 0.5) / 2.0
        }
    }

    /// Binomial family with cauchit link function
    pub struct BinomialCauchit;
    impl Family for BinomialCauchit {
        fn linkfun(mu: f64) -> f64 {
            qcauchy(mu)
        }

        fn linkinv(eta: f64) -> f64 {
            let thresh = qcauchy(f64::EPSILON);
            qcauchy(eta.clamp(-thresh, thresh))
        }

        fn variance(mu: f64) -> f64 {
            mu * (1.0 - mu)
        }

        fn mu_eta(eta: f64) -> f64 {
            dcauchy(eta).max(f64::EPSILON)
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            2.0 * (y_log_y(y, mu) + y_log_y(1.0 - y, 1.0 - mu))
        }

        fn mu_start(y: f64) -> f64 {
            (y + 0.5) / 2.0
        }
    }

    /// Binomial family with log link function
    pub struct BinomialLog;
    impl Family for BinomialLog {
        fn linkfun(mu: f64) -> f64 {
            mu.ln()
        }

        fn linkinv(eta: f64) -> f64 {
            eta.exp().max(f64::EPSILON)
        }

        fn variance(mu: f64) -> f64 {
            mu * (1.0 - mu)
        }

        fn mu_eta(eta: f64) -> f64 {
            eta.exp().max(f64::EPSILON)
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            2.0 * (y_log_y(y, mu) + y_log_y(1.0 - y, 1.0 - mu))
        }

        fn mu_start(y: f64) -> f64 {
            (y + 0.5) / 2.0
        }
    }

    /// Binomial family with complementary log-log link function
    pub struct BinomialComplementaryLogLog;
    impl Family for BinomialComplementaryLogLog {
        fn linkfun(mu: f64) -> f64 {
            (-(1.0 - mu).ln()).ln()
        }

        fn linkinv(eta: f64) -> f64 {
            (-(-eta).exp())
                .exp_m1()
                .clamp(f64::EPSILON, 1.0 - f64::EPSILON)
        }

        fn variance(mu: f64) -> f64 {
            mu * (1.0 - mu)
        }

        fn mu_eta(eta: f64) -> f64 {
            let eta = eta.min(700.0);
            (eta.exp() * (-eta.exp()).exp()).max(f64::EPSILON)
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            2.0 * (y_log_y(y, mu) + y_log_y(1.0 - y, 1.0 - mu))
        }

        fn mu_start(y: f64) -> f64 {
            (y + 0.5) / 2.0
        }
    }

    /// Gamma family with inverse link function
    pub struct GammaInverse;
    impl Family for GammaInverse {
        fn linkfun(mu: f64) -> f64 {
            1.0 / mu
        }

        fn linkinv(eta: f64) -> f64 {
            1.0 / eta
        }

        fn variance(mu: f64) -> f64 {
            mu.powi(2)
        }

        fn mu_eta(eta: f64) -> f64 {
            -1.0 / eta.powi(2)
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            -2.0 * ((if y == 0.0 { 1.0 } else { y / mu }).ln() - ((y - mu) / mu))
        }

        fn mu_start(y: f64) -> f64 {
            y
        }
    }

    /// Gamma family with identity link function
    pub struct GammaIdentity;
    impl Family for GammaIdentity {
        fn linkfun(mu: f64) -> f64 {
            mu
        }

        fn linkinv(eta: f64) -> f64 {
            eta
        }

        fn variance(mu: f64) -> f64 {
            mu.powi(2)
        }

        fn mu_eta(_eta: f64) -> f64 {
            1.0
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            -2.0 * ((if y == 0.0 { 1.0 } else { y / mu }).ln() - ((y - mu) / mu))
        }

        fn mu_start(y: f64) -> f64 {
            y
        }
    }

    /// Gamma family with log link function
    pub struct GammaLog;
    impl Family for GammaLog {
        fn linkfun(mu: f64) -> f64 {
            mu.ln()
        }

        fn linkinv(eta: f64) -> f64 {
            eta.exp().max(f64::EPSILON)
        }

        fn variance(mu: f64) -> f64 {
            mu.powi(2)
        }

        fn mu_eta(eta: f64) -> f64 {
            eta.exp().max(f64::EPSILON)
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            -2.0 * ((if y == 0.0 { 1.0 } else { y / mu }).ln() - ((y - mu) / mu))
        }

        fn mu_start(y: f64) -> f64 {
            y
        }
    }

    /// Poisson family with log link function
    pub struct PoissonLog;
    impl Family for PoissonLog {
        fn linkfun(mu: f64) -> f64 {
            mu.ln()
        }

        fn linkinv(eta: f64) -> f64 {
            eta.exp().max(f64::EPSILON)
        }

        fn variance(mu: f64) -> f64 {
            mu
        }

        fn mu_eta(eta: f64) -> f64 {
            eta.exp().max(f64::EPSILON)
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            2.0 * if y == 0.0 {
                mu
            } else {
                y * (y / mu).ln() - (y - mu)
            }
        }

        fn mu_start(y: f64) -> f64 {
            y + 0.1
        }
    }

    /// Poisson family with identity link function
    pub struct PoissonIdentity;
    impl Family for PoissonIdentity {
        fn linkfun(mu: f64) -> f64 {
            mu
        }

        fn linkinv(eta: f64) -> f64 {
            eta
        }

        fn variance(mu: f64) -> f64 {
            mu
        }

        fn mu_eta(_eta: f64) -> f64 {
            1.0
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            2.0 * if y == 0.0 {
                mu
            } else {
                y * (y / mu).ln() - (y - mu)
            }
        }

        fn mu_start(y: f64) -> f64 {
            y + 0.1
        }
    }

    /// Poisson family with sqrt link function
    pub struct PoissonSqrt;
    impl Family for PoissonSqrt {
        fn linkfun(mu: f64) -> f64 {
            mu.sqrt()
        }

        fn linkinv(eta: f64) -> f64 {
            eta.powi(2)
        }

        fn variance(mu: f64) -> f64 {
            mu
        }

        fn mu_eta(eta: f64) -> f64 {
            2.0 * eta
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            2.0 * if y == 0.0 {
                mu
            } else {
                y * (y / mu).ln() - (y - mu)
            }
        }

        fn mu_start(y: f64) -> f64 {
            y + 0.1
        }
    }

    /// Inverse Gaussian family with 1/mu^2 link function
    pub struct InverseGaussian;
    impl Family for InverseGaussian {
        fn linkfun(mu: f64) -> f64 {
            1.0 / mu.powi(2)
        }

        fn linkinv(eta: f64) -> f64 {
            1.0 / eta.sqrt()
        }

        fn variance(mu: f64) -> f64 {
            mu.powi(3)
        }

        fn mu_eta(eta: f64) -> f64 {
            -1.0 / (2.0 * eta.powf(1.5))
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            (y - mu).powi(2) / (y * mu.powi(2))
        }

        fn mu_start(y: f64) -> f64 {
            y
        }
    }

    /// Inverse Gaussian family with inverse link function
    pub struct InverseGaussianInverse;
    impl Family for InverseGaussianInverse {
        fn linkfun(mu: f64) -> f64 {
            1.0 / mu
        }

        fn linkinv(eta: f64) -> f64 {
            1.0 / eta
        }

        fn variance(mu: f64) -> f64 {
            mu.powi(3)
        }

        fn mu_eta(eta: f64) -> f64 {
            -1.0 / eta.powi(2)
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            (y - mu).powi(2) / (y * mu.powi(2))
        }

        fn mu_start(y: f64) -> f64 {
            y
        }
    }

    /// Inverse Gaussian family with identity link function
    pub struct InverseGaussianIdentity;
    impl Family for InverseGaussianIdentity {
        fn linkfun(mu: f64) -> f64 {
            mu
        }

        fn linkinv(eta: f64) -> f64 {
            eta
        }

        fn variance(mu: f64) -> f64 {
            mu.powi(3)
        }

        fn mu_eta(_eta: f64) -> f64 {
            1.0
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            (y - mu).powi(2) / (y * mu.powi(2))
        }

        fn mu_start(y: f64) -> f64 {
            y
        }
    }

    /// Inverse Gaussian family with log link function
    pub struct InverseGaussianLog;
    impl Family for InverseGaussianLog {
        fn linkfun(mu: f64) -> f64 {
            mu.ln()
        }

        fn linkinv(eta: f64) -> f64 {
            eta.exp().max(f64::EPSILON)
        }

        fn variance(mu: f64) -> f64 {
            mu.powi(3)
        }

        fn mu_eta(eta: f64) -> f64 {
            eta.exp().max(f64::EPSILON)
        }

        fn dev_resids(y: f64, mu: f64) -> f64 {
            (y - mu).powi(2) / (y * mu.powi(2))
        }

        fn mu_start(y: f64) -> f64 {
            y
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand_distr::{num_traits::float, Distribution};
    use test_log::test;

    macro_rules! assert_float_eq {
        ($a:expr, $b:expr, $tol:expr) => {
            assert!(($a - $b).abs() < $tol, "{:.22} != {:.22}", $a, $b);
        };
    }

    macro_rules! float_eq {
        ($a:expr, $b:expr) => {
            assert_float_eq!($a, $b, 1e-10);
        };
    }

    #[test]
    fn test_glm_irls() {
        let nrows = 50;
        let xs = faer::mat::from_column_major_slice(XS.as_slice(), nrows, 4);
        let m = Glm::irls::<family::BinomialLogit>(xs, YS.as_slice(), 1e-10, 25);
        float_eq!(m.intercept().coef(), -0.10480279218218244152716);
        float_eq!(m.slopes()[0].coef(), 0.06970776481172229199768);
        float_eq!(m.slopes()[1].coef(), 0.31341357257259599977672);
        float_eq!(m.slopes()[2].coef(), -0.52734374471258893546377);
        float_eq!(m.slopes()[3].coef(), 0.05905679790685783303594);
    }

    // #[test]
    // fn test_glm_irls_r() {
    //     let nrows = 50;
    //     let xs = faer::mat::from_column_major_slice(XS.as_slice(), nrows, 4);
    //     let m = Glm::irls_r::<family::BinomialLogit>(xs, YS.as_slice(), 1e-100, 25);
    //     float_eq!(m.intercept().coef(), -0.10480279218218244152716);
    //     float_eq!(m.slopes()[0].coef(), 0.06970776481172229199768);
    //     float_eq!(m.slopes()[1].coef(), 0.31341357257259599977672);
    //     float_eq!(m.slopes()[2].coef(), -0.52734374471258893546377);
    //     float_eq!(m.slopes()[3].coef(), 0.05905679790685783303594);
    // }

    #[test]
    fn test_glm_newton_raphson() {
        let nrows = 50;
        let xs = faer::mat::from_column_major_slice(XS.as_slice(), nrows, 4);
        let m = Glm::newton_raphson::<family::BinomialLogit>(xs, YS.as_slice(), 1e-10, 25);
        float_eq!(m.intercept().coef(), -0.10480279218218244152716);
        float_eq!(m.slopes()[0].coef(), 0.06970776481172229199768);
        float_eq!(m.slopes()[1].coef(), 0.31341357257259599977672);
        float_eq!(m.slopes()[2].coef(), -0.52734374471258893546377);
        float_eq!(m.slopes()[3].coef(), 0.05905679790685783303594);
    }

    #[test]
    fn test_glm_irls_predict() {
        let nrows = 50;
        let xs = faer::mat::from_column_major_slice(XS.as_slice(), nrows, 4);
        let m = Glm::irls::<family::BinomialLogit>(xs, YS.as_slice(), 1e-10, 25);
        float_eq!(
            m.predict::<family::BinomialLogit>(&[
                -0.7639264113390733523801,
                0.5045213234835747018181,
                -0.8257110454007502431395,
                1.1439926598276572988766
            ]),
            m.predicted()[0]
        );
    }

    #[test]
    fn test_glm_newton_raphson_predict() {
        let nrows = 50;
        let xs = faer::mat::from_column_major_slice(XS.as_slice(), nrows, 4);
        let m = Glm::newton_raphson::<family::BinomialLogit>(xs, YS.as_slice(), 1e-10, 25);
        float_eq!(
            m.predict::<family::BinomialLogit>(&[
                -0.7639264113390733523801,
                0.5045213234835747018181,
                -0.8257110454007502431395,
                1.1439926598276572988766
            ]),
            m.predicted()[0]
        );
    }

    #[rustfmt::skip]
    pub mod data {
        pub const XS: [f64; 200] = [-0.7639264113390734, -1.122168587439642, 1.3180190958562485, -1.1254934217874812, 2.019258865335717, 2.0405256792579354, 0.9151298525247888, 0.2859614374345147, 1.3355117857265448, 3.267764698168226, -1.3646094340798454, 0.14546463681676042, 0.1961781805014692, 0.34604731161419566, 0.2020400570545031, 1.1434862388781128, -1.9746744214828982, 1.6196726386583102, -1.1126042782603696, -0.5272037808142903, -1.8273071501576414, -1.0738179518221682, 1.477345633347955, -0.1204968532222195, 1.035770404158358, -1.3852168629078614, 0.6918608101348908, 1.2390422504049932, 0.32570620312614157, -0.32298724802109147, -0.6411112089349145, 1.3417031192578786, 0.8482172093549543, -0.7854830180864644, -0.2611358207803115, 0.2177597430118682, 0.4830736219034198, 0.6511955563288045, -0.9488112471833973, -0.5811431264026651, 0.18809370741356152, -0.11060383427343971, 0.23363506461376107, 2.10151698323362, 0.34108703556098036, 0.7703131529435837, -0.8383020596886386, -1.1884479892602393, -1.163038274876521, 0.1143167182239821, 0.5045213234835747, 0.45250846129217254, -0.6438357170493676, 0.13015140239485662, 0.07810955905291025, 1.2606619091136653, 1.4471218678893434, 0.2825413525242349, -0.29419429532384794, 0.743024939567625, -2.139445335067594, -1.7006001924447711, -0.19556993750969812, 0.4858778591333698, 1.2455557881401311, -0.30389318880473826, -0.7331983579587671, -0.34825348655629973, -1.3807719195223827, 0.31516446297499534, 1.8874506152347637, 0.8633768749405036, 1.6641143955791653, 0.08544640361943065, 0.6560949712192261, 0.8898345872873682, 0.5773637670422586, 0.7553864718047484, -0.08915687483881272, 0.8446766049871155, -0.32643094122205596, -1.059255788520573, 0.43778160892524376, 1.3205418952843777, -0.22135300807902908, 2.0050541186384563, -0.9882327646492486, -1.802461192301427, -0.96216340261943, -0.25680852143652616, 0.565744918056686, -1.2969068146720475, 0.18906115237200782, -0.5338737967533657, -0.26717513585464314, -0.5281695997392984, 0.5161418446187902, 0.9574603375889652, 0.6516956931630822, 0.35644199867900356, -0.8257110454007502, 0.8302169053857459, -0.2530830581836079, -0.734137934523698, -1.0901030823574396, 0.19355227208532028, 0.6325183820993368, 1.4662112593540295, -0.6188815996377673, 0.36662093340313534, -1.4497552889894483, -2.5064721455037398, -0.4960939926633383, 0.5190578413985156, -0.7100009342687456, 0.19200633251133847, 0.9876909512244467, 0.5295617820481348, -0.55796000317311, -0.04575111096778109, 1.9444983917146623, -0.5476338494084949, 0.6692476340840899, -0.3019418843416522, -0.07137121621252646, 1.7575717018993262, -1.7098232511507732, 1.3735728026032181, 0.5418886573314807, -0.6248465086585353, 0.6700207754318339, -0.007119067650844111, 1.1125864137777832, -0.21524887836806117, 0.019055364114858222, 1.6737514809117173, 0.05564712465556579, -0.5148210084727287, 0.9745281236284951, 0.6626479934841107, 0.9699653418147671, -0.3401798371976365, 1.5213960048586632, 1.2571010251655161, 1.536435904978547, 0.8376197731021151, 0.2834518786501805, -0.09048884964677842, 1.0293645224352295, -0.7801902520839322, 1.1439926598276573, -0.6233362529099903, 1.5962203784391535, -1.7625817506028694, 0.23757835483052037, 0.4546390409292788, -1.1833902915977506, -1.6596936934949689, 0.12350853258666475, 0.9962477050534115, 0.5529802569997323, 0.5705411800420922, -0.21422937459543162, -1.968436580058151, 0.36714288957415353, -1.5735282541664515, 0.19409220579851277, 0.5546008065338537, -1.6545355137247337, -0.3615864146673201, -0.9380250815907384, -0.5225091669154701, 0.47065146623824977, 0.3636738507963835, 0.37206175077482867, 0.35392305238002153, 0.013516510805018615, 0.2921958335896905, -0.09437113268679911, -1.7217683947661668, -0.7742387761491311, -0.5639597306581139, -0.8393984149243039, -0.04881063156536773, 2.4239027603424557, 1.2351183769382545, -1.8491914942121581, -0.3870160427570883, -0.17127086584621526, -0.26938349217000523, 0.02579956695052189, -0.2762535498877319, 0.9626503198828471, 1.052732068467081, -0.7552241706932443, 0.7593978364952113, 0.5761591189973172, 0.07539720472308677, -1.5453182804788623, 1.8092295355913308];
        pub const YS: [f64; 50] = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    }
    pub use data::*;
}

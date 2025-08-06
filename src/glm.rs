use std::mem::MaybeUninit;

use faer::{
    diag::DiagRef,
    get_global_parallelism,
    linalg::solvers::{DenseSolveCore, Solve, SolveLstsqCore},
    mat::{AsMatMut, AsMatRef},
    prelude::SolveLstsq,
    Col, ColRef, Mat, MatRef,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use tracing::{debug, trace, warn};

use crate::{calculate_adj_r2, coef::Coef, pnorm, should_disable_predicted, R2Simd};

#[derive(Debug, Clone)]
pub struct Glm {
    // the last element is the intercept
    coefs: Vec<Coef>,
    predicted: Vec<f64>,
    r2: f64,
    adj_r2: f64,
    r2_tjur: f64,
    n: u64,
    m: u64,
    aic: f64,
    weights: Vec<f64>,
    add_intercept: bool,
}

impl Glm {
    #[tracing::instrument(skip(xs, ys, colnames))]
    pub fn irls<F: Family>(
        xs: MatRef<'_, f64>,
        ys: &[f64],
        epsilon: f64,
        max_iterations: usize,
        add_intercept: bool,
        firth: bool,
        colnames: Option<&[String]>,
    ) -> Self {
        let ncols = xs.ncols();
        let mut mu = ys.iter().map(|y| F::mu_start(*y)).collect::<Vec<_>>();
        let mut delta = 1.0;
        let mut l = 0.0;
        let mut x = xs.to_owned();
        if add_intercept {
            x.resize_with(
                xs.nrows(),
                xs.ncols() + 1,
                #[inline(always)]
                |_, _| 1.0,
            );
        }
        let mut slopes = vec![0.0; xs.ncols()];
        let mut intercept = 0.0;
        let mut z = vec![0.0; ys.len()];
        let mut w = vec![0.0; ys.len()];
        let xt = x.transpose();
        let mut xtw = Mat::<f64>::zeros(x.ncols(), ys.len());
        let mut xtwx = Mat::zeros(x.ncols(), x.ncols());
        // each element of xtwz is the score function for the corresponding parameter
        let mut xtwz = Col::zeros(x.ncols());
        let mut iter = 0;
        let mut converged = true;

        // cholesky scratch stuff
        // let par = faer::Par::Seq;
        let par = get_global_parallelism();
        let n = xtwx.nrows();
        let mut chol_l = Mat::zeros(n, n);
        let mut chol_mem = faer::dyn_stack::MemBuffer::new(
            faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<f64>(
                n,
                faer::Par::Seq,
                Default::default(),
            ),
        );
        let chol_stack = faer::dyn_stack::MemStack::new(&mut chol_mem);

        let mut inv_mem = faer::dyn_stack::MemBuffer::new(
            faer::linalg::cholesky::llt::inverse::inverse_scratch::<f64>(n, par),
        );
        let inv_stack = faer::dyn_stack::MemStack::new(&mut inv_mem);

        // For Firth penalization
        let mut h_diag = if firth {
            vec![0.0; ys.len()]
        } else {
            Vec::new()
        };
        let mut xtwx_inv = Mat::zeros(n, n);
        let mut x_xtwx_inv = if firth {
            Mat::zeros(x.nrows(), n)
        } else {
            Mat::zeros(0, 0)
        };
        let mut firth_adjustment: Col<f64> = if firth {
            Col::zeros(x.ncols())
        } else {
            Col::zeros(0)
        };

        while delta > epsilon {
            for (i, mu) in mu.iter().enumerate() {
                // w[(i, i)] = 1.0 / (F::mu_eta(*mu).powi(2) * F::variance(*mu));
                w[i] = 1.0 / (F::mu_eta(*mu).powi(2) * F::variance(*mu));
            }

            // xtw.par_col_chunks_mut(1)
            //     .zip(w.par_iter())
            //     .enumerate()
            //     .for_each(|(j, (mut xtw, w))| {
            //         xtw.col_iter_mut().for_each(|x| {
            //             x.iter_mut()
            //                 .enumerate()
            //                 .for_each(|(i, x)| *x = xt[(i, j)] * w)
            //         })
            //     });
            (0..xtw.ncols()).for_each(|c| {
                let w = w[c];
                let xtw = xtw.col(c).try_as_col_major().unwrap().as_slice();
                let mut xtw =
                    unsafe { std::slice::from_raw_parts_mut(xtw.as_ptr().cast_mut(), xtw.len()) };
                let xt = xt.col(c);
                for r in 0..xtw.len() {
                    xtw[r] = xt[r] * w;
                }
            });
            // faer::linalg::matmul::matmul(
            //     xtw.as_mut(),
            //     faer::Accum::Replace,
            //     xt,
            //     &w,
            //     1.0,
            //     get_global_parallelism(),
            // );
            // faer::linalg::matmul::triangular::matmul(
            //     xtw.as_mut(),
            //     faer::linalg::matmul::triangular::BlockStructure::Rectangular,
            //     faer::Accum::Replace,
            //     xt,
            //     faer::linalg::matmul::triangular::BlockStructure::Rectangular,
            //     &w,
            //     faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
            //     1.0,
            //     get_global_parallelism(),
            // );
            faer::linalg::matmul::matmul(xtwx.as_mut(), faer::Accum::Replace, &xtw, &x, 1.0, par);

            // we inline this to use faer::Par::Seq
            let llt = {
                chol_l.as_mat_mut().copy_from_triangular_lower(&xtwx);

                let res = faer::linalg::cholesky::llt::factor::cholesky_in_place(
                    chol_l.as_mut(),
                    Default::default(),
                    par,
                    chol_stack,
                    Default::default(),
                );
                match res {
                    Ok(_) => {
                        faer::zip!(&mut chol_l).for_each_triangular_upper(
                            faer::linalg::zip::Diag::Skip,
                            |faer::unzip!(x)| {
                                *x = 0.0;
                            },
                        );
                        Ok(())
                    },
                    Err(e) => Err(e),
                }
            };

            // Apply Firth penalization if enabled
            if firth {
                // Calculate the inverse of xtwx (information matrix)
                match llt {
                    Ok(_) => {
                        // Create identity matrix
                        xtwx_inv.fill(0.0);
                        for j in 0..n {
                            xtwx_inv[(j, j)] = 1.0;
                        }

                        // Solve for inverse using Cholesky decomposition

                        faer::linalg::cholesky::llt::inverse::inverse(
                            xtwx_inv.as_mut(),
                            chol_l.as_ref(),
                            par,
                            inv_stack,
                        );

                        // make self adjoint (symmetric across the diagonal)
                        for j in 0..xtwx_inv.nrows() {
                            for i in 0..j {
                                xtwx_inv[(i, j)] = xtwx_inv[(j, i)];
                            }
                        }

                        faer::linalg::matmul::matmul(
                            x_xtwx_inv.as_mut(),
                            faer::Accum::Replace,
                            &x,
                            xtwx_inv.as_ref(),
                            1.0,
                            par,
                        );

                        // Calculate hat matrix diagonal elements (leverage values)
                        // We have x_xtwx_inv as X * (X'WX)^(-1)
                        // Now we need to calculate the diagonal of H = W * X * (X'WX)^(-1) * X'
                        for i in 0..ys.len() {
                            let mut h_i = 0.0;
                            for j in 0..n {
                                h_i += x_xtwx_inv[(i, j)] * xt[(j, i)];
                            }
                            h_diag[i] = w[i] * h_i;
                        }

                        // Calculate working variable
                        for i in 0..ys.len() {
                            let adj = (F::k3i_k2i(mu[i]) * h_diag[i]) / 2.0;
                            z[i] = F::linkfun(mu[i]) + (ys[i] + adj - mu[i]) * F::mu_eta(mu[i]);
                        }
                    },
                    Err(_) => {
                        warn!("Could not compute Firth adjustment due to matrix inversion failure");
                    },
                }
            } else {
                for ((z, mu), y) in z.iter_mut().zip(mu.iter()).zip(ys) {
                    *z = F::linkfun(*mu) + (y - mu) * F::mu_eta(*mu);
                }
            }
            faer::linalg::matmul::matmul(
                xtwz.as_mut(),
                faer::Accum::Replace,
                &xtw,
                ColRef::from_slice(z.as_slice()),
                1.0,
                par,
            );

            let beta = match llt {
                Ok(_) => {
                    faer::linalg::cholesky::llt::solve::solve_in_place(
                        chol_l.as_mat_ref(),
                        xtwz.as_mat_mut(),
                        par,
                        chol_stack,
                    );
                    xtwz.clone()
                },
                Err(_) => {
                    warn!("Using pseudo inverse");
                    xtwx.as_mat_ref()
                        .thin_svd()
                        .expect("could not compute thin SVD for pseudoinverse")
                        .pseudoinverse()
                        * &xtwz
                },
            };
            let b = beta
                .try_as_col_major()
                .expect("could not get slice")
                .as_slice();
            if add_intercept {
                slopes.as_mut_slice().copy_from_slice(&b[..xs.ncols()]);
                intercept = b[xs.ncols()];
            } else {
                slopes.as_mut_slice().copy_from_slice(b);
            }
            let eta = &x * beta;
            let eta = eta
                .try_as_col_major()
                .expect("could not get slice")
                .as_slice();
            for (mu, eta) in mu.iter_mut().zip(eta) {
                *mu = F::linkinv(*eta);
            }
            // let mut new_dev = 0.0;
            // for i in 0..ys.len() {
            //     new_dev += F::dev_resids(ys[i], mu[i]);
            // }
            // delta = (new_dev - dev).abs();
            // dev = new_dev;
            let old_ll = l;
            l = if firth {
                // Add the log determinant of the information matrix to the log-likelihood
                let log_det = match llt {
                    Ok(_) => {
                        // For Cholesky decomposition, det = prod(diag(L))^2
                        let mut det = 1.0;
                        for j in 0..n {
                            det *= chol_l[(j, j)];
                        }
                        2.0 * det.ln() // log(det) = 2 * sum(log(diag(L)))
                    },
                    Err(_) => 0.0, // In case of failure, don't adjust
                };
                ll(mu.as_slice(), ys) + 0.5 * log_det
            } else {
                ll(mu.as_slice(), ys)
            };
            delta = (l - old_ll).abs();
            if iter >= max_iterations {
                warn!("Did not converge after {} iterations", max_iterations);
                converged = false;
                break;
            }
            iter += 1;
            trace!(delta, slopes = ?slopes, intercept, "Iteration {}", iter);
        }
        if converged {
            debug!("Converged after {} iterations", iter);
        }

        // firth penalization already has the fisher information matrix so we don't need to
        // compute it again
        if !firth {
            // Create identity matrix
            xtwx_inv.fill(0.0);
            for j in 0..n {
                xtwx_inv[(j, j)] = 1.0;
            }

            // Solve for inverse using Cholesky decomposition

            faer::linalg::cholesky::llt::inverse::inverse(
                xtwx_inv.as_mut(),
                chol_l.as_ref(),
                par,
                inv_stack,
            );

            // make self adjoint (symmetric across the diagonal)
            for j in 0..xtwx_inv.nrows() {
                for i in 0..j {
                    xtwx_inv[(i, j)] = xtwx_inv[(j, i)];
                }
            }
        }

        let r2 = crate::r2(ys, &mu);
        let adj_r2 = calculate_adj_r2(r2, ys.len(), xs.ncols());

        let r2_tjur = crate::compute_r2_tjur(ys, &mu);
        let rank = if add_intercept { ncols + 1 } else { ncols };
        let aic = -2.0
            * ys.iter()
                .zip(mu.iter())
                .map(|(y, mu)| crate::dbinom(y.round(), 1.0, *mu, true))
                .sum::<f64>()
            + 2.0 * rank as f64;
        if should_disable_predicted() {
            mu = Vec::new();
        }

        let mut coefs = slopes
            .into_iter()
            .enumerate()
            .map(|(i, coef)| {
                let std_err = xtwx_inv[(i, i)].sqrt();
                let t = coef / std_err;
                let p = 2.0 * (1.0 - pnorm(t.abs()));
                let label = if let Some(colname) = colnames.as_ref().and_then(|cn| cn.get(i)) {
                    colname.to_string()
                } else {
                    format!("x[{}]", i)
                };
                Coef::new(label, coef, std_err, t, p)
            })
            .collect::<Vec<_>>();
        if add_intercept {
            let std_err = xtwx_inv[(ncols, ncols)].sqrt();
            let t = intercept / std_err;
            let p = 2.0 * (1.0 - pnorm(t.abs()));
            coefs.push(Coef::new_intercept(intercept, std_err, t, p));
        }

        Self {
            coefs,
            r2_tjur,
            predicted: mu,
            r2,
            adj_r2,
            n: ys.len() as u64,
            m: ncols as u64,
            aic,
            weights: w,
            add_intercept,
        }
    }

    // UNFINISHED AND NOT FUNCTIONAL, MEANT TO BE THE IRLS IMPLEMENTATION THAT R
    // USES WHICH SEEMS TO USE SIGNIFICANTLY LESS MATMULS
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
            MatRef::from_column_major_slice(xw_vec.as_slice(), xs.nrows(), x.ncols());
        let mut zw_vec = vec![0.0; ys.len()];
        let mut zw_mut =
            unsafe { std::slice::from_raw_parts_mut(zw_vec.as_mut_ptr(), zw_vec.len()) };
        let mut zw = ColRef::from_slice(&zw_vec);
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
            let b = beta
                .try_as_col_major()
                .expect("could not get slice")
                .as_slice();
            slopes.as_mut_slice().copy_from_slice(&b[..xs.ncols()]);
            intercept = b[xs.ncols()];
            let eta = &x * beta;
            let eta = eta
                .try_as_col_major()
                .expect("could not get slice")
                .as_slice();
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
                .chain(std::iter::once(Coef::new_intercept(
                    intercept, 0.0, 0.0, 0.0,
                )))
                .collect(),
            r2_tjur: crate::compute_r2_tjur(ys, &mu),
            predicted: mu,
            r2,
            adj_r2,
            n: ys.len() as u64,
            m: ncols as u64,
            aic: 0.0,
            weights: Vec::new(),
            add_intercept: true,
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
                faer::Accum::Replace,
                &x,
                ColRef::from_slice(beta.as_slice()),
                1.0,
                get_global_parallelism(),
            );
            for (mu, l) in mu.iter_mut().zip(
                linear_predictor
                    .try_as_col_major()
                    .expect("could not get slice")
                    .as_slice(),
            ) {
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
                faer::Accum::Replace,
                xt,
                ColRef::from_slice(ys_sub_mu.as_slice()),
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
                faer::Accum::Replace,
                &xtw,
                &x,
                1.0,
                get_global_parallelism(),
            );

            let beta_new = ColRef::from_slice(beta.as_slice())
                + match hessian.llt(faer::Side::Lower) {
                    Ok(chol) => chol.solve(&jacobian),
                    Err(_) => {
                        warn!("Using pseudo inverse");
                        hessian
                            .as_mat_ref()
                            .thin_svd()
                            .expect("could not compute thin SVD for pseudoinverse")
                            .pseudoinverse()
                            * &jacobian
                    },
                };

            if (&beta_new - ColRef::from_slice(beta.as_slice())).norm_l1() < epsilon {
                debug!("Converged after {} iterations", i);
                converged = true;
                beta.copy_from_slice(
                    beta_new
                        .try_as_col_major()
                        .expect("could not get slice")
                        .as_slice(),
                );
                break;
            }
            beta.copy_from_slice(
                beta_new
                    .try_as_col_major()
                    .expect("could not get slice")
                    .as_slice(),
            );
        }
        if !converged {
            warn!("Did not converge after {} iterations", max_iterations);
        }
        let r2 = R2Simd::new(ys, &mu).calculate();
        let adj_r2 = calculate_adj_r2(r2, ys.len(), xs.ncols());

        let predicted = if should_disable_predicted() {
            Vec::new()
        } else {
            (&x * ColRef::from_slice(beta.as_slice()))
                .try_as_col_major()
                .expect("could not get slice")
                .as_slice()
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
                .chain(std::iter::once(Coef::new_intercept(
                    beta[ncols],
                    0.0,
                    0.0,
                    0.0,
                )))
                .collect(),
            r2_tjur: crate::compute_r2_tjur(ys, &predicted),
            predicted,
            r2,
            adj_r2,
            n: ys.len() as u64,
            m: x.ncols() as u64,
            aic: 0.0,
            weights: Vec::new(),
            add_intercept: true,
        }
    }

    pub fn slopes(&self) -> &[Coef] {
        if self.add_intercept {
            &self.coefs[..self.coefs.len() - 1]
        } else {
            &self.coefs
        }
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

    pub fn r2_tjur(&self) -> f64 {
        self.r2_tjur
    }

    pub fn predict<F: Family>(&self, x: &[f64]) -> f64 {
        let mut v = self.intercept().coef();
        let slopes = self.slopes();
        for i in 0..self.slopes().len() {
            v += slopes[i].coef() * x[i];
        }
        F::linkinv(v)
    }

    pub fn coefs(&self) -> &[Coef] {
        &self.coefs
    }

    pub fn aic(&self) -> f64 {
        self.aic
    }

    // k should default to 2
    pub fn extract_aic(&self, _scale: f64, k: f64) -> (f64, f64) {
        let edf = (self.n - self.m) as f64;
        (edf, self.aic + (k - 2.0) * edf)
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}

#[inline(always)]
fn ll(p: &[f64], y: &[f64]) -> f64 {
    p.iter()
        .zip(y)
        .map(|(p, y)| y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        .sum()
}

pub trait Family {
    fn linkfun(mu: f64) -> f64;
    fn linkinv(eta: f64) -> f64;
    fn variance(mu: f64) -> f64;
    fn mu_eta(eta: f64) -> f64;
    fn dev_resids(y: f64, mu: f64) -> f64;
    fn mu_start(y: f64) -> f64;
    fn k3i_k2i(mu: f64) -> f64;
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
    use super::*;
    use crate::{dcauchy, dnorm, pnorm, qcauchy, qnorm};

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

        fn k3i_k2i(mu: f64) -> f64 {
            0.0
        }
    }

    // /// Gaussian family with log link function
    // pub struct GaussianLog;
    // impl Family for GaussianLog {
    //     fn linkfun(mu: f64) -> f64 {
    //         mu.ln()
    //     }
    //
    //     fn linkinv(eta: f64) -> f64 {
    //         eta.exp().max(f64::EPSILON)
    //     }
    //
    //     fn variance(_mu: f64) -> f64 {
    //         1.0
    //     }
    //
    //     fn mu_eta(eta: f64) -> f64 {
    //         eta.exp().max(f64::EPSILON)
    //     }
    //
    //     fn dev_resids(y: f64, mu: f64) -> f64 {
    //         (y - mu).powi(2)
    //     }
    //
    //     fn mu_start(y: f64) -> f64 {
    //         y
    //     }
    // }

    // /// Gaussian family with inverse link function
    // pub struct GaussianInverse;
    // impl Family for GaussianInverse {
    //     fn linkfun(mu: f64) -> f64 {
    //         1.0 / mu
    //     }
    //
    //     fn linkinv(eta: f64) -> f64 {
    //         1.0 / eta
    //     }
    //
    //     fn variance(mu: f64) -> f64 {
    //         1.0
    //     }
    //
    //     fn mu_eta(eta: f64) -> f64 {
    //         -1.0 / eta.powi(2)
    //     }
    //
    //     fn dev_resids(y: f64, mu: f64) -> f64 {
    //         (y - mu).powi(2)
    //     }
    //
    //     fn mu_start(y: f64) -> f64 {
    //         y
    //     }
    // }

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

        fn k3i_k2i(mu: f64) -> f64 {
            1.0 - 2.0 * mu
        }
    }

    // /// Binomial family with probit link function
    // pub struct BinomialProbit;
    // impl Family for BinomialProbit {
    //     fn linkfun(mu: f64) -> f64 {
    //         qnorm(mu)
    //     }
    //
    //     fn linkinv(eta: f64) -> f64 {
    //         let thresh = qnorm(f64::EPSILON);
    //         pnorm(eta.clamp(-thresh, thresh))
    //     }
    //
    //     fn variance(mu: f64) -> f64 {
    //         mu * (1.0 - mu)
    //     }
    //
    //     fn mu_eta(eta: f64) -> f64 {
    //         dnorm(eta).max(f64::EPSILON)
    //     }
    //
    //     fn dev_resids(y: f64, mu: f64) -> f64 {
    //         2.0 * (y_log_y(y, mu) + y_log_y(1.0 - y, 1.0 - mu))
    //     }
    //
    //     fn mu_start(y: f64) -> f64 {
    //         (y + 0.5) / 2.0
    //     }
    // }

    // /// Binomial family with cauchit link function
    // pub struct BinomialCauchit;
    // impl Family for BinomialCauchit {
    //     fn linkfun(mu: f64) -> f64 {
    //         qcauchy(mu)
    //     }
    //
    //     fn linkinv(eta: f64) -> f64 {
    //         let thresh = qcauchy(f64::EPSILON);
    //         qcauchy(eta.clamp(-thresh, thresh))
    //     }
    //
    //     fn variance(mu: f64) -> f64 {
    //         mu * (1.0 - mu)
    //     }
    //
    //     fn mu_eta(eta: f64) -> f64 {
    //         dcauchy(eta).max(f64::EPSILON)
    //     }
    //
    //     fn dev_resids(y: f64, mu: f64) -> f64 {
    //         2.0 * (y_log_y(y, mu) + y_log_y(1.0 - y, 1.0 - mu))
    //     }
    //
    //     fn mu_start(y: f64) -> f64 {
    //         (y + 0.5) / 2.0
    //     }
    // }

    // /// Binomial family with log link function
    // pub struct BinomialLog;
    // impl Family for BinomialLog {
    //     fn linkfun(mu: f64) -> f64 {
    //         mu.ln()
    //     }
    //
    //     fn linkinv(eta: f64) -> f64 {
    //         eta.exp().max(f64::EPSILON)
    //     }
    //
    //     fn variance(mu: f64) -> f64 {
    //         mu * (1.0 - mu)
    //     }
    //
    //     fn mu_eta(eta: f64) -> f64 {
    //         eta.exp().max(f64::EPSILON)
    //     }
    //
    //     fn dev_resids(y: f64, mu: f64) -> f64 {
    //         2.0 * (y_log_y(y, mu) + y_log_y(1.0 - y, 1.0 - mu))
    //     }
    //
    //     fn mu_start(y: f64) -> f64 {
    //         (y + 0.5) / 2.0
    //     }
    // }

    // /// Binomial family with complementary log-log link function
    // pub struct BinomialComplementaryLogLog;
    // impl Family for BinomialComplementaryLogLog {
    //     fn linkfun(mu: f64) -> f64 {
    //         (-(1.0 - mu).ln()).ln()
    //     }
    //
    //     fn linkinv(eta: f64) -> f64 {
    //         (-(-eta).exp())
    //             .exp_m1()
    //             .clamp(f64::EPSILON, 1.0 - f64::EPSILON)
    //     }
    //
    //     fn variance(mu: f64) -> f64 {
    //         mu * (1.0 - mu)
    //     }
    //
    //     fn mu_eta(eta: f64) -> f64 {
    //         let eta = eta.min(700.0);
    //         (eta.exp() * (-eta.exp()).exp()).max(f64::EPSILON)
    //     }
    //
    //     fn dev_resids(y: f64, mu: f64) -> f64 {
    //         2.0 * (y_log_y(y, mu) + y_log_y(1.0 - y, 1.0 - mu))
    //     }
    //
    //     fn mu_start(y: f64) -> f64 {
    //         (y + 0.5) / 2.0
    //     }
    // }

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

        fn k3i_k2i(mu: f64) -> f64 {
            mu
        }
    }

    // /// Gamma family with identity link function
    // pub struct GammaIdentity;
    // impl Family for GammaIdentity {
    //     fn linkfun(mu: f64) -> f64 {
    //         mu
    //     }
    //
    //     fn linkinv(eta: f64) -> f64 {
    //         eta
    //     }
    //
    //     fn variance(mu: f64) -> f64 {
    //         mu.powi(2)
    //     }
    //
    //     fn mu_eta(_eta: f64) -> f64 {
    //         1.0
    //     }
    //
    //     fn dev_resids(y: f64, mu: f64) -> f64 {
    //         -2.0 * ((if y == 0.0 { 1.0 } else { y / mu }).ln() - ((y - mu) / mu))
    //     }
    //
    //     fn mu_start(y: f64) -> f64 {
    //         y
    //     }
    // }

    // /// Gamma family with log link function
    // pub struct GammaLog;
    // impl Family for GammaLog {
    //     fn linkfun(mu: f64) -> f64 {
    //         mu.ln()
    //     }
    //
    //     fn linkinv(eta: f64) -> f64 {
    //         eta.exp().max(f64::EPSILON)
    //     }
    //
    //     fn variance(mu: f64) -> f64 {
    //         mu.powi(2)
    //     }
    //
    //     fn mu_eta(eta: f64) -> f64 {
    //         eta.exp().max(f64::EPSILON)
    //     }
    //
    //     fn dev_resids(y: f64, mu: f64) -> f64 {
    //         -2.0 * ((if y == 0.0 { 1.0 } else { y / mu }).ln() - ((y - mu) / mu))
    //     }
    //
    //     fn mu_start(y: f64) -> f64 {
    //         y
    //     }
    // }

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

        fn k3i_k2i(mu: f64) -> f64 {
            1.0
        }
    }

    // /// Poisson family with identity link function
    // pub struct PoissonIdentity;
    // impl Family for PoissonIdentity {
    //     fn linkfun(mu: f64) -> f64 {
    //         mu
    //     }
    //
    //     fn linkinv(eta: f64) -> f64 {
    //         eta
    //     }
    //
    //     fn variance(mu: f64) -> f64 {
    //         mu
    //     }
    //
    //     fn mu_eta(_eta: f64) -> f64 {
    //         1.0
    //     }
    //
    //     fn dev_resids(y: f64, mu: f64) -> f64 {
    //         2.0 * if y == 0.0 {
    //             mu
    //         } else {
    //             y * (y / mu).ln() - (y - mu)
    //         }
    //     }
    //
    //     fn mu_start(y: f64) -> f64 {
    //         y + 0.1
    //     }
    // }

    // /// Poisson family with sqrt link function
    // pub struct PoissonSqrt;
    // impl Family for PoissonSqrt {
    //     fn linkfun(mu: f64) -> f64 {
    //         mu.sqrt()
    //     }
    //
    //     fn linkinv(eta: f64) -> f64 {
    //         eta.powi(2)
    //     }
    //
    //     fn variance(mu: f64) -> f64 {
    //         mu
    //     }
    //
    //     fn mu_eta(eta: f64) -> f64 {
    //         2.0 * eta
    //     }
    //
    //     fn dev_resids(y: f64, mu: f64) -> f64 {
    //         2.0 * if y == 0.0 {
    //             mu
    //         } else {
    //             y * (y / mu).ln() - (y - mu)
    //         }
    //     }
    //
    //     fn mu_start(y: f64) -> f64 {
    //         y + 0.1
    //     }
    // }

    // /// Inverse Gaussian family with 1/mu^2 link function
    // pub struct InverseGaussian;
    // impl Family for InverseGaussian {
    //     fn linkfun(mu: f64) -> f64 {
    //         1.0 / mu.powi(2)
    //     }
    //
    //     fn linkinv(eta: f64) -> f64 {
    //         1.0 / eta.sqrt()
    //     }
    //
    //     fn variance(mu: f64) -> f64 {
    //         mu.powi(3)
    //     }
    //
    //     fn mu_eta(eta: f64) -> f64 {
    //         -1.0 / (2.0 * eta.powf(1.5))
    //     }
    //
    //     fn dev_resids(y: f64, mu: f64) -> f64 {
    //         (y - mu).powi(2) / (y * mu.powi(2))
    //     }
    //
    //     fn mu_start(y: f64) -> f64 {
    //         y
    //     }
    // }

    // /// Inverse Gaussian family with inverse link function
    // pub struct InverseGaussianInverse;
    // impl Family for InverseGaussianInverse {
    //     fn linkfun(mu: f64) -> f64 {
    //         1.0 / mu
    //     }
    //
    //     fn linkinv(eta: f64) -> f64 {
    //         1.0 / eta
    //     }
    //
    //     fn variance(mu: f64) -> f64 {
    //         mu.powi(3)
    //     }
    //
    //     fn mu_eta(eta: f64) -> f64 {
    //         -1.0 / eta.powi(2)
    //     }
    //
    //     fn dev_resids(y: f64, mu: f64) -> f64 {
    //         (y - mu).powi(2) / (y * mu.powi(2))
    //     }
    //
    //     fn mu_start(y: f64) -> f64 {
    //         y
    //     }
    // }

    // /// Inverse Gaussian family with identity link function
    // pub struct InverseGaussianIdentity;
    // impl Family for InverseGaussianIdentity {
    //     fn linkfun(mu: f64) -> f64 {
    //         mu
    //     }
    //
    //     fn linkinv(eta: f64) -> f64 {
    //         eta
    //     }
    //
    //     fn variance(mu: f64) -> f64 {
    //         mu.powi(3)
    //     }
    //
    //     fn mu_eta(_eta: f64) -> f64 {
    //         1.0
    //     }
    //
    //     fn dev_resids(y: f64, mu: f64) -> f64 {
    //         (y - mu).powi(2) / (y * mu.powi(2))
    //     }
    //
    //     fn mu_start(y: f64) -> f64 {
    //         y
    //     }
    // }

    // /// Inverse Gaussian family with log link function
    // pub struct InverseGaussianLog;
    // impl Family for InverseGaussianLog {
    //     fn linkfun(mu: f64) -> f64 {
    //         mu.ln()
    //     }
    //
    //     fn linkinv(eta: f64) -> f64 {
    //         eta.exp().max(f64::EPSILON)
    //     }
    //
    //     fn variance(mu: f64) -> f64 {
    //         mu.powi(3)
    //     }
    //
    //     fn mu_eta(eta: f64) -> f64 {
    //         eta.exp().max(f64::EPSILON)
    //     }
    //
    //     fn dev_resids(y: f64, mu: f64) -> f64 {
    //         (y - mu).powi(2) / (y * mu.powi(2))
    //     }
    //
    //     fn mu_start(y: f64) -> f64 {
    //         y
    //     }
    // }
}

#[cfg(test)]
mod tests {
    use rand_distr::{num_traits::float, Distribution};
    use test_log::test;

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
    fn test_glm_irls() {
        let nrows = 50;
        let xs = MatRef::from_column_major_slice(XS.as_slice(), nrows, 4);
        let m = Glm::irls::<family::BinomialLogit>(xs, YS.as_slice(), 1e-10, 25, true, false, None);
        float_eq!(m.intercept().coef(), INTERCEPT);
        float_eq!(m.slopes()[0].coef(), SLOPES[0]);
        float_eq!(m.slopes()[1].coef(), SLOPES[1]);
        float_eq!(m.slopes()[2].coef(), SLOPES[2]);
        float_eq!(m.slopes()[3].coef(), SLOPES[3]);
        float_eq!(m.aic(), 56.14718865822164417523);
    }

    #[test]
    fn test_glm_irls_null() {
        let nrows = 50;
        let xs = MatRef::from_column_major_slice([].as_slice(), nrows, 0);
        let m = Glm::irls::<family::BinomialLogit>(xs, YS.as_slice(), 1e-10, 25, true, false, None);
        float_eq!(m.intercept().coef(), -0.1603426500751793937205);
        assert_eq!(m.slopes().len(), 0);
        float_eq!(m.aic(), 70.9943758458399685196127);
    }

    #[test]
    fn test_glm_irls_no_intercept() {
        let nrows = 50;
        let xs = MatRef::from_column_major_slice(XS.as_slice(), nrows, 4);
        let m =
            Glm::irls::<family::BinomialLogit>(xs, YS.as_slice(), 1e-10, 25, false, false, None);
        assert_eq!(m.slopes().len(), 4);
        float_eq!(m.slopes()[0].coef(), NO_INTERCEPT_SLOPES[0]);
        float_eq!(m.slopes()[1].coef(), NO_INTERCEPT_SLOPES[1]);
        float_eq!(m.slopes()[2].coef(), NO_INTERCEPT_SLOPES[2]);
        float_eq!(m.slopes()[3].coef(), NO_INTERCEPT_SLOPES[3]);
    }

    // #[test]
    // fn test_glm_irls_r() {
    //     let nrows = 50;
    //     let xs = MatRef::from_column_major_slice(XS.as_slice(), nrows, 4);
    //     let m = Glm::irls_r::<family::BinomialLogit>(xs, YS.as_slice(), 1e-100,
    // 25);     float_eq!(m.intercept().coef(), -0.10480279218218244152716);
    //     float_eq!(m.slopes()[0].coef(), 0.06970776481172229199768);
    //     float_eq!(m.slopes()[1].coef(), 0.31341357257259599977672);
    //     float_eq!(m.slopes()[2].coef(), -0.52734374471258893546377);
    //     float_eq!(m.slopes()[3].coef(), 0.05905679790685783303594);
    // }

    #[test]
    fn test_glm_newton_raphson() {
        let nrows = 50;
        let xs = MatRef::from_column_major_slice(XS.as_slice(), nrows, 4);
        let m = Glm::newton_raphson::<family::BinomialLogit>(xs, YS.as_slice(), 1e-10, 25);
        float_eq!(m.intercept().coef(), INTERCEPT);
        float_eq!(m.slopes()[0].coef(), SLOPES[0]);
        float_eq!(m.slopes()[1].coef(), SLOPES[1]);
        float_eq!(m.slopes()[2].coef(), SLOPES[2]);
        float_eq!(m.slopes()[3].coef(), SLOPES[3]);
    }

    #[test]
    fn test_glm_irls_predict() {
        let nrows = 50;
        let xs = MatRef::from_column_major_slice(XS.as_slice(), nrows, 4);
        let m = Glm::irls::<family::BinomialLogit>(xs, YS.as_slice(), 1e-10, 25, true, false, None);
        float_eq!(
            m.predict::<family::BinomialLogit>(&[XS[0], XS[50], XS[100], XS[150]]),
            m.predicted()[0]
        );
    }

    #[test]
    fn test_glm_newton_raphson_predict() {
        let nrows = 50;
        let xs = MatRef::from_column_major_slice(XS.as_slice(), nrows, 4);
        let m = Glm::newton_raphson::<family::BinomialLogit>(xs, YS.as_slice(), 1e-10, 25);
        float_eq!(
            m.predict::<family::BinomialLogit>(&[XS[0], XS[50], XS[100], XS[150]]),
            m.predicted()[0]
        );
    }

    // #[test]
    // fn test_glm_irls_firth() {
    //     let nrows = 50;
    //     let xs = MatRef::from_column_major_slice(XS.as_slice(), nrows, 4);
    //     let m = Glm::irls::<family::BinomialLogit>(xs, YS.as_slice(), 1e-10, 25, false);
    //     println!("R2: {}", m.r2());
    //     println!("Slopes: {:?}", m.slopes());
    //     println!("Intercept: {:?}", m.intercept());
    //     let m = Glm::irls::<family::BinomialLogit>(xs, YS.as_slice(), 1e-10, 250, true);
    //     println!("R2: {}", m.r2());
    //     println!("Slopes: {:?}", m.slopes());
    //     println!("Intercept: {:?}", m.intercept());
    //     panic!();
    // }

    #[rustfmt::skip]
    pub mod data {
        pub const XS: [f64; 200] = [
            0.32, -0.78, 0.45, -1.22, 0.11, 0.67, -0.91, 1.02, -0.44, 0.85, -1.01, 0.33, 0.28,
            -0.67, -1.35, 0.60, -0.09, 1.21, 0.74, -0.23, -0.18, 0.91, 0.42, -0.60, 1.03, -1.11,
            0.07, -0.37, 0.19, -0.49, 0.80, -0.66, 0.23, -1.08, 0.55, -0.04, -1.26, 0.92, -0.73,
            0.17, -0.14, -0.97, 0.61, 0.09, -1.38, 0.13, -0.59, 0.25, -0.85, 1.16, -1.45, 0.56,
            1.10, -0.78, 0.29, -1.34, 0.62, 1.14, -0.92, -0.60, 0.15, -0.24, -0.43, 1.01, 0.70,
            -1.19, -0.17, 0.77, -1.42, 0.11, -1.06, 0.96, -0.35, 1.08, -0.38, -0.73, 0.44, 0.12,
            -0.88, 0.51, -1.27, -0.15, 1.22, 0.40, -0.94, -0.05, 1.36, -0.69, 1.00, -0.91, 0.84,
            1.30, -0.48, -1.09, 0.36, -0.66, 0.23, -1.17, -0.04, 0.67, 0.88, -1.34, 0.33, -0.90,
            0.77, -0.66, 1.19, -0.27, 0.94, -1.21, -0.85, 1.25, 0.17, -0.29, 0.46, -1.06, -0.73,
            0.39, -0.25, -1.28, 0.30, -0.18, 0.66, 1.12, -0.62, -1.25, 0.86, -0.45, 1.36, 0.06,
            -0.10, 0.93, -0.81, 0.41, 0.02, -1.17, -0.51, 0.73, 0.37, -0.96, -0.08, -0.65, 1.13,
            0.58, 0.90, -0.74, -1.05, -0.33, 0.53, 1.01, 1.21, -0.95, 0.12, -1.34, 0.05, 0.78,
            -0.57, -1.15, 0.73, 0.48, -1.28, 1.07, -0.19, -0.70, 1.10, -0.12, -1.01, 0.44, -0.81,
            0.63, 1.33, -0.31, 0.90, -1.07, -0.43, 0.19, 1.06, -0.68, -0.35, 0.22, 1.30, 0.14,
            -1.19, -0.09, 0.65, 0.38, -0.84, -0.60, 1.19, -1.23, 0.87, -0.50, -1.36, 0.33, -0.27,
            0.71, 1.18, -0.41, -1.10, 0.59,
        ];

        pub const YS: [f64; 50] = [
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
        ];
        pub const INTERCEPT: f64 = -0.2684343418101716;
        pub const SLOPES: [f64; 4] = [
            -0.021466882058221837,
            -0.5737590452688993,
            0.6169197849363219,
            1.868472448752581,
        ];
        pub const NO_INTERCEPT_SLOPES: [f64; 4] = [
            -0.001212187314425448555,
            -0.530392162536109323945,
            0.602901458579869542476,
            1.842422792116841456789,
        ];
    }
    pub use data::*;
}

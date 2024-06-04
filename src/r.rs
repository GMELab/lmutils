#![allow(dead_code)]
#![allow(refining_impl_trait)]
use faer::MatRef;
use faer_ext::IntoNalgebra;
use rayon::prelude::*;

/// Credit: code adapted from the R `qnorm` function https://github.com/wch/r-source/blob/5dec7823a19dc02fdcbb326912d1066951b12c2e/src/nmath/qnorm.c
#[inline]
fn r_d_lval(p: f64, _log_p: bool, lower_tail: bool) -> f64 {
    if lower_tail {
        p
    } else {
        0.5 - p + 0.5
    }
}
#[inline]
fn r_dt_qiv(p: f64, log_p: bool, lower_tail: bool) -> f64 {
    if log_p {
        if lower_tail {
            p.exp()
        } else {
            -p.exp_m1()
        }
    } else {
        r_d_lval(p, log_p, lower_tail)
    }
}
#[inline]
fn r_d_cval(p: f64, _log_p: bool, lower_tail: bool) -> f64 {
    if lower_tail {
        0.5 - p + 0.5
    } else {
        p
    }
}
#[inline]
fn r_dt_civ(p: f64, log_p: bool, lower_tail: bool) -> f64 {
    if log_p {
        if lower_tail {
            -p.exp_m1()
        } else {
            p.exp()
        }
    } else {
        r_d_cval(p, log_p, lower_tail)
    }
}
#[allow(clippy::excessive_precision)]
const M_2PI: f64 = 6.283185307179586476925286766559;
#[allow(clippy::excessive_precision)]
pub fn qnorm(p: f64, mean: f64, sd: f64, lower_tail: bool, log_p: bool) -> f64 {
    if p.is_nan() || mean.is_nan() || sd.is_nan() {
        return p + mean + sd;
    }
    if log_p {
        if p > 0.0 {
            return f64::NAN;
        }
        if p == 0.0 {
            if lower_tail {
                return f64::INFINITY;
            } else {
                return f64::NEG_INFINITY;
            }
        }
        if p == f64::NEG_INFINITY {
            if lower_tail {
                return f64::NEG_INFINITY;
            } else {
                return f64::INFINITY;
            }
        }
    } else {
        if !(0.0..=1.0).contains(&p) {
            return f64::NAN;
        }
        if p == 0.0 {
            if lower_tail {
                return f64::NEG_INFINITY;
            } else {
                return f64::INFINITY;
            }
        }
        if p == 1.0 {
            if lower_tail {
                return f64::INFINITY;
            } else {
                return f64::NEG_INFINITY;
            }
        }
    }
    if sd < 0.0 {
        return f64::NAN;
    }
    if sd == 0.0 {
        return mean;
    }
    let p_ = r_dt_qiv(p, log_p, lower_tail);
    let q = p_ - 0.5;
    let val = if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        q * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r
            + 67265.770927008700853f64)
            * r
            + 45921.953931549871457)
            * r
            + 13731.693765509461125)
            * r
            + 1971.5909503065514427)
            * r
            + 133.14166789178437745)
            * r
            + 3.387132872796366608)
            / (((((((r * 5226.495278852854561 + 28729.085735721942674) * r
                + 39307.89580009271061)
                * r
                + 21213.794301586595867)
                * r
                + 5394.1960214247511077)
                * r
                + 687.1870074920579083)
                * r
                + 42.313330701600911252)
                * r
                + 1.)
    } else {
        let lp = if log_p && ((lower_tail && q <= 0.0) || (!lower_tail && q > 0.0)) {
            p
        } else {
            f64::ln(if q > 0.0 {
                r_dt_civ(p, log_p, lower_tail)
            } else {
                p_
            })
        };
        let mut r = (-lp).sqrt();
        let val = if r <= 5.0 {
            r += -1.6;
            (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r
                + 0.24178072517745061177)
                * r
                + 1.27045825245236838258)
                * r
                + 3.64784832476320460504)
                * r
                + 5.7694972214606914055)
                * r
                + 4.6303378461565452959)
                * r
                + 1.42343711074968357734)
                / (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r
                    + 0.0151986665636164571966)
                    * r
                    + 0.14810397642748007459)
                    * r
                    + 0.68976733498510000455)
                    * r
                    + 1.6763848301838038494)
                    * r
                    + 2.05319162663775882187)
                    * r
                    + 1.)
        } else if r <= 27.0 {
            r += 5.0;
            (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r
                + 0.0012426609473880784386)
                * r
                + 0.026532189526576123093)
                * r
                + 0.29656057182850489123)
                * r
                + 1.7848265399172913358)
                * r
                + 5.4637849111641143699)
                * r
                + 6.6579046435011037772)
                / (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r
                    + 1.8463183175100546818e-5)
                    * r
                    + 7.868691311456132591e-4)
                    * r
                    + 0.0148753612908506148525)
                    * r
                    + 0.13692988092273580531)
                    * r
                    + 0.59983220655588793769)
                    * r
                    + 1.)
        } else if r > 6.4e8 {
            r * (2.0f64).sqrt()
        } else {
            // s2 = -ldexp(lp, 1);
            let s2 = -lp * 2.0;
            let mut x2 = s2 - (M_2PI * s2).ln();
            if r < 36000.0 {
                x2 = s2 - (M_2PI * x2).ln() - 2.0 / (2.0 + x2);
                if r < 840.0 {
                    x2 = s2 - (M_2PI * x2).ln()
                        + 2.0 * (-(1.0 - 1.0 / (4.0 + x2)) / (2.0 + x2)).ln_1p();
                    if r < 109.0 {
                        x2 = s2 - (M_2PI * x2).ln()
                            + 2.0
                                * (-(1.0 - (1.0 - 5.0 / (6.0 + x2)) / (4.0 + x2)) / (2.0 + x2))
                                    .ln_1p();
                        if r < 55.0 {
                            x2 = s2 - (M_2PI * x2).ln()
                                + 2.0
                                    * (-(1.0
                                        - (1.0 - (5.0 - 9.0 / (8.0 + x2)) / (6.0 + x2))
                                            / (4.0 + x2))
                                        / (2.0 + x2))
                                        .ln_1p();
                        }
                    }
                }
            }
            x2.sqrt()
        };
        if q < 0.0 {
            -val
        } else {
            val
        }
    };
    mean + sd * val
}

/// An implementation of the R `rank` function.
pub trait Rank {
    fn rank(self) -> impl IndexedParallelIterator<Item = f64>;
}

impl Rank for Vec<f64> {
    fn rank(self) -> impl IndexedParallelIterator<Item = f64> {
        let mut sorted = self.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
        let len = self.len();
        (0..len).into_par_iter().map(move |i| {
            let val = &self[i];
            let start = sorted
                .iter()
                .position(|x| x == val)
                .expect("value in unsorted list but not sorted");
            let mut end = start;
            while end < len && sorted[end] == *val {
                end += 1;
            }
            ((start + 1)..=end).map(|x| x as f64).sum::<f64>() / (end - start) as f64
        })
    }
}

impl<'a> Rank for &'a [f64] {
    fn rank(self) -> impl IndexedParallelIterator<Item = f64> + 'a {
        let mut sorted = self.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
        let len = self.len();
        (0..len).into_par_iter().map(move |i| {
            let val = &self[i];
            let start = sorted
                .iter()
                .position(|x| x == val)
                .expect("value in unsorted list but not sorted");
            let mut end = start;
            while end < len && sorted[end] == *val {
                end += 1;
            }
            ((start + 1)..=end).map(|x| x as f64).sum::<f64>() / (end - start) as f64
        })
    }
}

impl<'a> Rank for &'a mut [f64] {
    fn rank(self) -> impl IndexedParallelIterator<Item = f64> + 'a {
        self[..].rank()
    }
}

pub trait QuantNorm {
    fn quant_norm(self) -> impl IndexedParallelIterator<Item = f64>;
}

impl QuantNorm for Vec<f64> {
    fn quant_norm(self) -> impl IndexedParallelIterator<Item = f64> {
        let len = self.len();
        self.rank()
            .map(move |i| qnorm(i / ((len + 1) as f64), 0.0, 1.0, true, false))
    }
}

impl<'a> QuantNorm for &'a [f64] {
    fn quant_norm(self) -> impl IndexedParallelIterator<Item = f64> + 'a {
        let len = self.len();
        self.rank()
            .map(move |i| qnorm(i / ((len + 1) as f64), 0.0, 1.0, true, false))
    }
}

impl<'a> QuantNorm for &'a mut [f64] {
    fn quant_norm(self) -> impl IndexedParallelIterator<Item = f64> + 'a {
        let len = self.len();
        self.rank()
            .map(move |i| qnorm(i / ((len + 1) as f64), 0.0, 1.0, true, false))
    }
}

pub fn mean(x: &[f64]) -> f64 {
    let count = x.len() as f64;
    x.iter().sum::<f64>() / count
}

pub fn sd(x: &[f64]) -> f64 {
    let mean = mean(x);
    let count = x.len() as f64;
    (x.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1.0)).sqrt()
}

pub fn mean_sd(x: &[f64]) -> (f64, f64) {
    let mean = mean(x);
    let count = x.len() as f64;
    let sd = (x.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1.0)).sqrt();
    (mean, sd)
}

pub fn variance(x: &[f64]) -> f64 {
    let mean = mean(x);
    let count = x.len() as f64;
    x.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1.0)
}

pub fn standardization(x: &mut [f64]) {
    let (mean, sd) = mean_sd(x);
    for x in x.iter_mut() {
        *x = (*x - mean) / sd;
    }
}

pub fn rm_stratification<'a>(
    xs: MatRef<'a, f64>,
    ep: &'a [f64],
) -> impl IndexedParallelIterator<Item = f64> + 'a {
    let ep = ep.quant_norm().collect::<Vec<_>>();
    let Lm { residuals, .. } = lm(xs, &ep);
    residuals.quant_norm()
}

pub fn rm_heteroscedasticity(mut p: Vec<f64>, e: &[f64]) -> Vec<f64> {
    let e_len = e.len() as f64;
    let e_ranks = e.rank().map(|x| x / (e_len + 1.0)).collect::<Vec<_>>();
    let partition_size = 1.0 / 20.0 / 2.0;
    for i in 1..=20 {
        let f = i as f64;
        let lower_lower_bound = (f - 1.0) * partition_size;
        let lower_upper_bound = f * partition_size;
        let upper_lower_bound = 1.0 - f * partition_size;
        let upper_upper_bound = 1.0 - (f - 1.0) * partition_size;
        let indices = e_ranks
            .iter()
            .enumerate()
            .filter_map(|(i, x)| {
                if (*x >= lower_lower_bound && *x < lower_upper_bound)
                    || (*x > upper_lower_bound && *x <= upper_upper_bound)
                {
                    Some(i)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        // P_resid_final[select] <-  standardization( P_resid_final[select] )
        let mut values = indices.iter().map(|&i| p[i]).collect::<Vec<_>>();
        standardization(&mut values);
        for (i, &index) in indices.iter().enumerate() {
            p[index] = values[i];
        }
    }
    p
}

#[non_exhaustive]
pub struct Lm {
    pub residuals: Vec<f64>,
    pub r2: f64,
    pub adj_r2: f64,
}

pub fn lm(xs: MatRef<'_, f64>, ys: &[f64]) -> Lm {
    let ncols = xs.ncols();
    let a = xs.into_nalgebra();
    let a = a.insert_column(ncols, 1.0);
    let b = nalgebra::DVector::from_iterator(ys.len(), ys.iter().copied());
    let qr = a.clone().qr();
    let (q, r) = (qr.q(), qr.r());
    let x = r
        .try_inverse()
        .expect("could not find inverse or pseudo inverse of R")
        * q.transpose()
        * b;
    let intercept = x[ncols];
    let residuals = ys
        .iter()
        .enumerate()
        .map(|(i, y)| y - (intercept + (0..ncols).map(|j| x[j] * a[(i, j)]).sum::<f64>()))
        .collect::<Vec<_>>();
    let mean_y = mean(ys);
    let r2 = 1.0
        - residuals.iter().map(|x| x.powi(2)).sum::<f64>()
            / ys.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();
    let adj_r2 =
        1.0 - (1.0 - r2) * (ys.len() as f64 - 1.0) / (ys.len() as f64 - ncols as f64 - 1.0);
    Lm {
        residuals,
        r2,
        adj_r2,
    }
}

use faer::{Col, ColRef, Mat, MatRef};
use rand::seq::SliceRandom;
use tracing::warn;

/// Randomly split n samples into k folds for cross-validation.
/// The returned vector indicates which fold each sample belongs to.
pub fn split_folds(n: usize, k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).map(|i| i % k).collect();
    indices.shuffle(&mut rand::thread_rng());
    indices
}

#[derive(Clone, Debug)]
pub struct ElnetControl {
    max_iter: usize,
    epsilon: f64,
    devmax: f64,
    fdev: f64,
    nlambda: usize,
}

impl Default for ElnetControl {
    fn default() -> Self {
        ElnetControl {
            max_iter: 1000,
            epsilon: 1e-7,
            devmax: 0.999,
            fdev: 1e-5,
            nlambda: 100,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ElnetResult {
    pub beta: Col<f64>,
    pub iters: usize,
    pub lambda: f64,
    pub r2: f64,
    pub mse: f64,
}

/// Coordinate descent for Elastic Net regression.
/// X: n x p matrix
/// y: length n column vector
/// lambda: regularization
/// max_iter: number of passes
pub fn elnet(
    mut x: Mat<f64>,
    mut y: Col<f64>,
    alpha: f64,
    lambdas: Option<Vec<f64>>,
    control: ElnetControl,
) -> Vec<ElnetResult> {
    let p = x.ncols();
    let means = x
        .col_iter()
        .map(|col| crate::mean(col.try_as_col_major().unwrap().as_slice()))
        .collect::<Vec<f64>>();
    let stds = x
        .col_iter()
        .map(|col| crate::variance(col.try_as_col_major().unwrap().as_slice(), 1).sqrt())
        .collect::<Vec<f64>>();
    for (i, col) in x.col_iter_mut().enumerate() {
        let mean = means[i];
        let std = stds[i];
        if std > 0.0 {
            col.iter_mut().for_each(|x| *x = (*x - mean) / std);
        } else {
            col.iter_mut().for_each(|x| *x = 0.0);
        }
    }
    let lambdas =
        lambdas.unwrap_or_else(|| lambda_sequence(x.as_ref(), y.as_ref(), alpha, control.nlambda));
    let y_mean = crate::mean(y.try_as_col_major().unwrap().as_slice());
    let y_std = crate::variance(y.try_as_col_major().unwrap().as_slice(), 1).sqrt();
    for i in 0..y.nrows() {
        if y_std > 0.0 {
            y[i] = (y[i] - y_mean) / y_std;
        } else {
            y[i] = 0.0;
        }
    }

    x.resize_with(x.nrows(), x.ncols() + 1, |_, _| 1.0);
    let mut beta = Col::ones(x.ncols());
    let mut old_beta = beta.clone();
    let mut y_pred = &x * &beta; // Initial prediction
    let mut old_pred = x
        .col_iter()
        .map(|col| col.to_owned())
        .collect::<Vec<Col<f64>>>();
    let mut new_pred_j = Col::<f64>::zeros(x.nrows());
    let mut pred_j = Col::<f64>::zeros(x.nrows());

    let mut results = Vec::with_capacity(lambdas.len());
    let mut old_r2 = 0.0;

    'finish: for lambda in lambdas {
        let mut converged = false;
        'next: for i in 0..control.max_iter {
            old_beta.copy_from(&beta);
            for j in 0..=p {
                let x_j = x.col(j);
                let old_pred_j = old_pred[j].as_mut();

                faer::zip!(&mut pred_j, &y, &y_pred, &old_pred_j).for_each(
                    |faer::unzip!(pred_j, y, y_pred, old_pred_j)| {
                        *pred_j = *y - *y_pred + *old_pred_j;
                    },
                );
                let xtx = x_j.transpose() * x_j;
                let rho = (x_j.transpose() * &pred_j) / xtx;
                if j == p {
                    // intercept term
                    beta[j] = rho;
                } else {
                    // Elastic Net: soft threshold + ridge shrinkage
                    let z = 1.0 + lambda * (1.0 - alpha);
                    let s = soft_threshold(rho, lambda * alpha);
                    beta[j] = s / z;
                }

                // update y_pred
                // since x doesn't change and only one beta changes, we only need to
                // compute one column
                let beta_j = beta[j];
                faer::zip!(&mut new_pred_j, &x_j).for_each(|faer::unzip!(new_pred_j, x_j)| {
                    *new_pred_j = *x_j * beta_j;
                });
                faer::zip!(&mut y_pred, &old_pred_j, &new_pred_j).for_each(
                    |faer::unzip!(y_pred, old_pred_j, new_pred_j)| {
                        *y_pred = *y_pred - *old_pred_j + *new_pred_j;
                    },
                );
                std::mem::swap(&mut old_pred[j], &mut new_pred_j);
            }
            let mut beta_diff = 0.0;
            for i in 0..=p {
                beta_diff += (beta[i] - old_beta[i]).powi(2);
            }
            if beta_diff < control.epsilon {
                let r2 = crate::compute_r2(
                    y.try_as_col_major().unwrap().as_slice(),
                    y_pred.try_as_col_major().unwrap().as_slice(),
                );
                // we don't want this model
                if i > 1 && r2 < old_r2 {
                    println!(
                        "R-squared decreased from {} to {} at iteration {}",
                        old_r2, r2, i
                    );
                    // If R-squared is decreasing, we can stop
                    break 'finish;
                }
                let mut beta = beta.clone();
                // Scale back the beta coefficients
                for i in 0..p {
                    beta[i] *= y_std / stds[i];
                }
                beta[p] = beta[p] * y_std + y_mean
                    - beta
                        .iter()
                        .zip(means.iter())
                        .map(|(b, m)| b * m)
                        .sum::<f64>();

                results.push(ElnetResult {
                    beta,
                    iters: i + 1,
                    lambda,
                    r2,
                    mse: y
                        .iter()
                        .zip(y_pred.iter())
                        .map(|(yi, ypi)| (yi - ypi).powi(2))
                        .sum::<f64>()
                        / y.nrows() as f64,
                });
                if i > 1 {
                    if r2 > control.devmax {
                        // If R-squared is very high, we can stop
                        break 'finish;
                    }
                    if r2 - old_r2 < control.fdev * r2 {
                        break 'finish;
                    }
                }
                old_r2 = r2;
                converged = true;
                break 'next; // Break out of the lambda loop
            }
        }
        if !converged {
            warn!("Elastic Net did not converge for lambda {}", lambda);
        }
    }

    results
}

/// Cross-validated elastic net regression
/// Get the lambda sequence from the master model, we don't actually need the weights or anything.
/// Then, for each fold fit a model using the other folds as training data, allow that model to
/// generate it's own lambda sequence.
/// Then, align each model to the master lambda sequence and use that to determine the best model.
/// Then return the best model across all folds.
pub fn cv_elnet(
    x: MatRef<f64>,
    y: ColRef<f64>,
    alpha: f64,
    nfolds: usize,
    folds: Option<Vec<usize>>,
    lambdas: Option<Vec<f64>>,
    control: ElnetControl,
) -> ElnetResult {
    let folds = folds.unwrap_or_else(|| split_folds(x.nrows(), nfolds));
    let mut xs = x.to_owned();
    let means = x
        .col_iter()
        .map(|col| crate::mean(col.try_as_col_major().unwrap().as_slice()))
        .collect::<Vec<f64>>();
    let stds = x
        .col_iter()
        .map(|col| crate::variance(col.try_as_col_major().unwrap().as_slice(), 1).sqrt())
        .collect::<Vec<f64>>();
    for (i, col) in xs.col_iter_mut().enumerate() {
        let mean = means[i];
        let std = stds[i];
        if std > 0.0 {
            col.iter_mut().for_each(|x| *x = (*x - mean) / std);
        } else {
            col.iter_mut().for_each(|x| *x = 0.0);
        }
    }
    // this gets used as s in the call to lambda_interp
    let master_lambdas = lambdas
        .clone()
        .unwrap_or_else(|| lambda_sequence(xs.as_ref(), y.as_ref(), alpha, control.nlambda));
    println!(
        "Master lambda sequence: {:?}",
        master_lambdas
            .iter()
            .map(|l| format!("{:.6}", l))
            .collect::<Vec<_>>()
    );
    let mut results = Vec::with_capacity(nfolds * lambdas.as_ref().map_or(1, |l| l.len()));
    for fold in 0..nfolds {
        let fold = folds
            .iter()
            .enumerate()
            .filter(|(_, f)| **f == fold)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let mut x_train = Mat::zeros(x.nrows() - fold.len(), x.ncols());
        let mut y_train = Col::zeros(x.nrows() - fold.len());
        let mut x_test = Mat::zeros(fold.len(), x.ncols());
        let mut y_test = Col::zeros(fold.len());
        let mut train_idx = 0;
        for (i, &index) in fold.iter().enumerate() {
            x_test.row_mut(i).copy_from(&x.row(index));
            y_test[i] = y[index];
        }
        for i in 0..x.nrows() {
            if !fold.contains(&i) {
                x_train.row_mut(train_idx).copy_from(&x.row(i));
                y_train[train_idx] = y[i];
                train_idx += 1;
            }
        }

        let fold_results = elnet(x_train, y_train, alpha, None, control.clone());
        // let (left, right, frac) = lambda_interp(
        //     &fold_results.iter().map(|r| r.lambda).collect::<Vec<_>>(),
        //     &master_lambdas,
        // );
        // let mut interpolated_results = Vec::with_capacity(fold_results.len() - 1);
        // for i in 0..left.len().min(fold_results.len()) {
        //     if left[i] == right[i] {
        //         // If the left and right indices are the same, we can just use that result
        //         let y_pred = &x_test * fold_results[left[i]].beta.subrows(0, x_test.ncols())
        //             + Col::from_fn(x_test.nrows(), |_| {
        //                 fold_results[left[i]].beta[x_test.ncols()]
        //             });
        //         let mse = y_test
        //             .iter()
        //             .zip(y_pred.iter())
        //             .map(|(yi, ypi)| (yi - ypi).powi(2))
        //             .sum::<f64>()
        //             / y_test.nrows() as f64;
        //         interpolated_results.push(Result {
        //             beta: fold_results[left[i]].beta.clone(),
        //             iters: fold_results[left[i]].iters,
        //             lambda: master_lambdas[i],
        //             mse,
        //             r2: crate::compute_r2(
        //                 y_test.try_as_col_major().unwrap().as_slice(),
        //                 y_pred.try_as_col_major().unwrap().as_slice(),
        //             ),
        //         });
        //     } else {
        //         // Interpolate between the two results
        //         let beta_left = &fold_results[left[i]].beta;
        //         let beta_right = &fold_results[right[i]].beta;
        //         let beta = beta_left
        //             .iter()
        //             .zip(beta_right.iter())
        //             .map(|(bl, br)| bl * frac[i] + br * (1.0 - frac[i]))
        //             .collect::<Col<f64>>();
        //         let y_pred = &x_test * beta.subrows(0, x_test.ncols())
        //             + Col::from_fn(x_test.nrows(), |_| beta[x_test.ncols()]);
        //         let mse = y_test
        //             .iter()
        //             .zip(y_pred.iter())
        //             .map(|(yi, ypi)| (yi - ypi).powi(2))
        //             .sum::<f64>()
        //             / y_test.nrows() as f64;
        //         interpolated_results.push(Result {
        //             beta,
        //             iters: fold_results[left[i]].iters,
        //             lambda: master_lambdas[i],
        //             mse,
        //             r2: crate::compute_r2(
        //                 y_test.try_as_col_major().unwrap().as_slice(),
        //                 y_pred.try_as_col_major().unwrap().as_slice(),
        //             ),
        //         });
        //     }
        // }
        // results.extend(interpolated_results);
        results.extend(fold_results.into_iter().map(|r| {
            let y_pred = &x_test * r.beta.subrows(0, x_test.ncols())
                + Col::from_fn(x_test.nrows(), |_| r.beta[x_test.ncols()]);
            let mse = y_test
                .iter()
                .zip(y_pred.iter())
                .map(|(yi, ypi)| (yi - ypi).powi(2))
                .sum::<f64>()
                / y_test.nrows() as f64;
            ElnetResult {
                beta: r.beta,
                iters: r.iters,
                lambda: r.lambda,
                mse,
                r2: crate::compute_r2(
                    y_test.try_as_col_major().unwrap().as_slice(),
                    y_pred.try_as_col_major().unwrap().as_slice(),
                ),
            }
        }));
    }

    // let min = results
    //     .iter()
    //     .map(|r| r.mse)
    //     .min_by(|a, b| a.partial_cmp(b).unwrap())
    //     .unwrap();
    // println!(
    //     "{:#?}",
    //     results
    //         .iter()
    //         .filter(|r| r.mse <= min + 0.003)
    //         .collect::<Vec<_>>()
    // );

    // we want to return the best result across all folds
    results
        .into_iter()
        .min_by(|a, b| a.mse.partial_cmp(&b.mse).expect("No NaN values in mse"))
        .expect("No results found")
}

fn soft_threshold(z: f64, lambda: f64) -> f64 {
    if z > lambda {
        z - lambda
    } else if z < -lambda {
        z + lambda
    } else {
        0.0
    }
}

/// https://github.com/cran/glmnet/blob/14f90adf86f80d0fca8100d7a3af944091e3d27b/R/glmnetFlex.R#L238-L252
fn lambda_sequence(x: MatRef<f64>, y: ColRef<f64>, alpha: f64, n: usize) -> Vec<f64> {
    let lambda_min_ratio = if x.nrows() < x.ncols() { 1e-2 } else { 1e-4 };
    let y_mean = y.sum() / y.nrows() as f64;
    let y = Col::from_fn(y.nrows(), |i| (y[i] - y_mean) / y.nrows() as f64);
    let mut g = x.transpose() * y;
    for i in g.iter_mut() {
        *i = i.abs();
    }
    let lambda_max = g.max().unwrap() / alpha.max(0.001);
    let mut lambda_seq = Vec::with_capacity(n);
    let lambda_start = lambda_max.ln();
    let step = ((lambda_max * lambda_min_ratio).ln() - lambda_start) / (n as f64 - 1.0);
    for i in 0..n {
        let lambda = (lambda_start + i as f64 * step).exp();
        lambda_seq.push(lambda);
    }
    lambda_seq
}

fn approx(x: &[f64], y: &[f64], new_x: &[f64]) -> Vec<(f64, f64)> {
    assert!(x.len() == y.len(), "x and y must have the same length");
    let mut points = x.iter().copied().zip(y.iter().copied()).collect::<Vec<_>>();
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    points.dedup_by(|a, b| a.0 == b.0);
    if points.len() < 2 {
        panic!("At least two points are required for interpolation");
    }
    let mut new_x = new_x
        .iter()
        .copied()
        .enumerate()
        .map(|(i, x)| (i, x, 0.0))
        .collect::<Vec<_>>();
    new_x.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut i = 0;
    for (_, x_val, y_val) in &mut new_x {
        let x_val = *x_val;
        while i < points.len() - 1 && points[i + 1].0 < x_val {
            i += 1;
        }
        if i == 0 && x_val < points[0].0 {
            // If x_val is less than the first point
            *y_val = f64::NAN;
        } else if i == points.len() - 1 && x_val > points[i].0 {
            // If x_val is greater than the last point
            *y_val = f64::NAN;
        } else if i == points.len() - 1 || points[i].0 == x_val {
            // If x_val is exactly at a point or at the last point
            *y_val = points[i].1;
        } else {
            let x0 = points[i].0;
            let y0 = points[i].1;
            let x1 = points[i + 1].0;
            let y1 = points[i + 1].1;
            let slope = (y1 - y0) / (x1 - x0);
            *y_val = y0 + slope * (x_val - x0);
        }
    }
    new_x.sort_by(|a, b| a.0.cmp(&b.0));
    new_x
        .into_iter()
        .map(|(_, x_val, y_val)| (x_val, y_val))
        .collect()
}

/// https://github.com/cran/glmnet/blob/14f90adf86f80d0fca8100d7a3af944091e3d27b/R/lambda.interp.R
fn lambda_interp(lambda: &[f64], s: &[f64]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    if lambda.len() == 1 {
        let nums = s.len();
        return (vec![0; nums], vec![0; nums], vec![1.0; nums]);
    }
    let k = lambda.len();
    let mut sfrac = s
        .iter()
        .map(|&s| (lambda[0] - s) / (lambda[0] - lambda[k - 1]))
        .collect::<Vec<f64>>();
    let lambda = lambda
        .iter()
        .map(|&l| (lambda[0] - l) / (lambda[0] - lambda[k - 1]))
        .collect::<Vec<f64>>();
    let min_lambda = *lambda
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let max_lambda = *lambda
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    sfrac.iter_mut().for_each(|s| {
        if *s < min_lambda {
            *s = min_lambda;
        }
        if *s > max_lambda {
            *s = max_lambda;
        }
    });
    let coord: Vec<f64> = approx(
        &lambda,
        &(0..k).map(|i| i as f64).collect::<Vec<_>>(),
        &sfrac,
    )
    .into_iter()
    .map(|(_, y)| y)
    .collect();
    let left: Vec<_> = coord.iter().map(|&c| c.floor() as usize).collect();
    let right: Vec<_> = coord.iter().map(|&c| c.ceil() as usize).collect();
    sfrac.iter_mut().enumerate().for_each(|(i, s)| {
        *s = (*s - lambda[right[i]]) / (lambda[left[i]] - lambda[right[i]]);
    });
    for i in 0..sfrac.len() {
        if left[i] == right[i] || (lambda[left[i]] - lambda[right[i]]).abs() < f64::EPSILON {
            sfrac[i] = 1.0;
        }
    }
    (left, right, sfrac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approx() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 4.0, 10.0, 20.0];
        let new_x = vec![1.5, 2.5, 3.5];
        let result = approx(&x, &y, &new_x);
        assert_eq!(result.len(), 3);
        assert!((result[0].1 - 2.5).abs() < 1e-6);
        assert!((result[1].1 - 7.0).abs() < 1e-6);
        assert!((result[2].1 - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_lambda_interp() {
        let lambda = vec![0.1, 0.2, 0.3, 0.4];
        let s = vec![0.15, 0.21, 0.37];
        let (left, right, frac) = lambda_interp(&lambda, &s);
        assert_eq!(left, vec![0, 1, 2]);
        assert_eq!(right, vec![1, 2, 3]);
        assert!((frac[0] - 0.5).abs() < 1e-6);
        assert!((frac[1] - 0.9).abs() < 1e-6);
        assert!((frac[2] - 0.3).abs() < 1e-6);
    }
}

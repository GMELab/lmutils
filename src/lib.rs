#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![allow(dead_code, unused)]
mod calc;
mod error;
mod file;
mod mat;
mod matrix;

use std::{mem::MaybeUninit, panic::AssertUnwindSafe, sync::Mutex};

use rayon::prelude::*;
use tracing::{debug, debug_span, error, info, trace, warn};

pub use crate::{calc::*, error::*, file::*, matrix::*};

#[cfg_attr(coverage_nightly, coverage(off))]
#[doc(hidden)]
pub fn core_parallelize<T, F, R, E>(data: Vec<T>, out: Option<usize>, f: F) -> Result<Vec<R>, E>
where
    T: Send + Sync,
    for<'a> F: (Fn(usize, &'a mut T) -> Result<Vec<R>, E>) + Send + Sync,
    R: Send + Sync,
    E: std::error::Error,
{
    let core_parallelism = std::env::var("LMUTILS_CORE_PARALLELISM")
        .ok()
        .or_else(|| std::env::var("LMUTILS_NUM_MAIN_THREADS").ok())
        .and_then(|x| x.parse::<usize>().ok())
        .unwrap_or(16)
        .clamp(1, data.len().min(num_cpus::get()));
    let ignore_errors = std::env::var("LMUTILS_IGNORE_CORE_PARALLEL_ERRORS")
        .ok()
        .map(|x| x == "1")
        .unwrap_or(true);

    let mut results_uninit = if let Some(out) = out {
        // if there's a known output size, we can preallocate the results and guarantee
        // an order
        let mut results: Vec<MaybeUninit<R>> = Vec::with_capacity(out * data.len());
        results.extend((0..(out * data.len())).map(|_| MaybeUninit::uninit()));
        Some(results)
    } else {
        // if there isn't, then we can't guarantee an order, so we just return None
        None
    };
    let mut results_push = if out.is_none() {
        Some(Mutex::new(Vec::new()))
    } else {
        None
    };

    let data = Mutex::new(data.into_iter().enumerate().collect::<Vec<_>>());

    std::thread::scope(|s| {
        for _ in 0..core_parallelism {
            s.spawn(|| {
                loop {
                    let mut guard = data.lock().unwrap();
                    let d = guard.pop();
                    drop(guard);
                    if let Some((i, mut d)) = d {
                        rayon::scope(|s| {
                            s.spawn(|_| {
                                let s = debug_span!("core_scope");
                                let _e = s.enter();
                                let mut tries = 1;
                                #[allow(clippy::blocks_in_conditions)]
                                while std::panic::catch_unwind(AssertUnwindSafe(|| {
                                    let r = f(i, &mut d).unwrap();
                                    if let Some(out) = out {
                                        let results = unsafe {
                                            std::slice::from_raw_parts_mut(
                                                results_uninit
                                                    .as_ref()
                                                    .unwrap()
                                                    .as_ptr()
                                                    .add(i * out)
                                                    .cast_mut(),
                                                out,
                                            )
                                        };
                                        for (i, p) in r.into_iter().enumerate() {
                                            results[i].write(p);
                                        }
                                    } else {
                                        let mut results =
                                            results_push.as_ref().unwrap().lock().unwrap();
                                        results.extend(r);
                                    }
                                }))
                                .is_err()
                                {
                                    let duration = std::time::Duration::from_secs(4u64.pow(tries));
                                    warn!(
                                        "Error in core scope, retrying in {} seconds",
                                        duration.as_secs_f64()
                                    );
                                    std::thread::sleep(duration);
                                    tries += 1;
                                    if tries > 5 {
                                        if ignore_errors {
                                            error!("Error in core scope, ignoring");
                                            break;
                                        } else {
                                            error!("Error in core scope, too many retries");
                                            panic!("Error in core scope, too many retries");
                                        }
                                    }
                                }
                            })
                        })
                    } else {
                        break;
                    }
                }
            });
        }
    });

    Ok(if let Some(out) = out {
        // SAFETY: We have initialized all elements of the array.
        unsafe { std::mem::transmute::<Vec<MaybeUninit<R>>, Vec<R>>(results_uninit.unwrap()) }
    } else {
        results_push.unwrap().into_inner().unwrap()
    })
}

// Calculate R^2 and adjusted R^2 for a list of data and outcomes.
#[tracing::instrument(skip(data, outcomes, data_names))]
pub fn calculate_r2s(
    mut data: Vec<Matrix>,
    mut outcomes: Matrix,
    data_names: Option<Vec<&str>>,
) -> Result<Vec<R2>, crate::Error> {
    outcomes.remove_column_by_name_if_exists("eid");
    outcomes.remove_column_by_name_if_exists("IID");
    let colnames = outcomes
        .colnames()?
        .map(|x| x.into_iter().map(|x| x.to_string()).collect::<Vec<_>>());
    let or = outcomes.as_mat_ref()?;
    // let data = Mutex::new(
    //     data.into_iter()
    //         .enumerate()
    //         .map(|(i, m)| m.make_parallel_safe().map(|m| (i, m)))
    //         .collect::<Result<Vec<_>, _>>()?,
    // );
    // let ndata = data.lock().unwrap().len();
    // let mut results: Vec<MaybeUninit<R2>> = Vec::with_capacity(or.ncols() *
    // ndata); results.extend((0..(or.ncols() * ndata)).map(|_|
    // MaybeUninit::uninit()));
    let data_names = match data_names {
        Some(data_names) if data_names.iter().all(|x| *x == "NA") => None,
        x => x,
    };
    core_parallelize(data, Some(or.ncols()), |i, mat| {
        info!(
            "Calculating R^2 for data set {}",
            if let Some(data_names) = &data_names {
                data_names[i].to_string()
            } else {
                (i + 1).to_string()
            }
        );
        if !mat.is_loaded() {
            mat.into_owned()?;
        }
        if mat.has_column_loaded("eid") || mat.has_column_loaded("IID") {
            mat.remove_column_by_name_if_exists("eid")?;
            mat.remove_column_by_name_if_exists("IID")?;
        }
        let r = mat.as_mat_ref_loaded();
        let r2s = get_r2s(r, or)
            .into_iter()
            .enumerate()
            .map(|(j, mut r)| {
                if let Some(data_names) = &data_names {
                    r.data = Some(data_names[i].to_string());
                } else {
                    r.data = Some((i + 1).to_string());
                }
                r.outcome = colnames
                    .as_ref()
                    .and_then(|c| c.get(j).map(|c| c.to_string()))
                    .or_else(|| Some((j + 1).to_string()));
                r
            })
            .collect::<Vec<_>>();
        debug!("Writing results");
        info!(
            "Finished calculating R^2 for data set {}",
            if let Some(data_names) = &data_names {
                data_names[i].to_string()
            } else {
                i.to_string()
            }
        );
        Ok(r2s)
    })
}

#[tracing::instrument(skip(data, outcomes, data_names))]
pub fn column_p_values(
    mut data: Vec<Matrix>,
    mut outcomes: Matrix,
    data_names: Option<Vec<&str>>,
) -> Result<Vec<PValue>, crate::Error> {
    outcomes.remove_column_by_name_if_exists("eid");
    outcomes.remove_column_by_name_if_exists("IID");
    let colnames = outcomes
        .colnames()?
        .map(|x| x.into_iter().map(|x| x.to_string()).collect::<Vec<_>>());
    let or = outcomes.as_mat_ref()?;
    // let data = Mutex::new(
    //     data.into_iter()
    //         .enumerate()
    //         .map(|(i, m)| m.make_parallel_safe().map(|m| (i, m)))
    //         .collect::<Result<Vec<_>, _>>()?,
    // );
    // let ndata = or.nrows();
    // let mut results: Vec<MaybeUninit<PValue>> = Vec::with_capacity(or.ncols() *
    // ndata); results.extend((0..(or.ncols() * ndata)).map(|_|
    // MaybeUninit::uninit()));
    core_parallelize(data, None, |i, mat| {
        info!(
            "Calculating p-values for data set {}",
            if let Some(data_names) = &data_names {
                data_names[i].to_string()
            } else {
                (i + 1).to_string()
            }
        );
        if !mat.is_loaded() {
            mat.into_owned()?;
        }
        if mat.has_column_loaded("eid") || mat.has_column_loaded("IID") {
            mat.remove_column_by_name_if_exists("eid")?;
            mat.remove_column_by_name_if_exists("IID")?;
        }
        let data = mat.as_mat_ref_loaded();
        let p_values = (0..data.ncols())
            .into_par_iter()
            .flat_map(|x| {
                let xs = data.get(.., x).try_as_slice().unwrap();
                (0..or.ncols()).into_par_iter().map(move |y| {
                    let ys = or.get(.., y).try_as_slice().unwrap();
                    (x, y, p_value(xs, ys))
                })
            })
            .map(|(x, y, mut p)| {
                if let Some(data_names) = &data_names {
                    p.data = Some(data_names[i].to_string());
                } else {
                    p.data = Some((i + 1).to_string());
                }
                p.data_column = Some((x + 1) as u32);
                p.outcome = colnames
                    .as_ref()
                    .and_then(|c| c.get(y).map(|c| c.to_string()))
                    .or_else(|| Some((y + 1).to_string()));
                p
            })
            .collect::<Vec<_>>();
        info!(
            "Finished calculating p-values for data set {}",
            if let Some(data_names) = &data_names {
                data_names[i].to_string()
            } else {
                i.to_string()
            }
        );
        Ok(p_values)
    })
}

pub fn compute_r2(actual: &[f64], predicted: &[f64]) -> f64 {
    R2Simd::new(actual, predicted).calculate()
}

#[cfg(test)]
mod tests {
    use test_log::test;

    use super::*;

    #[test]
    fn test_calculate_r2s() {
        let mut m1 = Matrix::Owned(OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            None,
        ));
        let mut m2 = Matrix::Owned(OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            None,
        ));
        let mut m3 = Matrix::Owned(OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            None,
        ));
        let results = calculate_r2s(vec![m1, m2], m3, None).unwrap();
        assert_eq!(results.len(), 6);
    }

    #[test]
    fn test_calculate_r2s_names() {
        let mut m1 = Matrix::Owned(OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        ));
        let mut m2 = Matrix::Owned(OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Some(vec!["d".to_string(), "e".to_string(), "f".to_string()]),
        ));
        let mut m3 = Matrix::Owned(OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Some(vec!["g".to_string(), "h".to_string(), "i".to_string()]),
        ));
        let results = calculate_r2s(vec![m1, m2], m3, Some(vec!["j", "k"])).unwrap();
        assert_eq!(results.len(), 6);
    }

    #[test]
    fn test_column_p_values() {
        let mut m1 = Matrix::Owned(OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            None,
        ));
        let mut m2 = Matrix::Owned(OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            None,
        ));
        let mut m3 = Matrix::Owned(OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            None,
        ));
        let results = column_p_values(vec![m1, m2], m3, None).unwrap();
        assert_eq!(results.len(), 18);
    }

    #[test]
    fn test_column_p_values_names() {
        let mut m1 = Matrix::Owned(OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        ));
        let mut m2 = Matrix::Owned(OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Some(vec!["d".to_string(), "e".to_string(), "f".to_string()]),
        ));
        let mut m3 = Matrix::Owned(OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Some(vec!["g".to_string(), "h".to_string(), "i".to_string()]),
        ));
        let results = column_p_values(vec![m1, m2], m3, Some(vec!["j", "k"])).unwrap();
        assert_eq!(results.len(), 18);
    }
}

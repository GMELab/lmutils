mod calc;
mod errors;
mod file;
mod matrix;
mod transform;

use std::{mem::MaybeUninit, panic::AssertUnwindSafe, sync::Mutex};

use rayon::prelude::*;
use tracing::{debug, debug_span, error, info, trace};

pub use crate::{calc::*, errors::*, file::*, matrix::*, transform::*};

fn main_scope<T, F>(data: Mutex<Vec<T>>, f: F)
where
    T: Clone + Send + Sync,
    F: Fn(T) + Send + Sync,
{
    let blocks_at_once = std::env::var("LMUTILS_NUM_MAIN_THREADS")
        .ok()
        .and_then(|x| x.parse::<usize>().ok())
        .unwrap_or(16)
        .clamp(1, data.lock().unwrap().len());

    std::thread::scope(|s| {
        for _ in 0..blocks_at_once {
            s.spawn(|| loop {
                let mut guard = data.lock().unwrap();
                let data = guard.pop();
                drop(guard);
                if let Some(data) = data {
                    rayon::scope(|s| {
                        s.spawn(|_| {
                            let s = debug_span!("main_scope");
                            let _e = s.enter();
                            let mut tries = 1;
                            while std::panic::catch_unwind(AssertUnwindSafe(|| f(data.clone())))
                                .is_err()
                            {
                                let duration = std::time::Duration::from_secs(4u64.pow(tries));
                                error!(
                                    "Error in main scope, retrying in {} seconds",
                                    duration.as_secs_f64()
                                );
                                std::thread::sleep(duration);
                                tries += 1;
                                if tries > 5 {
                                    panic!("Error in main scope, too many retries");
                                }
                            }
                        })
                    })
                } else {
                    break;
                }
            });
        }
    });
}

/// Convert a file from one format to another.
/// `from` is the file to read.
/// `to` is the file to write.
/// `item_type` is the type of data to read and write.
/// Returns `Ok(())` if successful.
pub fn convert_file(from: &str, to: &str) -> Result<(), ConvertFileError> {
    let from: File = from.parse()?;
    let to: File = to.parse()?;
    let mut mat = from.read()?;
    to.write(&mut mat)?;
    Ok(())
}

/// Calculate R^2 and adjusted R^2 for a list of data and outcomes.
#[tracing::instrument(skip(data, outcomes, data_names))]
pub fn calculate_r2s<'a>(
    data: Vec<impl Transform<'a>>,
    outcomes: impl Transform<'a>,
    data_names: Option<Vec<&str>>,
) -> Result<Vec<R2>, ReadMatrixError> {
    let mut outcomes = outcomes.transform()?;
    outcomes.remove_column_by_name_if_exists("eid").unwrap();
    outcomes.remove_column_by_name_if_exists("IID").unwrap();
    let colnames = outcomes
        .colnames()?
        .map(|x| x.into_iter().map(|x| x.to_string()).collect::<Vec<_>>());
    let or = outcomes.as_mat_ref()?;
    let data = Mutex::new(
        data.into_iter()
            .enumerate()
            .map(|(i, m)| m.make_parallel_safe().map(|m| (i, m)))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let ndata = data.lock().unwrap().len();
    let mut results: Vec<MaybeUninit<R2>> = Vec::with_capacity(or.ncols() * ndata);
    results.extend((0..(or.ncols() * ndata)).map(|_| MaybeUninit::uninit()));
    main_scope(data, |(i, data)| {
        info!(
            "Calculating R^2 for data set {}",
            if let Some(data_names) = &data_names {
                data_names[i].to_string()
            } else {
                (i + 1).to_string()
            }
        );
        let mut mat = data.transform().unwrap();
        mat.remove_column_by_name_if_exists("eid").unwrap();
        mat.remove_column_by_name_if_exists("IID").unwrap();
        let r = mat.as_mat_ref().unwrap();
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
        let results = unsafe {
            std::slice::from_raw_parts_mut(
                results.as_ptr().add(i * or.ncols()).cast_mut(),
                or.ncols(),
            )
        };
        trace!("Made slice");
        for (i, p) in r2s.into_iter().enumerate() {
            trace!("Writing result {}", i);
            results[i].write(p);
            trace!("Wrote result {}", i)
        }
        trace!("Wrote results");
        info!(
            "Finished calculating R^2 for data set {}",
            if let Some(data_names) = &data_names {
                data_names[i].to_string()
            } else {
                i.to_string()
            }
        );
    });
    Ok(unsafe { std::mem::transmute::<_, Vec<R2>>(results) })
}

pub fn column_p_values<'a>(
    data: Vec<impl Transform<'a>>,
    outcomes: impl Transform<'a>,
    data_names: Option<Vec<&str>>,
) -> Result<Vec<PValue>, ReadMatrixError> {
    let mut outcomes = outcomes.transform()?;
    outcomes.remove_column_by_name_if_exists("eid").unwrap();
    outcomes.remove_column_by_name_if_exists("IID").unwrap();
    let colnames = outcomes
        .colnames()?
        .map(|x| x.into_iter().map(|x| x.to_string()).collect::<Vec<_>>());
    let or = outcomes.as_mat_ref()?;
    let data = Mutex::new(
        data.into_iter()
            .enumerate()
            .map(|(i, m)| m.make_parallel_safe().map(|m| (i, m)))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let ndata = or.nrows();
    let mut results: Vec<MaybeUninit<PValue>> = Vec::with_capacity(or.ncols() * ndata);
    results.extend((0..(or.ncols() * ndata)).map(|_| MaybeUninit::uninit()));
    main_scope(data, |(i, data)| {
        info!(
            "Calculating p-values for data set {}",
            if let Some(data_names) = &data_names {
                data_names[i].to_string()
            } else {
                (i + 1).to_string()
            }
        );
        let mut mat = data.transform().unwrap();
        mat.remove_column_by_name_if_exists("eid").unwrap();
        mat.remove_column_by_name_if_exists("IID").unwrap();
        let data = mat.as_mat_ref().unwrap();
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
        let results = unsafe {
            std::slice::from_raw_parts_mut(
                results.as_ptr().add(i * or.ncols()).cast_mut(),
                or.ncols(),
            )
        };
        for (i, p) in p_values.into_iter().enumerate() {
            results[i].write(p);
        }
        info!(
            "Finished calculating p-values for data set {}",
            if let Some(data_names) = &data_names {
                data_names[i].to_string()
            } else {
                i.to_string()
            }
        );
    });
    Ok(unsafe { std::mem::transmute::<_, Vec<PValue>>(results) })
}

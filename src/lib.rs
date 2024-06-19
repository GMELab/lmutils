mod calc;
mod errors;
mod file;
mod matrix;
pub mod r;
mod transform;

use std::sync::Mutex;

use rayon::prelude::*;

pub use crate::{calc::*, errors::*, file::*, matrix::*, transform::*};

fn main_scope<T, F>(data: Mutex<Vec<T>>, f: F)
where
    T: Send,
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
                            f(data);
                        })
                    });
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
pub fn convert_file(
    from: &str,
    to: &str,
    item_type: TransitoryType,
) -> Result<(), ConvertFileError> {
    let from: File = from.parse()?;
    let to: File = to.parse()?;
    let mat = from.read_transitory(item_type)?;
    to.write_transitory(&mat)?;
    Ok(())
}

/// Calculate R^2 and adjusted R^2 for a block and outcomes.
pub fn calculate_r2<'a>(
    data: impl Transform<'a>,
    outcomes: impl Transform<'a>,
) -> Result<Vec<R2>, ReadMatrixError> {
    let data = data.transform()?;
    let mut outcomes = outcomes.transform()?;
    outcomes.remove_column_by_name_if_exists("eid");
    outcomes.remove_column_by_name_if_exists("IID");
    let data = data.as_mat_ref()?;
    let outcomes = outcomes.as_mat_ref()?;
    Ok(get_r2s(data, outcomes))
}

/// Calculate R^2 and adjusted R^2 for a list of data and outcomes.
pub fn calculate_r2s<'a>(
    data: Vec<impl Transform<'a> + Send>,
    outcomes: impl Transform<'a>,
    data_names: Option<Vec<&str>>,
) -> Result<Vec<R2>, ReadMatrixError> {
    let mut outcomes = outcomes.transform()?;
    outcomes.remove_column_by_name_if_exists("eid");
    outcomes.remove_column_by_name_if_exists("IID");
    let colnames = outcomes
        .colnames()
        .map(|x| x.into_iter().map(|x| x.to_string()).collect::<Vec<_>>());
    let or = outcomes.as_mat_ref()?;
    let data = Mutex::new(
        data.into_iter()
            .enumerate()
            .map(|(i, m)| m.make_parallel_safe().map(|m| (i, m)))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let results = Mutex::new(Vec::new());
    main_scope(data, |(i, data)| {
        let mut mat = data.transform().unwrap();
        mat.remove_column_by_name_if_exists("eid");
        mat.remove_column_by_name_if_exists("IID");
        let r = mat.as_mat_ref().unwrap();
        let r2s = get_r2s(r, or)
            .into_iter()
            .enumerate()
            .map(|(j, mut r)| {
                if let Some(data_names) = &data_names {
                    r.data = Some(data_names[i].to_string());
                }
                r.outcome = colnames
                    .as_ref()
                    .and_then(|c| c.get(j).map(|c| c.to_string()))
                    .or_else(|| Some((j + 1).to_string()));
                r
            })
            .collect::<Vec<_>>();
        results.lock().unwrap().extend(r2s);
    });
    Ok(results.into_inner().unwrap())
}

pub fn column_p_values<'a>(
    data: Vec<impl Transform<'a>>,
    outcomes: impl Transform<'a>,
    data_names: Option<Vec<&str>>,
) -> Result<Vec<PValue>, ReadMatrixError> {
    let mut outcomes = outcomes.transform()?;
    outcomes.remove_column_by_name_if_exists("eid");
    outcomes.remove_column_by_name_if_exists("IID");
    let colnames = outcomes
        .colnames()
        .map(|x| x.into_iter().map(|x| x.to_string()).collect::<Vec<_>>());
    let or = outcomes.as_mat_ref()?;
    let data = Mutex::new(
        data.into_iter()
            .enumerate()
            .map(|(i, m)| m.make_parallel_safe().map(|m| (i, m)))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let results = Mutex::new(Vec::new());
    main_scope(data, |(i, data)| {
        let mut mat = data.transform().unwrap();
        mat.remove_column_by_name_if_exists("eid");
        mat.remove_column_by_name_if_exists("IID");
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
                }
                p.data_column = Some(x as u32);
                p.outcome = colnames
                    .as_ref()
                    .and_then(|c| c.get(y).map(|c| c.to_string()))
                    .or_else(|| Some((y + 1).to_string()));
                p
            })
            .collect::<Vec<_>>();
        results.lock().unwrap().extend(p_values);
    });
    Ok(results.into_inner().unwrap())
}

mod calc;
mod errors;
mod file;
mod matrix;
pub mod r;
mod transform;

use rayon::prelude::*;

pub use crate::{calc::*, errors::*, file::*, matrix::*, transform::*};

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
    let outcomes = outcomes.transform()?;
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
    let colnames = outcomes
        .colnames()
        .map(|x| x.into_iter().map(|x| x.to_string()).collect::<Vec<_>>());
    let or = outcomes.as_mat_ref()?;
    let data = data
        .into_iter()
        .map(|i| i.make_parallel_safe())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(data
        .into_par_iter()
        .enumerate()
        .map(|(i, data)| {
            let mat = data.transform()?;
            let r = mat.as_mat_ref()?;
            Ok(get_r2s(r, or)
                .into_iter()
                .enumerate()
                .map(|(j, mut r)| {
                    if let Some(data_names) = &data_names {
                        r.data = Some(data_names[i].to_string());
                    }
                    r.outcome = colnames
                        .as_ref()
                        .and_then(|c| c.get(j).map(|c| c.to_string()));
                    r
                })
                .collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>, ReadMatrixError>>()?
        .into_iter()
        .flatten()
        .collect())
}

mod calc;
mod errors;
mod file;
mod matrix;
mod r;
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
    to.write_transitory(&mat);
    Ok(())
}

/// Convert files from one format to another.
/// `files` is a list of tuples of file names to read and write.
/// `item_type` is the type of data to read and write.
/// Returns `Ok(())` if successful.
pub fn convert_files(
    files: &[(&str, &str)],
    item_type: TransitoryType,
) -> Result<(), ConvertFileError> {
    for (from, to) in files {
        convert_file(from, to, item_type)?;
    }
    Ok(())
}

/// Calculate R^2 and adjusted R^2 for a block and outcomes.
pub fn calculate_r2<'a>(
    data: impl Transform<'a>,
    outcomes: impl Transform<'a>,
) -> Result<Vec<R2>, ReadMatrixError> {
    let mut data = data.transform()?;
    let mut outcomes = outcomes.transform()?;
    let data = data.as_mat_ref()?;
    let outcomes = outcomes.as_mat_ref()?;
    Ok(get_r2s(data, outcomes))
}

/// Calculate R^2 and adjusted R^2 for a list of data and outcomes.
pub fn calculate_r2s<'a>(
    data: Vec<impl Transform<'a> + Send>,
    outcomes: impl Transform<'a>,
) -> Result<Vec<R2>, ReadMatrixError> {
    let mut outcomes = outcomes.transform()?;
    let outcomes = outcomes.as_mat_ref()?;
    let data = data
        .into_iter()
        .map(|i| i.make_parallel_safe())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(data
        .into_par_iter()
        .map(|i| {
            let mut i = i.transform()?;
            let i = i.as_mat_ref()?;
            Ok(get_r2s(i, outcomes).into_iter())
        })
        .collect::<Result<Vec<_>, ReadMatrixError>>()?
        .into_iter()
        .flatten()
        .collect())
}

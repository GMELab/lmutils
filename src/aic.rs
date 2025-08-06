use std::collections::HashSet;

use faer::{get_global_parallelism, set_global_parallelism, Mat};
use rayon::prelude::*;

use crate::{Error, Family, Glm, Matrix, DISABLE_PREDICTED};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum From {
    Null,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Backward,
    Both,
}

impl Direction {
    pub fn forward(&self) -> bool {
        matches!(self, Direction::Forward | Direction::Both)
    }

    pub fn backward(&self) -> bool {
        matches!(self, Direction::Backward | Direction::Both)
    }
}

#[tracing::instrument(skip(mat, outcome, model, colnames))]
fn step_aic_inner<F: Family>(
    mut mat: Matrix,
    outcome: &[f64],
    mut model: Glm,
    direction: Direction,
    steps: usize,
    colnames: Vec<String>,
) -> Result<Glm, Error> {
    let ncols = mat.ncols()?;
    let mut colnames = colnames.into_iter().collect::<HashSet<_>>();
    let mut working_set = model
        .coefs()
        .iter()
        // we don't want our intercept
        .filter(|x| !x.intercept())
        .map(|x| x.label().to_string())
        .collect::<HashSet<_>>();
    let mat = mat.to_owned_loaded();
    for _ in 0..steps {
        // we want to (in parallel) build our set of candidate models
        // if we can go forward, we want a model that includes each column not in the working set
        // if we can go backward, we want a model with each column in the working set removed

        let forward_candidates = if direction.forward() {
            Some(
                colnames
                    .difference(&working_set)
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .map(|x| {
                        let mut cols = working_set.clone();
                        cols.insert(x.to_string());
                        let mut mat = Matrix::Owned(mat.clone());
                        mat.subset_columns_by_name(&cols);
                        let mut colnames = mat
                            .colnames_loaded()
                            .unwrap()
                            .into_iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<_>>();
                        Ok(Glm::irls::<F>(
                            mat.as_mat_ref_loaded(),
                            outcome,
                            1e-6,
                            100,
                            true,
                            false,
                            Some(colnames.as_slice()),
                        ))
                    }),
            )
        } else {
            None
        };
        let backward_candidates = if direction.backward() {
            Some(
                working_set
                    .iter()
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .map(|x| {
                        let mut cols = working_set.clone();
                        cols.remove(x);
                        let mut mat = Matrix::Owned(mat.clone());
                        mat.subset_columns_by_name(&cols);
                        let mut colnames = mat
                            .colnames_loaded()
                            .unwrap()
                            .into_iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<_>>();
                        Ok(Glm::irls::<F>(
                            mat.as_mat_ref_loaded(),
                            outcome,
                            1e-6,
                            100,
                            true,
                            false,
                            Some(colnames.as_slice()),
                        ))
                    }),
            )
        } else {
            None
        };
        let candidates = match (forward_candidates, backward_candidates) {
            (Some(f), Some(b)) => f.chain(b).collect::<Result<Vec<_>, Error>>()?,
            (Some(f), None) => f.collect::<Result<Vec<_>, Error>>()?,
            (None, Some(b)) => b.collect::<Result<Vec<_>, Error>>()?,
            (None, None) => unreachable!(),
        };
        if candidates.is_empty() {
            tracing::debug!("No candidates found, stopping AIC step.");
            break;
        }

        let mut best_model: Option<Glm> = None;
        for candidate in candidates {
            if let Some(best) = &best_model {
                if candidate.aic() < best.aic() {
                    best_model = Some(candidate);
                }
            } else {
                best_model = Some(candidate);
            }
        }
        if let Some(best) = best_model {
            if best.aic() < model.aic() {
                model = best;
                // update our working set
                let new_working_set: HashSet<String> = model
                    .coefs()
                    .iter()
                    .filter(|x| !x.intercept())
                    .map(|x| x.label().to_string())
                    .collect();
                // log if we're adding or removing columns
                if new_working_set.len() > working_set.len() {
                    tracing::info!(
                        "Adding columns to working set: {:?}",
                        new_working_set.difference(&working_set)
                    );
                } else if new_working_set.len() < working_set.len() {
                    tracing::info!(
                        "Removing columns from working set: {:?}",
                        working_set.difference(&new_working_set)
                    );
                }
                working_set = new_working_set;
            } else {
                // we didn't find a better model, so we're done
                break;
            }
        } else {
            // no candidates, we're done
            break;
        }
    }

    Ok(model)
}

#[tracing::instrument(skip(mat, outcome))]
pub fn step_aic<F: Family>(
    mut mat: Matrix,
    outcome: &[f64],
    from: From,
    direction: Direction,
    steps: usize,
) -> Result<Glm, Error> {
    let nrows = mat.nrows()?;
    let colnames = mat
        .colnames()?
        .ok_or(Error::MissingColumnNames)?
        .into_iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>();
    mat.into_owned()?;
    let par = get_global_parallelism();
    set_global_parallelism(faer::Par::Seq);
    unsafe {
        DISABLE_PREDICTED = true;
    }

    let model = match from {
        From::Null => Glm::irls::<F>(
            Mat::ones(nrows, 0).as_ref(),
            outcome,
            1e-6,
            100,
            true,
            false,
            None,
        ),
        From::Full => Glm::irls::<F>(
            mat.as_mat_ref_loaded(),
            outcome,
            1e-6,
            100,
            true,
            false,
            Some(colnames.as_slice()),
        ),
    };
    let result = step_aic_inner::<F>(mat, outcome, model, direction, steps, colnames);
    set_global_parallelism(par);
    unsafe {
        DISABLE_PREDICTED = false;
    }
    result
}

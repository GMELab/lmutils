use std::collections::HashSet;

use crate::{file::FileType, matrix::OwnedMatrix, ReadMatrixError};
use log::debug;
use rayon::prelude::*;

use crate::matrix::Matrix;

pub trait Transform<'a> {
    /// Apply the transformation to the matrix.
    fn transform(self) -> Result<Matrix<'a>, ReadMatrixError>;

    /// If the matrix points to a .RData file, loading these in parallel is not thread safe so we
    /// need to load it first.
    fn make_parallel_safe(self) -> Result<Self, ReadMatrixError>
    where
        Self: Sized;

    /// Standardize the matrix.
    fn standardization(self) -> Standardization<'a, Self>
    where
        Self: Sized,
    {
        Standardization::new(self)
    }

    /// Remove NaN rows.
    fn remove_nan_rows(self) -> RemoveNanRows<'a, Self>
    where
        Self: Sized,
    {
        RemoveNanRows::new(self)
    }

    /// NaN to mean.
    fn nan_to_mean(self) -> NanToMean<'a, Self>
    where
        Self: Sized,
    {
        NanToMean::new(self)
    }

    /// Column sum is at least `min_sum`.
    fn min_sum(self, min_sum: f64) -> MinSum<'a, Self>
    where
        Self: Sized,
    {
        MinSum::new(self, min_sum)
    }
}

impl<'a> Transform<'a> for Matrix<'a> {
    fn transform(self) -> Result<Matrix<'a>, ReadMatrixError> {
        let mat = match self {
            Matrix::File(f) => Matrix::Owned(f.read_matrix(true)?),
            _ => self,
        };
        debug!("Loaded matrix");
        Ok(mat)
    }

    fn make_parallel_safe(self) -> Result<Matrix<'a>, ReadMatrixError>
    where
        Self: Sized,
    {
        Ok(match self {
            Matrix::R(r) => Matrix::R(r),
            Matrix::Owned(m) => Matrix::Owned(m),
            Matrix::File(f) => match f.file_type() {
                FileType::Rdata => Matrix::Owned(f.read_matrix(true)?),
                _ => Matrix::File(f),
            },
            Matrix::Ref(r) => Matrix::Ref(r),
        })
    }
}

macro_rules! simple_transform {
    ($name:ident) => {
        pub struct $name<'a, T>
        where
            T: Transform<'a>,
        {
            parent: T,
            __phantom: std::marker::PhantomData<&'a ()>,
        }

        impl<'a, T> $name<'a, T>
        where
            T: Transform<'a>,
        {
            fn new(parent: T) -> Self {
                $name {
                    parent,
                    __phantom: std::marker::PhantomData,
                }
            }
        }
    };
}

simple_transform!(Standardization);
simple_transform!(RemoveNanRows);
simple_transform!(NanToMean);

pub struct MinSum<'a, T>
where
    T: Transform<'a>,
{
    parent: T,
    min_sum: f64,
    __phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, T> MinSum<'a, T>
where
    T: Transform<'a>,
{
    fn new(parent: T, min_sum: f64) -> Self {
        MinSum {
            parent,
            min_sum,
            __phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T> Transform<'a> for Standardization<'a, T>
where
    T: Transform<'a>,
{
    fn transform(self) -> Result<Matrix<'a>, ReadMatrixError> {
        let mut mat = self.parent.transform()?;
        mat.as_mat_mut()?.par_col_chunks_mut(1).for_each(|col| {
            let mut mean = 0.0;
            let mut std = 0.0;
            faer::stats::row_mean(
                faer::row::from_mut(&mut mean),
                col.as_ref(),
                faer::stats::NanHandling::Ignore,
            );
            faer::stats::row_varm(
                faer::row::from_mut(&mut std),
                col.as_ref(),
                faer::row::from_ref(&mean),
                faer::stats::NanHandling::Ignore,
            );
            let std = std.sqrt();
            for x in col.col_mut(0).iter_mut() {
                *x = (*x - mean) / std;
            }
        });
        debug!("Standardized matrix");
        Ok(mat)
    }

    fn make_parallel_safe(self) -> Result<Self, ReadMatrixError> {
        Ok(Self {
            parent: self.parent.make_parallel_safe()?,
            __phantom: std::marker::PhantomData,
        })
    }
}

impl<'a, T> Transform<'a> for RemoveNanRows<'a, T>
where
    T: Transform<'a>,
{
    fn transform(self) -> Result<Matrix<'a>, ReadMatrixError> {
        let mat = self.parent.transform()?;
        let rows = mat
            .as_mat_ref()?
            .par_row_chunks(1)
            .enumerate()
            .filter(|(_, row)| row.is_all_finite())
            .map(|(i, _)| i)
            .collect::<HashSet<_>>();
        debug!("Removed {} rows with NaN values", rows.len());
        Matrix::remove_rows(mat, &rows)
    }

    fn make_parallel_safe(self) -> Result<Self, ReadMatrixError> {
        Ok(Self {
            parent: self.parent.make_parallel_safe()?,
            __phantom: std::marker::PhantomData,
        })
    }
}

impl<'a, T> Transform<'a> for NanToMean<'a, T>
where
    T: Transform<'a>,
{
    fn transform(self) -> Result<Matrix<'a>, ReadMatrixError> {
        let mut mat = self.parent.transform()?;
        mat.as_mat_mut()?.par_col_chunks_mut(1).for_each(|col| {
            let mut mean = 0.0;
            faer::stats::row_mean(
                faer::row::from_mut(&mut mean),
                col.as_ref(),
                faer::stats::NanHandling::Ignore,
            );
            // let mut count = 0.0;
            // let mut sum = 0.0;
            // for i in slice.iter() {
            //     if i.is_finite() {
            //         count += 1.0;
            //         sum += *i;
            //     }
            // }
            // let mean = sum / count;
            for x in col.col_mut(0).iter_mut() {
                if !x.is_finite() {
                    *x = mean;
                }
            }
        });
        debug!("Replaced NaN values with mean");
        Ok(mat)
    }

    fn make_parallel_safe(self) -> Result<Self, ReadMatrixError> {
        Ok(Self {
            parent: self.parent.make_parallel_safe()?,
            __phantom: std::marker::PhantomData,
        })
    }
}

impl<'a, T> Transform<'a> for MinSum<'a, T>
where
    T: Transform<'a>,
{
    fn transform(self) -> Result<Matrix<'a>, ReadMatrixError> {
        let mut mat = self.parent.transform()?;
        let min_sum = self.min_sum;
        let cols = mat
            .as_mat_ref()?
            .par_col_chunks(1)
            .enumerate()
            .filter(|(_, col)| col.sum() < min_sum)
            .map(|(i, _)| i)
            .collect::<HashSet<_>>();
        // if cols.is_empty() {
        //     return Ok(mat);
        // }
        let m = mat.as_mat_ref()?;
        let nrows = m.nrows();
        let ncols = m.ncols();
        let data = m
            .col_iter()
            .enumerate()
            .filter(|(i, _)| !cols.contains(i))
            .flat_map(|(_, c)| c.iter().copied())
            .collect();
        let mat = Matrix::Owned(OwnedMatrix {
            data,
            rows: nrows,
            cols: ncols - cols.len(),
            colnames: mat.colnames().map(|x| {
                x.iter()
                    .enumerate()
                    .filter(|(i, _)| !cols.contains(i))
                    .map(|(_, x)| x.to_string())
                    .collect()
            }),
        });
        debug!(
            "Removed {} columns with sum less than {}",
            cols.len(),
            min_sum
        );
        Ok(mat)
    }

    fn make_parallel_safe(self) -> Result<Self, ReadMatrixError> {
        Ok(Self {
            parent: self.parent.make_parallel_safe()?,
            min_sum: self.min_sum,
            __phantom: std::marker::PhantomData,
        })
    }
}

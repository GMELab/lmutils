use std::collections::HashSet;

use crate::{calc::standardization, file::FileType, matrix::OwnedMatrix, mean, ReadMatrixError};
use rayon::prelude::*;
use tracing::debug;

use crate::matrix::Matrix;

pub trait Transform<'a>: Clone + Send + Sync {
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
    fn transform(mut self) -> Result<Matrix<'a>, ReadMatrixError> {
        self.into_owned()?;
        debug!("Loaded matrix");
        Ok(self)
    }

    fn make_parallel_safe(self) -> Result<Matrix<'a>, ReadMatrixError>
    where
        Self: Sized,
    {
        Ok(match self {
            Matrix::File(f) => match f.file_type() {
                FileType::Rdata => f.read()?,
                _ => Matrix::File(f),
            },
            _ => self,
        })
    }
}

macro_rules! simple_transform {
    ($name:ident) => {
        #[derive(Clone)]
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

#[derive(Clone)]
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
            let col = col.col_mut(0);
            standardization(col.try_as_slice_mut().unwrap());
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
        let mut mat = self.parent.transform()?;
        let rows = mat
            .as_mat_ref()?
            .par_row_chunks(1)
            .enumerate()
            .filter(|(_, row)| row.is_all_finite())
            .map(|(i, _)| i)
            .collect::<HashSet<_>>();
        debug!("Removed {} rows with NaN values", rows.len());
        mat.remove_rows(&rows).unwrap();
        Ok(mat)
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
            let col = col.col_mut(0);
            let m = mean(col.as_ref().try_as_slice().unwrap());
            for x in col.iter_mut() {
                if !x.is_finite() {
                    *x = m;
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
        debug!("Computing column sums");
        let m = mat.as_mat_ref()?;
        let cols = m
            .par_col_chunks(1)
            .enumerate()
            .filter(|(_, col)| col.sum() >= min_sum)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if cols.len() == m.ncols() {
            return Ok(mat);
        }
        let nrows = m.nrows();
        let ncols = cols.len();
        debug!("Computing data");
        let mut data = Vec::with_capacity(nrows * ncols);
        for i in &cols {
            data.extend_from_slice(m.col(*i).try_as_slice().unwrap());
        }
        let mat = Matrix::Owned(OwnedMatrix {
            data,
            nrows,
            ncols,
            colnames: mat.colnames()?.map(|x| {
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

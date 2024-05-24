use std::collections::HashSet;

use crate::{file::FileType, matrix::OwnedMatrix, r::standardization, ReadMatrixError};
use faer::linalg::zip::MaybeContiguous;
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
        Ok(match self {
            Matrix::File(f) => Matrix::Owned(f.read_matrix(true)?),
            _ => self,
        })
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
        mat.as_mat_mut()?.par_col_chunks_mut(1).for_each(|mut col| {
            let slice: &mut [f64] =
                unsafe { std::mem::transmute(col.get_slice_unchecked((0, 0), col.nrows())) };
            standardization(slice);
        });
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
        mat.as_mat_mut()?.par_col_chunks_mut(1).for_each(|mut col| {
            let slice: &mut [f64] =
                unsafe { std::mem::transmute(col.get_slice_unchecked((0, 0), col.nrows())) };
            let mean = slice.iter().filter(|x| x.is_finite()).sum::<f64>() / slice.len() as f64;
            for x in slice.iter_mut() {
                if !x.is_finite() {
                    *x = mean;
                }
            }
        });
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
        let mat = self.parent.transform()?;
        let min_sum = self.min_sum;
        let cols = mat
            .as_mat_ref()?
            .par_col_chunks(1)
            .enumerate()
            .filter(|(_, col)| col.sum() < min_sum)
            .map(|(i, _)| i)
            .collect::<HashSet<_>>();
        let mat = mat.as_mat_ref()?;
        let nrows = mat.nrows();
        let ncols = mat.ncols();
        let data = mat
            .par_col_chunks(1)
            .enumerate()
            .filter(|(i, _)| !cols.contains(i))
            .flat_map(|(_, c)| {
                (0..c.nrows())
                    .into_par_iter()
                    .map(move |j| unsafe { *c.get_unchecked(j, 0) })
            })
            .collect();
        Ok(Matrix::Owned(OwnedMatrix {
            data,
            rows: nrows,
            cols: ncols - cols.len(),
        }))
    }

    fn make_parallel_safe(self) -> Result<Self, ReadMatrixError> {
        Ok(Self {
            parent: self.parent.make_parallel_safe()?,
            min_sum: self.min_sum,
            __phantom: std::marker::PhantomData,
        })
    }
}

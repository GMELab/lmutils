use std::{collections::HashSet, str::FromStr};

use crate::{
    errors::{CombineMatricesError, FileParseError},
    file::File,
    MatParseError, ReadMatrixError,
};
use extendr_api::{
    AsTypedSlice, FromRobj, IntoRobj, MatrixConversions, RMatrix, Rinternals, Robj, Rstr,
};
use faer::{
    reborrow::{IntoConst, Reborrow},
    MatMut, MatRef,
};
use rayon::prelude::*;

pub enum Matrix<'a> {
    R(RMatrix<f64>),
    Owned(OwnedMatrix<f64>),
    File(File),
    Ref(MatMut<'a, f64>),
}

/// SAFETY: This is only safe when not pointing to a .RData file.
unsafe impl<'a> Send for Matrix<'a> {}

impl<'a> Matrix<'a> {
    pub fn as_ref(&mut self) -> Result<Matrix<'_>, ReadMatrixError> {
        Ok(Matrix::Ref(self.as_mat_mut()?))
    }

    pub fn as_mat_ref(&self) -> Result<MatRef<'_, f64>, ReadMatrixError> {
        Ok(unsafe {
            (*(self as *const Self as *mut Self))
                .as_mat_mut()?
                .into_const()
        })
    }

    pub fn as_mat_mut(&mut self) -> Result<MatMut<'_, f64>, ReadMatrixError> {
        Ok(match self {
            Matrix::R(r) => unsafe {
                let ptr = r.data().as_ptr() as *mut f64;
                faer::mat::from_raw_parts_mut(ptr, r.nrows(), r.ncols(), 1, r.nrows() as isize)
            },
            Matrix::Owned(m) => {
                let rows = m.rows();
                let cols = m.cols();
                faer::mat::from_column_major_slice_mut(m.data.as_mut(), rows, cols)
            },
            Matrix::File(f) => {
                let m = f.read_matrix(true)?;
                *self = Matrix::Owned(m);
                self.as_mat_mut()?
            },
            Matrix::Ref(r) => r.as_mut(),
        })
    }

    pub fn combine(self, mut others: Vec<Matrix<'_>>) -> Result<Self, CombineMatricesError> {
        if others.is_empty() {
            return Ok(self);
        }
        let self_ = self.as_mat_ref()?;
        let others = others
            .iter_mut()
            .map(|x| x.as_mat_ref())
            .collect::<Result<Vec<_>, _>>()?;
        if others.iter().any(|i| i.nrows() != self_.nrows()) {
            return Err(CombineMatricesError::MatrixDimensionsMismatch);
        }
        let mut data = Vec::with_capacity(
            self_.nrows() * self_.ncols()
                + (others.iter().map(|i| i.ncols() * i.nrows()).sum::<usize>()),
        );
        for i in &others {
            for c in 0..i.ncols() {
                unsafe {
                    data.extend(
                        i.get_unchecked(.., c)
                            .try_as_slice()
                            .expect("could not get slice"),
                    )
                };
            }
        }
        Ok(Self::Owned(OwnedMatrix::new(
            self_.nrows(),
            self_.ncols() + others.iter().map(|i| i.ncols()).sum::<usize>(),
            data,
        )))
    }

    pub fn remove_rows(self, removing: &HashSet<usize>) -> Result<Self, ReadMatrixError> {
        let mat = self.as_mat_ref()?;
        let nrows = mat.nrows();
        let ncols = mat.ncols();
        let data = mat
            .par_row_chunks(1)
            .enumerate()
            .filter(|(i, _)| !removing.contains(i))
            .flat_map(|(_, r)| {
                (0..r.ncols())
                    .into_par_iter()
                    .map(move |j| unsafe { *r.get_unchecked(0, j) })
            })
            .collect();
        Ok(Matrix::Owned(OwnedMatrix {
            data,
            rows: nrows - removing.len(),
            cols: ncols,
        }))
    }

    pub fn into_robj(self) -> Result<Robj, ReadMatrixError> {
        Ok(match self {
            Matrix::R(r) => r.into_robj(),
            Matrix::Owned(m) => m.to_rmatrix().into_robj(),
            Matrix::File(f) => f.read_matrix(true)?.into_matrix().into_robj()?,
            Matrix::Ref(r) => r.to_rmatrix().into_robj(),
        })
    }

    pub fn to_owned(self) -> Result<OwnedMatrix<f64>, ReadMatrixError> {
        Ok(match self {
            Matrix::R(r) => OwnedMatrix::from_rmatrix(r),
            Matrix::Owned(m) => m,
            Matrix::File(f) => f.read_matrix(true)?,
            Matrix::Ref(r) => OwnedMatrix::new(
                r.nrows(),
                r.ncols(),
                (0..r.ncols())
                    .flat_map(|i| {
                        r.rb()
                            .get(.., i)
                            .try_as_slice()
                            .expect("matrix should have row stride 1")
                            .iter()
                            .copied()
                    })
                    .collect(),
            ),
        })
    }
}

#[derive(
    Debug, serde::Serialize, serde::Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
#[archive(check_bytes)]
pub struct OwnedMatrix<T>
where
    T: MatEmpty + Clone,
{
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: Vec<T>,
}

impl<T> OwnedMatrix<T>
where
    T: MatEmpty + Clone,
{
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert!(rows * cols == data.len());
        Self { rows, cols, data }
    }

    pub fn transpose(self) -> Self {
        let mut data = vec![T::empty(); self.data.len()];
        self.data.into_iter().enumerate().for_each(|(i, x)| {
            let row = i / self.cols;
            let col = i % self.cols;
            let i = col * self.rows + row;
            data[i] = x;
        });
        Self {
            data,
            rows: self.cols,
            cols: self.rows,
        }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn remove_rows(self, removing: &HashSet<usize>) -> Self {
        let Self { rows, cols, data } = self;
        Self {
            rows: rows - removing.len(),
            cols,
            data: data
                .into_iter()
                .zip((0..self.rows).cycle().take(self.rows * self.cols))
                .filter(|(_, i)| !removing.contains(i))
                .map(|(i, _)| i)
                .collect(),
        }
    }
}

pub trait MatEmpty {
    fn empty() -> Self;
}

impl MatEmpty for f64 {
    fn empty() -> Self {
        0.0
    }
}

impl MatEmpty for i32 {
    fn empty() -> Self {
        0
    }
}

impl MatEmpty for String {
    fn empty() -> Self {
        String::new()
    }
}

pub trait MatParse<T, E>
where
    MatParseError: From<E>,
{
    fn mat_parse(&self) -> Result<T, MatParseError>;
}

impl<T, E> MatParse<T, E> for Rstr
where
    T: FromStr<Err = E>,
    MatParseError: From<E>,
{
    fn mat_parse(&self) -> Result<T, MatParseError> {
        Ok(self.as_str().parse()?)
    }
}

impl<T, E> MatParse<T, E> for &str
where
    T: FromStr<Err = E>,
    MatParseError: From<E>,
{
    fn mat_parse(&self) -> Result<T, MatParseError> {
        Ok(self.parse()?)
    }
}

pub trait FromRMatrix<T, R>
where
    for<'a> Robj: AsTypedSlice<'a, R>,
    T: MatEmpty + Clone,
{
    fn from_rmatrix(r: RMatrix<R>) -> OwnedMatrix<T>;
}

impl FromRMatrix<f64, f64> for OwnedMatrix<f64> {
    fn from_rmatrix(r: RMatrix<f64>) -> OwnedMatrix<f64> {
        let data = r.data().to_vec();
        OwnedMatrix::new(r.nrows(), r.ncols(), data)
    }
}

impl FromRMatrix<String, Rstr> for OwnedMatrix<String> {
    fn from_rmatrix(r: RMatrix<Rstr>) -> OwnedMatrix<String> {
        let data = r.data().iter().map(|x| x.to_string()).collect::<Vec<_>>();
        OwnedMatrix::new(r.nrows(), r.ncols(), data)
    }
}

pub trait ToRMatrix<T, R>
where
    for<'a> Robj: AsTypedSlice<'a, R>,
    T: MatEmpty + Clone,
{
    fn to_rmatrix(&self) -> RMatrix<R>;
}

impl ToRMatrix<f64, f64> for OwnedMatrix<f64> {
    fn to_rmatrix(&self) -> RMatrix<f64> {
        RMatrix::new_matrix(self.rows, self.cols, |r, c| self.data[r * self.cols + c])
    }
}

impl ToRMatrix<String, Rstr> for OwnedMatrix<String> {
    fn to_rmatrix(&self) -> RMatrix<Rstr> {
        todo!("Rstr does not implement ToVectorValue https://github.com/extendr/extendr/issues/770")
        // RMatrix::new_matrix(self.rows, self.cols, |r, c| (self.data[r * self.cols + c]))
    }
}

impl<'a> ToRMatrix<f64, f64> for MatRef<'a, f64> {
    fn to_rmatrix(&self) -> RMatrix<f64> {
        RMatrix::new_matrix(self.nrows(), self.ncols(), |r, c| unsafe {
            *self.get_unchecked(r, c)
        })
    }
}

impl<'a> ToRMatrix<f64, f64> for MatMut<'a, f64> {
    fn to_rmatrix(&self) -> RMatrix<f64> {
        let self_ = self.rb();
        RMatrix::new_matrix(self.nrows(), self.ncols(), |r, c| unsafe {
            *self_.get_unchecked(r, c)
        })
    }
}

#[derive(Copy, Clone, Debug)]
pub enum TransitoryType {
    Float,
    Str,
}

impl<'a> FromRobj<'a> for TransitoryType {
    fn from_robj(robj: &'a Robj) -> std::result::Result<Self, &'static str> {
        match robj.as_integer() {
            Some(0) => Ok(Self::Float),
            Some(1) => Ok(Self::Str),
            _ => Err("Invalid item type"),
        }
    }
}

pub enum TransitoryMatrix {
    Float(OwnedMatrix<f64>),
    Str(OwnedMatrix<String>),
}

impl TransitoryMatrix {
    pub fn remove_rows(self, removing: &HashSet<usize>) -> Self {
        match self {
            Self::Float(m) => Self::Float(m.remove_rows(removing)),
            Self::Str(m) => Self::Str(m.remove_rows(removing)),
        }
    }
}

impl<'a> FromRobj<'a> for Matrix<'a> {
    fn from_robj(robj: &'a Robj) -> std::result::Result<Self, &'static str> {
        if robj.is_matrix() {
            Ok(Self::R(robj.as_matrix::<f64>().ok_or("Invalid matrix")?))
        } else {
            Err("Invalid item type")
        }
    }
}

pub trait IntoMatrix<'a> {
    fn into_matrix(self) -> Matrix<'a>;
}

impl<'a> IntoMatrix<'a> for RMatrix<f64> {
    fn into_matrix(self) -> Matrix<'a> {
        Matrix::R(self)
    }
}

impl<'a> IntoMatrix<'a> for OwnedMatrix<f64> {
    fn into_matrix(self) -> Matrix<'a> {
        Matrix::Owned(self)
    }
}

impl<'a> IntoMatrix<'a> for File {
    fn into_matrix(self) -> Matrix<'a> {
        Matrix::File(self)
    }
}

pub trait TryIntoMatrix<'a> {
    type Err;

    fn try_into_matrix(self) -> Result<Matrix<'a>, Self::Err>;
}

impl<'a> TryIntoMatrix<'a> for Robj {
    type Err = &'static str;

    fn try_into_matrix(self) -> Result<Matrix<'a>, Self::Err> {
        if self.is_matrix() {
            Ok(Matrix::R(self.as_matrix::<f64>().ok_or("Invalid matrix")?))
        } else {
            Err("Invalid item type")
        }
    }
}

impl<'a> TryIntoMatrix<'a> for MatMut<'a, f64> {
    type Err = ();

    fn try_into_matrix(self) -> Result<Matrix<'a>, Self::Err> {
        if self.row_stride() == 1 {
            Ok(Matrix::Ref(self))
        } else {
            Err(())
        }
    }
}

impl<'a> TryIntoMatrix<'a> for &str {
    type Err = FileParseError;

    fn try_into_matrix(self) -> Result<Matrix<'a>, Self::Err> {
        Ok(Matrix::File(self.parse()?))
    }
}

impl<'a, T> TryIntoMatrix<'a> for T
where
    T: IntoMatrix<'a>,
{
    type Err = ();

    fn try_into_matrix(self) -> Result<Matrix<'a>, Self::Err> {
        Ok(self.into_matrix())
    }
}

impl<'a, T> From<T> for Matrix<'a>
where
    T: IntoMatrix<'a>,
{
    fn from(t: T) -> Self {
        t.into_matrix()
    }
}

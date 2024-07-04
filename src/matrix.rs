use core::panic;
use std::{collections::HashSet, mem::MaybeUninit, str::FromStr};

use crate::{
    errors::{CombineMatricesError, FileParseError},
    file::File,
    ExtendMatrixError, MatParseError, ReadMatrixError,
};
use extendr_api::{
    wrapper, AsStrIter, AsTypedSlice, Attributes, FromRobj, IntoRobj, MatrixConversions, RMatrix,
    Rinternals, Robj, Rstr,
};
use faer::{
    reborrow::{IntoConst, Reborrow},
    MatMut, MatRef,
};
use rayon::prelude::*;

#[derive(Debug)]
pub enum Matrix<'a> {
    R(RMatrix<f64>),
    Owned(OwnedMatrix<f64>),
    File(File),
    Ref(MatMut<'a, f64>),
}

/// SAFETY: This is only safe when not pointing to a .RData file.
unsafe impl<'a> Send for Matrix<'a> {}
unsafe impl<'a> Sync for Matrix<'a> {}

impl<'a> Matrix<'a> {
    pub fn as_ref(&mut self) -> Result<Matrix<'_>, ReadMatrixError> {
        Ok(Matrix::Ref(self.as_mat_mut()?))
    }

    /// Safety: This function is safe unless this function is being called in parallel and this
    /// Matrix points to a file.
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

    pub fn combine(mut self, mut others: Vec<Matrix<'_>>) -> Result<Self, CombineMatricesError> {
        if others.is_empty() {
            return Ok(self);
        }
        let colnames =
            if self.colnames().is_some() && others.iter_mut().all(|i| i.colnames().is_some()) {
                let mut colnames = self
                    .colnames()
                    .unwrap()
                    .into_iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>();
                for i in &mut others {
                    colnames.extend(i.colnames().unwrap().iter().map(|x| x.to_string()));
                }
                Some(colnames)
            } else {
                None
            };
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
        for c in 0..self_.ncols() {
            unsafe {
                data.extend(
                    self_
                        .get_unchecked(.., c)
                        .try_as_slice()
                        .expect("could not get slice"),
                )
            };
        }
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
            colnames,
        )))
    }

    pub fn extend(mut self, mut others: Vec<Matrix<'_>>) -> Result<Self, ExtendMatrixError> {
        if others.is_empty() {
            return Ok(self);
        }
        let colnames = self.colnames();
        if others.iter_mut().any(|i| i.colnames() != colnames) {
            return Err(ExtendMatrixError::ColumnNamesMismatch);
        }
        let colnames = colnames.map(|x| x.iter().map(|x| x.to_string()).collect());
        let self_ = self.as_mat_ref()?;
        let others = others
            .iter_mut()
            .map(|x| x.as_mat_ref())
            .collect::<Result<Vec<_>, _>>()?;
        if others.iter().any(|i| i.ncols() != self_.ncols()) {
            return Err(ExtendMatrixError::MatrixDimensionsMismatch);
        }
        let ncols = self_.ncols();
        let nrows = self_.nrows() + others.iter().map(|i| i.nrows()).sum::<usize>();
        let mats: Vec<MatRef<f64>> = [&[self_], others.as_slice()].concat();
        let data = vec![MaybeUninit::<f64>::uninit(); nrows * ncols];
        mats.par_iter().enumerate().for_each(|(i, m)| {
            let rows_before = mats.iter().take(i).map(|m| m.nrows()).sum::<usize>();
            (0..ncols).into_par_iter().for_each(|c| unsafe {
                let src = m
                    .get_unchecked(.., c)
                    .try_as_slice()
                    .expect("could not get slice");
                let dst = data
                    .as_ptr()
                    .add(nrows * c + rows_before)
                    .cast::<f64>()
                    .cast_mut();
                let slice = std::slice::from_raw_parts_mut(dst, m.nrows());
                slice.copy_from_slice(src);
            });
        });
        Ok(Self::Owned(OwnedMatrix::new(
            nrows,
            ncols,
            unsafe { std::mem::transmute(data) },
            colnames,
        )))
    }

    pub fn remove_rows(mut self, removing: &HashSet<usize>) -> Result<Self, ReadMatrixError> {
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
            colnames: self
                .colnames()
                .map(|x| x.into_iter().map(|x| x.to_string()).collect()),
        }))
    }

    pub fn into_robj(self) -> Result<Robj, ReadMatrixError> {
        Ok(self.to_rmatrix().into_robj())
    }

    pub fn to_owned(self) -> Result<OwnedMatrix<f64>, ReadMatrixError> {
        Ok(match self {
            Matrix::R(r) => OwnedMatrix::from_rmatrix(&r),
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
                None,
            ),
        })
    }

    pub fn into_owned(&mut self) -> Result<(), ReadMatrixError> {
        match self {
            Matrix::R(r) => {
                *self = Matrix::Owned(OwnedMatrix::from_rmatrix(r));
                Ok(())
            },
            Matrix::Owned(_) => Ok(()),
            Matrix::File(f) => {
                *self = Matrix::Owned(f.read_matrix(true)?);
                Ok(())
            },
            Matrix::Ref(r) => {
                let r = r.as_ref();
                *self = Matrix::Owned(OwnedMatrix::new(
                    r.nrows(),
                    r.ncols(),
                    (0..r.ncols())
                        .flat_map(|i| {
                            r.get(.., i)
                                .try_as_slice()
                                .expect("matrix should have row stride 1")
                                .iter()
                                .copied()
                        })
                        .collect(),
                    None,
                ));
                Ok(())
            },
        }
    }

    pub fn colnames(&mut self) -> Option<Vec<&str>> {
        match self {
            Matrix::R(r) => colnames(r),
            Matrix::Owned(m) => m.colnames().map(|x| x.iter().map(|x| x.as_str()).collect()),
            Matrix::File(f) => {
                let m = f.read_matrix(true).ok()?;
                *self = Matrix::Owned(m);
                self.colnames()
            },
            Matrix::Ref(_) => None,
        }
    }

    pub fn from_slice(data: &'a mut [f64], rows: usize, cols: usize) -> Self {
        Self::Ref(faer::mat::from_column_major_slice_mut(data, rows, cols))
    }

    pub fn remove_column_by_name_if_exists(&mut self, name: &str) {
        let colnames = self.colnames();
        if colnames.is_none() {
            return;
        }
        let exists = colnames
            .expect("colnames should be present")
            .iter()
            .any(|x| *x == name);
        if exists {
            self.into_owned().expect("could not convert to owned");
            match self {
                Matrix::Owned(m) => m.remove_column_by_name_if_exists(name),
                _ => unreachable!(),
            }
        }
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
    pub(crate) colnames: Option<Vec<String>>,
    pub(crate) data: Vec<T>,
}

impl<T> OwnedMatrix<T>
where
    T: MatEmpty + Clone,
{
    pub fn new(rows: usize, cols: usize, data: Vec<T>, colnames: Option<Vec<String>>) -> Self {
        assert!(rows * cols == data.len());
        Self {
            rows,
            cols,
            data,
            colnames,
        }
    }

    pub fn transpose(self) -> Self {
        let mut data = vec![T::empty(); self.data.len()];
        self.data.into_iter().enumerate().for_each(|(i, x)| {
            let new_row = i / self.rows;
            let new_col = i % self.rows;
            let i = new_col * self.cols + new_row;
            data[i] = x;
        });
        Self {
            data,
            rows: self.cols,
            cols: self.rows,
            colnames: None,
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

    #[inline]
    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    #[inline]
    pub fn colnames(&self) -> Option<&[String]> {
        self.colnames.as_deref()
    }

    pub fn remove_rows(self, removing: &HashSet<usize>) -> Self {
        let Self { rows, data, .. } = self;
        Self {
            rows: rows - removing.len(),
            data: data
                .into_iter()
                .zip((0..self.rows).cycle().take(self.rows * self.cols))
                .filter(|(_, i)| !removing.contains(i))
                .map(|(i, _)| i)
                .collect(),
            ..self
        }
    }

    pub fn remove_rows_mut(&mut self, removing: &HashSet<usize>)
    where
        T: Copy,
    {
        self.data = self
            .data
            .iter()
            .zip((0..self.rows).cycle().take(self.rows * self.cols))
            .filter(|(_, i)| !removing.contains(i))
            .map(|(i, _)| *i)
            .collect();
        self.rows -= removing.len();
    }

    pub fn col(&self, col: usize) -> &[T] {
        &self.data[(col * self.rows)..(col * self.rows + self.rows)]
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.data.get(col * self.rows + row)
    }

    pub fn is_col_sorted(&self, col: usize) -> bool
    where
        T: PartialOrd + Send + Sync,
    {
        self.col(col).par_windows(2).all(|w| w[0] <= w[1])
    }

    pub fn sort_by_column(&mut self, by: &str)
    where
        T: PartialOrd + Copy + Send + Sync,
    {
        let colnames = self.colnames().expect("colnames should be present");
        let by_col_idx = colnames
            .iter()
            .position(|x| x == by)
            .expect("column not found");
        if self.is_col_sorted(by_col_idx) {
            return;
        }
        let mut order = {
            self.col(by_col_idx)
                .iter()
                .copied()
                .enumerate()
                .collect::<Vec<_>>()
        };
        // check if the column is sorted
        order.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("could not compare"));
        self.sort_by_order(order.into_iter().map(|(i, _)| i).collect());
    }

    pub fn sort_by_order(&mut self, order: Vec<usize>)
    where
        T: Copy + Send + Sync,
    {
        self.data.par_chunks_exact_mut(self.rows).for_each(|row| {
            let sorted = order.iter().map(|i| row[*i]).collect::<Vec<_>>();
            row.copy_from_slice(&sorted);
        });
    }

    pub fn dedup_by_column(&mut self, by: &str)
    where
        T: PartialEq + Copy + Send + Sync + PartialOrd,
    {
        let colnames = self.colnames().expect("colnames should be present");
        let by_col_idx = colnames
            .iter()
            .position(|x| x == by)
            .expect("column not found");
        let mut removing = HashSet::new();
        let mut col = self.col(by_col_idx).iter().enumerate().collect::<Vec<_>>();
        col.sort_by(|a, b| a.1.partial_cmp(b.1).expect("could not compare"));
        for i in 1..col.len() {
            if col[i - 1].1 == col[i].1 {
                removing.insert(col[i].0);
            }
        }
        self.remove_rows_mut(&removing);
    }

    pub fn match_to(&mut self, other: &[T], col: &str)
    where
        T: PartialOrd + Copy + Send + Sync + Default,
    {
        let self_col_idx = self
            .colnames()
            .expect("colnames should be present")
            .iter()
            .position(|x| x == col)
            .expect("column not found");
        self.sort_by_column(col);
        let other_is_sorted = 'a: {
            let mut i = 1;
            while i < other.len() {
                if other[i - 1] > other[i] {
                    break 'a false;
                }
                i += 1;
            }
            true
        };
        if other_is_sorted {
            let mut i = 0;
            let mut removing = HashSet::new();
            for j in other {
                while self.get(i, self_col_idx) < Some(j) {
                    removing.insert(i);
                    i += 1;
                }
                if self.get(i, self_col_idx) == Some(j) {
                    i += 1;
                } else {
                    panic!("could not find match for index {}", i);
                }
            }
            for i in i..self.rows {
                removing.insert(i);
            }
            self.remove_rows_mut(&removing);
        } else {
            fn binary_search<T>(data: &[T], x: &T) -> Option<usize>
            where
                T: PartialOrd,
            {
                let mut low = 0;
                let mut high = data.len();
                while low < high {
                    let mid = low + (high - low) / 2;
                    if data[mid] < *x {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
                if low < data.len() && data[low] == *x {
                    Some(low)
                } else {
                    None
                }
            }

            let data = vec![Default::default(); other.len() * self.cols];
            let self_col = self.col(self_col_idx);
            other.par_iter().enumerate().for_each(|(idx, j)| {
                // SAFETY: no iteration of this iterator will mutably access
                // overlapping data
                let data: &mut [T] = unsafe {
                    std::slice::from_raw_parts_mut(data.as_slice().as_ptr() as *mut T, data.len())
                };
                let i = binary_search(self_col, j);
                if let Some(i) = i {
                    for k in 0..self.cols {
                        data[other.len() * k + idx] = *self.get(i, k).expect("could not get value");
                    }
                } else {
                    panic!("could not find match for index {}", idx);
                }
            });
            self.data = data;
            self.rows = other.len();
        }
    }

    pub fn merge(self, other: &Self, by: &str) -> Self
    where
        T: PartialOrd,
    {
        if self.colnames().is_none() || other.colnames().is_none() {
            return self;
        }
        let Self {
            colnames: self_colnames,
            ..
        } = self;
        let mut colnames = self_colnames.expect("colnames should be present");
        let other_colnames = other.colnames().expect("colnames should be present");
        let self_by_col_idx = colnames
            .iter()
            .position(|x| x == by)
            .expect("column not found");
        let mut self_by_col = self.data
            [(self_by_col_idx * self.rows)..(self_by_col_idx * self.rows + self.rows)]
            .iter()
            .enumerate()
            .collect::<Vec<_>>();
        self_by_col.sort_by(|a, b| a.1.partial_cmp(b.1).expect("could not compare"));
        let other_by_col_idx = other_colnames
            .iter()
            .position(|x| x == by)
            .expect("column not found");
        let mut other_by_col = other.data
            [(other_by_col_idx * other.rows)..(other_by_col_idx * other.rows + other.rows)]
            .iter()
            .enumerate()
            .collect::<Vec<_>>();
        other_by_col.sort_by(|a, b| a.1.partial_cmp(b.1).expect("could not compare"));
        // join the data from the first matrix with the data from the second matrix on the column
        // `by`
        // data is stored in column-major order
        let mut self_by_col_iter = self_by_col.into_iter();
        let mut other_by_col_iter = other_by_col.into_iter();
        let mut matches = Vec::with_capacity(self.rows.min(other.rows));
        while let (Some(self_by), Some(other_by)) =
            (self_by_col_iter.next(), other_by_col_iter.next())
        {
            if self_by.1 == other_by.1 {
                matches.push((self_by.0, other_by.0));
            } else if self_by.1 < other_by.1 {
                for self_by in self_by_col_iter.by_ref() {
                    if self_by.1 == other_by.1 {
                        matches.push((self_by.0, other_by.0));
                        break;
                    } else if self_by.1 > other_by.1 {
                        break;
                    }
                }
            } else {
                for other_by in other_by_col_iter.by_ref() {
                    if self_by.1 == other_by.1 {
                        matches.push((self_by.0, other_by.0));
                        break;
                    } else if self_by.1 < other_by.1 {
                        break;
                    }
                }
            }
        }
        let mut data: Vec<T> = Vec::with_capacity(matches.len() * (self.cols + other.cols - 1));
        for i in 0..self.cols {
            let col = &self.data[(i * self.rows)..(i * self.rows + self.rows)];
            data.extend(matches.iter().map(|(r, _)| col[*r].clone()));
        }
        for i in 0..other.cols {
            if i == other_by_col_idx {
                continue;
            }
            let col = &other.data[(i * other.rows)..(i * other.rows + other.rows)];
            data.extend(matches.iter().map(|(_, r)| col[*r].clone()));
        }
        colnames.extend(other_colnames.iter().filter(|x| x != &by).cloned());
        Self {
            rows: matches.len(),
            cols: self.cols + other.cols - 1,
            data,
            colnames: Some(colnames),
        }
    }

    pub fn remove_column_by_name_if_exists(&mut self, name: &str) {
        if self.colnames().is_none() {
            return;
        }
        let col_idx = self
            .colnames
            .as_ref()
            .expect("colnames should be present")
            .iter()
            .position(|x| x == name);
        if let Some(col_idx) = col_idx {
            self.colnames
                .as_mut()
                .expect("colnames should be present")
                .remove(col_idx);
            self.data
                .drain((col_idx * self.rows)..(col_idx * self.rows + self.rows));
            self.cols -= 1;
        }
    }

    pub fn remove_column_by_name(&mut self, name: &str) {
        if self.colnames.is_none() {
            return;
        }
        let col_idx = self
            .colnames
            .as_ref()
            .expect("colnames should be present")
            .iter()
            .position(|x| x == name)
            .expect("column not found");
        self.colnames
            .as_mut()
            .expect("colnames should be present")
            .remove(col_idx);
        self.data
            .drain((col_idx * self.rows)..(col_idx * self.rows + self.rows));
        self.cols -= 1;
    }
}

impl OwnedMatrix<f64> {
    pub fn as_mat_ref(&self) -> MatRef<'_, f64> {
        faer::mat::from_column_major_slice(self.data.as_slice(), self.rows, self.cols)
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

fn colnames<T>(r: &RMatrix<T>) -> Option<Vec<&str>> {
    r.dimnames().map(|mut dimnames| {
        dimnames
            .nth(1)
            .unwrap()
            .as_str_iter()
            .unwrap()
            .collect::<Vec<_>>()
    })
}

pub trait FromRMatrix<T, R>
where
    for<'a> Robj: AsTypedSlice<'a, R>,
    T: MatEmpty + Clone,
{
    fn from_rmatrix(r: &RMatrix<R>) -> OwnedMatrix<T>;
}

impl FromRMatrix<f64, f64> for OwnedMatrix<f64> {
    fn from_rmatrix(r: &RMatrix<f64>) -> OwnedMatrix<f64> {
        let data = r.data().to_vec();
        OwnedMatrix::new(
            r.nrows(),
            r.ncols(),
            data,
            colnames(r).map(|x| x.iter().map(|x| x.to_string()).collect()),
        )
    }
}

impl FromRMatrix<String, Rstr> for OwnedMatrix<String> {
    fn from_rmatrix(r: &RMatrix<Rstr>) -> OwnedMatrix<String> {
        let data = r.data().iter().map(|x| x.to_string()).collect::<Vec<_>>();
        OwnedMatrix::new(
            r.nrows(),
            r.ncols(),
            data,
            colnames(r).map(|x| x.iter().map(|x| x.to_string()).collect()),
        )
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
        use extendr_api::prelude::*;

        let mat = RMatrix::new_matrix(
            self.rows,
            self.cols,
            #[inline(always)]
            |r, c| self.data[c * self.rows + r],
        );
        let mut dimnames = List::from_values([NULL, NULL]);
        if let Some(colnames) = self.colnames() {
            dimnames.set_elt(1, colnames.into_robj()).unwrap();
        }
        mat.set_attrib(wrapper::symbol::dimnames_symbol(), dimnames)
            .unwrap();
        mat
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

impl<'a> Matrix<'a> {
    pub fn to_rmatrix(&self) -> Result<RMatrix<f64>, ReadMatrixError> {
        Ok(match self {
            Matrix::R(r) => r.as_matrix().unwrap(),
            Matrix::Owned(m) => m.to_rmatrix(),
            Matrix::File(f) => f.read_matrix(true)?.to_rmatrix(),
            Matrix::Ref(r) => r.to_rmatrix(),
        })
    }
}

#[derive(Copy, Clone, Debug)]
pub enum TransitoryType {
    Float = 0,
    Str = 1,
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

impl<'a> IntoMatrix<'a> for RMatrix<i32> {
    fn into_matrix(self) -> Matrix<'a> {
        let data = self.data().iter().map(|x| *x as f64).collect();
        Matrix::Owned(OwnedMatrix::new(
            self.nrows(),
            self.ncols(),
            data,
            colnames(&self).map(|x| x.iter().map(|x| x.to_string()).collect()),
        ))
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

use core::panic;
use std::{collections::HashSet, mem::MaybeUninit, str::FromStr};

use crate::{
    errors::{CombineColumnsError, FileParseError},
    file::File,
    CombineRowsError, DedupByColumnError, DedupByColumnNameError, JoinByColumnNameError, JoinError,
    MatParseError, MatchToByColumnNameError, MatchToColumnError, MatrixFromRobjError,
    ReadMatrixError, RemoveColumnByNameError, RemoveColumnsError, RemoveRowsError,
    SortByColumnError, SortByColumnNameError, SortByOrderError,
};
use extendr_api::{
    io::Load, single_threaded, wrapper, AsStrIter, Attributes, Conversions, IntoRobj,
    MatrixConversions, RMatrix, Rinternals, Robj,
};
use faer::{MatMut, MatRef};
use rayon::prelude::*;
use tracing::{debug, trace, warn};

#[derive(Debug, Clone, Copy)]
pub enum Join {
    /// Inner join, only rows that are present in both matrices are kept
    Inner = 0,
    /// Left join, all rows from the left matrix must be matched
    Left = 1,
    /// Right join, all rows from the right matrix must be matched
    Right = 2,
}

impl std::fmt::Display for Join {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Join::Inner => write!(f, "inner"),
            Join::Left => write!(f, "left"),
            Join::Right => write!(f, "right"),
        }
    }
}

#[derive(Debug)]
pub enum Matrix<'a> {
    R(RMatrix<f64>),
    Owned(OwnedMatrix),
    File(File),
    Ref(MatMut<'a, f64>),
}

impl<'a> Clone for Matrix<'a> {
    fn clone(&self) -> Self {
        match self {
            Matrix::R(m) => Matrix::R((*m).as_matrix().unwrap()),
            Matrix::Owned(m) => Matrix::Owned(m.clone()),
            Matrix::File(f) => Matrix::File(f.clone()),
            Matrix::Ref(m) => {
                let data = vec![MaybeUninit::<f64>::uninit(); m.nrows() * m.ncols()];
                m.as_ref().par_col_chunks(1).enumerate().for_each(|(i, c)| {
                    // SAFETY: No two threads will write to the same location
                    let data = unsafe {
                        std::slice::from_raw_parts_mut(
                            data.as_ptr().add(i * m.nrows()).cast::<f64>().cast_mut(),
                            data.len(),
                        )
                    };
                    data.copy_from_slice(c.col(0).try_as_slice().expect("could not get slice"));
                });
                Matrix::Owned(OwnedMatrix::new(
                    m.nrows(),
                    m.ncols(),
                    // SAFETY: The data is initialized now
                    unsafe { std::mem::transmute(data) },
                    None,
                ))
            },
        }
    }
}

// SAFETY: This is always safe except when calling into R, for example by loading an .RData file
unsafe impl<'a> Send for Matrix<'a> {}
unsafe impl<'a> Sync for Matrix<'a> {}

impl<'a> Matrix<'a> {
    pub fn as_mat_ref(&mut self) -> Result<MatRef<'_, f64>, ReadMatrixError> {
        Ok(match self {
            m @ Matrix::File(_) => m.into_owned()?,
            m => m,
        }
        .as_mat_ref_loaded())
    }

    pub fn as_mat_ref_loaded(&self) -> MatRef<'_, f64> {
        match self {
            Matrix::R(m) => faer::mat::from_column_major_slice(m.data(), m.nrows(), m.ncols()),
            Matrix::Owned(m) => {
                faer::mat::from_column_major_slice(m.data.as_slice(), m.nrows, m.ncols)
            },
            Matrix::File(_) => panic!("cannot call this function on a file"),
            Matrix::Ref(m) => m.as_ref(),
        }
    }

    pub fn as_mat_mut(&mut self) -> Result<MatMut<'_, f64>, ReadMatrixError> {
        Ok(match self {
            // SAFETY: We know that the data is valid
            Matrix::R(m) => unsafe {
                faer::mat::from_raw_parts_mut(
                    m.data().as_ptr().cast_mut(),
                    m.nrows(),
                    m.ncols(),
                    1,
                    m.nrows() as isize,
                )
            },
            Matrix::Owned(m) => {
                faer::mat::from_column_major_slice_mut(m.data.as_mut(), m.nrows, m.ncols)
            },
            m @ Matrix::File(_) => m.into_owned()?.as_mat_mut()?,
            Matrix::Ref(m) => m.as_mut(),
        })
    }

    pub fn as_owned_ref(&mut self) -> Result<&OwnedMatrix, ReadMatrixError> {
        self.into_owned().map(|m| match m {
            Matrix::Owned(m) => &*m,
            _ => unreachable!(),
        })
    }

    pub fn to_rmatrix(&mut self) -> Result<RMatrix<f64>, ReadMatrixError> {
        Ok(match self {
            Matrix::R(m) => m.clone().into_robj().clone().as_matrix().unwrap(),
            Matrix::Owned(m) => m.to_rmatrix(),
            m @ Matrix::File(_) => m.into_owned()?.to_rmatrix()?,
            Matrix::Ref(m) => {
                let m = m.as_ref();
                // SAFETY: We know that r and c are within bounds
                RMatrix::new_matrix(m.nrows(), m.ncols(), |r, c| unsafe {
                    *(m.get_unchecked(r, c))
                })
            },
        })
    }

    pub fn into_robj(mut self) -> Result<Robj, ReadMatrixError> {
        Ok(self.to_rmatrix().into_robj())
    }

    pub fn from_rdata(mut reader: impl std::io::Read) -> Result<Self, ReadMatrixError> {
        let mut buf = [0; 5];
        reader.read_exact(&mut buf)?;
        if buf != *b"RDX3\n" {
            return Err(ReadMatrixError::InvalidRdataFile);
        }
        let obj = single_threaded(|| {
            Robj::from_reader(&mut reader, extendr_api::io::PstreamFormat::XdrFormat, None)
        })?;
        let mat = obj
            .as_pairlist()
            .ok_or(ReadMatrixError::InvalidRdataFile)?
            .into_iter()
            .next()
            .ok_or(ReadMatrixError::InvalidRdataFile)?
            .1;
        Ok(Matrix::from_robj(mat)?)
    }

    pub fn from_robj(r: Robj) -> Result<Self, MatrixFromRobjError> {
        if r.is_matrix() {
            let float = RMatrix::<f64>::try_from(r);
            match float {
                Ok(float) => Ok(float.into()),
                Err(extendr_api::Error::TypeMismatch(r)) => {
                    Ok(RMatrix::<i32>::try_from(r)?.into_matrix())
                },
                Err(e) => Err(e.into()),
            }
        } else if r.is_string() {
            Ok(File::from_str(r.as_str().expect("i is a string"))?.into())
        } else if r.is_integer() {
            let v = r
                .as_integer_slice()
                .expect("data is an integer vector")
                .iter()
                .map(|i| *i as f64)
                .collect::<Vec<_>>();
            Ok(Matrix::Owned(OwnedMatrix::new(v.len(), 1, v, None)))
        } else if r.is_real() {
            let v = r.as_real_vector().expect("data is a real vector");
            Ok(Matrix::Owned(OwnedMatrix::new(v.len(), 1, v, None)))
        } else if r.is_list() {
            if r.class()
                .map(|x| x.into_iter().any(|c| c == "data.frame"))
                .unwrap_or(false)
            {
                use extendr_api::prelude::*;

                return Ok(single_threaded(|| {
                    extendr_api::R!("as.matrix(sapply({{r}}, as.double))")
                        .map(|x| x.as_matrix::<f64>().unwrap().into_matrix())
                })?);
            } else {
                Err(MatrixFromRobjError::InvalidItemType)
            }
        } else {
            Err(MatrixFromRobjError::InvalidItemType)
        }
    }

    pub fn to_owned(self) -> Result<OwnedMatrix, ReadMatrixError> {
        Ok(match self {
            Matrix::File(m) => m.read()?.to_owned_loaded(),
            m => m.to_owned_loaded(),
        })
    }

    pub fn to_owned_loaded(self) -> OwnedMatrix {
        match self {
            Matrix::File(_) => panic!("cannot call this function on a file"),
            mut m => {
                m.into_owned().expect("could not convert to owned");
                match m {
                    Matrix::Owned(m) => m,
                    _ => unreachable!(),
                }
            },
        }
    }

    #[inline]
    #[tracing::instrument(skip(self))]
    pub fn into_owned(&mut self) -> Result<&mut Self, ReadMatrixError> {
        match self {
            Matrix::R(m) => {
                *self = Matrix::Owned(OwnedMatrix::from_rmatrix(m));
                Ok(self)
            },
            Matrix::Owned(_) => Ok(self),
            Matrix::File(m) => {
                *self = m.read()?;
                Ok(self)
            },
            Matrix::Ref(m) => {
                let data = vec![MaybeUninit::<f64>::uninit(); m.nrows() * m.ncols()];
                m.as_ref().par_col_chunks(1).enumerate().for_each(|(i, c)| {
                    // SAFETY: No two threads will write to the same location
                    let data = unsafe {
                        std::slice::from_raw_parts_mut(
                            data.as_ptr().add(i * m.nrows()).cast::<f64>().cast_mut(),
                            data.len(),
                        )
                    };
                    data.copy_from_slice(c.col(0).try_as_slice().expect("could not get slice"));
                });
                *self = Matrix::Owned(OwnedMatrix::new(
                    m.nrows(),
                    m.ncols(),
                    // SAFETY: The data is initialized now
                    unsafe { std::mem::transmute(data) },
                    None,
                ));
                Ok(self)
            },
        }
    }

    pub fn colnames(&mut self) -> Result<Option<Vec<&str>>, ReadMatrixError> {
        Ok(match self {
            Matrix::R(m) => colnames(m),
            Matrix::Owned(m) => m.colnames().map(|x| x.iter().map(|x| x.as_str()).collect()),
            Matrix::File(f) => {
                let m = f.read()?;
                *self = m;
                self.colnames()?
            },
            Matrix::Ref(_) => None,
        })
    }

    pub fn from_slice(data: &'a mut [f64], rows: usize, cols: usize) -> Self {
        Matrix::Ref(faer::mat::from_column_major_slice_mut(data, rows, cols))
    }

    pub fn from_rmatrix(r: RMatrix<f64>) -> Self {
        Matrix::R(r)
    }

    pub fn from_owned(m: OwnedMatrix) -> Self {
        Matrix::Owned(m)
    }

    pub fn from_file(f: File) -> Self {
        Matrix::File(f)
    }

    pub fn from_mat_mut(m: MatMut<'a, f64>) -> Self {
        assert!(m.row_stride() == 1);
        Matrix::Ref(m)
    }

    #[tracing::instrument(skip(self, others))]
    pub fn combine_columns(
        &mut self,
        others: &mut [Matrix<'a>],
    ) -> Result<&mut Self, CombineColumnsError> {
        if others.is_empty() {
            warn!("attempted to combine by columns with empty others");
            return Ok(self);
        }

        // Only retain the column names if they are present in all matrices
        let colnames = if self.colnames()?.is_some()
            && others
                .iter_mut()
                .map(|i| i.colnames())
                .collect::<Result<Vec<Option<Vec<&str>>>, _>>()?
                .into_iter()
                .all(|i| i.is_some())
        {
            let mut colnames = self
                .colnames()?
                .unwrap()
                .into_iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>();
            for i in others.iter_mut() {
                colnames.extend(i.colnames()?.unwrap().iter().map(|x| x.to_string()));
            }
            Some(colnames)
        } else {
            None
        };

        // Ensure that all the matrices are properly sized
        let others = others
            .iter_mut()
            .map(|x| x.as_mat_ref())
            .collect::<Result<Vec<_>, _>>()?;
        let nrows = self.nrows()?;
        if others.iter().any(|i| i.nrows() != nrows) {
            return Err(CombineColumnsError::MatrixDimensionsMismatch);
        }
        let ncols = self.ncols()? + others.iter().map(|i| i.ncols()).sum::<usize>();
        let data = vec![MaybeUninit::<f64>::uninit(); nrows * ncols];
        debug!("nrows: {}, ncols: {}", nrows, ncols);

        // Combine the matrices
        let mats = [&[self.as_mat_ref_loaded()], others.as_slice()].concat();
        mats.par_iter().enumerate().for_each(|(i, m)| {
            let cols_before = mats.iter().take(i).map(|m| m.ncols()).sum::<usize>();
            // SAFETY: No two threads will write to the same location
            (0..m.ncols()).into_par_iter().for_each(|c| unsafe {
                let src = m
                    .get_unchecked(.., c)
                    .try_as_slice()
                    .expect("could not get slice");
                let dst = data
                    .as_ptr()
                    .add(m.nrows() * (c + cols_before))
                    .cast::<f64>()
                    .cast_mut();
                let slice = std::slice::from_raw_parts_mut(dst, m.nrows());
                slice.copy_from_slice(src);
            });
        });
        *self = Matrix::Owned(OwnedMatrix::new(
            nrows,
            ncols,
            // SAFETY: The data is initialized now
            unsafe { std::mem::transmute(data) },
            colnames,
        ));
        Ok(self)
    }

    #[tracing::instrument(skip(self, others))]
    pub fn combine_rows(
        &mut self,
        others: &mut [Matrix<'a>],
    ) -> Result<&mut Self, CombineRowsError> {
        if others.is_empty() {
            warn!("attempted to combine by rows with empty others");
            return Ok(self);
        }
        let colnames = self.colnames()?;
        if others
            .iter_mut()
            .map(|i| i.colnames())
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .any(|i| *i != colnames)
        {
            return Err(CombineRowsError::ColumnNamesMismatch);
        }
        let colnames = colnames.map(|x| x.iter().map(|x| x.to_string()).collect());
        let ncols = self.ncols()?;
        let others = others
            .iter_mut()
            .map(|x| x.as_mat_ref())
            .collect::<Result<Vec<_>, _>>()?;
        if others.iter().any(|i| i.ncols() != ncols) {
            return Err(CombineRowsError::MatrixDimensionsMismatch);
        }
        let nrows = self.nrows()? + others.iter().map(|i| i.nrows()).sum::<usize>();
        debug!("nrows: {}, ncols: {}", nrows, ncols);
        let mats = [&[self.as_mat_ref_loaded()], others.as_slice()].concat();
        let data = vec![MaybeUninit::<f64>::uninit(); nrows * ncols];
        mats.par_iter().enumerate().for_each(|(i, m)| {
            let rows_before = mats.iter().take(i).map(|m| m.nrows()).sum::<usize>();
            (0..ncols).into_par_iter().for_each(|c| unsafe {
                let src = m
                    .get_unchecked(.., c)
                    .try_as_slice()
                    .expect("could not get slice");
                debug!("{i} {c} src: {:?}", src);
                let dst = data
                    .as_ptr()
                    .add(nrows * c + rows_before)
                    .cast::<f64>()
                    .cast_mut();
                debug!("{i} {c} dst: {:?}", dst);
                let slice = std::slice::from_raw_parts_mut(dst, m.nrows());
                debug!("{i} {c} slice: {:?}", slice);
                slice.copy_from_slice(src);
                debug!("{i} {c} slice: {:?}", slice);
            });
        });
        *self = Matrix::Owned(OwnedMatrix::new(
            nrows,
            ncols,
            // SAFETY: The data is initialized now
            unsafe { std::mem::transmute(data) },
            colnames,
        ));
        Ok(self)
    }

    #[tracing::instrument(skip(self))]
    pub fn remove_rows(&mut self, removing: &HashSet<usize>) -> Result<&mut Self, RemoveRowsError> {
        for i in removing.iter() {
            if *i >= self.nrows()? {
                return Err(RemoveRowsError::RowIndexOutOfBounds(*i));
            }
        }
        let new_nrows = self.nrows()? - removing.len();
        let m = self.as_mat_ref()?;
        let data = vec![MaybeUninit::<f64>::uninit(); new_nrows * m.ncols()];
        (0..m.ncols()).into_par_iter().for_each(|c| {
            let mut j = 0;
            for i in 0..m.nrows() {
                if removing.contains(&i) {
                    continue;
                }
                let src = m.get(i, c);
                unsafe {
                    let dst = data
                        .as_ptr()
                        .add(c * new_nrows + j)
                        .cast::<f64>()
                        .cast_mut();
                    dst.write(*src);
                }
                j += 1;
            }
        });
        *self = Matrix::Owned(OwnedMatrix::new(
            new_nrows,
            self.ncols()?,
            // SAFETY: The data is initialized now
            unsafe { std::mem::transmute(data) },
            self.colnames()?
                .map(|x| x.iter().map(|x| x.to_string()).collect()),
        ));
        Ok(self)
    }

    #[tracing::instrument(skip(self))]
    pub fn remove_columns(
        &mut self,
        removing: &HashSet<usize>,
    ) -> Result<&mut Self, RemoveColumnsError> {
        let m = self.as_mat_ref()?;
        for i in removing.iter() {
            if *i >= m.ncols() {
                return Err(RemoveColumnsError::ColumnIndexOutOfBounds(*i));
            }
        }
        let new_ncols = m.ncols() - removing.len();
        let data = vec![MaybeUninit::<f64>::uninit(); m.nrows() * new_ncols];
        debug!("nrows: {}, ncols: {}", m.nrows(), new_ncols);
        (0..m.ncols())
            .filter(|x| !removing.contains(x))
            .collect::<Vec<_>>()
            .into_par_iter()
            .enumerate()
            // SAFETY: No two threads will write to the same location
            .for_each(|(n, o)| unsafe {
                let src = m.get_unchecked(.., o).try_as_slice().expect("could not get slice");
                let dst = data.as_ptr().add(n * m.nrows()).cast::<f64>().cast_mut();
                let slice = std::slice::from_raw_parts_mut(dst, m.nrows());
                slice.copy_from_slice(src);
            });
        *self = Matrix::Owned(OwnedMatrix::new(
            m.nrows(),
            new_ncols,
            // SAFETY: The data is initialized now
            unsafe { std::mem::transmute(data) },
            self.colnames()?.map(|x| {
                x.iter()
                    .enumerate()
                    .filter_map(|(i, x)| {
                        if removing.contains(&i) {
                            None
                        } else {
                            Some(x.to_string())
                        }
                    })
                    .collect()
            }),
        ));
        Ok(self)
    }

    #[tracing::instrument(skip(self))]
    pub fn remove_column_by_name(
        &mut self,
        name: &str,
    ) -> Result<&mut Self, RemoveColumnByNameError> {
        let colnames = self.colnames()?;
        if colnames.is_none() {
            return Err(RemoveColumnByNameError::MissingColumnNames);
        }
        let exists = colnames
            .expect("colnames should be present")
            .iter()
            .position(|x| *x == name);
        if let Some(i) = exists {
            self.remove_columns(&[i].iter().copied().collect())?;
        } else {
            return Err(RemoveColumnByNameError::ColumnNameNotFound(
                name.to_string(),
            ));
        }
        Ok(self)
    }

    #[tracing::instrument(skip(self))]
    pub fn remove_column_by_name_if_exists(
        &mut self,
        name: &str,
    ) -> Result<&mut Self, ReadMatrixError> {
        match self.remove_column_by_name(name) {
            Ok(_) => Ok(self),
            Err(RemoveColumnByNameError::ColumnNameNotFound(_)) => Ok(self),
            Err(RemoveColumnByNameError::ReadMatrixError(e)) => Err(e),
            _ => unreachable!(),
        }
    }

    #[tracing::instrument(skip(self))]
    pub fn transpose(&mut self) -> Result<&mut Self, ReadMatrixError> {
        let m = self.as_mat_ref()?;
        let new_data = vec![MaybeUninit::<f64>::uninit(); m.nrows() * m.ncols()];
        m.par_col_chunks(1).enumerate().for_each(|(new_row, c)| {
            c.col(0).iter().enumerate().for_each(|(new_col, x)| {
                let i = new_col * m.ncols() + new_row;
                unsafe {
                    new_data.as_ptr().add(i).cast_mut().cast::<f64>().write(*x);
                };
            });
        });
        *self = Matrix::Owned(OwnedMatrix::new(
            m.ncols(),
            m.nrows(),
            // SAFETY: The data is initialized now
            unsafe { std::mem::transmute(new_data) },
            None,
        ));
        Ok(self)
    }

    #[tracing::instrument(skip(self))]
    pub fn sort_by_column(&mut self, by: usize) -> Result<&mut Self, SortByColumnError> {
        if by >= self.ncols()? {
            return Err(SortByColumnError::ColumnIndexOutOfBounds(by));
        }
        let col = self.col(by)?.unwrap();
        let mut order = col.iter().copied().enumerate().collect::<Vec<_>>();
        order.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("could not compare"));
        Ok(self.sort_by_order(
            order
                .into_iter()
                .map(|(i, _)| i)
                .collect::<Vec<_>>()
                .as_slice(),
        )?)
    }

    #[tracing::instrument(skip(self))]
    pub fn sort_by_column_name(&mut self, by: &str) -> Result<&mut Self, SortByColumnNameError> {
        let colnames = self.colnames()?;
        if colnames.is_none() {
            return Err(SortByColumnNameError::MissingColumnNames);
        }
        let by_col_idx = colnames
            .expect("colnames should be present")
            .iter()
            .position(|x| *x == by);
        if let Some(i) = by_col_idx {
            self.sort_by_column(i)?;
            Ok(self)
        } else {
            Err(SortByColumnNameError::ColumnNameNotFound(by.to_string()))
        }
    }

    #[tracing::instrument(skip(self))]
    pub fn sort_by_order(&mut self, order: &[usize]) -> Result<&mut Self, SortByOrderError> {
        if order.len() != self.nrows()? {
            return Err(SortByOrderError::OrderLengthMismatch(order.len()));
        }
        for i in order.iter() {
            if *i >= self.nrows()? {
                return Err(SortByOrderError::RowIndexOutOfBounds(*i));
            }
        }
        let data = vec![MaybeUninit::<f64>::uninit(); self.as_mat_ref()?.nrows() * self.ncols()?];
        let m = self.as_mat_ref()?;
        m.par_col_chunks(1).enumerate().for_each(|(i, c)| {
            let col = c.col(0);
            let slice = unsafe {
                std::slice::from_raw_parts_mut(
                    data.as_ptr().add(i * m.nrows()).cast::<f64>().cast_mut(),
                    m.nrows(),
                )
            };
            for (i, o) in order.iter().enumerate() {
                slice[i] = col[*o];
            }
        });
        *self = Matrix::Owned(OwnedMatrix::new(
            m.nrows(),
            m.ncols(),
            // SAFETY: The data is initialized now
            unsafe { std::mem::transmute(data) },
            self.colnames()?
                .map(|x| x.iter().map(|x| x.to_string()).collect()),
        ));
        Ok(self)
    }

    #[tracing::instrument(skip(self))]
    pub fn dedup_by_column(&mut self, by: usize) -> Result<&mut Self, DedupByColumnError> {
        if by >= self.ncols()? {
            return Err(DedupByColumnError::ColumnIndexOutOfBounds(by));
        }
        let mut col = self.col(by)?.unwrap().to_vec();
        col.sort_by(|a, b| a.partial_cmp(b).expect("could not compare"));
        let mut removing = HashSet::new();
        for i in 1..col.len() {
            if col[i] == col[i - 1] {
                removing.insert(i);
            }
        }
        Ok(self
            .remove_rows(&removing)
            // by doing this we avoid nesting another error that's impossible to occur inside
            // DedupByColumnError
            .expect("all indices should be valid"))
    }

    #[tracing::instrument(skip(self))]
    pub fn dedup_by_column_name(&mut self, by: &str) -> Result<&mut Self, DedupByColumnNameError> {
        let colnames = self.colnames()?;
        if colnames.is_none() {
            return Err(DedupByColumnNameError::MissingColumnNames);
        }
        let by_col_idx = colnames
            .expect("colnames should be present")
            .iter()
            .position(|x| *x == by);
        if let Some(i) = by_col_idx {
            self.dedup_by_column(i)?;
            Ok(self)
        } else {
            Err(DedupByColumnNameError::ColumnNameNotFound(by.to_string()))
        }
    }

    #[tracing::instrument(skip(self, other))]
    pub fn match_to(
        &mut self,
        other: &[f64],
        col: usize,
        join: Join,
    ) -> Result<&mut Self, MatchToColumnError> {
        if col >= self.ncols()? {
            return Err(MatchToColumnError::ColumnIndexOutOfBounds(col));
        }
        let col = self
            .col(col)?
            .unwrap()
            .iter()
            .enumerate()
            .collect::<Vec<_>>();
        let mut other = other.iter().enumerate().collect::<Vec<_>>();
        other.sort_by(|a, b| a.1.partial_cmp(b.1).expect("could not compare"));
        let mut order = Vec::with_capacity(match join {
            Join::Inner => col.len().min(other.len()),
            Join::Left => col.len(),
            Join::Right => other.len(),
        });
        let mut i = 0;
        let mut j = 0;
        while i < col.len() && j < other.len() {
            match col[i].1.partial_cmp(other[j].1) {
                Some(std::cmp::Ordering::Less) => i += 1,
                Some(std::cmp::Ordering::Equal) => {
                    order.push((other[j].0, col[i].0));
                    i += 1;
                    j += 1;
                },
                Some(std::cmp::Ordering::Greater) => j += 1,
                None => panic!("could not compare"),
            }
        }
        match join {
            Join::Inner => (),
            Join::Left => {
                if order.len() < col.len() {
                    return Err(MatchToColumnError::NotAllRowsMatched(join));
                }
            },
            Join::Right => {
                if order.len() < other.len() {
                    return Err(MatchToColumnError::NotAllRowsMatched(join));
                }
            },
        }
        let m = self.as_mat_ref()?;
        let data = vec![MaybeUninit::<f64>::uninit(); m.ncols() * order.len()];
        order.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("could not compare"));
        let order = order.into_iter().map(|(_, o)| o).collect::<Vec<_>>();
        m.par_col_chunks(1).enumerate().for_each(|(i, c)| {
            let col = c.col(0);
            // SAFETY: No two threads will write to the same location
            let slice = unsafe {
                std::slice::from_raw_parts_mut(
                    data.as_ptr().add(i * order.len()).cast::<f64>().cast_mut(),
                    m.nrows(),
                )
            };
            for (i, o) in order.iter().enumerate() {
                slice[i] = col[*o];
            }
        });
        *self = Matrix::Owned(OwnedMatrix::new(
            order.len(),
            m.ncols(),
            // SAFETY: The data is initialized now
            unsafe { std::mem::transmute(data) },
            self.colnames()?
                .map(|x| x.iter().map(|x| x.to_string()).collect()),
        ));

        // if self is sorted, then we can just go by pairs otherwise binary search
        Ok(self)
    }

    #[tracing::instrument(skip(self))]
    pub fn match_to_by_column_name(
        &mut self,
        other: &[f64],
        col: &str,
        join: Join,
    ) -> Result<&mut Self, MatchToByColumnNameError> {
        let colnames = self.colnames()?;
        if colnames.is_none() {
            return Err(MatchToByColumnNameError::MissingColumnNames);
        }
        let by_col_idx = colnames
            .expect("colnames should be present")
            .iter()
            .position(|x| *x == col);
        if let Some(i) = by_col_idx {
            self.match_to(other, i, join)?;
            Ok(self)
        } else {
            Err(MatchToByColumnNameError::ColumnNameNotFound(
                col.to_string(),
            ))
        }
    }

    #[tracing::instrument(skip(self, other))]
    pub fn join(
        &mut self,
        other: &mut Matrix<'a>,
        self_by: usize,
        other_by: usize,
        join: Join,
    ) -> Result<&mut Self, JoinError> {
        if self_by >= self.ncols()? {
            return Err(JoinError::ColumnIndexOutOfBounds(self_by));
        }
        if other_by >= other.ncols()? {
            return Err(JoinError::ColumnIndexOutOfBounds(other_by));
        }
        let mut self_col = self
            .col(self_by)?
            .unwrap()
            .iter()
            .enumerate()
            .collect::<Vec<_>>();
        self_col.sort_by(|a, b| a.1.partial_cmp(b.1).expect("could not compare"));
        let mut other_col = other
            .col(other_by)?
            .unwrap()
            .iter()
            .enumerate()
            .collect::<Vec<_>>();
        other_col.sort_by(|a, b| a.1.partial_cmp(b.1).expect("could not compare"));
        let mut order = Vec::with_capacity(match join {
            Join::Inner => self_col.len().min(other_col.len()),
            Join::Left => self_col.len(),
            Join::Right => other_col.len(),
        });
        let mut i = 0;
        let mut j = 0;
        while i < self_col.len() && j < other_col.len() {
            match self_col[i].1.partial_cmp(other_col[j].1) {
                Some(std::cmp::Ordering::Less) => i += 1,
                Some(std::cmp::Ordering::Equal) => {
                    order.push((self_col[i].0, other_col[j].0));
                    i += 1;
                    j += 1;
                },
                Some(std::cmp::Ordering::Greater) => j += 1,
                None => panic!("could not compare"),
            }
        }
        order.sort_by_key(|x| x.0);
        trace!("order: {:?}", order);
        match join {
            Join::Inner => (),
            Join::Left => {
                if order.len() < self_col.len() {
                    return Err(JoinError::NotAllRowsMatched(join));
                }
            },
            Join::Right => {
                if order.len() < other_col.len() {
                    return Err(JoinError::NotAllRowsMatched(join));
                }
            },
        }
        let self_m = self.as_mat_ref()?;
        let other_m = other.as_mat_ref()?;
        let ncols = self_m.ncols() + other_m.ncols() - 1;
        let data = vec![MaybeUninit::<f64>::uninit(); ncols * order.len()];
        debug!("nrows: {}, ncols: {}", order.len(), ncols);
        let self_cols = (0..self_m.ncols()).collect::<Vec<_>>();
        let mut other_cols = (0..other_m.ncols()).collect::<Vec<_>>();
        other_cols.remove(other_by);
        rayon::scope(|s| {
            s.spawn(|_| {
                self_cols.par_iter().enumerate().for_each(|(i, &c)| {
                    let col = self_m.col(c);
                    let slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            data.as_ptr().add(i * order.len()).cast::<f64>().cast_mut(),
                            self_m.nrows(),
                        )
                    };
                    for (i, o) in order.iter().enumerate() {
                        slice[i] = col[o.0];
                    }
                });
            });
            s.spawn(|_| {
                other_cols.par_iter().enumerate().for_each(|(i, &c)| {
                    let col = other_m.col(c);
                    let slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            data.as_ptr()
                                .add((i + self_m.ncols()) * order.len())
                                .cast::<f64>()
                                .cast_mut(),
                            other_m.nrows(),
                        )
                    };
                    for (i, o) in order.iter().enumerate() {
                        slice[i] = col[o.1];
                    }
                });
            });
        });
        let mut self_colnames = self
            .colnames()?
            .map(|x| x.iter().map(|x| x.to_string()).collect::<Vec<_>>());
        let other_colnames = other.colnames()?;
        if let (Some(self_colnames), Some(mut other_colnames)) =
            (&mut self_colnames, other_colnames)
        {
            other_colnames.remove(other_by);
            self_colnames.extend(other_colnames.into_iter().map(|x| x.to_string()));
        }
        *self = Matrix::Owned(OwnedMatrix::new(
            order.len(),
            ncols,
            // SAFETY: The data is initialized now
            unsafe { std::mem::transmute(data) },
            self_colnames,
        ));

        Ok(self)
    }

    #[tracing::instrument(skip(self, other))]
    pub fn join_by_column_name(
        &mut self,
        other: &mut Matrix<'a>,
        by: &str,
        join: Join,
    ) -> Result<&mut Self, JoinByColumnNameError> {
        let self_colnames = self.colnames()?;
        let other_colnames = other.colnames()?;
        if self_colnames.is_none() || other_colnames.is_none() {
            return Err(JoinByColumnNameError::MissingColumnNames);
        }
        let self_by_col_idx = self_colnames
            .expect("colnames should be present")
            .iter()
            .position(|x| *x == by);
        let other_by_col_idx = other_colnames
            .expect("colnames should be present")
            .iter()
            .position(|x| *x == by);
        if let (Some(i), Some(j)) = (self_by_col_idx, other_by_col_idx) {
            self.join(other, i, j, join)?;
            Ok(self)
        } else {
            Err(JoinByColumnNameError::ColumnNameNotFound(by.to_string()))
        }
    }

    pub fn nrows(&mut self) -> Result<usize, ReadMatrixError> {
        self.as_mat_ref().map(|x| x.nrows())
    }

    pub fn nrows_loaded(&self) -> usize {
        self.as_mat_ref_loaded().nrows()
    }

    pub fn ncols(&mut self) -> Result<usize, ReadMatrixError> {
        self.as_mat_ref().map(|x| x.ncols())
    }

    pub fn ncols_loaded(&self) -> usize {
        self.as_mat_ref_loaded().ncols()
    }

    pub fn data(&mut self) -> Result<&[f64], ReadMatrixError> {
        self.as_owned_ref().map(|x| x.data.as_slice())
    }

    pub fn col(&mut self, col: usize) -> Result<Option<&[f64]>, ReadMatrixError> {
        if col >= self.ncols()? {
            return Ok(None);
        }
        self.as_mat_ref().map(|x| {
            Some(unsafe {
                x.get_unchecked(.., col)
                    .try_as_slice()
                    .expect("could not get slice")
            })
        })
    }

    pub fn col_loaded(&self, col: usize) -> Option<&[f64]> {
        if col >= self.ncols_loaded() {
            return None;
        }
        Some(unsafe {
            self.as_mat_ref_loaded()
                .get_unchecked(.., col)
                .try_as_slice()
                .expect("could not get slice")
        })
    }

    pub fn get(&mut self, row: usize, col: usize) -> Result<Option<f64>, ReadMatrixError> {
        let nrows = self.nrows()?;
        let ncols = self.ncols()?;
        self.as_mat_ref().map(|x| {
            if row >= nrows || col > ncols {
                None
            } else {
                Some(unsafe { *x.get_unchecked(row, col) })
            }
        })
    }

    pub fn get_loaded(&self, row: usize, col: usize) -> Option<f64> {
        let nrows = self.nrows_loaded();
        let ncols = self.ncols_loaded();
        if row >= nrows || col > ncols {
            None
        } else {
            Some(unsafe { *self.as_mat_ref_loaded().get_unchecked(row, col) })
        }
    }
}

impl FromStr for Matrix<'_> {
    type Err = FileParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Matrix::File(s.parse()?))
    }
}

#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[archive(check_bytes)]
pub struct OwnedMatrix {
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    pub(crate) colnames: Option<Vec<String>>,
    pub(crate) data: Vec<f64>,
}

impl OwnedMatrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>, colnames: Option<Vec<String>>) -> Self {
        assert!(rows * cols == data.len());
        Self {
            nrows: rows,
            ncols: cols,
            data,
            colnames,
        }
    }

    pub fn transpose(self) -> Self {
        let mut data = Vec::with_capacity(self.nrows * self.ncols);
        data.extend((0..(self.nrows * self.ncols)).map(|_| MaybeUninit::<f64>::uninit()));
        self.data.into_par_iter().enumerate().for_each(|(i, x)| {
            let new_row = i / self.nrows;
            let new_col = i % self.nrows;
            let i = new_col * self.ncols + new_row;
            unsafe {
                data.as_ptr().add(i).cast_mut().cast::<f64>().write(x);
            };
        });
        Self {
            data: unsafe { std::mem::transmute(data) },
            nrows: self.ncols,
            ncols: self.nrows,
            colnames: None,
        }
    }

    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline(always)]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    #[inline(always)]
    pub fn into_data(self) -> Vec<f64> {
        self.data
    }

    #[inline(always)]
    pub fn colnames(&self) -> Option<&[String]> {
        self.colnames.as_deref()
    }

    pub fn from_rmatrix(r: &RMatrix<f64>) -> Self {
        let data = r.data().to_vec();
        Self {
            nrows: r.nrows(),
            ncols: r.ncols(),
            data,
            colnames: colnames(r).map(|x| x.iter().map(|x| x.to_string()).collect()),
        }
    }

    pub fn to_rmatrix(&self) -> RMatrix<f64> {
        use extendr_api::prelude::*;

        let mat = RMatrix::new_matrix(
            self.nrows,
            self.ncols,
            #[inline(always)]
            |r, c| self.data[c * self.nrows + r],
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

pub fn parse(s: &str) -> Result<f64, MatParseError> {
    if s == "NA" {
        return Ok(f64::NAN);
    }
    s.parse().map_err(MatParseError::from)
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

impl<'a> IntoMatrix<'a> for OwnedMatrix {
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

#[cfg(test)]
mod tests {
    use super::*;
    use test_log::test;

    #[test]
    fn test_combine_columns_success() {
        let mut m1 = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m2 = OwnedMatrix::new(
            3,
            2,
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            Some(vec!["c".to_string(), "d".to_string()]),
        )
        .into_matrix();
        let m3 = OwnedMatrix::new(
            3,
            2,
            vec![13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            Some(vec!["e".to_string(), "f".to_string()]),
        )
        .into_matrix();
        let m = m1.combine_columns(&mut [m2, m3]).unwrap();
        assert_eq!(
            m.data().unwrap(),
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0
            ],
        );
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &[
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
                "e".to_string(),
                "f".to_string()
            ]
        );
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 6);
    }

    #[test]
    fn test_combine_columns_dimensions_mismatch() {
        let mut m1 = OwnedMatrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], None).into_matrix();
        let m2 = OwnedMatrix::new(2, 2, vec![19.0, 20.0, 21.0, 22.0], None).into_matrix();
        let res = m1.combine_columns(&mut [m2]).unwrap_err();
        assert!(matches!(res, CombineColumnsError::MatrixDimensionsMismatch));
    }

    #[test]
    fn test_combine_columns_no_colnames() {
        let mut m1 = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m2 =
            OwnedMatrix::new(3, 2, vec![19.0, 20.0, 21.0, 22.0, 23.0, 24.0], None).into_matrix();
        let m = m1.combine_columns(&mut [m2]).unwrap();
        assert_eq!(
            m.data().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
        );
        assert!(m.colnames().unwrap().is_none());
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 4);
    }

    #[test]
    fn test_combine_rows_success() {
        let mut m1 = OwnedMatrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], None).into_matrix();
        let m2 = OwnedMatrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], None).into_matrix();
        let m3 =
            OwnedMatrix::new(3, 2, vec![13.0, 14.0, 15.0, 16.0, 17.0, 18.0], None).into_matrix();
        let m = m1.combine_rows(&mut [m2, m3]).unwrap();
        assert_eq!(
            m.data().unwrap(),
            &[
                1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 13.0, 14.0, 15.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0,
                16.0, 17.0, 18.0
            ],
        );
        assert!(m.colnames().unwrap().is_none());
        assert_eq!(m.nrows().unwrap(), 9);
        assert_eq!(m.ncols().unwrap(), 2);
    }

    #[test]
    fn test_combine_rows_dimensions_mismatch() {
        let mut m1 = OwnedMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], None).into_matrix();
        let m2 = OwnedMatrix::new(2, 2, vec![19.0, 20.0, 21.0, 22.0], None).into_matrix();
        let res = m1.combine_rows(&mut [m2]).unwrap_err();
        assert!(matches!(res, CombineRowsError::MatrixDimensionsMismatch));
    }

    #[test]
    fn test_combine_rows_column_names_mismatch() {
        let mut m1 = OwnedMatrix::new(
            2,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m2 = OwnedMatrix::new(
            2,
            3,
            vec![19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            Some(vec!["c".to_string(), "d".to_string()]),
        )
        .into_matrix();
        let m = m1.combine_rows(&mut [m2]).unwrap_err();
        assert!(matches!(m, CombineRowsError::ColumnNamesMismatch));
    }

    #[test]
    fn test_remove_rows_success() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let mut removing = HashSet::new();
        removing.insert(1);
        let m = m.remove_rows(&removing).unwrap();
        assert_eq!(m.data().unwrap(), &[1.0, 3.0, 4.0, 6.0]);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
        assert_eq!(m.nrows().unwrap(), 2);
        assert_eq!(m.ncols().unwrap(), 2);
    }

    #[test]
    fn test_remove_rows_index_out_of_bounds() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let mut removing = HashSet::new();
        removing.insert(3);
        let m = m.remove_rows(&removing).unwrap_err();
        assert!(matches!(m, RemoveRowsError::RowIndexOutOfBounds(3)));
    }

    #[test]
    fn test_remove_columns_success() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let mut removing = HashSet::new();
        removing.insert(1);
        let m = m.remove_columns(&removing).unwrap();
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(m.colnames().unwrap().unwrap(), &["a".to_string()]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 1);
    }

    #[test]
    fn test_remove_columns_index_out_of_bounds() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let mut removing = HashSet::new();
        removing.insert(2);
        let m = m.remove_columns(&removing).unwrap_err();
        assert!(matches!(m, RemoveColumnsError::ColumnIndexOutOfBounds(2)));
    }

    #[test]
    fn test_remove_column_by_name_success() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.remove_column_by_name("a").unwrap();
        assert_eq!(m.data().unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(m.colnames().unwrap().unwrap(), &["b".to_string()]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 1);
    }

    #[test]
    fn test_remove_column_by_name_column_not_found() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.remove_column_by_name("c").unwrap_err();
        assert!(matches!(m, RemoveColumnByNameError::ColumnNameNotFound(_)));
    }

    #[test]
    fn test_remove_column_by_name_if_exists_success() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.remove_column_by_name_if_exists("a").unwrap();
        assert_eq!(m.data().unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(m.colnames().unwrap().unwrap(), &["b".to_string()]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 1);
    }

    #[test]
    fn test_remove_column_by_name_if_exists_column_not_found() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.remove_column_by_name_if_exists("c").unwrap();
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
    }

    #[test]
    fn test_transpose() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.transpose().unwrap();
        assert_eq!(m.data().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(m.nrows().unwrap(), 2);
        assert_eq!(m.ncols().unwrap(), 3);
        assert!(m.colnames().unwrap().is_none(),);
    }

    #[test]
    fn test_sort_by_column_success() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.sort_by_column(0).unwrap();
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_sort_by_column_already_sorted() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.sort_by_column(0).unwrap();
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_sort_by_column_index_out_of_bounds() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.sort_by_column(2).unwrap_err();
        assert!(matches!(m, SortByColumnError::ColumnIndexOutOfBounds(2)));
    }

    #[test]
    fn test_sort_by_column_name_success() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.sort_by_column_name("a").unwrap();
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_sort_by_column_name_no_colnames() {
        let mut m = OwnedMatrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], None).into_matrix();
        let m = m.sort_by_column_name("a").unwrap_err();
        assert!(matches!(m, SortByColumnNameError::MissingColumnNames));
    }

    #[test]
    fn test_sort_by_column_name_not_found() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.sort_by_column_name("c").unwrap_err();
        assert!(matches!(m, SortByColumnNameError::ColumnNameNotFound(_)));
    }

    #[test]
    fn test_sort_by_order() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.sort_by_order(&[2, 0, 1]).unwrap();
        assert_eq!(m.data().unwrap(), &[1.0, 3.0, 2.0, 4.0, 6.0, 5.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_sort_by_order_out_of_bounds() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.sort_by_order(&[2, 0, 3]).unwrap_err();
        assert!(matches!(m, SortByOrderError::RowIndexOutOfBounds(3)));
    }

    #[test]
    fn test_sort_by_order_length_missmatch() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.sort_by_order(&[2, 0]).unwrap_err();
        assert!(matches!(m, SortByOrderError::OrderLengthMismatch(2)));
    }

    #[test]
    fn test_dedup_by_column_success() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 2.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.dedup_by_column(0).unwrap();
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 4.0, 5.0]);
        assert_eq!(m.nrows().unwrap(), 2);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_dedup_by_column_index_out_of_bounds() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 2.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.dedup_by_column(2).unwrap_err();
        assert!(matches!(m, DedupByColumnError::ColumnIndexOutOfBounds(2)));
    }

    #[test]
    fn test_dedup_by_column_name_success() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 2.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.dedup_by_column_name("a").unwrap();
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 4.0, 5.0]);
        assert_eq!(m.nrows().unwrap(), 2);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_dedup_by_column_name_no_colnames() {
        let mut m = OwnedMatrix::new(3, 2, vec![1.0, 2.0, 2.0, 4.0, 5.0, 6.0], None).into_matrix();
        let m = m.dedup_by_column_name("a").unwrap_err();
        assert!(matches!(m, DedupByColumnNameError::MissingColumnNames));
    }

    #[test]
    fn test_dedup_by_column_name_not_found() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 2.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.dedup_by_column_name("c").unwrap_err();
        assert!(matches!(m, DedupByColumnNameError::ColumnNameNotFound(_)));
    }

    #[test]
    fn test_match_to_success_inner() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let other = [5.0, 8.0, 1.0, 2.0, 7.0];
        let m = m1.match_to(&other, 0, Join::Inner).unwrap();
        assert_eq!(m.data().unwrap(), &[5.0, 1.0, 2.0, 5.0, 1.0, 2.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_match_to_success_left() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let other = [5.0, 1.0, 6.0, 2.0, 3.0, 4.0, 5.0, 7.0];
        let m = m1.match_to(&other, 0, Join::Left).unwrap();
        assert_eq!(
            m.data().unwrap(),
            &[5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0],
        );
        assert_eq!(m.nrows().unwrap(), 5);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_match_to_success_right() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let other = [5.0, 1.0, 2.0];
        let m = m1.match_to(&other, 0, Join::Right).unwrap();
        assert_eq!(m.data().unwrap(), &[5.0, 1.0, 2.0, 5.0, 1.0, 2.0],);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_match_to_success_empty_other() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let other = [];
        let m = m1.match_to(&other, 0, Join::Inner).unwrap();
        assert!(m.data().unwrap().is_empty());
        assert_eq!(m.nrows().unwrap(), 0);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_match_to_index_out_of_bounds() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let other = [5.0, 1.0, 6.0, 2.0, 3.0, 4.0, 5.0, 7.0];
        let m = m1.match_to(&other, 2, Join::Left).unwrap_err();
        assert!(matches!(m, MatchToColumnError::ColumnIndexOutOfBounds(2)));
    }

    #[test]
    fn test_match_to_not_all_rows_matched_left() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let other = [5.0, 6.0, 2.0, 3.0, 4.0, 5.0];
        let m = m1.match_to(&other, 0, Join::Left).unwrap_err();
        assert!(matches!(
            m,
            MatchToColumnError::NotAllRowsMatched(Join::Left)
        ));
    }

    #[test]
    fn test_match_to_not_all_rows_matched_right() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let other = [5.0, 6.0, 2.0, 3.0, 4.0, 5.0];
        let m = m1.match_to(&other, 0, Join::Right).unwrap_err();
        assert!(matches!(
            m,
            MatchToColumnError::NotAllRowsMatched(Join::Right)
        ));
    }

    #[test]
    fn test_match_to_by_column_name_success_inner() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let other = [5.0, 8.0, 1.0, 2.0, 7.0];
        let m = m1
            .match_to_by_column_name(&other, "a", Join::Inner)
            .unwrap();
        assert_eq!(m.data().unwrap(), &[5.0, 1.0, 2.0, 5.0, 1.0, 2.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_match_to_by_column_name_success_left() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let other = [5.0, 1.0, 6.0, 2.0, 3.0, 4.0, 5.0, 7.0];
        let m = m1.match_to_by_column_name(&other, "a", Join::Left).unwrap();
        assert_eq!(
            m.data().unwrap(),
            &[5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0],
        );
        assert_eq!(m.nrows().unwrap(), 5);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_match_to_by_column_name_success_right() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let other = [5.0, 1.0, 2.0];
        let m = m1
            .match_to_by_column_name(&other, "a", Join::Right)
            .unwrap();
        assert_eq!(m.data().unwrap(), &[5.0, 1.0, 2.0, 5.0, 1.0, 2.0],);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_match_to_by_column_name_success_empty_other() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let other = [];
        let m = m1
            .match_to_by_column_name(&other, "a", Join::Inner)
            .unwrap();
        assert!(m.data().unwrap().is_empty());
        assert_eq!(m.nrows().unwrap(), 0);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_match_to_by_column_name_column_name_not_found() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let other = [5.0, 8.0, 1.0, 2.0, 7.0];
        let m = m1
            .match_to_by_column_name(&other, "c", Join::Inner)
            .unwrap_err();
        assert!(matches!(m, MatchToByColumnNameError::ColumnNameNotFound(_)));
    }

    #[test]
    fn test_match_to_by_column_name_no_colnames() {
        let mut m1 = OwnedMatrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], None).into_matrix();
        let other = [5.0, 8.0, 1.0, 2.0, 7.0];
        let m = m1
            .match_to_by_column_name(&other, "a", Join::Inner)
            .unwrap_err();
        assert!(matches!(m, MatchToByColumnNameError::MissingColumnNames));
    }

    #[test]
    fn test_join_success_inner() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let mut m2 = OwnedMatrix::new(
            3,
            2,
            vec![5.0, 6.0, 2.0, 7.0, 3.0, 4.0],
            Some(vec!["a".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m1.join(&mut m2, 0, 0, Join::Inner).unwrap();
        assert_eq!(
            m.data().unwrap(),
            &[6.0, 2.0, 5.0, 6.0, 2.0, 5.0, 3.0, 4.0, 7.0]
        );
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 3);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_join_success_left() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let mut m2 = OwnedMatrix::new(
            7,
            2,
            vec![
                5.0, 6.0, 2.0, 7.0, 3.0, 4.0, 1.0, 5.0, 6.0, 2.0, 7.0, 3.0, 4.0, 1.0,
            ],
            Some(vec!["a".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m1.join(&mut m2, 0, 0, Join::Left).unwrap();
        assert_eq!(
            m.data().unwrap(),
            &[6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0]
        );
        assert_eq!(m.nrows().unwrap(), 5);
        assert_eq!(m.ncols().unwrap(), 3);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_join_success_right() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let mut m2 = OwnedMatrix::new(
            3,
            2,
            vec![5.0, 6.0, 2.0, 7.0, 3.0, 4.0],
            Some(vec!["a".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m1.join(&mut m2, 0, 0, Join::Right).unwrap();
        assert_eq!(
            m.data().unwrap(),
            &[6.0, 2.0, 5.0, 6.0, 2.0, 5.0, 3.0, 4.0, 7.0]
        );
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 3);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_join_index_out_of_bounds() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let mut m2 = OwnedMatrix::new(
            3,
            2,
            vec![5.0, 6.0, 2.0, 7.0, 3.0, 4.0],
            Some(vec!["a".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m1.join(&mut m2, 0, 2, Join::Inner).unwrap_err();
        assert!(matches!(m, JoinError::ColumnIndexOutOfBounds(2)));
    }

    #[test]
    fn test_join_not_all_rows_matched_left() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![8.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let mut m2 = OwnedMatrix::new(
            7,
            2,
            vec![
                5.0, 6.0, 2.0, 7.0, 3.0, 4.0, 1.0, 5.0, 6.0, 2.0, 7.0, 3.0, 4.0, 1.0,
            ],
            Some(vec!["a".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m1.join(&mut m2, 0, 0, Join::Left).unwrap_err();
        assert!(matches!(m, JoinError::NotAllRowsMatched(Join::Left)));
    }

    #[test]
    fn test_join_not_all_rows_matched_right() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let mut m2 = OwnedMatrix::new(
            3,
            2,
            vec![8.0, 6.0, 2.0, 7.0, 3.0, 4.0],
            Some(vec!["a".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m1.join(&mut m2, 0, 0, Join::Right).unwrap_err();
        assert!(matches!(m, JoinError::NotAllRowsMatched(Join::Right)));
    }

    #[test]
    fn test_join_by_column_name_success_inner() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let mut m2 = OwnedMatrix::new(
            3,
            2,
            vec![5.0, 6.0, 2.0, 7.0, 3.0, 4.0],
            Some(vec!["a".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m1.join_by_column_name(&mut m2, "a", Join::Inner).unwrap();
        assert_eq!(
            m.data().unwrap(),
            &[6.0, 2.0, 5.0, 6.0, 2.0, 5.0, 3.0, 4.0, 7.0]
        );
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 3);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_join_by_column_name_success_left() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let mut m2 = OwnedMatrix::new(
            7,
            2,
            vec![
                5.0, 6.0, 2.0, 7.0, 3.0, 4.0, 1.0, 5.0, 6.0, 2.0, 7.0, 3.0, 4.0, 1.0,
            ],
            Some(vec!["a".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m1.join_by_column_name(&mut m2, "a", Join::Left).unwrap();
        assert_eq!(
            m.data().unwrap(),
            &[6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0]
        );
        assert_eq!(m.nrows().unwrap(), 5);
        assert_eq!(m.ncols().unwrap(), 3);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_join_by_column_name_success_right() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let mut m2 = OwnedMatrix::new(
            3,
            2,
            vec![5.0, 6.0, 2.0, 7.0, 3.0, 4.0],
            Some(vec!["a".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m1.join_by_column_name(&mut m2, "a", Join::Right).unwrap();
        assert_eq!(
            m.data().unwrap(),
            &[6.0, 2.0, 5.0, 6.0, 2.0, 5.0, 3.0, 4.0, 7.0]
        );
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 3);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }
}

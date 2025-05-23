use std::{
    collections::{HashMap, HashSet},
    f64,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    str::FromStr,
};

#[cfg(feature = "r")]
use extendr_api::{
    io::Load, scalar::Scalar, single_threaded, wrapper, AsStrIter, Attributes, Conversions,
    IntoRobj, MatrixConversions, RMatrix, Rinternals, Robj, Rtype,
};
use faer::{c64, linalg::qr, ColMut, Mat, MatMut, MatRef, RowMut};
use rand_distr::Distribution;
use rayon::prelude::*;
use regex::Regex;
use tracing::{debug, error, info, trace};

use crate::{file::File, mean, standardize_column, standardize_row, Error};

#[derive(Debug, Clone, Copy)]
pub enum Join {
    /// Inner join, only rows that are present in both matrices are kept
    Inner = 0,
    /// Left join, all rows from the left matrix must be matched
    Left = 1,
    /// Right join, all rows from the right matrix must be matched
    Right = 2,
}

const INVALID_JOIN_TYPE: &str = "invalid join type, must be one of 0, 1, or 2";

#[cfg(feature = "r")]
impl TryFrom<Robj> for Join {
    type Error = &'static str;

    #[cfg_attr(coverage_nightly, coverage(off))]
    fn try_from(obj: Robj) -> Result<Self, Self::Error> {
        let val = if obj.is_integer() {
            obj.as_integer().unwrap()
        } else if obj.is_real() {
            obj.as_real().unwrap() as i32
        } else if obj.is_logical() {
            obj.as_logical().unwrap().inner()
        } else {
            return Err(INVALID_JOIN_TYPE);
        };
        match val {
            0 => Ok(Join::Inner),
            1 => Ok(Join::Left),
            2 => Ok(Join::Right),
            _ => Err(INVALID_JOIN_TYPE),
        }
    }
}

impl std::fmt::Display for Join {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Join::Inner => write!(f, "inner"),
            Join::Left => write!(f, "left"),
            Join::Right => write!(f, "right"),
        }
    }
}

#[doc(hidden)]
pub trait DerefMatrix: Deref<Target = Matrix> + DerefMut + std::fmt::Debug {}

impl<T> DerefMatrix for T where T: Deref<Target = Matrix> + DerefMut + std::fmt::Debug {}

pub enum Matrix {
    #[cfg(feature = "r")]
    R(RMatrix<f64>),
    Owned(OwnedMatrix),
    File(File),
    Dyn(Box<dyn DerefMatrix>),
    Transform(
        #[allow(clippy::type_complexity)]
        Vec<Box<dyn for<'a> FnOnce(&'a mut Matrix) -> Result<&'a mut Matrix, Error>>>,
        Box<Matrix>,
    ),
}

impl PartialEq for Matrix {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            #[cfg(feature = "r")]
            (Matrix::R(a), Matrix::R(b)) => a == b,
            (Matrix::Owned(a), Matrix::Owned(b)) => a == b,
            (Matrix::File(a), Matrix::File(b)) => a == b,
            (Matrix::Dyn(a), Matrix::Dyn(b)) => ***a == ***b,
            (Matrix::Transform(_, a), Matrix::Transform(_, b)) => a == b,
            _ => false,
        }
    }
}

impl std::fmt::Debug for Matrix {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "r")]
            Matrix::R(m) => write!(f, "Matrix::R({:?})", m),
            Matrix::Owned(m) => write!(f, "Matrix::Owned({:?})", m),
            Matrix::File(m) => write!(f, "Matrix::File({:?})", m),
            Matrix::Dyn(m) => write!(f, "Matrix::Dyn({:?})", m),
            Matrix::Transform(t, m) => write!(f, "Matrix::Transform({:?}, {:?})", t.len(), m),
        }
    }
}

// SAFETY: This is always safe except when calling into R, for example by
// loading an .RData file
unsafe impl Send for Matrix {}
unsafe impl Sync for Matrix {}

impl Matrix {
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn as_mat_ref(&mut self) -> Result<MatRef<'_, f64>, crate::Error> {
        Ok(match self {
            m @ (Matrix::File(_) | Matrix::Transform(..)) => m.into_owned()?,
            m => m,
        }
        .as_mat_ref_loaded())
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn as_mat_ref_loaded(&self) -> MatRef<'_, f64> {
        match self {
            #[cfg(feature = "r")]
            Matrix::R(m) => MatRef::from_column_major_slice(m.data(), m.nrows(), m.ncols()),
            Matrix::Owned(m) => {
                MatRef::from_column_major_slice(m.data.as_slice(), m.nrows, m.ncols)
            },
            Matrix::File(_) => panic!("cannot call this function on a file"),
            Matrix::Dyn(m) => m.as_mat_ref_loaded(),
            Matrix::Transform(..) => panic!("cannot call this function on a transform"),
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn as_mat_mut(&mut self) -> Result<MatMut<'_, f64>, crate::Error> {
        Ok(match self {
            // SAFETY: We know that the data is valid
            #[cfg(feature = "r")]
            Matrix::R(m) => unsafe {
                MatMut::from_raw_parts_mut(
                    m.data().as_ptr().cast_mut(),
                    m.nrows(),
                    m.ncols(),
                    1,
                    m.nrows() as isize,
                )
            },
            Matrix::Owned(m) => {
                MatMut::from_column_major_slice_mut(m.data.as_mut(), m.nrows, m.ncols)
            },
            m @ (Matrix::File(_) | Matrix::Transform(..)) => m.into_owned()?.as_mat_mut()?,
            Matrix::Dyn(m) => m.as_mat_mut()?,
        })
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn as_owned_ref(&mut self) -> Result<&OwnedMatrix, crate::Error> {
        self.into_owned()?;
        match self {
            Matrix::Owned(m) => Ok(&*m),
            Matrix::Dyn(m) => m.as_owned_ref(),
            _ => unreachable!(),
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn as_owned_mut(&mut self) -> Result<&mut OwnedMatrix, crate::Error> {
        self.into_owned()?;
        match self {
            Matrix::Owned(m) => Ok(m),
            Matrix::Dyn(m) => m.as_owned_mut(),
            _ => unreachable!(),
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn is_loaded(&self) -> bool {
        match self {
            #[cfg(feature = "r")]
            Matrix::R(_) => true,
            Matrix::Owned(_) => true,
            Matrix::File(_) | Matrix::Transform(..) => false,
            Matrix::Dyn(m) => m.is_loaded(),
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    #[cfg(feature = "r")]
    pub fn to_rmatrix(&mut self) -> Result<RMatrix<f64>, crate::Error> {
        Ok(match self {
            #[cfg(feature = "r")]
            Matrix::R(m) => m.clone().into_robj().clone().as_matrix().unwrap(),
            Matrix::Owned(m) => {
                use extendr_api::prelude::*;

                let mut mat = RMatrix::new_matrix(
                    m.nrows,
                    m.ncols,
                    #[inline(always)]
                    |r, c| m.data[c * m.nrows + r],
                );
                let mut dimnames = List::from_values([NULL, NULL]);
                if let Some(colnames) = &m.colnames {
                    dimnames.set_elt(1, colnames.into_robj()).unwrap();
                }
                mat.set_attrib(wrapper::symbol::dimnames_symbol(), dimnames)
                    .unwrap();
                mat
            },
            m @ (Matrix::File(_) | Matrix::Transform(..)) => m.into_owned()?.to_rmatrix()?,
            Matrix::Dyn(m) => m.to_rmatrix()?,
        })
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    #[cfg(feature = "r")]
    pub fn into_robj(&mut self) -> Result<Robj, crate::Error> {
        Ok(self.to_rmatrix().into_robj())
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    #[cfg(feature = "r")]
    pub fn from_rdata(mut reader: impl std::io::Read) -> Result<Self, crate::Error> {
        let mut buf = [0; 5];
        reader.read_exact(&mut buf)?;
        if buf != *b"RDX3\n" {
            return Err(crate::Error::InvalidRdataFile);
        }
        let obj = Robj::from_reader(
            &mut reader,
            extendr_api::io::PstreamFormat::R_pstream_xdr_format,
            None,
        )?;
        let mat = obj
            .as_pairlist()
            .ok_or(crate::Error::InvalidRdataFile)?
            .into_iter()
            .next()
            .ok_or(crate::Error::InvalidRdataFile)?
            .1;
        Matrix::from_robj(mat)
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    #[cfg(feature = "r")]
    pub fn from_rds(mut reader: impl std::io::Read) -> Result<Self, crate::Error> {
        let obj = Robj::from_reader(
            &mut reader,
            extendr_api::io::PstreamFormat::R_pstream_xdr_format,
            None,
        )?;
        Matrix::from_robj(obj)
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    #[cfg(feature = "r")]
    pub fn from_robj(r: Robj) -> Result<Self, crate::Error> {
        fn r_int_to_f64(r: i32) -> f64 {
            if r == i32::MIN {
                f64::NAN
            } else {
                r as f64
            }
        }

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
                .map(|i| r_int_to_f64(*i))
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

                let df = r.as_list().expect("data is a list");
                struct Par(pub Robj);
                unsafe impl Send for Par {}
                let mut names = df.iter().map(|(n, r)| (n, Par(r))).collect::<Vec<_>>();
                let (names, data) = names.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
                let data = data
                    .into_par_iter()
                    .map(|Par(r)| {
                        if r.is_string() {
                            Ok(r.as_str_iter()
                                .unwrap()
                                .map(|x| x.parse::<f64>())
                                .collect::<std::result::Result<Vec<_>, _>>()?)
                        } else if r.is_integer() {
                            Ok(r.as_integer_slice()
                                .unwrap()
                                .iter()
                                .map(|x| r_int_to_f64(*x))
                                .collect())
                        } else if r.is_real() {
                            Ok(r.as_real_vector().unwrap().to_vec())
                        } else if r.is_logical() {
                            Ok(r.as_logical_slice()
                                .unwrap()
                                .iter()
                                .map(|x| r_int_to_f64(x.inner()))
                                .collect())
                        } else {
                            Err(crate::Error::InvalidItemType)
                        }
                    })
                    .collect::<std::result::Result<Vec<_>, crate::Error>>()?;
                let ncols = data.len();
                let nrows = data[0].len();
                for i in data.iter().skip(1) {
                    if i.len() != nrows {
                        return Err(crate::Error::UnequalColumnLengths);
                    }
                }
                Ok(Matrix::Owned(OwnedMatrix::new(
                    nrows,
                    ncols,
                    data.concat(),
                    Some(names.into_iter().map(|x| x.to_string()).collect()),
                )))
            } else {
                Err(crate::Error::InvalidItemType)
            }
        } else {
            Err(crate::Error::InvalidItemType)
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    #[doc(hidden)]
    pub fn to_owned_loaded(self) -> OwnedMatrix {
        if let Matrix::Owned(m) = self {
            m
        } else {
            panic!("matrix is not owned");
        }
    }

    #[tracing::instrument(skip(self))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn into_owned(&mut self) -> Result<&mut Self, crate::Error> {
        match self {
            #[cfg(feature = "r")]
            Matrix::R(_) => {
                let colnames = self
                    .colnames()?
                    .map(|x| x.into_iter().map(|x| x.to_string()).collect());
                let Matrix::R(m) = self else { unreachable!() };
                *self = Matrix::Owned(OwnedMatrix::new(
                    m.nrows(),
                    m.ncols(),
                    m.data().to_vec(),
                    colnames,
                ));
                Ok(self)
            },
            Matrix::Owned(_) => Ok(self),
            Matrix::File(m) => {
                *self = m.read()?;
                Ok(self)
            },
            Matrix::Dyn(m) => {
                m.into_owned()?;
                Ok(self)
            },
            Matrix::Transform(..) => self.transform(),
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn colnames(&mut self) -> Result<Option<Vec<&str>>, crate::Error> {
        Ok(match self {
            #[cfg(feature = "r")]
            Matrix::R(m) => m.dimnames().and_then(|mut dimnames| {
                dimnames
                    .nth(1)
                    .unwrap()
                    .as_str_iter()
                    .map(|x| x.collect::<Vec<_>>())
            }),
            Matrix::Owned(m) => m
                .colnames
                .as_deref()
                .map(|x| x.iter().map(|x| x.as_str()).collect()),
            m @ (Matrix::File(_) | Matrix::Transform(..)) => m.into_owned()?.colnames()?,
            Matrix::Dyn(_) => None,
        })
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn colnames_loaded(&self) -> Option<Vec<&str>> {
        match self {
            #[cfg(feature = "r")]
            Matrix::R(m) => m.dimnames().and_then(|mut dimnames| {
                dimnames
                    .nth(1)
                    .unwrap()
                    .as_str_iter()
                    .map(|x| x.collect::<Vec<_>>())
            }),
            Matrix::Owned(m) => m
                .colnames
                .as_deref()
                .map(|x| x.iter().map(|x| x.as_str()).collect()),
            Matrix::File(_) => None,
            Matrix::Dyn(_) => None,
            Matrix::Transform(..) => None,
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn set_colnames(&mut self, colnames: Vec<String>) -> Result<&mut Self, crate::Error> {
        if colnames.len() != self.ncols()? {
            return Err(crate::Error::ColumnNamesMismatch);
        }
        match self {
            #[cfg(feature = "r")]
            Matrix::R(m) => {
                use extendr_api::prelude::*;

                let mut dimnames = List::from_values([NULL, NULL]);
                dimnames.set_elt(1, colnames.into_robj()).unwrap();
                m.set_attrib(wrapper::symbol::dimnames_symbol(), dimnames)
                    .unwrap();
                Ok(self)
            },
            Matrix::Owned(m) => {
                m.colnames = Some(colnames);
                Ok(self)
            },
            m @ (Matrix::File(_) | Matrix::Transform(..)) => m.into_owned()?.set_colnames(colnames),
            Matrix::Dyn(m) => m.set_colnames(colnames),
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn from_slice(data: &[f64], rows: usize, cols: usize) -> Self {
        Matrix::Owned(OwnedMatrix::new(rows, cols, data.to_vec(), None))
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    #[cfg(feature = "r")]
    pub fn from_rmatrix(r: RMatrix<f64>) -> Self {
        Matrix::R(r)
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn from_owned(m: OwnedMatrix) -> Self {
        Matrix::Owned(m)
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn from_file(f: File) -> Self {
        Matrix::File(f)
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn from_deref(m: impl DerefMatrix + 'static) -> Matrix {
        Matrix::Dyn(Box::new(m))
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn from_mat_ref(m: faer::MatRef<'_, f64>) -> Self {
        let data = vec![MaybeUninit::<f64>::uninit(); m.nrows() * m.ncols()];
        m.par_col_chunks(1).enumerate().for_each(|(i, c)| {
            let col = c.col(0);
            let slice = unsafe {
                std::slice::from_raw_parts_mut(
                    data.as_ptr().add(i * m.nrows()).cast::<f64>().cast_mut(),
                    m.nrows(),
                )
            };
            for (i, x) in col.iter().enumerate() {
                slice[i] = *x;
            }
        });
        Matrix::Owned(OwnedMatrix::new(
            m.nrows(),
            m.ncols(),
            // SAFETY: The data is initialized now
            unsafe {
                std::mem::transmute::<std::vec::Vec<std::mem::MaybeUninit<f64>>, std::vec::Vec<f64>>(
                    data,
                )
            },
            None,
        ))
    }

    pub fn generate_normal_matrix(rows: usize, cols: usize, mean: f64, std_dev: f64) -> Self {
        let data = rand_distr::Normal::new(mean, std_dev)
            .unwrap()
            .sample_iter(rand::thread_rng())
            .take(rows * cols)
            .collect::<Vec<_>>();
        Matrix::Owned(OwnedMatrix::new(rows, cols, data, None))
    }

    pub fn generate_standard_normal_matrix(rows: usize, cols: usize) -> Self {
        Self::generate_normal_matrix(rows, cols, 0.0, 1.0)
    }

    pub fn eigen(&mut self, symmetric: Option<bool>) -> Result<Eigen, crate::Error> {
        Eigen::new(self, symmetric)
    }

    pub fn is_symmetric(&mut self) -> Result<bool, crate::Error> {
        let m = self.as_mat_ref()?;
        if m.nrows() != m.ncols() {
            return Err(crate::Error::MatrixDimensionsMismatch);
        }
        let mut is_symmetric = true;
        'outer: for i in 1..m.nrows() {
            for j in 0..i {
                if (m.get(i, j) - m.get(j, i)).abs() >= 100.0 * f64::EPSILON {
                    is_symmetric = false;
                    break 'outer;
                }
            }
        }
        Ok(is_symmetric)
    }
}

pub enum Eigen {
    Real { values: Vec<f64>, vectors: Vec<f64> },
    Complex { values: Vec<c64>, vectors: Vec<c64> },
}

impl Eigen {
    pub fn new(m: &mut Matrix, symmetric: Option<bool>) -> Result<Self, crate::Error> {
        enum E {
            Generic(faer::linalg::solvers::Eigen<f64>),
            SelfAdjoint(faer::linalg::solvers::SelfAdjointEigen<f64>),
        }
        let symmetric = match symmetric {
            Some(s) => s,
            None => m.is_symmetric()?,
        };
        let m = m.as_mat_ref()?;
        if m.nrows() != m.ncols() {
            return Err(crate::Error::MatrixDimensionsMismatch);
        }
        if symmetric {
            let eigen = m.self_adjoint_eigen(faer::Side::Lower)?;
            let s = eigen.S();
            let u = eigen.U();
            let mut values = Vec::with_capacity(m.nrows());
            for i in 0..m.nrows() {
                values.push(s[i]);
            }
            let mut zero = 0.0;
            let mut vectors: Vec<f64> = vec![zero; m.nrows() * m.nrows()];
            u.par_col_chunks(1).enumerate().for_each(|(i, c)| {
                let col = c.col(0);
                let mut vector = unsafe {
                    std::slice::from_raw_parts_mut(
                        vectors.as_ptr().cast_mut().add(i * m.nrows()),
                        m.nrows(),
                    )
                };
                for (j, x) in col.iter().enumerate() {
                    vector[j] = *x;
                }
            });
            Ok(Eigen::Real { values, vectors })
        } else {
            let eigen = m.eigen()?;
            let s = eigen.S();
            let u = eigen.U();
            let complex = (0..m.nrows()).into_par_iter().any(|i| {
                if s[i].im != 0.0 {
                    // is complex
                    return true;
                }
                let col = u.col(i);
                for j in 0..m.nrows() {
                    if col[j].im != 0.0 {
                        return true;
                    }
                }
                false
            });
            if complex {
                let mut values = Vec::with_capacity(m.nrows());
                for i in 0..m.nrows() {
                    values.push(s[i]);
                }
                let mut zero = c64::new(0.0, 0.0);
                let mut vectors: Vec<c64> = vec![zero; m.nrows() * m.nrows()];
                u.par_col_chunks(1).enumerate().for_each(|(i, c)| {
                    let col = c.col(0);
                    let mut vector = unsafe {
                        std::slice::from_raw_parts_mut(
                            vectors.as_ptr().cast_mut().add(i * m.nrows()),
                            m.nrows(),
                        )
                    };
                    for (j, x) in col.iter().enumerate() {
                        vector[j] = *x;
                    }
                });
                Ok(Eigen::Complex { values, vectors })
            } else {
                let mut values = Vec::with_capacity(m.nrows());
                for i in 0..m.nrows() {
                    values.push(s[i].re);
                }
                let mut zero = 0.0;
                let mut vectors: Vec<f64> = vec![zero; m.nrows() * m.nrows()];
                u.par_col_chunks(1).enumerate().for_each(|(i, c)| {
                    let col = c.col(0);
                    let mut vector = unsafe {
                        std::slice::from_raw_parts_mut(
                            vectors.as_ptr().cast_mut().add(i * m.nrows()),
                            m.nrows(),
                        )
                    };
                    for (j, x) in col.iter().enumerate() {
                        vector[j] = x.re;
                    }
                });
                Ok(Eigen::Real { values, vectors })
            }
        }
    }
}

impl Matrix {
    pub fn transform(&mut self) -> Result<&mut Self, crate::Error> {
        if let Matrix::Transform(fns, mat) = self {
            let slf = std::mem::replace(self, Matrix::Owned(OwnedMatrix::new(0, 0, vec![], None)));
            if let Matrix::Transform(fns, mat) = slf {
                let mut mat = *mat;
                for f in fns {
                    f(&mut mat)?;
                }
                *self = mat;
            }
        }
        Ok(self)
    }

    fn add_transformation(
        &mut self,
        f: impl for<'a> FnOnce(&'a mut Matrix) -> Result<&'a mut Matrix, Error> + 'static,
    ) -> &mut Self {
        match self {
            Matrix::Transform(fns, _) => fns.push(Box::new(f)),
            _ => {
                let m =
                    std::mem::replace(self, Matrix::Owned(OwnedMatrix::new(0, 0, vec![], None)));
                *self = Matrix::Transform(vec![], Box::new(m));
                self.add_transformation(f);
            },
        }
        self
    }

    pub fn t_combine_columns(&mut self, mut others: Vec<Self>) -> &mut Self {
        self.add_transformation(move |m| m.combine_columns(others.as_mut_slice()))
    }

    #[tracing::instrument(skip(self, others))]
    pub fn combine_columns(&mut self, others: &mut [Self]) -> Result<&mut Self, crate::Error> {
        if others.is_empty() {
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
            return Err(crate::Error::MatrixDimensionsMismatch);
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
                    .try_as_col_major()
                    .expect("could not get slice")
                    .as_slice();
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
            unsafe {
                std::mem::transmute::<std::vec::Vec<std::mem::MaybeUninit<f64>>, std::vec::Vec<f64>>(
                    data,
                )
            },
            colnames,
        ));
        Ok(self)
    }

    pub fn t_combine_rows(&mut self, mut others: Vec<Self>) -> &mut Self {
        self.add_transformation(move |m| m.combine_rows(others.as_mut_slice()))
    }

    #[tracing::instrument(skip(self, others))]
    pub fn combine_rows(&mut self, others: &mut [Self]) -> Result<&mut Self, crate::Error> {
        if others.is_empty() {
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
            return Err(crate::Error::ColumnNamesMismatch);
        }
        let colnames = colnames.map(|x| x.iter().map(|x| x.to_string()).collect());
        let ncols = self.ncols()?;
        let others = others
            .iter_mut()
            .map(|x| x.as_mat_ref())
            .collect::<Result<Vec<_>, _>>()?;
        if others.iter().any(|i| i.ncols() != ncols) {
            return Err(crate::Error::MatrixDimensionsMismatch);
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
                    .try_as_col_major()
                    .expect("could not get slice")
                    .as_slice();
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
            unsafe {
                std::mem::transmute::<std::vec::Vec<std::mem::MaybeUninit<f64>>, std::vec::Vec<f64>>(
                    data,
                )
            },
            colnames,
        ));
        Ok(self)
    }

    pub fn t_remove_rows(&mut self, removing: HashSet<usize>) -> &mut Self {
        self.add_transformation(move |m| m.remove_rows(&removing))
    }

    #[tracing::instrument(skip(self, removing))]
    pub fn remove_rows(&mut self, removing: &HashSet<usize>) -> Result<&mut Self, crate::Error> {
        if removing.is_empty() {
            return Ok(self);
        }
        for i in removing.iter() {
            if *i >= self.nrows()? {
                return Err(crate::Error::RowIndexOutOfBounds(*i));
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
            unsafe {
                std::mem::transmute::<std::vec::Vec<std::mem::MaybeUninit<f64>>, std::vec::Vec<f64>>(
                    data,
                )
            },
            self.colnames()?
                .map(|x| x.iter().map(|x| x.to_string()).collect()),
        ));
        Ok(self)
    }

    pub fn t_remove_columns(&mut self, removing: HashSet<usize>) -> &mut Self {
        self.add_transformation(move |m| m.remove_columns(&removing))
    }

    #[tracing::instrument(skip(self, removing))]
    pub fn remove_columns(&mut self, removing: &HashSet<usize>) -> Result<&mut Self, crate::Error> {
        if removing.is_empty() {
            return Ok(self);
        }
        let m = self.as_mat_ref()?;
        for i in removing.iter() {
            if *i >= m.ncols() {
                return Err(crate::Error::ColumnIndexOutOfBounds(*i));
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
                let src = m.get_unchecked(.., o).try_as_col_major().expect("could not get slice").as_slice();
                let dst = data.as_ptr().add(n * m.nrows()).cast::<f64>().cast_mut();
                let slice = std::slice::from_raw_parts_mut(dst, m.nrows());
                slice.copy_from_slice(src);
            });
        *self = Matrix::Owned(OwnedMatrix::new(
            m.nrows(),
            new_ncols,
            // SAFETY: The data is initialized now
            unsafe {
                std::mem::transmute::<std::vec::Vec<std::mem::MaybeUninit<f64>>, std::vec::Vec<f64>>(
                    data,
                )
            },
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

    pub fn t_remove_column_by_name(&mut self, name: &str) -> &mut Self {
        let name = name.to_string();
        self.add_transformation(move |m| m.remove_column_by_name(&name))
    }

    #[tracing::instrument(skip(self))]
    pub fn remove_column_by_name(&mut self, name: &str) -> Result<&mut Self, crate::Error> {
        let colnames = self.colnames()?;
        if colnames.is_none() {
            return Err(crate::Error::MissingColumnNames);
        }
        let exists = colnames
            .expect("colnames should be present")
            .iter()
            .position(|x| *x == name);
        if let Some(i) = exists {
            self.remove_columns(&[i].iter().copied().collect())?;
        } else {
            return Err(crate::Error::ColumnNameNotFound(name.to_string()));
        }
        Ok(self)
    }

    pub fn t_remove_columns_by_name(&mut self, names: HashSet<String>) -> &mut Self {
        self.add_transformation(move |m| m.remove_columns_by_name(&names))
    }

    #[tracing::instrument(skip(self, names))]
    pub fn remove_columns_by_name(
        &mut self,
        names: &HashSet<String>,
    ) -> Result<&mut Self, crate::Error> {
        let names = HashSet::<&str>::from_iter(names.iter().map(|x| x.as_str()));
        let colnames = self.colnames()?;
        if colnames.is_none() {
            return Err(crate::Error::MissingColumnNames);
        }
        let colnames = colnames.expect("colnames should be present");
        let removing = colnames
            .iter()
            .enumerate()
            .filter_map(|(i, x)| if names.contains(x) { Some(i) } else { None })
            .collect();
        self.remove_columns(&removing)
    }

    pub fn t_remove_column_by_name_if_exists(&mut self, name: &str) -> &mut Self {
        let name = name.to_string();
        self.add_transformation(move |m| m.remove_column_by_name_if_exists(&name))
    }

    #[tracing::instrument(skip(self))]
    pub fn remove_column_by_name_if_exists(
        &mut self,
        name: &str,
    ) -> Result<&mut Self, crate::Error> {
        match self.remove_column_by_name(name) {
            Ok(_) => Ok(self),
            Err(crate::Error::ColumnNameNotFound(_) | crate::Error::MissingColumnNames) => Ok(self),
            Err(e) => Err(e),
        }
    }

    pub fn t_transpose(&mut self) -> &mut Self {
        self.add_transformation(|m| m.transpose())
    }

    #[tracing::instrument(skip(self))]
    pub fn transpose(&mut self) -> Result<&mut Self, crate::Error> {
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
            unsafe {
                std::mem::transmute::<std::vec::Vec<std::mem::MaybeUninit<f64>>, std::vec::Vec<f64>>(
                    new_data,
                )
            },
            None,
        ));
        Ok(self)
    }

    pub fn t_sort_by_column(&mut self, by: usize) -> &mut Self {
        self.add_transformation(move |m| m.sort_by_column(by))
    }

    #[tracing::instrument(skip(self))]
    pub fn sort_by_column(&mut self, by: usize) -> Result<&mut Self, crate::Error> {
        if by >= self.ncols()? {
            return Err(crate::Error::ColumnIndexOutOfBounds(by));
        }
        let col = self.col(by)?.unwrap();
        let mut order = col.iter().copied().enumerate().collect::<Vec<_>>();
        order.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("could not compare"));
        self.sort_by_order(
            order
                .into_iter()
                .map(|(i, _)| i)
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    pub fn t_sort_by_column_name(&mut self, by: &str) -> &mut Self {
        let by = by.to_string();
        self.add_transformation(move |m| m.sort_by_column_name(&by))
    }

    #[tracing::instrument(skip(self))]
    pub fn sort_by_column_name(&mut self, by: &str) -> Result<&mut Self, crate::Error> {
        let colnames = self.colnames()?;
        if colnames.is_none() {
            return Err(crate::Error::MissingColumnNames);
        }
        let by_col_idx = colnames
            .expect("colnames should be present")
            .iter()
            .position(|x| *x == by);
        if let Some(i) = by_col_idx {
            self.sort_by_column(i)?;
            Ok(self)
        } else {
            Err(crate::Error::ColumnNameNotFound(by.to_string()))
        }
    }

    pub fn t_sort_by_order(&mut self, order: Vec<usize>) -> &mut Self {
        self.add_transformation(move |m| m.sort_by_order(&order))
    }

    #[tracing::instrument(skip(self, order))]
    pub fn sort_by_order(&mut self, order: &[usize]) -> Result<&mut Self, crate::Error> {
        if order.len() != self.nrows()? {
            return Err(crate::Error::OrderLengthMismatch(order.len()));
        }
        for i in order.iter() {
            if *i >= self.nrows()? {
                return Err(crate::Error::RowIndexOutOfBounds(*i));
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
            unsafe {
                std::mem::transmute::<std::vec::Vec<std::mem::MaybeUninit<f64>>, std::vec::Vec<f64>>(
                    data,
                )
            },
            self.colnames()?
                .map(|x| x.iter().map(|x| x.to_string()).collect()),
        ));
        Ok(self)
    }

    pub fn t_dedup_by_column(&mut self, by: usize) -> &mut Self {
        self.add_transformation(move |m| m.dedup_by_column(by))
    }

    #[tracing::instrument(skip(self))]
    pub fn dedup_by_column(&mut self, by: usize) -> Result<&mut Self, crate::Error> {
        if by >= self.ncols()? {
            return Err(crate::Error::ColumnIndexOutOfBounds(by));
        }
        let mut col = self.col(by)?.unwrap().to_vec();
        col.sort_by(|a, b| a.partial_cmp(b).expect("could not compare"));
        let mut removing = HashSet::new();
        for i in 1..col.len() {
            if col[i] == col[i - 1] {
                removing.insert(i);
            }
        }
        self
            .remove_rows(&removing)
            // by doing this we avoid nesting another error that's impossible to occur inside
            // crate::Error
            .expect("all indices should be valid");
        Ok(self)
    }

    pub fn t_dedup_by_column_name(&mut self, by: &str) -> &mut Self {
        let by = by.to_string();
        self.add_transformation(move |m| m.dedup_by_column_name(&by))
    }

    #[tracing::instrument(skip(self))]
    pub fn dedup_by_column_name(&mut self, by: &str) -> Result<&mut Self, crate::Error> {
        let colnames = self.colnames()?;
        if colnames.is_none() {
            return Err(crate::Error::MissingColumnNames);
        }
        let by_col_idx = colnames
            .expect("colnames should be present")
            .iter()
            .position(|x| *x == by);
        if let Some(i) = by_col_idx {
            self.dedup_by_column(i)?;
            Ok(self)
        } else {
            Err(crate::Error::ColumnNameNotFound(by.to_string()))
        }
    }

    pub fn t_match_to(&mut self, with: Vec<f64>, by: usize, join: Join) -> &mut Self {
        self.add_transformation(move |m| m.match_to(&with, by, join))
    }

    #[tracing::instrument(skip(self, with))]
    pub fn match_to(
        &mut self,
        with: &[f64],
        by: usize,
        join: Join,
    ) -> Result<&mut Self, crate::Error> {
        if by >= self.ncols()? {
            return Err(crate::Error::ColumnIndexOutOfBounds(by));
        }
        let mut col = self
            .col(by)?
            .unwrap()
            .iter()
            .enumerate()
            .collect::<Vec<_>>();
        col.sort_by(|a, b| a.1.partial_cmp(b.1).expect("could not compare"));
        let mut other = with.iter().enumerate().collect::<Vec<_>>();
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
                    return Err(crate::Error::NotAllRowsMatched(join));
                }
            },
            Join::Right => {
                if order.len() < other.len() {
                    return Err(crate::Error::NotAllRowsMatched(join));
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
            unsafe {
                std::mem::transmute::<std::vec::Vec<std::mem::MaybeUninit<f64>>, std::vec::Vec<f64>>(
                    data,
                )
            },
            self.colnames()?
                .map(|x| x.iter().map(|x| x.to_string()).collect()),
        ));

        // if self is sorted, then we can just go by pairs otherwise binary search
        Ok(self)
    }

    pub fn t_match_to_by_column_name(
        &mut self,
        other: Vec<f64>,
        col: &str,
        join: Join,
    ) -> &mut Self {
        let col = col.to_string();
        self.add_transformation(move |m| m.match_to_by_column_name(&other, &col, join))
    }

    #[tracing::instrument(skip(self, other))]
    pub fn match_to_by_column_name(
        &mut self,
        other: &[f64],
        col: &str,
        join: Join,
    ) -> Result<&mut Self, crate::Error> {
        let colnames = self.colnames()?;
        if colnames.is_none() {
            return Err(crate::Error::MissingColumnNames);
        }
        let by_col_idx = colnames
            .expect("colnames should be present")
            .iter()
            .position(|x| *x == col);
        if let Some(i) = by_col_idx {
            self.match_to(other, i, join)?;
            Ok(self)
        } else {
            Err(crate::Error::ColumnNameNotFound(col.to_string()))
        }
    }

    pub fn t_join(
        &mut self,
        mut other: Self,
        self_by: usize,
        other_by: usize,
        join: Join,
    ) -> &mut Self {
        self.add_transformation(move |m| m.join(&mut other, self_by, other_by, join))
    }

    #[tracing::instrument(skip(self, other))]
    pub fn join(
        &mut self,
        other: &mut Matrix,
        self_by: usize,
        other_by: usize,
        join: Join,
    ) -> Result<&mut Self, crate::Error> {
        if self_by >= self.ncols()? {
            return Err(crate::Error::ColumnIndexOutOfBounds(self_by));
        }
        if other_by >= other.ncols()? {
            return Err(crate::Error::ColumnIndexOutOfBounds(other_by));
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
                    return Err(crate::Error::NotAllRowsMatched(join));
                }
            },
            Join::Right => {
                if order.len() < other_col.len() {
                    return Err(crate::Error::NotAllRowsMatched(join));
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
            unsafe {
                std::mem::transmute::<std::vec::Vec<std::mem::MaybeUninit<f64>>, std::vec::Vec<f64>>(
                    data,
                )
            },
            self_colnames,
        ));

        Ok(self)
    }

    pub fn t_join_by_column_name(&mut self, mut other: Matrix, by: &str, join: Join) -> &mut Self {
        let by = by.to_string();
        self.add_transformation(move |m| m.join_by_column_name(&mut other, &by, join))
    }

    #[tracing::instrument(skip(self, other))]
    pub fn join_by_column_name(
        &mut self,
        other: &mut Matrix,
        by: &str,
        join: Join,
    ) -> Result<&mut Self, crate::Error> {
        let self_colnames = self.colnames()?;
        let other_colnames = other.colnames()?;
        if self_colnames.is_none() || other_colnames.is_none() {
            if self_colnames.is_none() {
                debug!("self colnames are missing");
            } else {
                debug!("other colnames are missing");
            }
            return Err(crate::Error::MissingColumnNames);
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
            Err(crate::Error::ColumnNameNotFound(by.to_string()))
        }
    }

    pub fn t_standardize_columns(&mut self) -> &mut Self {
        self.add_transformation(|m| m.standardize_columns())
    }

    #[tracing::instrument(skip(self))]
    pub fn standardize_columns(&mut self) -> Result<&mut Self, crate::Error> {
        debug!("Standardizing matrix");
        self.as_mat_mut()?
            .par_col_chunks_mut(1)
            .for_each(|c| standardize_column(c.col_mut(0)));
        debug!("Standardized matrix");
        Ok(self)
    }

    pub fn t_standardize_rows(&mut self) -> &mut Self {
        self.add_transformation(|m| m.standardize_rows())
    }

    #[tracing::instrument(skip(self))]
    pub fn standardize_rows(&mut self) -> Result<&mut Self, crate::Error> {
        debug!("Standardizing matrix");
        self.as_mat_mut()?
            .par_row_chunks_mut(1)
            .for_each(|r| standardize_row(r.row_mut(0)));
        debug!("Standardized matrix");
        Ok(self)
    }

    pub fn t_remove_nan_rows(&mut self) -> &mut Self {
        self.add_transformation(|m| m.remove_nan_rows())
    }

    #[tracing::instrument(skip(self))]
    pub fn remove_nan_rows(&mut self) -> Result<&mut Self, crate::Error> {
        let removing = self
            .as_mat_ref()?
            .par_row_chunks(1)
            .enumerate()
            .filter(|(_, row)| !row.is_all_finite())
            .map(|(i, _)| i)
            .collect::<HashSet<_>>();
        debug!("Removed {} rows with NaN values", removing.len());
        self.remove_rows(&removing)
    }

    pub fn t_remove_nan_columns(&mut self) -> &mut Self {
        self.add_transformation(|m| m.remove_nan_columns())
    }

    #[tracing::instrument(skip(self))]
    pub fn remove_nan_columns(&mut self) -> Result<&mut Self, crate::Error> {
        let removing = self
            .as_mat_ref()?
            .par_col_chunks(1)
            .enumerate()
            .filter(|(_, col)| !col.is_all_finite())
            .map(|(i, _)| i)
            .collect::<HashSet<_>>();
        debug!("Removed {} columns with NaN values", removing.len());
        self.remove_columns(&removing)
    }

    pub fn t_nan_to_value(&mut self, val: f64) -> &mut Self {
        self.add_transformation(move |m| m.nan_to_value(val))
    }

    #[tracing::instrument(skip(self))]
    pub fn nan_to_value(&mut self, val: f64) -> Result<&mut Self, crate::Error> {
        self.as_mat_mut()?.par_col_chunks_mut(1).for_each(|c| {
            c.col_mut(0).iter_mut().for_each(|x| {
                if !x.is_finite() {
                    *x = val;
                }
            })
        });
        Ok(self)
    }

    pub fn t_nan_to_column_mean(&mut self) -> &mut Self {
        self.add_transformation(move |m| m.nan_to_column_mean())
    }

    #[tracing::instrument(skip(self))]
    pub fn nan_to_column_mean(&mut self) -> Result<&mut Self, crate::Error> {
        self.as_mat_mut()?.par_col_chunks_mut(1).for_each(|c| {
            let col = c.col_mut(0);
            let m = mean::mean(
                col.as_ref()
                    .try_as_col_major()
                    .expect("could not get slice")
                    .as_slice(),
            );
            col.iter_mut().for_each(|x| {
                if !x.is_finite() {
                    *x = m;
                }
            })
        });
        Ok(self)
    }

    pub fn t_nan_to_row_mean(&mut self) -> &mut Self {
        self.add_transformation(move |m| m.nan_to_row_mean())
    }

    #[tracing::instrument(skip(self))]
    pub fn nan_to_row_mean(&mut self) -> Result<&mut Self, crate::Error> {
        self.as_mat_mut()?.par_row_chunks_mut(1).for_each(|r| {
            let row = r.row_mut(0);
            let mut m = 0.0;
            faer::stats::col_mean(
                ColMut::from_mut(&mut m),
                row.as_ref().as_mat(),
                faer::stats::NanHandling::Ignore,
            );
            row.iter_mut().for_each(|x| {
                if !x.is_finite() {
                    *x = m;
                }
            })
        });
        Ok(self)
    }

    pub fn t_min_column_sum(&mut self, sum: f64) -> &mut Self {
        self.add_transformation(move |m| m.min_column_sum(sum))
    }

    #[tracing::instrument(skip(self))]
    pub fn min_column_sum(&mut self, sum: f64) -> Result<&mut Self, crate::Error> {
        let removing = self
            .as_mat_mut()?
            .par_col_chunks_mut(1)
            .enumerate()
            .filter(|(_, c)| c.sum() < sum)
            .map(|(i, _)| i)
            .collect::<HashSet<_>>();
        debug!("Removed {} columns with sum < {}", removing.len(), sum);
        self.remove_columns(&removing)
    }

    pub fn t_max_column_sum(&mut self, sum: f64) -> &mut Self {
        self.add_transformation(move |m| m.max_column_sum(sum))
    }

    #[tracing::instrument(skip(self))]
    pub fn max_column_sum(&mut self, sum: f64) -> Result<&mut Self, crate::Error> {
        let removing = self
            .as_mat_mut()?
            .par_col_chunks_mut(1)
            .enumerate()
            .filter(|(_, c)| c.sum() > sum)
            .map(|(i, _)| i)
            .collect();
        self.remove_columns(&removing)
    }

    pub fn t_min_row_sum(&mut self, sum: f64) -> &mut Self {
        self.add_transformation(move |m| m.min_row_sum(sum))
    }

    #[tracing::instrument(skip(self))]
    pub fn min_row_sum(&mut self, sum: f64) -> Result<&mut Self, crate::Error> {
        let removing = self
            .as_mat_mut()?
            .par_row_chunks_mut(1)
            .enumerate()
            .filter(|(_, r)| r.sum() < sum)
            .map(|(i, _)| i)
            .collect();
        self.remove_rows(&removing)
    }

    pub fn t_max_row_sum(&mut self, sum: f64) -> &mut Self {
        self.add_transformation(move |m| m.max_row_sum(sum))
    }

    #[tracing::instrument(skip(self))]
    pub fn max_row_sum(&mut self, sum: f64) -> Result<&mut Self, crate::Error> {
        let removing = self
            .as_mat_mut()?
            .par_row_chunks_mut(1)
            .enumerate()
            .filter(|(_, r)| r.sum() > sum)
            .map(|(i, _)| i)
            .collect();
        self.remove_rows(&removing)
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn t_rename_column(&mut self, old: &str, new: &str) -> &mut Self {
        let old = old.to_string();
        let new = new.to_string();
        self.add_transformation(move |m| m.rename_column(&old, &new))
    }

    #[tracing::instrument(skip(self))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn rename_column(&mut self, old: &str, new: &str) -> Result<&mut Self, crate::Error> {
        let colnames = self.colnames()?;
        if colnames.is_none() {
            return Err(crate::Error::MissingColumnNames);
        }
        let idx = colnames
            .as_ref()
            .expect("colnames should be present")
            .iter()
            .position(|x| *x == old);
        if let Some(i) = idx {
            let mut colnames = colnames
                .expect("colnames should be present")
                .into_iter()
                .map(|x| {
                    if x == old {
                        new.to_string()
                    } else {
                        x.to_string()
                    }
                })
                .collect::<Vec<_>>();
            self.into_owned()?;
            self.as_owned_mut()?.colnames = Some(colnames);
            Ok(self)
        } else {
            Err(crate::Error::ColumnNameNotFound(old.to_string()))
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn t_rename_column_if_exists(&mut self, old: &str, new: &str) -> &mut Self {
        let old = old.to_string();
        let new = new.to_string();
        self.add_transformation(move |m| m.rename_column_if_exists(&old, &new))
    }

    #[tracing::instrument(skip(self))]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn rename_column_if_exists(
        &mut self,
        old: &str,
        new: &str,
    ) -> Result<&mut Self, crate::Error> {
        match self.rename_column(old, new) {
            Ok(_) => Ok(self),
            Err(crate::Error::ColumnNameNotFound(_) | crate::Error::MissingColumnNames) => Ok(self),
            Err(e) => Err(e),
        }
    }

    pub fn t_remove_duplicate_columns(&mut self) -> &mut Self {
        self.add_transformation(move |m| m.remove_duplicate_columns())
    }

    #[tracing::instrument(skip(self))]
    pub fn remove_duplicate_columns(&mut self) -> Result<&mut Self, crate::Error> {
        let ncols = self.ncols()?;
        let cols = (0..ncols)
            .into_par_iter()
            .map(|x| self.col_loaded(x))
            .collect::<Vec<_>>();
        let cols = (0..ncols)
            .into_par_iter()
            .flat_map(|i| ((i + 1)..ncols).into_par_iter().map(move |j| (i, j)))
            .filter_map(|(i, j)| if cols[i] == cols[j] { Some(j) } else { None })
            .collect::<HashSet<_>>();
        self.remove_columns(&cols)
    }

    pub fn t_remove_identical_columns(&mut self) -> &mut Self {
        self.add_transformation(move |m| m.remove_identical_columns())
    }

    #[tracing::instrument(skip(self))]
    pub fn remove_identical_columns(&mut self) -> Result<&mut Self, crate::Error> {
        let ncols = self.ncols()?;
        let cols = (0..ncols)
            .into_par_iter()
            .map(|x| self.col_loaded(x).unwrap())
            .collect::<Vec<_>>();
        let cols = (0..ncols)
            .into_par_iter()
            .filter_map(|i| {
                let col = cols[i];
                let first = col[0];
                for i in col.iter().skip(1) {
                    if *i != first {
                        return None;
                    }
                }
                Some(i)
            })
            .collect::<HashSet<_>>();
        self.remove_columns(&cols)
    }

    pub fn t_min_non_nan(&mut self, val: usize) -> &mut Self {
        self.add_transformation(move |m| m.min_non_nan(val))
    }

    #[tracing::instrument(skip(self))]
    pub fn min_non_nan(&mut self, val: usize) -> Result<&mut Self, crate::Error> {
        let removing = self
            .as_mat_mut()?
            .par_col_chunks_mut(1)
            .enumerate()
            .filter(|(_, c)| c.as_ref().col(0).iter().filter(|x| x.is_finite()).count() < val)
            .map(|(i, _)| i)
            .collect();
        self.remove_columns(&removing)
    }

    pub fn t_max_non_nan(&mut self, val: usize) -> &mut Self {
        self.add_transformation(move |m| m.max_non_nan(val))
    }

    #[tracing::instrument(skip(self))]
    pub fn max_non_nan(&mut self, val: usize) -> Result<&mut Self, crate::Error> {
        let removing = self
            .as_mat_mut()?
            .par_col_chunks_mut(1)
            .enumerate()
            .filter(|(_, c)| c.as_ref().col(0).iter().filter(|x| x.is_finite()).count() > val)
            .map(|(i, _)| i)
            .collect();
        self.remove_columns(&removing)
    }

    pub fn t_subset_columns(&mut self, cols: HashSet<usize>) -> &mut Self {
        self.add_transformation(move |m| m.subset_columns(&cols))
    }

    #[tracing::instrument(skip(self))]
    pub fn subset_columns(&mut self, cols: &HashSet<usize>) -> Result<&mut Self, crate::Error> {
        let ncols = self.ncols()?;
        let removing = (0..ncols)
            .into_par_iter()
            .filter(|x| !cols.contains(x))
            .collect::<HashSet<_>>();
        self.remove_columns(&removing)
    }

    pub fn t_subset_columns_by_name(&mut self, cols: HashSet<String>) -> &mut Self {
        self.add_transformation(move |m| m.subset_columns_by_name(&cols))
    }

    #[tracing::instrument(skip(self))]
    pub fn subset_columns_by_name(
        &mut self,
        cols: &HashSet<String>,
    ) -> Result<&mut Self, crate::Error> {
        let colnames = self.colnames()?;
        if colnames.is_none() {
            return Err(crate::Error::MissingColumnNames);
        }
        let colnames = colnames
            .expect("colnames should be present")
            .into_iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>();
        let cols = colnames
            .iter()
            .enumerate()
            .filter_map(|(i, x)| if cols.contains(x) { Some(i) } else { None })
            .collect::<HashSet<_>>();
        self.subset_columns(&cols)
    }

    pub fn t_rename_columns_with_regex(&mut self, regex: &str, replacement: &str) -> &mut Self {
        let regex = regex.to_string();
        let replacement = replacement.to_string();
        self.add_transformation(move |m| m.rename_columns_with_regex(&regex, &replacement))
    }

    #[tracing::instrument(skip(self))]
    pub fn rename_columns_with_regex(
        &mut self,
        regex: &str,
        replacement: &str,
    ) -> Result<&mut Self, crate::Error> {
        let colnames = self.colnames()?;
        if colnames.is_none() {
            return Err(crate::Error::MissingColumnNames);
        }
        let re = Regex::new(regex)?;
        let colnames = colnames
            .expect("colnames should be present")
            .iter()
            .map(|x| re.replace_all(x, replacement).to_string())
            .collect::<Vec<_>>();
        self.set_colnames(colnames)?;
        Ok(self)
    }

    pub fn t_scale_columns(&mut self, scale: Vec<f64>) -> &mut Self {
        self.add_transformation(move |m| m.scale_columns(&scale))
    }

    #[tracing::instrument(skip(self))]
    pub fn scale_columns(&mut self, scale: &[f64]) -> Result<&mut Self, crate::Error> {
        let ncols = self.ncols()?;
        if scale.len() != 1 && scale.len() != ncols {
            return Err(crate::Error::InvalidScaleLength(scale.len()));
        }
        if scale.len() == 1 {
            crate::scale::scale_scalar_in_place(self.as_mat_mut()?, scale[0]);
        } else {
            crate::scale::scale_vector_in_place(self.as_mat_mut()?, scale);
        }

        Ok(self)
    }
}

impl Matrix {
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn nrows(&mut self) -> Result<usize, crate::Error> {
        self.as_mat_ref().map(|x| x.nrows())
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn nrows_loaded(&self) -> usize {
        self.as_mat_ref_loaded().nrows()
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn ncols(&mut self) -> Result<usize, crate::Error> {
        self.as_mat_ref().map(|x| x.ncols())
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn ncols_loaded(&self) -> usize {
        self.as_mat_ref_loaded().ncols()
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn data(&mut self) -> Result<&[f64], crate::Error> {
        self.as_owned_ref().map(|x| x.data.as_slice())
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn as_mut_slice(&mut self) -> Result<&mut [f64], crate::Error> {
        match self {
            Matrix::Owned(m) => Ok(&mut m.data),
            #[cfg(feature = "r")]
            Matrix::R(m) => Ok(unsafe {
                std::slice::from_raw_parts_mut(m.data().as_ptr().cast_mut(), m.data().len())
            }),
            ref m => self.into_owned()?.as_mut_slice(),
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn col(&mut self, col: usize) -> Result<Option<&[f64]>, crate::Error> {
        if col >= self.ncols()? {
            return Ok(None);
        }
        self.as_mat_ref().map(|x| {
            Some(unsafe {
                x.get_unchecked(.., col)
                    .try_as_col_major()
                    .expect("could not get slice")
                    .as_slice()
            })
        })
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn col_loaded(&self, col: usize) -> Option<&[f64]> {
        if col >= self.ncols_loaded() {
            return None;
        }
        Some(unsafe {
            self.as_mat_ref_loaded()
                .get_unchecked(.., col)
                .try_as_col_major()
                .expect("could not get slice")
                .as_slice()
        })
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn get(&mut self, row: usize, col: usize) -> Result<Option<f64>, crate::Error> {
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

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn get_loaded(&self, row: usize, col: usize) -> Option<f64> {
        let nrows = self.nrows_loaded();
        let ncols = self.ncols_loaded();
        if row >= nrows || col > ncols {
            None
        } else {
            Some(unsafe { *self.as_mat_ref_loaded().get_unchecked(row, col) })
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn column_index(&mut self, name: &str) -> Result<usize, crate::Error> {
        let colnames = self.colnames()?;
        if colnames.is_none() {
            return Err(crate::Error::MissingColumnNames);
        }
        let idx = colnames
            .expect("colnames should be present")
            .iter()
            .position(|x| *x == name);
        match idx {
            Some(i) => Ok(i),
            None => Err(crate::Error::ColumnNameNotFound(name.to_string())),
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn has_column(&mut self, name: &str) -> Result<bool, crate::Error> {
        self.column_index(name).map(|_| true).or_else(|e| match e {
            crate::Error::ColumnNameNotFound(_) => Ok(false),
            e => Err(e),
        })
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn has_column_loaded(&self, name: &str) -> bool {
        self.colnames_loaded()
            .map(|x| x.contains(&name))
            .unwrap_or(false)
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn column_by_name(&mut self, name: &str) -> Result<Option<&[f64]>, crate::Error> {
        let col = self.column_index(name)?;
        self.col(col)
    }
}

impl FromStr for Matrix {
    type Err = crate::Error;

    #[cfg_attr(coverage_nightly, coverage(off))]
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

impl PartialEq for OwnedMatrix {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn eq(&self, other: &Self) -> bool {
        self.nrows == other.nrows
            && self.ncols == other.ncols
            && self.colnames == other.colnames
            && self.data.len() == other.data.len()
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl OwnedMatrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>, colnames: Option<Vec<String>>) -> Self {
        assert!(rows * cols == data.len());
        if let Some(colnames) = &colnames {
            assert_eq!(cols, colnames.len());
        }
        Self {
            nrows: rows,
            ncols: cols,
            data,
            colnames,
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    #[doc(hidden)]
    pub fn into_data(self) -> Vec<f64> {
        self.data
    }
}

pub trait IntoMatrix {
    fn into_matrix(self) -> Matrix;
}

#[cfg(feature = "r")]
impl IntoMatrix for RMatrix<f64> {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn into_matrix(self) -> Matrix {
        Matrix::R(self)
    }
}

#[cfg(feature = "r")]
impl IntoMatrix for RMatrix<i32> {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn into_matrix(self) -> Matrix {
        Matrix::from_robj(self.into_robj()).unwrap()
    }
}

impl IntoMatrix for OwnedMatrix {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn into_matrix(self) -> Matrix {
        Matrix::Owned(self)
    }
}

impl IntoMatrix for File {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn into_matrix(self) -> Matrix {
        Matrix::File(self)
    }
}

impl IntoMatrix for MatRef<'_, f64> {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn into_matrix(self) -> Matrix {
        Matrix::from_mat_ref(self)
    }
}

impl IntoMatrix for MatMut<'_, f64> {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn into_matrix(self) -> Matrix {
        Matrix::from_mat_ref(self.as_ref())
    }
}

impl IntoMatrix for Mat<f64> {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn into_matrix(self) -> Matrix {
        Matrix::from_mat_ref(self.as_ref())
    }
}

pub trait TryIntoMatrix {
    type Err;

    fn try_into_matrix(self) -> Result<Matrix, Self::Err>;
}

#[cfg(feature = "r")]
impl TryIntoMatrix for Robj {
    type Err = crate::Error;

    #[cfg_attr(coverage_nightly, coverage(off))]
    fn try_into_matrix(self) -> Result<Matrix, Self::Err> {
        Matrix::from_robj(self)
    }
}

impl TryIntoMatrix for &str {
    type Err = crate::Error;

    #[cfg_attr(coverage_nightly, coverage(off))]
    fn try_into_matrix(self) -> Result<Matrix, Self::Err> {
        Ok(Matrix::File(self.parse()?))
    }
}

impl<T> TryIntoMatrix for T
where
    T: IntoMatrix,
{
    type Err = ();

    #[cfg_attr(coverage_nightly, coverage(off))]
    fn try_into_matrix(self) -> Result<Matrix, Self::Err> {
        Ok(self.into_matrix())
    }
}

impl<T> From<T> for Matrix
where
    T: IntoMatrix,
{
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn from(t: T) -> Self {
        t.into_matrix()
    }
}

#[cfg(test)]
mod tests {
    use faer::traits::pulp::num_complex::Complex;
    use test_log::test;

    use super::*;

    macro_rules! assert_float_eq {
        ($a:expr, $b:expr, $tol:expr) => {
            assert!(($a - $b).abs() < $tol, "{:.22} != {:.22}", $a, $b);
        };
    }

    macro_rules! float_eq {
        ($a:expr, $b:expr) => {
            assert_float_eq!($a, $b, 1e-12);
        };
    }

    macro_rules! rough_eq {
        ($a:expr, $b:expr) => {
            assert_float_eq!($a, $b, 1e-3);
        };
    }

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
        let m = m1.t_combine_columns(vec![m2, m3]);
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
        assert!(matches!(res, Error::MatrixDimensionsMismatch));
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
        let m = m1.t_combine_columns(vec![m2]);
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
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m3 = OwnedMatrix::new(
            3,
            2,
            vec![13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m1.t_combine_rows(vec![m2, m3]);
        assert_eq!(
            m.data().unwrap(),
            &[
                1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 13.0, 14.0, 15.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0,
                16.0, 17.0, 18.0
            ],
        );
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
        assert_eq!(m.nrows().unwrap(), 9);
        assert_eq!(m.ncols().unwrap(), 2);
    }

    #[test]
    fn test_combine_rows_dimensions_mismatch() {
        let mut m1 = OwnedMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], None).into_matrix();
        let m2 = OwnedMatrix::new(2, 2, vec![19.0, 20.0, 21.0, 22.0], None).into_matrix();
        let res = m1.combine_rows(&mut [m2]).unwrap_err();
        assert!(matches!(res, Error::MatrixDimensionsMismatch));
    }

    #[test]
    fn test_combine_rows_column_names_mismatch() {
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
            vec![19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            Some(vec!["c".to_string(), "d".to_string()]),
        )
        .into_matrix();
        let m = m1.combine_rows(&mut [m2]).unwrap_err();
        assert!(matches!(m, Error::ColumnNamesMismatch));
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
        let m = m.t_remove_rows(removing);
        assert_eq!(m.data().unwrap(), &[1.0, 3.0, 4.0, 6.0]);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
        assert_eq!(m.nrows().unwrap(), 2);
        assert_eq!(m.ncols().unwrap(), 2);
    }

    #[test]
    fn test_remove_rows_empty() {
        let mut m = OwnedMatrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], None).into_matrix();
        let removing = HashSet::new();
        let m = m.t_remove_rows(removing);
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(m.colnames().unwrap().is_none());
        assert_eq!(m.nrows().unwrap(), 3);
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
        assert!(matches!(m, Error::RowIndexOutOfBounds(3)));
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
        let m = m.t_remove_columns(removing);
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(m.colnames().unwrap().unwrap(), &["a".to_string()]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 1);
    }

    #[test]
    fn test_remove_columns_empty() {
        let mut m = OwnedMatrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], None).into_matrix();
        let removing = HashSet::new();
        let m = m.t_remove_columns(removing);
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(m.colnames().unwrap().is_none());
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
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
        assert!(matches!(m, Error::ColumnIndexOutOfBounds(2)));
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
        let m = m.t_remove_column_by_name("a");
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
        assert!(matches!(m, Error::ColumnNameNotFound(_)));
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
        let m = m.t_remove_column_by_name_if_exists("a");
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
        let m = m.t_remove_column_by_name_if_exists("c");
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
    }

    #[test]
    fn test_remove_columns_by_name_success() {
        let mut m = OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m
            .t_remove_columns_by_name(HashSet::from_iter(["a", "c"].iter().map(|x| x.to_string())));
        assert_eq!(m.data().unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(m.colnames().unwrap().unwrap(), &["b".to_string()]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 1);
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
        let m = m.t_transpose();
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
        let m = m.t_sort_by_column(0);
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
        assert!(matches!(m, Error::ColumnIndexOutOfBounds(2)));
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
        let m = m.t_sort_by_column_name("a");
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
        assert!(matches!(m, Error::MissingColumnNames));
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
        assert!(matches!(m, Error::ColumnNameNotFound(_)));
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
        let m = m.t_sort_by_order(vec![2, 0, 1]);
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
        assert!(matches!(m, Error::RowIndexOutOfBounds(3)));
    }

    #[test]
    fn test_sort_by_order_length_mismatch() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.sort_by_order(&[2, 0]).unwrap_err();
        assert!(matches!(m, Error::OrderLengthMismatch(2)));
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
        let m = m.t_dedup_by_column(0);
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
        assert!(matches!(m, Error::ColumnIndexOutOfBounds(2)));
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
        let m = m.t_dedup_by_column_name("a");
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
        assert!(matches!(m, Error::MissingColumnNames));
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
        assert!(matches!(m, Error::ColumnNameNotFound(_)));
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
        let other = vec![5.0, 8.0, 1.0, 2.0, 7.0];
        let m = m1.t_match_to(other, 0, Join::Inner);
        assert_eq!(m.data().unwrap(), &[5.0, 1.0, 2.0, 5.0, 1.0, 2.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_match_to_success_not_sorted() {
        let mut m1 = OwnedMatrix::new(
            5,
            2,
            vec![5.0, 2.0, 1.0, 4.0, 3.0, 5.0, 2.0, 1.0, 4.0, 3.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let other = vec![5.0, 1.0, 6.0, 2.0, 3.0, 4.0, 5.0, 7.0];
        let m = m1.t_match_to(other, 0, Join::Inner);
        assert_eq!(
            m.data().unwrap(),
            &[5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(m.nrows().unwrap(), 5);
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
        let other = vec![5.0, 1.0, 6.0, 2.0, 3.0, 4.0, 5.0, 7.0];
        let m = m1.t_match_to(other, 0, Join::Left);
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
        let other = vec![5.0, 1.0, 2.0];
        let m = m1.t_match_to(other, 0, Join::Right);
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
        let other = vec![];
        let m = m1.t_match_to(other, 0, Join::Inner);
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
        assert!(matches!(m, Error::ColumnIndexOutOfBounds(2)));
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
        assert!(matches!(m, Error::NotAllRowsMatched(Join::Left)));
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
        assert!(matches!(m, Error::NotAllRowsMatched(Join::Right)));
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
        let other = vec![5.0, 8.0, 1.0, 2.0, 7.0];
        let m = m1.t_match_to_by_column_name(other, "a", Join::Inner);
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
        let other = vec![5.0, 1.0, 6.0, 2.0, 3.0, 4.0, 5.0, 7.0];
        let m = m1.t_match_to_by_column_name(other, "a", Join::Left);
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
        let other = vec![5.0, 1.0, 2.0];
        let m = m1.t_match_to_by_column_name(other, "a", Join::Right);
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
        let other = vec![];
        let m = m1.t_match_to_by_column_name(other, "a", Join::Inner);
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
        assert!(matches!(m, Error::ColumnNameNotFound(_)));
    }

    #[test]
    fn test_match_to_by_column_name_no_colnames() {
        let mut m1 = OwnedMatrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], None).into_matrix();
        let other = [5.0, 8.0, 1.0, 2.0, 7.0];
        let m = m1
            .match_to_by_column_name(&other, "a", Join::Inner)
            .unwrap_err();
        assert!(matches!(m, Error::MissingColumnNames));
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
        let m = m1.t_join(m2, 0, 0, Join::Inner);
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
        let m = m1.t_join(m2, 0, 0, Join::Left);
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
        let m = m1.t_join(m2, 0, 0, Join::Right);
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
        assert!(matches!(m, Error::ColumnIndexOutOfBounds(2)));
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
        assert!(matches!(m, Error::NotAllRowsMatched(Join::Left)));
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
        assert!(matches!(m, Error::NotAllRowsMatched(Join::Right)));
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
        let m = m1.t_join_by_column_name(m2, "a", Join::Inner);
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
        let m = m1.t_join_by_column_name(m2, "a", Join::Left);
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
        let m = m1.t_join_by_column_name(m2, "a", Join::Right);
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
    fn test_standardize_columns() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.t_standardize_columns();
        assert_eq!(m.data().unwrap(), &[-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_standardize_rows() {
        let mut m = OwnedMatrix::new(
            2,
            3,
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m.t_standardize_rows();
        assert_eq!(m.data().unwrap(), &[-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]);
        assert_eq!(m.nrows().unwrap(), 2);
        assert_eq!(m.ncols().unwrap(), 3);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_remove_nan_rows() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.t_remove_nan_rows();
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 4.0, 5.0]);
        assert_eq!(m.nrows().unwrap(), 2);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_remove_nan_columns() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.t_remove_nan_columns();
        assert_eq!(m.data().unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 1);
        assert_eq!(m.colnames().unwrap().unwrap(), &["b".to_string()]);
    }

    #[test]
    fn test_nan_to_value() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.t_nan_to_value(0.0);
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 0.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_nan_to_column_mean() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, f64::NAN, 4.0, f64::NAN, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.t_nan_to_column_mean();
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 1.5, 4.0, 5.0, 6.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_nan_to_column_mean_all_nan() {
        let mut m = OwnedMatrix::new(3, 2, vec![f64::NAN; 6], None).into_matrix();
        let m = m.t_nan_to_column_mean();
        assert_eq!(m.data().unwrap(), &[0.0; 6]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert!(m.colnames().unwrap().is_none());
    }

    #[test]
    fn test_nan_to_row_mean() {
        let mut m = OwnedMatrix::new(
            3,
            4,
            vec![
                1.0,
                2.0,
                f64::NAN,
                4.0,
                5.0,
                6.0,
                7.0,
                f64::NAN,
                9.0,
                11.0,
                11.0,
                12.0,
            ],
            Some(vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ]),
        )
        .into_matrix();
        let m = m.t_nan_to_row_mean();
        assert_eq!(
            m.data().unwrap(),
            &[1.0, 2.0, 9.0, 4.0, 5.0, 6.0, 7.0, 6.0, 9.0, 11.0, 11.0, 12.0]
        );
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 4);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &[
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        );
    }

    #[test]
    fn test_min_column_sum() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.t_min_column_sum(7.0);
        assert_eq!(m.data().unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 1);
        assert_eq!(m.colnames().unwrap().unwrap(), &["b".to_string()]);
    }

    #[test]
    fn test_max_column_sum() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.t_max_column_sum(7.0);
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 1);
        assert_eq!(m.colnames().unwrap().unwrap(), &["a".to_string()]);
    }

    #[test]
    fn test_min_row_sum() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.t_min_row_sum(7.0);
        assert_eq!(m.data().unwrap(), &[2.0, 3.0, 5.0, 6.0]);
        assert_eq!(m.nrows().unwrap(), 2);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_max_row_sum() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.t_max_row_sum(7.0);
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 4.0, 5.0]);
        assert_eq!(m.nrows().unwrap(), 2);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_remove_duplicate_columns() {
        let mut m = OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 1.0, 4.0, 5.0, 6.0, 1.0, 2.0, 1.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m.t_remove_duplicate_columns();
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 1.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn test_remove_identical_columns() {
        let mut m = OwnedMatrix::new(
            3,
            3,
            vec![1.0, 1.0, 1.0, 4.0, 5.0, 6.0, 1.0, 2.0, 1.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m.t_remove_identical_columns();
        assert_eq!(m.data().unwrap(), &[4.0, 5.0, 6.0, 1.0, 2.0, 1.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_min_non_nan() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, f64::NAN, f64::NAN, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.t_min_non_nan(2);
        assert_eq!(m.data().unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 1);
        assert_eq!(m.colnames().unwrap().unwrap(), &["b".to_string()]);
    }

    #[test]
    fn test_max_non_nan() {
        let mut m = OwnedMatrix::new(
            3,
            2,
            vec![1.0, f64::NAN, f64::NAN, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        )
        .into_matrix();
        let m = m.t_max_non_nan(2);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 1);
        assert_eq!(m.colnames().unwrap().unwrap(), &["a".to_string()]);
    }

    #[test]
    fn test_subset_columns() {
        let mut m = OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m.t_subset_columns([0, 2].into_iter().collect::<HashSet<_>>());
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_subset_columns_by_name() {
        let mut m = OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m.t_subset_columns_by_name(["a", "c", "d"].iter().map(|s| s.to_string()).collect());
        assert_eq!(m.data().unwrap(), &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 2);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_rename_columns_with_regex() {
        let mut m = OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m.t_rename_columns_with_regex("a", "x");
        assert_eq!(
            m.data().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 3);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["x".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_eigen_symmetric_real() {
        let mut m = OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 3.0, 6.0, 9.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let Eigen::Real { values, vectors } = m.eigen(None).unwrap() else {
            panic!("Expected real decomposition");
        };
        assert_eq!(values.len(), 3);
        float_eq!(values[0], -4.3296658397194226e-16);
        float_eq!(values[1], 0.6992647456322797);
        float_eq!(values[2], 14.300735254367696);
        assert_eq!(vectors.len(), 9);
        let expected = [
            0.9486832980505138,
            -1.3877787807814457e-15,
            -0.3162277660168371,
            0.17781910596911185,
            -0.8269242138935418,
            0.5334573179073411,
            0.26149639682478465,
            0.5623133863572407,
            0.7844891904743533,
        ];
        for (i, &v) in vectors.iter().enumerate() {
            float_eq!(v, expected[i]);
        }
    }

    #[test]
    fn test_eigen_not_symmetric_real() {
        let mut m = OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let Eigen::Real { values, vectors } = m.eigen(None).unwrap() else {
            panic!("Expected real decomposition");
        };
        assert_eq!(values.len(), 3);
        float_eq!(values[0], 16.116843969807025);
        float_eq!(values[1], -1.116843969807056);
        float_eq!(values[2], 0.0);
        assert_eq!(vectors.len(), 9);
        let expected = [
            -0.46454727338767027,
            -0.5707955312285774,
            -0.6770437890694855,
            -0.9178859873651294,
            -0.24901002745731335,
            0.4198659324505014,
            0.4082482904638624,
            -0.8164965809277261,
            0.40824829046386313,
        ];
        for (i, &v) in vectors.iter().enumerate() {
            float_eq!(v, expected[i]);
        }
    }

    #[test]
    fn test_is_symmetric() {
        let mut m = OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 3.0, 6.0, 9.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        )
        .into_matrix();
        assert!(m.is_symmetric().unwrap());
    }

    #[test]
    fn test_is_not_symmetric() {
        let mut m = OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        )
        .into_matrix();
        assert!(!m.is_symmetric().unwrap());
    }

    #[test]
    fn test_scale_columns_scalar() {
        let mut m = OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m.t_scale_columns(vec![2.0]);
        assert_eq!(
            m.data().unwrap(),
            &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
        );
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 3);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_scale_columns_vector() {
        let mut m = OwnedMatrix::new(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        )
        .into_matrix();
        let m = m.t_scale_columns(vec![2.0, 3.0, 4.0]);
        assert_eq!(
            m.data().unwrap(),
            &[2.0, 4.0, 6.0, 12.0, 15.0, 18.0, 28.0, 32.0, 36.0]
        );
        assert_eq!(m.nrows().unwrap(), 3);
        assert_eq!(m.ncols().unwrap(), 3);
        assert_eq!(
            m.colnames().unwrap().unwrap(),
            &["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }
}

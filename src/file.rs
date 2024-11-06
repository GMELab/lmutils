use core::str;
use std::{
    io::{Read, Write},
    mem::MaybeUninit,
    num::ParseFloatError,
    os::fd::AsRawFd,
    path::PathBuf,
    str::FromStr,
};

use cfg_if::cfg_if;
use extendr_api::{io::Save, pairlist, Pairlist};
use libc::WSTOPSIG;
use rayon::{
    iter::IntoParallelIterator,
    prelude::{IndexedParallelIterator, ParallelIterator},
    str::ParallelString,
};
use tracing::info;

use crate::{IntoMatrix, Matrix, OwnedMatrix};

#[derive(Clone, Debug, PartialEq)]
pub struct File {
    path:      PathBuf,
    file_type: FileType,
    gz:        bool,
}

impl File {
    pub fn new(path: impl Into<PathBuf>, file_type: FileType, gz: bool) -> Self {
        Self {
            path: path.into(),
            file_type,
            gz,
        }
    }

    #[inline(always)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    #[inline(always)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn file_type(&self) -> FileType {
        self.file_type
    }

    #[inline(always)]
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn gz(&self) -> bool {
        self.gz
    }

    pub fn read(&self) -> Result<Matrix, crate::Error> {
        #[cfg(unix)]
        if self.file_type == FileType::Rdata && std::env::var("LMUTILS_FD").is_err() {
            use std::{io::Seek, os::unix::process::CommandExt};

            let tmp_path = std::env::current_dir()
                .unwrap_or_else(|_| std::env::temp_dir())
                .join(format!(".lmutils.{}.tmp", rand::random::<u64>()));
            cfg_if!(
                if #[cfg(libc_2_27)] {
                    let mut file = memfile::MemFile::create_default(&tmp_path.to_string_lossy())?;
                    let fd = file.as_fd().as_raw_fd();
                    let new_fd = unsafe { libc::dup(fd) };
                }
            );
            cfg_if!(
                if #[cfg(libc_2_27)] {
                    let p2 = new_fd.to_string();
                } else {
                    let p2 = format!("'{}'", tmp_path.to_string_lossy());
                }
            );
            let output = unsafe {
                std::process::Command::new("Rscript")
                    .arg("-e")
                    .arg(format!(
                        "lmutils::internal_lmutils_file_into_fd('{}', {})",
                        self.path.to_string_lossy(),
                        p2,
                    ))
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .pre_exec(move || unsafe {
                        cfg_if!(if #[cfg(libc_2_27)] {
                            libc::dup2(fd, new_fd);
                        });
                        Ok(())
                    })
                    .output()?
            };
            if output.status.code().is_none() || output.status.code().unwrap() != 0 {
                tracing::error!("failed to read {}", self.path.display());
                tracing::error!("STDOUT: {}", String::from_utf8_lossy(&output.stdout));
                tracing::error!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
                return Err(crate::Error::Rscript(output.status.code().unwrap_or(-1)));
            }
            cfg_if!(
                if #[cfg(not(libc_2_27))] {
                    let mut file = std::fs::File::open(&tmp_path)?;
                }
            );
            file.rewind()?;
            let mat = Self::new("", FileType::Mat, false).read_from_reader(file);
            cfg_if!(
                if #[cfg(not(libc_2_27))] {
                    if std::fs::exists(&tmp_path)? {
                        std::fs::remove_file(tmp_path)?;
                    }
                }
            );
            return mat;
        }
        let file = std::fs::File::open(&self.path)?;
        if self.gz || self.file_type == FileType::Rdata {
            let decoder = flate2::read::GzDecoder::new(std::io::BufReader::new(
                // 1024 * 1024 * 100,
                file,
            ));
            self.read_from_reader(std::io::BufReader::new(decoder))
        } else {
            self.read_from_reader(std::io::BufReader::new(file))
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))] // We can't test RData, so we exclude this from
                                                 // coverage even though everything else is tested
    pub fn read_from_reader(&self, mut reader: impl std::io::Read) -> Result<Matrix, crate::Error> {
        Ok(match self.file_type {
            FileType::Csv => Self::read_text_file(reader, b',')?,
            FileType::Tsv => Self::read_text_file(reader, b'\t')?,
            FileType::Json => Matrix::Owned(serde_json::from_reader(reader)?),
            FileType::Txt => Self::read_text_file(reader, b' ')?,
            FileType::Rdata => Matrix::from_rdata(&mut reader)?,
            FileType::Rkyv => {
                let mut bytes = vec![];
                reader.read_to_end(&mut bytes)?;
                Matrix::Owned(unsafe { rkyv::from_bytes_unchecked(&bytes)? })
            },
            FileType::Cbor => Matrix::Owned(serde_cbor::from_reader(reader)?),
            FileType::Mat => Self::read_mat(reader)?,
        })
    }

    #[doc(hidden)]
    pub fn read_text_file(mut reader: impl std::io::Read, sep: u8) -> Result<Matrix, crate::Error> {
        let mut file = vec![];
        reader.read_to_end(&mut file)?;
        let mut nrows = 0;
        let file = file.trim_ascii();
        for b in file {
            if *b == b'\n' {
                nrows += 1;
            }
        }

        let mut header = vec![];
        let mut header_start = 0;
        for (i, b) in file.iter().enumerate() {
            if *b == b'\r' {
                continue;
            }
            if *b == sep || *b == b'\n' {
                header
                    .push(unsafe { str::from_utf8_unchecked(&file[header_start..i]).to_string() });
                header_start = i + 1;
                if *b == b'\n' {
                    break;
                }
            }
        }

        let ncols = header.len();

        let mut data = vec![MaybeUninit::<f64>::uninit(); nrows * ncols];
        let file = file[header_start..].trim_ascii();
        let mut lines = vec![MaybeUninit::uninit(); nrows];
        let mut start = 0;
        let mut next = 0;
        for (i, b) in file.iter().enumerate() {
            if *b == b'\n' {
                lines[next].write(file[start..i].trim_ascii());
                next += 1;
                start = i + 1;
            }
        }
        lines[next].write(file[start..].trim_ascii());

        let lines = unsafe { std::mem::transmute::<Vec<MaybeUninit<&[u8]>>, Vec<&[u8]>>(lines) };

        lines
            .into_par_iter()
            .enumerate()
            .map(|(row, line)| {
                let mut col = 0;
                let mut data_start = 0;
                for (i, b) in line.iter().enumerate() {
                    if *b == sep {
                        let field = unsafe { str::from_utf8_unchecked(&line[data_start..i]) };
                        if field == "NA" || field.is_empty() {
                            unsafe {
                                *data
                                    .as_ptr()
                                    .add(col * nrows + row)
                                    .cast_mut()
                                    .cast::<f64>() = f64::NAN;
                            }
                        } else {
                            let field = field.parse()?;
                            unsafe {
                                *data
                                    .as_ptr()
                                    .add(col * nrows + row)
                                    .cast_mut()
                                    .cast::<f64>() = field;
                            }
                        }
                        data_start = i + 1;
                        col += 1;
                    }
                }
                let field = unsafe { str::from_utf8_unchecked(&line[data_start..]) };
                if field == "NA" || field.is_empty() {
                    unsafe {
                        *data
                            .as_ptr()
                            .add(col * nrows + row)
                            .cast_mut()
                            .cast::<f64>() = f64::NAN;
                    }
                } else {
                    let field = field.parse()?;
                    unsafe {
                        *data
                            .as_ptr()
                            .add(col * nrows + row)
                            .cast_mut()
                            .cast::<f64>() = field;
                    }
                }
                col += 1;
                if col != ncols {
                    return Err(crate::Error::IncompleteFile);
                }
                Ok(())
            })
            .collect::<Result<(), _>>()?;

        Ok(Matrix::Owned(OwnedMatrix::new(
            nrows,
            ncols,
            unsafe {
                std::mem::transmute::<std::vec::Vec<std::mem::MaybeUninit<f64>>, std::vec::Vec<f64>>(
                    data,
                )
            },
            Some(header),
        )))
    }

    pub fn read_mat(mut reader: impl std::io::Read) -> Result<Matrix, crate::Error> {
        let mut prefix = [0; 3];
        reader.read_exact(&mut prefix)?;
        if prefix != [b'M', b'A', b'T'] {
            return Err(crate::Error::InvalidMatFile);
        }
        let mut version = [0; 1];
        reader.read_exact(&mut version)?;
        match version[0] {
            1 => {
                let mut buf = [0; 8];
                reader.read_exact(&mut buf)?;
                let nrows = u64::from_le_bytes(buf);
                reader.read_exact(&mut buf)?;
                let ncols = u64::from_le_bytes(buf);
                let usize_max = usize::MAX as u64;
                if nrows > usize_max || ncols > usize_max {
                    return Err(crate::Error::MatrixTooLarge);
                }
                match nrows.checked_mul(ncols) {
                    None => return Err(crate::Error::MatrixTooLarge),
                    Some(n) if n > usize_max => return Err(crate::Error::MatrixTooLarge),
                    _ => (),
                }
                let ncols = ncols as usize;
                let nrows = nrows as usize;
                let mut len = unsafe { nrows.unchecked_mul(ncols) };
                let mut buf = [0; 1];
                reader.read_exact(&mut buf)?;
                let mut colnames = None;
                if buf[0] == 1 {
                    let mut names = Vec::with_capacity(ncols);
                    let mut buf = [0; 2];
                    for _ in 0..ncols {
                        reader.read_exact(&mut buf)?;
                        let len = u16::from_le_bytes(buf) as usize;
                        let mut name = vec![MaybeUninit::<u8>::uninit(); len];
                        let mut slice = unsafe {
                            std::slice::from_raw_parts_mut(name.as_mut_ptr().cast::<u8>(), len)
                        };
                        reader.read_exact(slice)?;
                        names.push(unsafe {
                            String::from_utf8_unchecked(std::mem::transmute::<
                                std::vec::Vec<std::mem::MaybeUninit<u8>>,
                                std::vec::Vec<u8>,
                            >(name))
                        });
                    }
                    colnames = Some(names);
                }

                let mut data = vec![MaybeUninit::<f64>::uninit(); len];
                cfg_if!(
                    if #[cfg(target_endian = "little")] {
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u8>(), len * 8)
                        };
                        reader.read_exact(slice)?;
                    } else {
                        let mut buf = [0; 8];
                        for i in 0..len {
                            reader.read_exact(&mut buf)?;
                            let val = f64::from_le_bytes(buf);
                            unsafe {
                                *data.as_ptr().add(i).cast_mut().cast::<f64>() = val;
                            }
                        }
                    }
                );
                Ok(Matrix::Owned(OwnedMatrix::new(
                    nrows,
                    ncols,
                    unsafe {
                        std::mem::transmute::<
                            std::vec::Vec<std::mem::MaybeUninit<f64>>,
                            std::vec::Vec<f64>,
                        >(data)
                    },
                    colnames,
                )))
            },
            v => Err(crate::Error::UnsupportedMatFileVersion(v)),
        }
    }

    pub fn write(&self, mat: &mut Matrix) -> Result<(), crate::Error> {
        #[cfg(any(unix, target_os = "wasi"))]
        if self.file_type == FileType::Rdata && std::env::var("LMUTILS_FD").is_err() {
            use std::{
                io::Seek,
                os::{fd::FromRawFd, unix::process::CommandExt},
            };

            let tmp_path = std::env::current_dir()
                .unwrap_or_else(|_| std::env::temp_dir())
                .join(format!(".lmutils.{}.tmp", rand::random::<u64>()));
            cfg_if!(
                if #[cfg(libc_2_27)] {
                    let mut file = memfile::MemFile::create_default(&tmp_path.to_string_lossy())?;
                    let fd = file.as_fd().as_raw_fd();
                } else {
                    let mut file = std::fs::File::create(&tmp_path)?;
                    let fd = file.as_raw_fd();
                }
            );
            Self::new("", FileType::Mat, false).write_matrix_to_writer(&mut file, mat)?;
            file.rewind()?;
            let new_fd = unsafe { libc::dup(fd) };
            let output = unsafe {
                std::process::Command::new("Rscript")
                    .arg("-e")
                    .arg(format!(
                        "lmutils::internal_lmutils_fd_into_file('{}', {}, {})",
                        self.path.to_string_lossy(),
                        new_fd,
                        if (cfg!(libc_2_27)) { "TRUE" } else { "FALSE" }
                    ))
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .pre_exec(move || unsafe {
                        libc::dup2(fd, new_fd);
                        Ok(())
                    })
                    .output()?
            };
            cfg_if!(
                if #[cfg(not(libc_2_27))] {
                    if std::fs::exists(&tmp_path)? {
                        std::fs::remove_file(tmp_path)?;
                    }
                }
            );
            if output.status.code().is_none() || output.status.code().unwrap() != 0 {
                tracing::error!("failed to read {}", self.path.display());
                tracing::error!("STDOUT: {}", String::from_utf8_lossy(&output.stdout));
                tracing::error!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
                return Err(crate::Error::Rscript(output.status.code().unwrap_or(-1)));
            }
            return Ok(());
        }
        let file = std::fs::File::create(&self.path)?;
        if self.gz || self.file_type == FileType::Rdata {
            let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
            self.write_matrix_to_writer(std::io::BufWriter::new(encoder), mat)
        } else {
            self.write_matrix_to_writer(std::io::BufWriter::new(file), mat)
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))] // We can't test RData, so we exclude this from
                                                 // coverage even though everything else is tested
    pub fn write_matrix_to_writer(
        &self,
        mut writer: impl std::io::Write,
        mat: &mut Matrix,
    ) -> Result<(), crate::Error> {
        match self.file_type {
            FileType::Csv => Self::write_text_file(writer, mat, b',')?,
            FileType::Tsv => Self::write_text_file(writer, mat, b'\t')?,
            FileType::Json => serde_json::to_writer(writer, mat.as_owned_ref()?)?,
            FileType::Txt => Self::write_text_file(writer, mat, b' ')?,
            FileType::Rdata => {
                let mat = mat.to_rmatrix();
                let pl = pairlist!(mat = mat);
                writer.write_all(b"RDX3\n")?;
                pl.to_writer(
                    &mut writer,
                    extendr_api::io::PstreamFormat::XdrFormat,
                    3,
                    None,
                )?;
            },
            FileType::Rkyv => {
                let bytes = rkyv::to_bytes::<_, 256>(mat.as_owned_ref()?)?;
                writer.write_all(&bytes)?;
            },
            FileType::Cbor => serde_cbor::to_writer(writer, mat.as_owned_ref()?)?,
            FileType::Mat => Self::write_mat(writer, mat)?,
        }
        Ok(())
    }

    #[doc(hidden)]
    pub fn write_text_file(
        mut writer: impl std::io::Write,
        mat: &mut Matrix,
        sep: u8,
    ) -> Result<(), crate::Error> {
        let mut writer = std::io::BufWriter::with_capacity(128 * 1024, writer);
        let mat = mat.as_owned_ref()?;
        if let Some(colnames) = &mat.colnames {
            for (i, colname) in colnames.iter().enumerate() {
                writer.write_all(colname.as_bytes())?;
                if i != colnames.len() - 1 {
                    writer.write_all(&[sep])?;
                }
            }
            writer.write_all(b"\n")?;
        }
        let rows = (0..mat.nrows)
            .into_par_iter()
            .map(|i| {
                let mut buf = vec![];
                for j in 0..mat.ncols {
                    buf.extend_from_slice(mat.data[i + j * mat.nrows].to_string().as_bytes());
                    if j != mat.ncols - 1 {
                        buf.push(sep);
                    }
                }
                if i != mat.nrows - 1 {
                    buf.push(b'\n');
                }
                Ok(buf)
            })
            .collect::<Result<Vec<Vec<u8>>, crate::Error>>()?;
        for row in rows {
            writer.write_all(&row)?;
        }
        Ok(())
    }

    pub fn write_mat(
        mut writer: impl std::io::Write,
        mat: &mut Matrix,
    ) -> Result<(), crate::Error> {
        mat.into_owned()?;
        writer.write_all(b"MAT")?;
        writer.write_all(&[1])?;
        writer.write_all(&mat.nrows_loaded().to_le_bytes())?;
        writer.write_all(&mat.ncols_loaded().to_le_bytes())?;
        if let Some(colnames) = &mat.colnames_loaded() {
            writer.write_all(&[1])?;
            for colname in colnames {
                let len = colname.len() as u16;
                writer.write_all(&len.to_le_bytes())?;
                writer.write_all(colname.as_bytes())?;
            }
        } else {
            writer.write_all(&[0])?;
        }
        cfg_if!(
            if #[cfg(target_endian = "little")] {
                let data = mat.data()?;
                writer.write_all(unsafe {
                    std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 8)
                })?;
            } else {
                for val in mat.data.iter() {
                    writer.write_all(&val.to_le_bytes())?;
                }
            }
        );
        Ok(())
    }

    pub fn from_path(path: impl Into<PathBuf>) -> Result<Self, crate::Error> {
        let path = path.into();
        let extension = path
            .file_name()
            .ok_or(crate::Error::NoFileName)?
            .to_str()
            .ok_or(crate::Error::InvalidFileName)?
            .split('.')
            .filter(|x| !x.is_empty())
            .collect::<Vec<&str>>();
        if extension.len() < 2 {
            return Err(crate::Error::NoFileExtension);
        }
        let gz = extension[extension.len() - 1] == "gz";
        let extension = extension[extension.len() - if gz { 2 } else { 1 }];
        let file_type = FileType::from_str(extension)?;
        Ok(Self {
            path,
            file_type,
            gz,
        })
    }
}

impl FromStr for File {
    type Err = crate::Error;

    #[cfg_attr(coverage_nightly, coverage(off))]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_path(s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    /// Comma-separated values, row major.
    /// Expects the first row to be the column names.
    Csv,
    /// Tab-separated values, row major.
    /// Expects the first row to be the column names.
    Tsv,
    /// Serialized Matrix type.
    Json,
    /// Space-separated values, row major.
    /// Expects the first row to be the column names.
    Txt,
    /// RData file.
    Rdata,
    /// Serialized matrix type.
    Rkyv,
    /// Serialied matrix type.
    Cbor,
    /// Lmutils mat file format.
    Mat,
}

impl FromStr for FileType {
    type Err = crate::Error;

    #[cfg_attr(coverage_nightly, coverage(off))]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "csv" => Self::Csv,
            "tsv" => Self::Tsv,
            "json" => Self::Json,
            "txt" => Self::Txt,
            "rdata" | "RData" => Self::Rdata,
            "rkyv" => Self::Rkyv,
            "cbor" => Self::Cbor,
            "mat" => Self::Mat,
            _ => return Err(crate::Error::UnsupportedFileType(s.to_string())),
        })
    }
}

#[cfg(test)]
mod tests {
    use test_log::test;

    use super::*;

    #[test]
    fn test_csv() {
        let mut mat = Matrix::Owned(OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        ));
        let file = crate::File::new("tests/test.csv", crate::FileType::Csv, false);
        file.write(&mut mat).unwrap();
        let mat2 = file.read().unwrap();
        assert_eq!(mat, mat2);
    }

    #[test]
    fn test_tsv() {
        let mut mat = Matrix::Owned(OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        ));
        let file = crate::File::new("tests/test.tsv", crate::FileType::Tsv, false);
        file.write(&mut mat).unwrap();
        let mat2 = file.read().unwrap();
        assert_eq!(mat, mat2);
    }

    #[test]
    fn test_json() {
        let mut mat = Matrix::Owned(OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        ));
        let file = crate::File::new("tests/test.json", crate::FileType::Json, false);
        file.write(&mut mat).unwrap();
        let mat2 = file.read().unwrap();
        assert_eq!(mat, mat2);
    }

    #[test]
    fn test_txt() {
        let mut mat = Matrix::Owned(OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        ));
        let file = crate::File::new("tests/test.txt", crate::FileType::Txt, false);
        file.write(&mut mat).unwrap();
        let mat2 = file.read().unwrap();
        assert_eq!(mat, mat2);
    }

    #[test]
    fn test_rkyv() {
        let mut mat = Matrix::Owned(OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        ));
        let file = crate::File::new("tests/test.rkyv", crate::FileType::Rkyv, false);
        file.write(&mut mat).unwrap();
        let mat2 = file.read().unwrap();
        assert_eq!(mat, mat2);
    }

    #[test]
    fn test_cbor() {
        let mut mat = Matrix::Owned(OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        ));
        let file = crate::File::new("tests/test.cbor", crate::FileType::Cbor, false);
        file.write(&mut mat).unwrap();
        let mat2 = file.read().unwrap();
        assert_eq!(mat, mat2);
    }

    // #[test]
    // fn test_rdata() {
    //     let mut mat = Matrix::Owned(OwnedMatrix::new(
    //         3,
    //         2,
    //         vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    //         Some(vec!["a".to_string(), "b".to_string()]),
    //     ));
    //     let file = crate::File::new("tests/mat-f64.RData",
    // crate::FileType::Rdata, false);     file.write(&mut mat).unwrap();
    //     let mat2 = file.read().unwrap();
    //     assert_eq!(mat, mat2);
    // }

    #[test]
    fn test_mat() {
        let mut mat = Matrix::Owned(OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        ));
        let file = crate::File::new("tests/test.mat", crate::FileType::Mat, false);
        file.write(&mut mat).unwrap();
        let mat2 = file.read().unwrap();
        assert_eq!(mat, mat2);
    }

    #[test]
    fn test_from_path() {
        let file = crate::File::from_path("tests/test.csv").unwrap();
        assert_eq!(file.file_type, crate::FileType::Csv);
        let file = crate::File::from_path("tests/test.tsv").unwrap();
        assert_eq!(file.file_type, crate::FileType::Tsv);
        let file = crate::File::from_path("tests/test.json").unwrap();
        assert_eq!(file.file_type, crate::FileType::Json);
        let file = crate::File::from_path("tests/test.txt").unwrap();
        assert_eq!(file.file_type, crate::FileType::Txt);
        let file = crate::File::from_path("tests/test.rdata").unwrap();
        assert_eq!(file.file_type, crate::FileType::Rdata);
        let file = crate::File::from_path("tests/test.rkyv").unwrap();
        assert_eq!(file.file_type, crate::FileType::Rkyv);
        let file = crate::File::from_path("tests/test.cbor").unwrap();
        assert_eq!(file.file_type, crate::FileType::Cbor);
        let file = crate::File::from_path("tests/test.mat").unwrap();
        assert_eq!(file.file_type, crate::FileType::Mat);
    }

    #[test]
    fn test_from_path_gz() {
        let file = crate::File::from_path("tests/test.csv.gz").unwrap();
        assert_eq!(file.file_type, crate::FileType::Csv);
        assert!(file.gz);
        let file = crate::File::from_path("tests/test.tsv.gz").unwrap();
        assert_eq!(file.file_type, crate::FileType::Tsv);
        assert!(file.gz);
        let file = crate::File::from_path("tests/test.json.gz").unwrap();
        assert_eq!(file.file_type, crate::FileType::Json);
        assert!(file.gz);
        let file = crate::File::from_path("tests/test.txt.gz").unwrap();
        assert_eq!(file.file_type, crate::FileType::Txt);
        assert!(file.gz);
        let file = crate::File::from_path("tests/test.rdata.gz").unwrap();
        assert_eq!(file.file_type, crate::FileType::Rdata);
        assert!(file.gz);
        let file = crate::File::from_path("tests/test.rkyv.gz").unwrap();
        assert_eq!(file.file_type, crate::FileType::Rkyv);
        assert!(file.gz);
        let file = crate::File::from_path("tests/test.cbor.gz").unwrap();
        assert_eq!(file.file_type, crate::FileType::Cbor);
        assert!(file.gz);
        let file = crate::File::from_path("tests/test.mat.gz").unwrap();
        assert_eq!(file.file_type, crate::FileType::Mat);
    }

    #[test]
    fn test_from_path_invalid() {
        assert!(matches!(
            crate::File::from_path("tests/test").unwrap_err(),
            crate::Error::NoFileExtension
        ));
        assert!(matches!(
            crate::File::from_path("tests/test.").unwrap_err(),
            crate::Error::NoFileExtension
        ));
        assert!(matches!(
            crate::File::from_path("tests/test.invalid").unwrap_err(),
            crate::Error::UnsupportedFileType(_)
        ));
    }

    #[test]
    fn test_gz() {
        let mut mat = Matrix::Owned(OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        ));
        let file = crate::File::new("tests/test.csv.gz", crate::FileType::Csv, true);
        file.write(&mut mat).unwrap();
        let mat2 = file.read().unwrap();
        assert_eq!(mat, mat2);
    }

    #[test]
    fn test_csv_nan() {
        let mut mat = Matrix::Owned(OwnedMatrix::new(
            3,
            2,
            vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string()]),
        ));
        let file = crate::File::new("tests/mat.csv", crate::FileType::Csv, false);
        let mat2 = file.read().unwrap();
        assert_eq!(mat, mat2);
    }

    #[test]
    fn test_file_not_found() {
        let file = crate::File::new("tests/does_not_exist.csv", crate::FileType::Csv, false);
        assert!(matches!(file.read(), Err(crate::Error::Io(_))));
    }
}

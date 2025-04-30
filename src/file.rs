use core::str;
use std::{
    io::{Read, Write},
    mem::MaybeUninit,
    num::ParseFloatError,
    path::PathBuf,
    str::FromStr,
};

use cfg_if::cfg_if;
#[cfg(feature = "r")]
use extendr_api::{io::Save, pairlist, Pairlist};
use rayon::{
    iter::IntoParallelIterator,
    prelude::{IndexedParallelIterator, ParallelIterator},
    str::ParallelString,
};
use tracing::info;

use crate::{
    mat::{read_mat, write_mat},
    IntoMatrix, Matrix, OwnedMatrix,
};

#[derive(Clone, Debug, PartialEq)]
pub struct File {
    path: PathBuf,
    file_type: FileType,
    gz: bool,
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
        #[cfg(all(unix, feature = "r"))]
        if (self.file_type == FileType::Rdata || self.file_type == FileType::Rds)
            && std::env::var("LMUTILS_FD").is_err()
        {
            use std::{io::Seek, os::fd::AsRawFd, os::unix::process::CommandExt};

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
                        "devtools::load_all();lmutils::internal_lmutils_file_into_fd('{}', {})",
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
        #[cfg(feature = "r")]
        if (self.file_type == FileType::Rdata || self.file_type == FileType::Rds) {
            let decoder = flate2::read::GzDecoder::new(file);
            return self.read_from_reader(decoder);
        }
        if self.gz {
            let decoder = flate2::read::GzDecoder::new(file);
            self.read_from_reader(decoder)
        } else {
            self.read_from_reader(file)
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
            #[cfg(feature = "r")]
            FileType::Rdata => Matrix::from_rdata(&mut reader)?,
            #[cfg(feature = "r")]
            FileType::Rds => Matrix::from_rds(&mut reader)?,
            FileType::Rkyv => {
                let mut bytes = vec![];
                reader.read_to_end(&mut bytes)?;
                Matrix::Owned(unsafe { rkyv::from_bytes_unchecked(&bytes)? })
            },
            FileType::Cbor => Matrix::Owned(serde_cbor::from_reader(reader)?),
            FileType::Mat => read_mat(reader)?,
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

    pub fn write(&self, mat: &mut Matrix) -> Result<(), crate::Error> {
        #[cfg(all(unix, feature = "r"))]
        if (self.file_type == FileType::Rdata || self.file_type == FileType::Rds)
            && std::env::var("LMUTILS_FD").is_err()
        {
            use std::{
                io::Seek,
                os::{
                    fd::{AsRawFd, FromRawFd},
                    unix::process::CommandExt,
                },
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
        #[cfg(feature = "r")]
        if (self.file_type == FileType::Rdata || self.file_type == FileType::Rds) {
            let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
            return self.write_matrix_to_writer(encoder, mat);
        }
        if self.gz {
            let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
            self.write_matrix_to_writer(encoder, mat)
        } else {
            self.write_matrix_to_writer(file, mat)
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
            #[cfg(feature = "r")]
            FileType::Rdata => {
                let mat = mat.to_rmatrix()?;
                let pl = pairlist!(mat = mat);
                writer.write_all(b"RDX3\n")?;
                pl.to_writer(
                    &mut writer,
                    extendr_api::io::PstreamFormat::R_pstream_xdr_format,
                    3,
                    None,
                )?;
            },
            #[cfg(feature = "r")]
            FileType::Rds => {
                let mat = mat.to_rmatrix()?;
                mat.to_writer(
                    &mut writer,
                    extendr_api::io::PstreamFormat::R_pstream_xdr_format,
                    3,
                    None,
                )?;
            },
            FileType::Rkyv => {
                let bytes = rkyv::to_bytes::<_, 256>(mat.as_owned_ref()?)?;
                writer.write_all(&bytes)?;
            },
            FileType::Cbor => serde_cbor::to_writer(writer, mat.as_owned_ref()?)?,
            FileType::Mat => write_mat(writer, mat)?,
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
    #[cfg(feature = "r")]
    Rdata,
    /// RDS file
    #[cfg(feature = "r")]
    Rds,
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
            #[cfg(feature = "r")]
            "rdata" | "RData" => Self::Rdata,
            #[cfg(feature = "r")]
            "rds" | "Rds" => Self::Rds,
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

    // #[test]
    // fn test_rds() {
    //     let mut mat = Matrix::Owned(OwnedMatrix::new(
    //         3,
    //         2,
    //         vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    //         Some(vec!["a".to_string(), "b".to_string()]),
    //     ));
    //     let file = crate::File::new("tests/test.rds", crate::FileType::Rds, false);
    //     file.write(&mut mat).unwrap();
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
        #[cfg(feature = "r")]
        {
            let file = crate::File::from_path("tests/test.rdata").unwrap();
            assert_eq!(file.file_type, crate::FileType::Rdata);
            let file = crate::File::from_path("tests/test.rds").unwrap();
            assert_eq!(file.file_type, crate::FileType::Rds);
        }
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
        #[cfg(feature = "r")]
        {
            let file = crate::File::from_path("tests/test.rdata.gz").unwrap();
            assert_eq!(file.file_type, crate::FileType::Rdata);
            assert!(file.gz);
            let file = crate::File::from_path("tests/test.rds.gz").unwrap();
            assert_eq!(file.file_type, crate::FileType::Rds);
            assert!(file.gz);
        }
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

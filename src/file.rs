use std::{path::PathBuf, str::FromStr};

use extendr_api::{io::Save, pairlist, Pairlist};

use crate::{
    errors::FileParseError, parse, IntoMatrix, Matrix, OwnedMatrix, ReadMatrixError,
    WriteMatrixError,
};

#[derive(Clone, Debug)]
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
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    #[inline(always)]
    pub fn file_type(&self) -> FileType {
        self.file_type
    }

    #[inline(always)]
    pub fn gz(&self) -> bool {
        self.gz
    }

    pub fn read(&self) -> Result<Matrix<'static>, ReadMatrixError> {
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

    pub fn read_from_reader(
        &self,
        mut reader: impl std::io::Read,
    ) -> Result<Matrix<'static>, ReadMatrixError> {
        Ok(match self.file_type {
            FileType::Csv => Self::read_text_file(reader, b',')?,
            FileType::Tsv => Self::read_text_file(reader, b'\t')?,
            FileType::Json => Matrix::Owned(serde_json::from_reader(reader)?),
            FileType::Txt => Self::read_text_file(reader, b' ')?,
            FileType::Rdata => Matrix::from_rdata(&mut reader)?,
            FileType::Rkyv => {
                let mut bytes = vec![];
                reader.read_to_end(&mut bytes)?;
                Matrix::Owned(unsafe {
                    rkyv::from_bytes_unchecked(&bytes)
                        .map_err(|e| ReadMatrixError::RkyvError(e.to_string()))?
                })
            },
            FileType::Cbor => Matrix::Owned(serde_cbor::from_reader(reader)?),
        })
    }

    fn read_text_file(
        reader: impl std::io::Read,
        sep: u8,
    ) -> Result<Matrix<'static>, ReadMatrixError> {
        let mut data = vec![];
        let mut reader = csv::ReaderBuilder::new().delimiter(sep).from_reader(reader);
        let headers = reader.headers()?;
        // Check if headers are numeric.
        let colnames = if headers.iter().next().unwrap().parse::<f64>().is_err() {
            Some(headers.iter().map(|x| x.to_string()).collect())
        } else {
            None
        };
        for i in headers.iter() {
            if i.parse::<f64>().is_err() {
                break;
            }
            data.push(parse(i)?);
        }
        let cols = headers.len();
        let _ = headers;
        for result in reader.records() {
            let record = result?;
            for field in record.iter() {
                data.push(parse(field)?);
            }
        }
        let mut mat = OwnedMatrix::new(cols, data.len() / cols, data, None).transpose();
        mat.colnames = colnames;
        Ok(mat.into_matrix())
    }

    pub fn write(&self, mat: &mut Matrix<'_>) -> Result<(), WriteMatrixError> {
        let file = std::fs::File::create(&self.path)?;
        if self.gz || self.file_type == FileType::Rdata {
            let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
            self.write_matrix_to_writer(std::io::BufWriter::new(encoder), mat)
        } else {
            self.write_matrix_to_writer(std::io::BufWriter::new(file), mat)
        }
    }

    pub fn write_matrix_to_writer(
        &self,
        mut writer: impl std::io::Write,
        mat: &mut Matrix<'_>,
    ) -> Result<(), WriteMatrixError> {
        let mat = mat.as_owned_ref()?;
        match self.file_type {
            FileType::Csv => Self::write_text_file(writer, mat, b',')?,
            FileType::Tsv => Self::write_text_file(writer, mat, b'\t')?,
            FileType::Json => serde_json::to_writer(writer, &mat)?,
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
                let bytes = rkyv::to_bytes::<_, 256>(mat)
                    .map_err(|e| WriteMatrixError::RkyvError(e.to_string()))?;
                writer.write_all(&bytes)?;
            },
            FileType::Cbor => serde_cbor::to_writer(writer, &mat)?,
        }
        Ok(())
    }

    fn write_text_file(
        writer: impl std::io::Write,
        mat: &OwnedMatrix,
        sep: u8,
    ) -> Result<(), WriteMatrixError> {
        let mut writer = csv::WriterBuilder::new().delimiter(sep).from_writer(writer);
        if let Some(colnames) = mat.colnames() {
            writer.write_record(colnames)?;
        }
        for i in 0..mat.nrows {
            writer.write_record(
                mat.data
                    .iter()
                    .skip(i)
                    .step_by(mat.nrows)
                    .take(mat.ncols)
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>(),
            )?;
        }
        Ok(())
    }

    pub fn from_path(path: impl Into<PathBuf>) -> Result<Self, FileParseError> {
        let path = path.into();
        let extension = path
            .file_name()
            .ok_or(FileParseError::NoFileName)?
            .to_str()
            .ok_or(FileParseError::InvalidFileName)?
            .split('.')
            .collect::<Vec<&str>>();
        if extension.len() < 2 {
            return Err(FileParseError::NoFileExtension);
        }
        let gz = extension[extension.len() - 1] == "gz";
        let extension = extension[extension.len() - if gz { 2 } else { 1 }];
        let file_type = match extension {
            "csv" => FileType::Csv,
            "tsv" => FileType::Tsv,
            "json" => FileType::Json,
            "txt" => FileType::Txt,
            "rdata" | "RData" => FileType::Rdata,
            "rkyv" => FileType::Rkyv,
            "cbor" => FileType::Cbor,
            _ => return Err(FileParseError::UnsupportedFileType(extension.to_string())),
        };
        Ok(Self {
            path,
            file_type,
            gz,
        })
    }
}

impl FromStr for File {
    type Err = FileParseError;

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
}

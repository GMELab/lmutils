use std::{path::PathBuf, str::FromStr};

use extendr_api::{io::Save, pairlist, Pairlist};

use crate::{IntoMatrix, Matrix, OwnedMatrix};

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

    pub fn read<'a>(&self) -> Result<Matrix<'a>, crate::Error> {
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

    pub fn read_from_reader<'a>(
        &self,
        mut reader: impl std::io::Read,
    ) -> Result<Matrix<'a>, crate::Error> {
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
        })
    }

    fn read_text_file<'a>(reader: impl std::io::Read, sep: u8) -> Result<Matrix<'a>, crate::Error> {
        let mut data = vec![];
        let mut reader = csv::ReaderBuilder::new().delimiter(sep).from_reader(reader);
        let headers = reader.headers()?;
        // Check if headers are numeric.
        let colnames = Some(headers.iter().map(|x| x.to_string()).collect());
        let cols = headers.len();
        let _ = headers;
        for result in reader.records() {
            let record = result?;
            for field in record.iter() {
                data.push({
                    if field == "NA" {
                        Ok(f64::NAN)
                    } else {
                        field.parse().map_err(crate::Error::from)
                    }
                }?);
            }
        }
        let mut mat = OwnedMatrix::new(cols, data.len() / cols, data, None).into_matrix();
        mat.transpose();
        let mut mat = mat.to_owned()?;
        mat.colnames = colnames;
        Ok(mat.into_matrix())
    }

    pub fn write(&self, mat: &mut Matrix<'_>) -> Result<(), crate::Error> {
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
        }
        Ok(())
    }

    fn write_text_file(
        writer: impl std::io::Write,
        mat: &mut Matrix<'_>,
        sep: u8,
    ) -> Result<(), crate::Error> {
        let mat = mat.as_owned_ref()?;
        let mut writer = csv::WriterBuilder::new().delimiter(sep).from_writer(writer);
        if let Some(colnames) = &mat.colnames {
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

    pub fn from_path(path: impl Into<PathBuf>) -> Result<Self, crate::Error> {
        let path = path.into();
        let extension = path
            .file_name()
            .ok_or(crate::Error::NoFileName)?
            .to_str()
            .ok_or(crate::Error::InvalidFileName)?
            .split('.')
            .collect::<Vec<&str>>();
        if extension.len() < 2 {
            return Err(crate::Error::NoFileExtension);
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
            _ => return Err(crate::Error::UnsupportedFileType(extension.to_string())),
        };
        Ok(Self {
            path,
            file_type,
            gz,
        })
    }
}

impl FromStr for File {
    type Err = crate::Error;

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

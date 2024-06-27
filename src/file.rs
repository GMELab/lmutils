use std::{fmt::Display, path::PathBuf, str::FromStr};

use extendr_api::{
    io::{Load, Save},
    pairlist, AsTypedSlice, Conversions, MatrixConversions, Pairlist, Robj, Rstr, ToVectorValue,
};

use crate::{
    errors::FileParseError,
    matrix::{
        FromRMatrix, MatEmpty, MatParse, OwnedMatrix, ToRMatrix, TransitoryMatrix, TransitoryType,
    },
    MatParseError, ReadMatrixError, WriteMatrixError,
};

#[derive(Debug)]
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

    #[inline]
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    #[inline]
    pub fn file_type(&self) -> &FileType {
        &self.file_type
    }

    #[inline]
    pub fn gz(&self) -> bool {
        self.gz
    }

    pub fn read_transitory(
        &self,
        trans: TransitoryType,
    ) -> Result<TransitoryMatrix, ReadMatrixError> {
        Ok(match trans {
            TransitoryType::Float => TransitoryMatrix::Float(self.read_matrix::<f64, _, _>(true)?),
            TransitoryType::Str => TransitoryMatrix::Str(self.read_matrix::<String, _, _>(true)?),
        })
    }

    pub fn read_matrix<T, R, E>(
        &self,
        rkyv_validate: bool,
    ) -> Result<OwnedMatrix<T>, ReadMatrixError>
    where
        for<'a> T: MatEmpty + Clone + serde::Deserialize<'a> + ToVectorValue + rkyv::Archive,
        for<'a> Robj: AsTypedSlice<'a, R>,
        for<'a> <T as rkyv::Archive>::Archived:
            rkyv::CheckBytes<rkyv::validation::validators::DefaultValidator<'a>>,
        [<T as rkyv::Archive>::Archived]:
            rkyv::DeserializeUnsized<[T], rkyv::de::deserializers::SharedDeserializeMap>,
        for<'a> &'a str: MatParse<T, E>,
        Rstr: MatParse<T, E>,
        MatParseError: From<E>,
        OwnedMatrix<T>: FromRMatrix<T, R>,
    {
        let file = std::fs::File::open(&self.path)?;
        if self.gz || self.file_type == FileType::Rdata {
            let decoder = flate2::read::GzDecoder::new(std::io::BufReader::new(
                // 1024 * 1024 * 100,
                file,
            ));
            self.read_matrix_from_reader(std::io::BufReader::new(decoder), rkyv_validate)
        } else {
            self.read_matrix_from_reader(std::io::BufReader::new(file), rkyv_validate)
        }
    }

    pub fn read_matrix_from_reader<T, R, E>(
        &self,
        mut reader: impl std::io::Read,
        _rkyv_validate: bool,
    ) -> Result<OwnedMatrix<T>, ReadMatrixError>
    where
        for<'a> T: MatEmpty + Clone + serde::Deserialize<'a> + ToVectorValue + rkyv::Archive,
        for<'a> Robj: AsTypedSlice<'a, R>,
        for<'a> <T as rkyv::Archive>::Archived:
            rkyv::CheckBytes<rkyv::validation::validators::DefaultValidator<'a>>,
        [<T as rkyv::Archive>::Archived]:
            rkyv::DeserializeUnsized<[T], rkyv::de::deserializers::SharedDeserializeMap>,
        for<'a> &'a str: MatParse<T, E>,
        Rstr: MatParse<T, E>,
        MatParseError: From<E>,
        OwnedMatrix<T>: FromRMatrix<T, R>,
    {
        let mut data = vec![];
        Ok(match self.file_type {
            FileType::Csv => {
                let mut reader = csv::Reader::from_reader(reader);
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
                    data.push(i.mat_parse()?);
                }
                let cols = headers.len();
                let _ = headers;
                for result in reader.records() {
                    let record = result?;
                    for field in record.iter() {
                        data.push(field.mat_parse()?);
                    }
                }
                let mut mat = OwnedMatrix::new(cols, data.len() / cols, data, None).transpose();
                mat.colnames = colnames;
                mat
            },
            FileType::Tsv => {
                let mut reader = csv::ReaderBuilder::new()
                    .delimiter(b'\t')
                    .from_reader(reader);
                let headers = reader.headers()?;
                // Check if headers are numeric.
                let colnames = if headers.iter().next().unwrap().parse::<f64>().is_err() {
                    Some(headers.iter().map(|x| x.to_string()).collect())
                } else {
                    None
                };
                let cols = headers.len();
                let _ = headers;
                for result in reader.records() {
                    let record = result?;
                    for field in record.iter() {
                        data.push(field.mat_parse()?);
                    }
                }
                let mut mat = OwnedMatrix::new(cols, data.len() / cols, data, None).transpose();
                mat.colnames = colnames;
                mat
            },
            FileType::Json => serde_json::from_reader(reader)?,
            FileType::Txt => {
                let mut reader = csv::ReaderBuilder::new()
                    .delimiter(b' ')
                    .from_reader(reader);
                let headers = reader.headers()?;
                // Check if headers are numeric.
                let colnames = if headers.iter().next().unwrap().parse::<f64>().is_err() {
                    Some(headers.iter().map(|x| x.to_string()).collect())
                } else {
                    None
                };
                let cols = headers.len();
                let _ = headers;
                for result in reader.records() {
                    let record = result?;
                    for field in record.iter() {
                        data.push(field.mat_parse()?);
                    }
                }
                let mut mat = OwnedMatrix::new(cols, data.len() / cols, data, None).transpose();
                mat.colnames = colnames;
                mat
            },
            FileType::Rdata => {
                let mut buf = [0; 5];
                reader.read_exact(&mut buf)?;
                if buf != *b"RDX3\n" {
                    return Err(ReadMatrixError::InvalidRdataFile);
                }
                let obj = Robj::from_reader(
                    &mut reader,
                    extendr_api::io::PstreamFormat::XdrFormat,
                    None,
                )?;
                let mat = obj
                    .as_pairlist()
                    .ok_or(ReadMatrixError::InvalidRdataFile)?
                    .into_iter()
                    .next()
                    .ok_or(ReadMatrixError::InvalidRdataFile)?
                    .1
                    .as_matrix()
                    .ok_or(ReadMatrixError::InvalidRdataFile)?;
                OwnedMatrix::from_rmatrix(&mat)
            },
            FileType::Rkyv => {
                let mut bytes = vec![];
                reader.read_to_end(&mut bytes)?;
                // if rkyv_validate {
                //     rkyv::from_bytes(&bytes)
                //         .map_err(|e| ReadMatrixError::RkyvError(e.to_string()))?
                // } else {
                unsafe {
                    rkyv::from_bytes_unchecked(&bytes)
                        .map_err(|e| ReadMatrixError::RkyvError(e.to_string()))?
                }
                // }
            },
            FileType::Cbor => serde_cbor::from_reader(reader)?,
        })
    }

    pub fn write_transitory(&self, mat: &TransitoryMatrix) -> Result<(), WriteMatrixError> {
        match mat {
            TransitoryMatrix::Float(mat) => self.write_matrix(mat),
            TransitoryMatrix::Str(mat) => self.write_matrix(mat),
        }
    }

    pub fn write_matrix<T, R>(&self, mat: &OwnedMatrix<T>) -> Result<(), WriteMatrixError>
    where
        T: MatEmpty + Clone + Display + serde::Serialize + rkyv::Archive + ToVectorValue,
        for<'a> Robj: AsTypedSlice<'a, R>,
        T: rkyv::Serialize<
            rkyv::ser::serializers::CompositeSerializer<
                rkyv::ser::serializers::AlignedSerializer<rkyv::AlignedVec>,
                rkyv::ser::serializers::FallbackScratch<
                    rkyv::ser::serializers::HeapScratch<256>,
                    rkyv::ser::serializers::AllocScratch,
                >,
                rkyv::ser::serializers::SharedSerializeMap,
            >,
        >,
        OwnedMatrix<T>: ToRMatrix<T, R>,
    {
        let file = std::fs::File::create(&self.path)?;
        if self.gz || self.file_type == FileType::Rdata {
            let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
            self.write_matrix_to_writer(std::io::BufWriter::new(encoder), mat)
        } else {
            self.write_matrix_to_writer(std::io::BufWriter::new(file), mat)
        }
    }

    pub fn write_matrix_to_writer<T, R>(
        &self,
        mut writer: impl std::io::Write,
        mat: &OwnedMatrix<T>,
    ) -> Result<(), WriteMatrixError>
    where
        T: MatEmpty + Clone + Display + serde::Serialize + rkyv::Archive + ToVectorValue,
        for<'a> Robj: AsTypedSlice<'a, R>,
        T: rkyv::Serialize<
            rkyv::ser::serializers::CompositeSerializer<
                rkyv::ser::serializers::AlignedSerializer<rkyv::AlignedVec>,
                rkyv::ser::serializers::FallbackScratch<
                    rkyv::ser::serializers::HeapScratch<256>,
                    rkyv::ser::serializers::AllocScratch,
                >,
                rkyv::ser::serializers::SharedSerializeMap,
            >,
        >,
        OwnedMatrix<T>: ToRMatrix<T, R>,
    {
        match self.file_type {
            FileType::Csv => {
                let mut writer = csv::Writer::from_writer(writer);
                if let Some(colnames) = mat.colnames() {
                    writer.write_record(colnames)?;
                }
                for i in 0..mat.rows {
                    writer.write_record(
                        mat.data
                            .iter()
                            .skip(i)
                            .step_by(mat.cols)
                            .take(mat.cols)
                            .map(|x| x.to_string())
                            .collect::<Vec<String>>(),
                    )?;
                }
            },
            FileType::Tsv => {
                let mut writer = csv::WriterBuilder::new()
                    .delimiter(b'\t')
                    .from_writer(writer);
                if let Some(colnames) = mat.colnames() {
                    writer.write_record(colnames)?;
                }
                for i in 0..mat.rows {
                    writer.write_record(
                        mat.data
                            .iter()
                            .skip(i)
                            .step_by(mat.cols)
                            .take(mat.cols)
                            .map(|x| x.to_string())
                            .collect::<Vec<String>>(),
                    )?;
                }
            },
            FileType::Json => serde_json::to_writer(writer, &mat)?,
            FileType::Txt => {
                let mut writer = csv::WriterBuilder::new()
                    .delimiter(b' ')
                    .from_writer(writer);
                if let Some(colnames) = mat.colnames() {
                    writer.write_record(colnames)?;
                }
                for i in 0..mat.rows {
                    writer.write_record(
                        mat.data
                            .iter()
                            .skip(i)
                            .step_by(mat.cols)
                            .take(mat.cols)
                            .map(|x| x.to_string())
                            .collect::<Vec<String>>(),
                    )?;
                }
            },
            FileType::Rdata => {
                let mat = ToRMatrix::to_rmatrix(mat);
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

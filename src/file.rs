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
};

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

    pub fn read_transitory(&self, trans: TransitoryType) -> TransitoryMatrix {
        match trans {
            TransitoryType::Float => TransitoryMatrix::Float(self.read_matrix::<f64, _>(true)),
            TransitoryType::Str => TransitoryMatrix::Str(self.read_matrix::<String, _>(true)),
        }
    }

    pub fn read_matrix<T, R>(&self, rkyv_validate: bool) -> OwnedMatrix<T>
    where
        for<'a> T: MatEmpty + Clone + serde::Deserialize<'a> + ToVectorValue + rkyv::Archive,
        for<'a> Robj: AsTypedSlice<'a, R>,
        for<'a> <T as rkyv::Archive>::Archived:
            rkyv::CheckBytes<rkyv::validation::validators::DefaultValidator<'a>>,
        [<T as rkyv::Archive>::Archived]:
            rkyv::DeserializeUnsized<[T], rkyv::de::deserializers::SharedDeserializeMap>,
        for<'a> &'a str: MatParse<T>,
        Rstr: MatParse<T>,
        OwnedMatrix<T>: FromRMatrix<T, R>,
    {
        let file = std::fs::File::open(&self.path).unwrap();
        if self.gz || self.file_type == FileType::Rdata {
            let decoder = flate2::read::GzDecoder::new(file);
            self.read_matrix_from_reader(std::io::BufReader::new(decoder), rkyv_validate)
        } else {
            self.read_matrix_from_reader(std::io::BufReader::new(file), rkyv_validate)
        }
    }

    pub fn read_matrix_from_reader<T, R>(
        &self,
        mut reader: impl std::io::Read,
        rkyv_validate: bool,
    ) -> OwnedMatrix<T>
    where
        for<'a> T: MatEmpty + Clone + serde::Deserialize<'a> + ToVectorValue + rkyv::Archive,
        for<'a> Robj: AsTypedSlice<'a, R>,
        for<'a> <T as rkyv::Archive>::Archived:
            rkyv::CheckBytes<rkyv::validation::validators::DefaultValidator<'a>>,
        [<T as rkyv::Archive>::Archived]:
            rkyv::DeserializeUnsized<[T], rkyv::de::deserializers::SharedDeserializeMap>,
        for<'a> &'a str: MatParse<T>,
        Rstr: MatParse<T>,
        OwnedMatrix<T>: FromRMatrix<T, R>,
    {
        let mut data = vec![];
        match self.file_type {
            FileType::Csv => {
                let mut reader = csv::Reader::from_reader(reader);
                let headers = reader.headers().unwrap();
                // Check if headers are numeric.
                for i in headers.iter() {
                    if i.parse::<f64>().is_err() {
                        break;
                    }
                    data.push(i.mat_parse());
                }
                let cols = headers.len();
                let _ = headers;
                for result in reader.records() {
                    let record = result.unwrap();
                    for field in record.iter() {
                        data.push(field.mat_parse());
                    }
                }
                OwnedMatrix::new(data.len() / cols, cols, data).transpose()
            },
            FileType::Tsv => {
                let mut reader = csv::ReaderBuilder::new()
                    .delimiter(b'\t')
                    .from_reader(reader);
                let headers = reader.headers().unwrap();
                // Check if headers are numeric.
                for i in headers.iter() {
                    if i.to_string().parse::<f64>().is_err() {
                        break;
                    }
                    data.push(i.mat_parse());
                }
                let cols = headers.len();
                let _ = headers;
                for result in reader.records() {
                    let record = result.unwrap();
                    for field in record.iter() {
                        data.push(field.mat_parse());
                    }
                }
                OwnedMatrix::new(data.len() / cols, cols, data).transpose()
            },
            FileType::Json => serde_json::from_reader(reader).unwrap(),
            FileType::Txt => {
                let mut reader = csv::ReaderBuilder::new()
                    .delimiter(b' ')
                    .from_reader(reader);
                let headers = reader.headers().unwrap();
                // Check if headers are numeric.
                for i in headers.iter() {
                    if i.to_string().parse::<f64>().is_err() {
                        break;
                    }
                    data.push(i.mat_parse());
                }
                let cols = headers.len();
                let _ = headers;
                for result in reader.records() {
                    let record = result.unwrap();
                    for field in record.iter() {
                        data.push(field.mat_parse());
                    }
                }
                OwnedMatrix::new(data.len() / cols, cols, data).transpose()
            },
            FileType::Rdata => {
                let mut buf = [0; 5];
                reader.read_exact(&mut buf).unwrap();
                if buf != *b"RDX3\n" {
                    panic!("Invalid RData file");
                }
                let obj =
                    Robj::from_reader(&mut reader, extendr_api::io::PstreamFormat::XdrFormat, None)
                        .unwrap();
                let mat = obj
                    .as_pairlist()
                    .unwrap()
                    .into_iter()
                    .next()
                    .unwrap()
                    .1
                    .as_matrix()
                    .unwrap();
                OwnedMatrix::from_rmatrix(mat)
            },
            FileType::Rkyv => {
                let mut bytes = vec![];
                reader.read_to_end(&mut bytes).unwrap();
                if rkyv_validate {
                    rkyv::from_bytes(&bytes).unwrap()
                } else {
                    unsafe { rkyv::from_bytes_unchecked(&bytes).unwrap() }
                }
            },
            FileType::Cbor => serde_cbor::from_reader(reader).unwrap(),
        }
    }

    pub fn write_transitory(&self, mat: &TransitoryMatrix) {
        match mat {
            TransitoryMatrix::Float(mat) => self.write_matrix(mat),
            TransitoryMatrix::Str(mat) => self.write_matrix(mat),
        }
    }

    pub fn write_matrix<T, R>(&self, mat: &OwnedMatrix<T>)
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
        let file = std::fs::File::create(&self.path).unwrap();
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
    ) where
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
                for i in 0..mat.rows {
                    writer
                        .write_record(
                            mat.data
                                .iter()
                                .skip(i)
                                .step_by(mat.cols)
                                .take(mat.cols)
                                .map(|x| x.to_string())
                                .collect::<Vec<String>>(),
                        )
                        .unwrap();
                }
            },
            FileType::Tsv => {
                let mut writer = csv::WriterBuilder::new()
                    .delimiter(b'\t')
                    .from_writer(writer);
                for i in 0..mat.rows {
                    writer
                        .write_record(
                            mat.data
                                .iter()
                                .skip(i)
                                .step_by(mat.cols)
                                .take(mat.cols)
                                .map(|x| x.to_string())
                                .collect::<Vec<String>>(),
                        )
                        .unwrap();
                }
            },
            FileType::Json => serde_json::to_writer(writer, &mat).unwrap(),
            FileType::Txt => {
                let mut writer = csv::WriterBuilder::new()
                    .delimiter(b' ')
                    .from_writer(writer);
                for i in 0..mat.rows {
                    writer
                        .write_record(
                            mat.data
                                .iter()
                                .skip(i)
                                .step_by(mat.cols)
                                .take(mat.cols)
                                .map(|x| x.to_string())
                                .collect::<Vec<String>>(),
                        )
                        .unwrap();
                }
            },
            FileType::Rdata => {
                let mat = ToRMatrix::to_rmatrix(mat);
                let pl = pairlist!(mat = mat);
                writer.write_all(b"RDX3\n").unwrap();
                pl.to_writer(
                    &mut writer,
                    extendr_api::io::PstreamFormat::XdrFormat,
                    3,
                    None,
                )
                .unwrap();
            },
            FileType::Rkyv => {
                let bytes = rkyv::to_bytes::<_, 256>(mat).unwrap();
                writer.write_all(&bytes).unwrap();
            },
            FileType::Cbor => serde_cbor::to_writer(writer, &mat).unwrap(),
        }
    }
}

impl FromStr for File {
    type Err = FileParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let path = PathBuf::from(s);
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
        let gz = extension.len() > 2 && extension[2] == "gz";
        let extension = extension[1];
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    /// Comma-separated values, row major.
    Csv,
    /// Tab-separated values, row major.
    Tsv,
    /// Serialized Matrix type.
    Json,
    /// Space-separated values, row major.
    Txt,
    /// RData file.
    Rdata,
    /// Serialized matrix type.
    Rkyv,
    /// Serialied matrix type.
    Cbor,
}

use std::convert::Infallible;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum FileParseError {
    #[error("no file name")]
    NoFileName,
    #[error("invalid file name")]
    InvalidFileName,
    #[error("No file extension")]
    NoFileExtension,
    #[error("unsupported file type: {0}")]
    UnsupportedFileType(String),
    #[error("invalid file extension: {0}")]
    InvalidFileExtension(String),
}

#[derive(Error, Debug)]
pub enum MatParseError {
    #[error("file parse error: {0}")]
    FileParseError(#[from] FileParseError),
    #[error("parse float error: {0}")]
    ParseFloatError(#[from] std::num::ParseFloatError),
    #[error("other error: {0}")]
    OtherError(String),
}

impl From<Infallible> for MatParseError {
    fn from(_: Infallible) -> Self {
        unreachable!("infallible")
    }
}

#[derive(Error, Debug)]
pub enum ConvertFileError {
    #[error("file parse error: {0}")]
    FileParseError(#[from] FileParseError),
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
}

#[derive(Error, Debug)]
pub enum CombineMatricesError {
    #[error("matrix dimensions do not match")]
    MatrixDimensionsMismatch,
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
}

#[derive(Error, Debug)]
pub enum ReadMatrixError {
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("csv error: {0}")]
    CsvError(#[from] csv::Error),
    #[error("invalid rdata file")]
    InvalidRdataFile,
    #[error("matrix parse error: {0}")]
    MatParseError(#[from] MatParseError),
    #[error("json error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("r error: {0}")]
    RError(String),
    #[error("rkyv error: {0}")]
    RkyvError(String),
    #[error("cbor error: {0}")]
    CborError(#[from] serde_cbor::Error),
}

impl From<extendr_api::Error> for ReadMatrixError {
    fn from(err: extendr_api::Error) -> Self {
        ReadMatrixError::RError(err.to_string())
    }
}

#[derive(Error, Debug)]
pub enum WriteMatrixError {
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("csv error: {0}")]
    CsvError(#[from] csv::Error),
    #[error("json error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("r error: {0}")]
    RError(String),
    #[error("rkyv error: {0}")]
    RkyvError(String),
    #[error("cbor error: {0}")]
    CborError(#[from] serde_cbor::Error),
}

impl From<extendr_api::Error> for WriteMatrixError {
    fn from(err: extendr_api::Error) -> Self {
        WriteMatrixError::RError(err.to_string())
    }
}

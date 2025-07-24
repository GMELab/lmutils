use std::convert::Infallible;

use rkyv::{
    de::deserializers::SharedDeserializeMapError,
    ser::serializers::{AllocScratchError, CompositeSerializerError, SharedSerializeMapError},
};

use crate::Join;

#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("no file name")]
    NoFileName,
    #[error("invalid file name")]
    InvalidFileName,
    #[error("No file extension")]
    NoFileExtension,
    #[error("incomplete file")]
    IncompleteFile,
    #[error("unsupported file type: {0}")]
    UnsupportedFileType(String),
    #[error("invalid file extension: {0}")]
    InvalidFileExtension(String),
    #[error("parse float error: {0}")]
    ParseFloatError(#[from] std::num::ParseFloatError),
    #[error("matrix dimensions do not match")]
    MatrixDimensionsMismatch,
    #[error("column names do not match")]
    ColumnNamesMismatch,
    #[error("row index {0} out of bounds")]
    RowIndexOutOfBounds(usize),
    #[error("column index {0} out of bounds")]
    ColumnIndexOutOfBounds(usize),
    #[error("column name {0} not found")]
    ColumnNameNotFound(String),
    #[error("missing column names")]
    MissingColumnNames,
    #[error("order length {0} does not match matrix length")]
    OrderLengthMismatch(usize),
    #[error("not all rows matched with join {0}")]
    NotAllRowsMatched(Join),
    #[error("invalid item type")]
    InvalidItemType,
    #[error("invalid rdata file")]
    InvalidRdataFile,
    #[error("invalid rds file")]
    InvalidRdsFile,
    #[error("invalid mat file")]
    InvalidMatFile,
    #[error("unsupported mat file version {0}")]
    UnsupportedMatFileVersion(u8),
    #[error("matrix in mat file too large")]
    MatrixTooLarge,
    #[error("unequal column lengths")]
    UnequalColumnLengths,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("r error: {0}")]
    R(String),
    #[error("rscript error: {0}")]
    Rscript(i32),
    #[error("rkyv deserialize error: {0}")]
    RkyvDeserialize(#[from] SharedDeserializeMapError),
    #[error("rkyv serialize error: {0}")]
    RkyvSerialize(
        #[from] CompositeSerializerError<Infallible, AllocScratchError, SharedSerializeMapError>,
    ),
    #[error("cbor error: {0}")]
    Cbor(#[from] serde_cbor::Error),
    #[error("regex error: {0}")]
    Regex(#[from] regex::Error),
    #[error("eigen error")]
    Eigen(faer::linalg::evd::EvdError),
    #[error("invalid scale length {0}")]
    InvalidScaleLength(usize),
    #[error("plink file missing chromosome on line {0}")]
    PlinkMissingChromosome(usize),
    #[error("plink file missing identifier on line {0}")]
    PlinkMissingIdentifier(usize),
    #[error("plink file missing position on line {0}")]
    PlinkMissingPosition(usize),
    #[error("plink file missing coordinate on line {0}")]
    PlinkMissingCoordinate(usize),
    #[error("plink file invalid coordinate on line {0}")]
    PlinkInvalidCoordinate(usize),
}

#[cfg(feature = "r")]
impl From<extendr_api::Error> for Error {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn from(err: extendr_api::Error) -> Self {
        Error::R(err.to_string())
    }
}

#[cfg(feature = "r")]
impl From<Error> for extendr_api::Error {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn from(e: Error) -> Self {
        extendr_api::Error::Other(e.to_string())
    }
}

impl From<faer::linalg::evd::EvdError> for Error {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn from(err: faer::linalg::evd::EvdError) -> Self {
        Error::Eigen(err)
    }
}

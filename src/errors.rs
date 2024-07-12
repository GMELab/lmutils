use std::convert::Infallible;

use thiserror::Error;

use crate::Join;

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
    #[error("write matrix error: {0}")]
    WriteMatrixError(#[from] WriteMatrixError),
}

#[derive(Error, Debug)]
pub enum CombineColumnsError {
    #[error("matrix dimensions do not match")]
    MatrixDimensionsMismatch,
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
}

#[derive(Error, Debug)]
pub enum CombineRowsError {
    #[error("matrix dimensions do not match")]
    MatrixDimensionsMismatch,
    #[error("column names do not match")]
    ColumnNamesMismatch,
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
}

#[derive(Error, Debug)]
pub enum RemoveRowsError {
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
    #[error("row index {0} out of bounds")]
    RowIndexOutOfBounds(usize),
}

#[derive(Error, Debug)]
pub enum RemoveColumnsError {
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
    #[error("column index {0} out of bounds")]
    ColumnIndexOutOfBounds(usize),
}

#[derive(Error, Debug)]
pub enum RemoveColumnByNameError {
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
    #[error("remove columns error: {0}")]
    RemoveColumnsError(#[from] RemoveColumnsError),
    #[error("column name {0} not found")]
    ColumnNameNotFound(String),
    #[error("missing column names")]
    MissingColumnNames,
}

#[derive(Error, Debug)]
pub enum SortByOrderError {
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
    #[error("order index {0} out of bounds")]
    RowIndexOutOfBounds(usize),
    #[error("order length {0} does not match matrix length")]
    OrderLengthMismatch(usize),
}

#[derive(Error, Debug)]
pub enum SortByColumnError {
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
    #[error("column index {0} out of bounds")]
    ColumnIndexOutOfBounds(usize),
    #[error("sort by order error: {0}")]
    SortByOrderError(#[from] SortByOrderError),
}

#[derive(Error, Debug)]
pub enum SortByColumnNameError {
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
    #[error("column name {0} not found")]
    ColumnNameNotFound(String),
    #[error("sort by column error: {0}")]
    SortByColumnError(#[from] SortByColumnError),
    #[error("missing column names")]
    MissingColumnNames,
}

#[derive(Error, Debug)]
pub enum DedupByColumnError {
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
    #[error("column index {0} out of bounds")]
    ColumnIndexOutOfBounds(usize),
}

#[derive(Error, Debug)]
pub enum DedupByColumnNameError {
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
    #[error("column name {0} not found")]
    ColumnNameNotFound(String),
    #[error("dedup by column error: {0}")]
    DedupByColumnError(#[from] DedupByColumnError),
    #[error("missing column names")]
    MissingColumnNames,
}

#[derive(Error, Debug)]
pub enum MatchToColumnError {
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
    #[error("column index {0} out of bounds")]
    ColumnIndexOutOfBounds(usize),
    #[error("not all rows matched with join {0}")]
    NotAllRowsMatched(Join),
}

#[derive(Error, Debug)]
pub enum MatchToByColumnNameError {
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
    #[error("column name {0} not found")]
    ColumnNameNotFound(String),
    #[error("match to column error: {0}")]
    MatchToColumnError(#[from] MatchToColumnError),
    #[error("missing column names")]
    MissingColumnNames,
}

#[derive(Error, Debug)]
pub enum JoinError {
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
    #[error("column index {0} out of bounds")]
    ColumnIndexOutOfBounds(usize),
    #[error("not all rows matched with join {0}")]
    NotAllRowsMatched(Join),
}

#[derive(Error, Debug)]
pub enum JoinByColumnNameError {
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
    #[error("column name {0} not found")]
    ColumnNameNotFound(String),
    #[error("join error: {0}")]
    JoinError(#[from] JoinError),
    #[error("missing column names")]
    MissingColumnNames,
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
    #[error("matrix from robj error: {0}")]
    MatrixFromRobjError(#[from] MatrixFromRobjError),
}

#[derive(Error, Debug)]
pub enum WriteMatrixError {
    #[error("read matrix error: {0}")]
    ReadMatrixError(#[from] ReadMatrixError),
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

#[derive(Error, Debug)]
pub enum MatrixFromRobjError {
    #[error("invalid item type")]
    InvalidItemType,
    #[error("r error: {0}")]
    RError(#[from] extendr_api::Error),
    #[error("file parse error: {0}")]
    FileParseError(#[from] FileParseError),
}

macro_rules! rerror {
    ($($err:ty),*) => {
        $(
            impl From<extendr_api::Error> for $err {
                fn from(err: extendr_api::Error) -> Self {
                    <$err>::RError(err.to_string())
                }
            }
        )*
    };
}

rerror!(ReadMatrixError, WriteMatrixError);

macro_rules! others {
    ($($err:ty),*) => {
        $(
            impl From<$err> for extendr_api::Error {
                fn from(err: $err) -> Self {
                    extendr_api::Error::Other(err.to_string())
                }
            }
        )*
    };
}

others!(
    FileParseError,
    MatParseError,
    ConvertFileError,
    CombineColumnsError,
    CombineRowsError,
    RemoveRowsError,
    RemoveColumnsError,
    RemoveColumnByNameError,
    SortByOrderError,
    SortByColumnError,
    SortByColumnNameError,
    DedupByColumnError,
    DedupByColumnNameError,
    MatchToColumnError,
    MatchToByColumnNameError,
    JoinError,
    JoinByColumnNameError,
    ReadMatrixError,
    WriteMatrixError,
    MatrixFromRobjError
);

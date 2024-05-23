use thiserror::Error;

#[derive(Error, Debug)]
pub enum FileParseError {
    #[error("No file name")]
    NoFileName,
    #[error("Invalid file name")]
    InvalidFileName,
    #[error("No file extension")]
    NoFileExtension,
    #[error("Unsupported file type: {0}")]
    UnsupportedFileType(String),
    #[error("Invalid file extension: {0}")]
    InvalidFileExtension(String),
}

#[derive(Error, Debug)]
pub enum ConvertFileError {
    #[error("file parse error: {0}")]
    FileParseError(#[from] FileParseError),
}

#[derive(Error, Debug)]
pub enum CombineMatricesError {
    #[error("Matrix dimensions do not match")]
    MatrixDimensionsMismatch,
}

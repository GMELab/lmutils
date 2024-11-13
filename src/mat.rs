use std::mem::MaybeUninit;

use cfg_if::cfg_if;

use crate::{Matrix, OwnedMatrix};

pub fn read_mat(mut reader: impl std::io::Read) -> Result<Matrix, crate::Error> {
    let mut prefix = [0; 3];
    reader.read_exact(&mut prefix)?;
    if prefix != [b'M', b'A', b'T'] {
        return Err(crate::Error::InvalidMatFile);
    }
    let mut version = [0; 1];
    reader.read_exact(&mut version)?;
    match version[0] {
        FloatMatrix::VERSION => FloatMatrix::read(reader),
        v => Err(crate::Error::UnsupportedMatFileVersion(v)),
    }
}

pub fn write_mat(mut writer: impl std::io::Write, mat: &mut Matrix) -> Result<(), crate::Error> {
    mat.into_owned()?;
    writer.write_all(b"MAT")?;
    writer.write_all(&[FloatMatrix::VERSION])?;
    FloatMatrix::write(writer, mat);
    Ok(())
}

trait Mat {
    const VERSION: u8;

    fn read(reader: impl std::io::Read) -> Result<Matrix, crate::Error>;
    fn write(writer: impl std::io::Write, mat: &mut Matrix) -> Result<(), crate::Error>;

    fn read_header(mut reader: impl std::io::Read) -> Result<(usize, usize), crate::Error> {
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
        Ok((nrows as usize, ncols as usize))
    }

    fn read_colnames(
        mut reader: impl std::io::Read,
        ncols: usize,
    ) -> Result<Option<Vec<String>>, crate::Error> {
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
                let mut slice =
                    unsafe { std::slice::from_raw_parts_mut(name.as_mut_ptr().cast::<u8>(), len) };
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
        Ok(colnames)
    }

    fn write_colnames(mut writer: impl std::io::Write, mat: &Matrix) -> Result<(), crate::Error> {
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
        Ok(())
    }
}

/// Format (little endian):
/// - 3 bytes: "MAT"
/// - 1 byte: version
/// - 8 bytes: nrows
/// - 8 bytes: ncols
/// - 1 byte: colnames flag
/// - if colnames flag is 1:
///  - for each column:
///    - 2 bytes: length of column name
///    - n bytes: column name
/// - nrows * ncols * 8 bytes: data
struct FloatMatrix;

impl Mat for FloatMatrix {
    const VERSION: u8 = 1;

    fn read(mut reader: impl std::io::Read) -> Result<Matrix, crate::Error> {
        let (nrows, ncols) = Self::read_header(&mut reader)?;
        let mut len = unsafe { nrows.unchecked_mul(ncols) };
        let colnames = Self::read_colnames(&mut reader, ncols)?;

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
                std::mem::transmute::<std::vec::Vec<std::mem::MaybeUninit<f64>>, std::vec::Vec<f64>>(
                    data,
                )
            },
            colnames,
        )))
    }

    fn write(mut writer: impl std::io::Write, mat: &mut Matrix) -> Result<(), crate::Error> {
        writer.write_all(&mat.nrows_loaded().to_le_bytes())?;
        writer.write_all(&mat.ncols_loaded().to_le_bytes())?;
        Self::write_colnames(&mut writer, mat)?;
        let data = mat.data()?;
        cfg_if!(
            if #[cfg(target_endian = "little")] {
                writer.write_all(unsafe {
                    std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 8)
                })?;
            } else {
                for val in data.iter() {
                    writer.write_all(&val.to_le_bytes())?;
                }
            }
        );
        Ok(())
    }
}

/// Format (little endian):
/// - 3 bytes: "MAT"
/// - 1 byte: version
/// - 8 bytes: nrows
/// - 8 bytes: ncols
/// - 1 byte: colnames flag
/// - if colnames flag is 1:
///   - for each column:
///     - 2 bytes: length of column name
///     - n bytes: column name
/// - ceil(nrows * ncols / 8) bytes: data
struct BinaryMatrix;

impl Mat for BinaryMatrix {
    const VERSION: u8 = 2;

    fn read(mut reader: impl std::io::Read) -> Result<Matrix, crate::Error> {
        let (nrows, ncols) = Self::read_header(&mut reader)?;
        let mut len = unsafe { nrows.unchecked_mul(ncols) };
        let colnames = Self::read_colnames(&mut reader, ncols)?;

        let mut data = vec![MaybeUninit::<f64>::uninit(); len];
        let mut buf = [0; 1];
        for i in 0..(len / 8) {
            reader.read_exact(&mut buf)?;
            for j in 0..8 {
                let val = (buf[0] >> j) & 1;
                unsafe {
                    *data.as_ptr().add((i * 8) + j).cast_mut().cast::<f64>() = val as f64;
                }
            }
        }
        if len % 8 != 0 {
            reader.read_exact(&mut buf)?;
            for j in 0..(len % 8) {
                let val = (buf[0] >> j) & 1;
                println!("{:?}", val);
                unsafe {
                    *data
                        .as_ptr()
                        .add(((len / 8) * 8) + j)
                        .cast_mut()
                        .cast::<f64>() = val as f64;
                }
            }
        }
        Ok(Matrix::Owned(OwnedMatrix::new(
            nrows,
            ncols,
            unsafe {
                std::mem::transmute::<std::vec::Vec<std::mem::MaybeUninit<f64>>, std::vec::Vec<f64>>(
                    data,
                )
            },
            colnames,
        )))
    }

    fn write(mut writer: impl std::io::Write, mat: &mut Matrix) -> Result<(), crate::Error> {
        writer.write_all(&mat.nrows_loaded().to_le_bytes())?;
        writer.write_all(&mat.ncols_loaded().to_le_bytes())?;
        Self::write_colnames(&mut writer, mat)?;
        let mut data = mat.data()?;
        let mut bits = 0u8;
        for chunk in data.chunks(8) {
            for (i, &val) in chunk.iter().enumerate() {
                bits |= (val as u8) << i;
            }
            writer.write_all(&[bits])?;
            bits = 0;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::OwnedMatrix;

    #[test]
    fn float_matrix() {
        let mut mat = Matrix::Owned(OwnedMatrix::new(
            2,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        ));
        let mut buf = Vec::new();
        FloatMatrix::write(&mut buf, &mut mat).unwrap();
        let mut cursor = Cursor::new(buf);
        let mat2 = FloatMatrix::read(&mut cursor).unwrap();
        assert_eq!(mat, mat2);
    }

    #[test]
    fn binary_matrix() {
        let mut mat = Matrix::Owned(OwnedMatrix::new(
            3,
            3,
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        ));
        let mut buf = Vec::new();
        BinaryMatrix::write(&mut buf, &mut mat).unwrap();
        let mut cursor = Cursor::new(buf);
        let mat2 = BinaryMatrix::read(&mut cursor).unwrap();
        assert_eq!(mat, mat2);
    }
}

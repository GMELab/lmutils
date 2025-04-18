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
        FloatMatrix::VERSION => FloatMatrix.read(reader),
        BinaryMatrix::VERSION => BinaryMatrix {
            zero: 0.0,
            one: 1.0,
        }
        .read(reader),
        BinaryColumnMatrix::VERSION => BinaryColumnMatrix.read(reader),
        v => Err(crate::Error::UnsupportedMatFileVersion(v)),
    }
}

pub fn write_mat(mut writer: impl std::io::Write, mat: &mut Matrix) -> Result<(), crate::Error> {
    mat.into_owned()?;
    writer.write_all(b"MAT")?;
    let ncols = mat.ncols_loaded();
    let nrows = mat.nrows_loaded();
    let data = mat.data()?;
    let mut unique: [u64; 2] = [data[0].to_bits(), 0];
    let mut iter = data.iter();
    // get second unique value
    for i in &mut iter {
        if i.to_bits() != unique[0] {
            unique[1] = i.to_bits();
            break;
        }
    }
    for i in &mut iter {
        if i.to_bits() != unique[0] && i.to_bits() != unique[1] {
            // more than two unique values, but could still be two unique per column so
            // we can't write as a float matrix yet
            for col in 0..ncols {
                let mut unique = [data[col * nrows].to_bits(), 0];
                let mut iter = data[col * nrows..(col + 1) * nrows].iter();
                for i in &mut iter {
                    if i.to_bits() != unique[0] {
                        unique[1] = i.to_bits();
                        break;
                    }
                }
                for i in &mut iter {
                    if i.to_bits() != unique[0] && i.to_bits() != unique[1] {
                        // more than two unique values in a column, can't write as a binary column
                        // matrix
                        writer.write_all(&[FloatMatrix::VERSION])?;
                        FloatMatrix.write(writer, mat);
                        return Ok(());
                    }
                }
            }
            writer.write_all(&[BinaryColumnMatrix::VERSION])?;
            BinaryColumnMatrix.write(writer, mat);
            return Ok(());
        }
    }
    // otherwise, write as binary matrix
    writer.write_all(&[BinaryMatrix::VERSION])?;
    BinaryMatrix {
        zero: f64::from_bits(unique[0]),
        one: f64::from_bits(unique[1]),
    }
    .write(writer, mat);
    Ok(())
}

trait Mat {
    const VERSION: u8;

    fn read(&self, reader: impl std::io::Read) -> Result<Matrix, crate::Error>;
    fn write(&self, writer: impl std::io::Write, mat: &mut Matrix) -> Result<(), crate::Error>;

    fn read_header(&self, mut reader: impl std::io::Read) -> Result<(usize, usize), crate::Error> {
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
        Ok((nrows as usize, ncols as usize))
    }

    fn read_colnames(
        &self,
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

    fn write_colnames(
        &self,
        mut writer: impl std::io::Write,
        mat: &Matrix,
    ) -> Result<(), crate::Error> {
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

    fn read(&self, mut reader: impl std::io::Read) -> Result<Matrix, crate::Error> {
        let (nrows, ncols) = self.read_header(&mut reader)?;
        let mut len = unsafe { nrows.unchecked_mul(ncols) };
        let colnames = self.read_colnames(&mut reader, ncols)?;

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

    fn write(&self, mut writer: impl std::io::Write, mat: &mut Matrix) -> Result<(), crate::Error> {
        writer.write_all(&mat.nrows_loaded().to_le_bytes())?;
        writer.write_all(&mat.ncols_loaded().to_le_bytes())?;
        self.write_colnames(&mut writer, mat)?;
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
/// - 8 bytes: 0-bit packed data
/// - 8 bytes: 1-bit packed data
/// - ceil(nrows * ncols / 8) bytes: data
struct BinaryMatrix {
    zero: f64,
    one: f64,
}

impl Mat for BinaryMatrix {
    const VERSION: u8 = 2;

    #[allow(clippy::uninit_vec)]
    fn read(&self, mut reader: impl std::io::Read) -> Result<Matrix, crate::Error> {
        let (nrows, ncols) = self.read_header(&mut reader)?;
        let mut len = unsafe { nrows.unchecked_mul(ncols) };
        let colnames = self.read_colnames(&mut reader, ncols)?;

        let mut zero = [0; 8];
        reader.read_exact(&mut zero)?;
        let zero = f64::from_le_bytes(zero);
        let mut one = [0; 8];
        reader.read_exact(&mut one)?;
        let one = f64::from_le_bytes(one);
        let mut buf = Vec::with_capacity(len / 8 + (len % 8 != 0) as usize);
        unsafe {
            buf.set_len(buf.capacity());
        }
        reader.read_exact(&mut buf)?;
        let mut data = Vec::with_capacity(len);
        let mut spare = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr(), len) };
        crate::unpack(spare, &buf, zero, one);
        unsafe {
            data.set_len(len);
        }
        Ok(Matrix::Owned(OwnedMatrix::new(
            nrows, ncols, data, colnames,
        )))
    }

    fn write(&self, mut writer: impl std::io::Write, mat: &mut Matrix) -> Result<(), crate::Error> {
        writer.write_all(&mat.nrows_loaded().to_le_bytes())?;
        writer.write_all(&mat.ncols_loaded().to_le_bytes())?;
        self.write_colnames(&mut writer, mat)?;
        writer.write_all(&self.zero.to_le_bytes())?;
        writer.write_all(&self.one.to_le_bytes())?;
        let mut data = mat.data()?;
        let mut buf = Vec::with_capacity(data.len() / 8 + (data.len() % 8 != 0) as usize);
        let mut spare = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr(), buf.capacity()) };
        crate::pack(spare, data, self.zero, self.one);
        unsafe {
            buf.set_len(buf.capacity());
        }
        writer.write_all(&buf)?;
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
/// - for each column:
///   - 8 bytes: 0-bit packed data
///   - 8 bytes: 1-bit packed data
///   - ceil(nrows / 8) bytes: data
struct BinaryColumnMatrix;

impl Mat for BinaryColumnMatrix {
    const VERSION: u8 = 3;

    #[allow(clippy::uninit_vec)]
    fn read(&self, mut reader: impl std::io::Read) -> Result<Matrix, crate::Error> {
        let (nrows, ncols) = self.read_header(&mut reader)?;
        let mut len = unsafe { nrows.unchecked_mul(ncols) };
        let colnames = self.read_colnames(&mut reader, ncols)?;

        let mut data = Vec::<f64>::with_capacity(len);
        for i in 0..ncols {
            let mut zero = [0; 8];
            reader.read_exact(&mut zero)?;
            let zero = f64::from_le_bytes(zero);
            let mut one = [0; 8];
            reader.read_exact(&mut one)?;
            let one = f64::from_le_bytes(one);
            let mut buf = Vec::with_capacity(nrows / 8 + (nrows % 8 != 0) as usize);
            unsafe {
                buf.set_len(buf.capacity());
            }
            reader.read_exact(&mut buf)?;
            let mut spare =
                unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().add(i * nrows), nrows) };
            crate::unpack(spare, &buf, zero, one);
        }
        unsafe {
            data.set_len(len);
        }
        Ok(Matrix::Owned(OwnedMatrix::new(
            nrows, ncols, data, colnames,
        )))
    }

    fn write(&self, mut writer: impl std::io::Write, mat: &mut Matrix) -> Result<(), crate::Error> {
        let nrows = mat.nrows_loaded();
        let ncols = mat.ncols_loaded();
        writer.write_all(&nrows.to_le_bytes())?;
        writer.write_all(&ncols.to_le_bytes())?;
        self.write_colnames(&mut writer, mat)?;
        let mut data = mat.data()?;
        for col in 0..ncols {
            let zero = data[col * nrows].to_bits();
            let mut one = 0.0;
            for i in data[col * nrows..(col + 1) * nrows].iter() {
                if i.to_bits() != zero {
                    one = *i;
                    break;
                }
            }
            writer.write_all(&zero.to_le_bytes())?;
            writer.write_all(&one.to_le_bytes())?;
            let mut buf = Vec::with_capacity(nrows / 8 + (nrows % 8 != 0) as usize);
            let mut spare =
                unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr(), buf.capacity()) };
            crate::pack(
                spare,
                &data[col * nrows..(col + 1) * nrows],
                f64::from_bits(zero),
                one,
            );
            unsafe {
                buf.set_len(buf.capacity());
            }
            writer.write_all(&buf)?;
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
            4,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        ));
        let mut buf = Vec::new();
        write_mat(&mut buf, &mut mat).unwrap();
        assert_eq!(
            buf,
            [
                // header
                b'M',
                b'A',
                b'T',
                FloatMatrix::VERSION,
                // nrows
                4,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                // ncols
                3,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                // colnames flag
                1,
                // colnames
                1,
                0,
                b'a',
                1,
                0,
                b'b',
                1,
                0,
                b'c',
                // data
                0,
                0,
                0,
                0,
                0,
                0,
                240,
                63,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                64,
                0,
                0,
                0,
                0,
                0,
                0,
                8,
                64,
                0,
                0,
                0,
                0,
                0,
                0,
                16,
                64,
                0,
                0,
                0,
                0,
                0,
                0,
                240,
                63,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                64,
                0,
                0,
                0,
                0,
                0,
                0,
                8,
                64,
                0,
                0,
                0,
                0,
                0,
                0,
                16,
                64,
                0,
                0,
                0,
                0,
                0,
                0,
                240,
                63,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                64,
                0,
                0,
                0,
                0,
                0,
                0,
                8,
                64,
                0,
                0,
                0,
                0,
                0,
                0,
                16,
                64,
            ]
        );
        let mut cursor = Cursor::new(buf);
        let mut mat2 = read_mat(&mut cursor).unwrap();
        assert_eq!(mat, mat2);
    }

    #[test]
    fn binary_matrix() {
        let mut mat = Matrix::Owned(OwnedMatrix::new(
            2,
            3,
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        ));
        let mut buf = Vec::new();
        write_mat(&mut buf, &mut mat).unwrap();
        assert_eq!(
            buf,
            [
                // header
                b'M',
                b'A',
                b'T',
                BinaryMatrix::VERSION,
                // nrows
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                // ncols
                3,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                // colnames flag
                1,
                // colnames
                1,
                0,
                b'a',
                1,
                0,
                b'b',
                1,
                0,
                b'c',
                // zero
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                // one
                0,
                0,
                0,
                0,
                0,
                0,
                240,
                63,
                // data
                0b00101010
            ]
        );
        let mut cursor = Cursor::new(buf);
        let mat2 = read_mat(&mut cursor).unwrap();
        assert_eq!(mat, mat2);
    }

    #[test]
    fn binary_column_matrix() {
        let mut mat = Matrix::Owned(OwnedMatrix::new(
            2,
            3,
            vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0],
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        ));
        let mut buf = Vec::new();
        write_mat(&mut buf, &mut mat).unwrap();
        assert_eq!(
            buf,
            [
                // header
                b'M',
                b'A',
                b'T',
                BinaryColumnMatrix::VERSION,
                // nrows
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                // ncols
                3,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                // colnames flag
                1,
                // colnames
                1,
                0,
                b'a',
                1,
                0,
                b'b',
                1,
                0,
                b'c',
                // column 1
                // zero
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                // one
                0,
                0,
                0,
                0,
                0,
                0,
                240,
                63,
                // data
                0b10,
                // column 2
                // zero
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                // one
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                64,
                // data
                0b10,
                // column 3
                // zero
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                // one
                0,
                0,
                0,
                0,
                0,
                0,
                8,
                64,
                // data
                0b10,
            ]
        );
        let mut cursor = Cursor::new(buf);
        let mat2 = read_mat(&mut cursor).unwrap();
        assert_eq!(mat, mat2);
    }
}

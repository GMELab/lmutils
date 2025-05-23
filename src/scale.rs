use faer::{Mat, MatMut, MatRef};

pub fn scale_scalar_in_place(mut mat: MatMut<f64>, scale: f64) {
    if mat.ncols() * mat.nrows() < 5_000_000 {
        scale_scalar_in_place_sync(mat.as_mut(), scale);
    } else {
        scale_scalar_in_place_par(mat.as_mut(), scale);
    }
}

pub fn scale_scalar_in_place_sync(mut mat: MatMut<f64>, scale: f64) {
    for i in 0..mat.ncols() {
        for j in 0..mat.nrows() {
            mat[(j, i)] *= scale;
        }
    }
}

pub fn scale_scalar_in_place_par(mut mat: MatMut<f64>, scale: f64) {
    ieu::execute(mat.ncols(), |i| {
        #[allow(invalid_reference_casting)]
        let slice = mat.as_ref().col(i).try_as_col_major().unwrap().as_slice();
        let mut col =
            unsafe { std::slice::from_raw_parts_mut(slice.as_ptr().cast_mut(), slice.len()) };
        for x in col.iter_mut() {
            *x *= scale;
        }
    });
}

pub fn scale_vector_in_place(mat: MatMut<f64>, scale: &[f64]) {
    assert_eq!(scale.len(), mat.ncols());
    if mat.ncols() * mat.nrows() < 5_000_000 {
        scale_vector_in_place_sync(mat, scale);
    } else {
        scale_vector_in_place_par(mat, scale);
    }
}

pub fn scale_vector_in_place_sync(mut mat: MatMut<f64>, scale: &[f64]) {
    assert_eq!(scale.len(), mat.ncols());
    for i in 0..mat.ncols() {
        let scale = scale[i];
        for j in 0..mat.nrows() {
            mat[(j, i)] *= scale;
        }
    }
}

pub fn scale_vector_in_place_par(mut mat: MatMut<f64>, scale: &[f64]) {
    assert_eq!(scale.len(), mat.ncols());
    ieu::execute(mat.ncols(), |i| {
        #[allow(invalid_reference_casting)]
        let slice = mat.as_ref().col(i).try_as_col_major().unwrap().as_slice();
        let mut col =
            unsafe { std::slice::from_raw_parts_mut(slice.as_ptr().cast_mut(), slice.len()) };
        let scale = scale[i];
        for x in col.iter_mut() {
            *x *= scale;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_scalar_in_place_sync() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut mat = MatMut::from_column_major_slice_mut(data.as_mut_slice(), 5, 1);
        scale_scalar_in_place_sync(mat, 2.0);
        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_scale_vector_in_place_sync() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut mat = MatMut::from_column_major_slice_mut(data.as_mut_slice(), 5, 2);
        scale_vector_in_place_sync(mat, &[2.0, 3.0]);
        assert_eq!(
            data,
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 18.0, 21.0, 24.0, 27.0, 30.0]
        );
    }

    #[test]
    fn test_scale_scalar_in_place_par() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut mat = MatMut::from_column_major_slice_mut(data.as_mut_slice(), 5, 1);
        scale_scalar_in_place_par(mat, 2.0);
        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_scale_vector_in_place_par() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut mat = MatMut::from_column_major_slice_mut(data.as_mut_slice(), 5, 2);
        scale_vector_in_place_par(mat, &[2.0, 3.0]);
        assert_eq!(
            data,
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 18.0, 21.0, 24.0, 27.0, 30.0]
        );
    }
}

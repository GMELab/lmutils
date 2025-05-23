use faer::{Mat, MatMut, MatRef};

pub fn scale_scalar(mat: MatRef<f64>, scale: f64) -> Mat<f64> {
    let mut mat = mat.to_owned();
    scale_scalar_in_place(mat.as_mut(), scale);
    mat
}

pub fn scale_scalar_in_place(mut mat: MatMut<f64>, scale: f64) {
    if mat.ncols() * mat.nrows() < 5_000_000 {
        scale_scalar_in_place_sync(mat.as_mut(), scale);
    } else {
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
}

pub fn scale_scalar_in_place_sync(mut mat: MatMut<f64>, scale: f64) {
    for i in 0..mat.ncols() {
        for j in 0..mat.nrows() {
            mat[(j, i)] *= scale;
        }
    }
}

pub fn scale_vector(mat: MatRef<f64>, scale: &[f64]) -> Mat<f64> {
    assert_eq!(scale.len(), mat.ncols());
    let mut mat = mat.to_owned();
    scale_vector_in_place(mat.as_mut(), scale);
    mat
}

pub fn scale_vector_in_place(mat: MatMut<f64>, scale: &[f64]) {
    assert_eq!(scale.len(), mat.ncols());
    if mat.ncols() * mat.nrows() < 5_000_000 {
        scale_vector_in_place_sync(mat, scale);
    } else {
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

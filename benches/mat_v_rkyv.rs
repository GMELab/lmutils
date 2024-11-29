use diol::prelude::*;
use lmutils::{File, IntoMatrix, OwnedMatrix};

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        list![
            write_rkyv,
            write_rkyv_gz,
            write_mat_float,
            write_mat_float_gz,
            write_mat_binary,
            write_mat_binary_gz,
            load_rkyv,
            load_rkyv_gz,
            load_mat_float,
            load_mat_float_gz,
            load_mat_binary,
            load_mat_binary_gz,
        ],
        [100, 1000],
    );
    bench.run()?;
    Ok(())
}

fn write_rkyv(bencher: Bencher, len: usize) {
    let mut mat = OwnedMatrix::new(
        len,
        len,
        (0..len * len).map(|x| x as f64).collect::<Vec<_>>(),
        Some((0..len).map(|x| x.to_string()).collect::<Vec<_>>()),
    )
    .into_matrix();
    let file = File::new("mat.rkyv", lmutils::FileType::Rkyv, false);
    bencher.bench(|| {
        file.write(&mut mat).unwrap();
    });
}

fn write_mat_float(bencher: Bencher, len: usize) {
    let mut mat = OwnedMatrix::new(
        len,
        len,
        (0..len * len).map(|x| x as f64).collect::<Vec<_>>(),
        Some((0..len).map(|x| x.to_string()).collect::<Vec<_>>()),
    )
    .into_matrix();
    let file = File::new("mat.mat", lmutils::FileType::Mat, false);
    bencher.bench(|| {
        file.write(&mut mat).unwrap();
    });
}

fn write_rkyv_gz(bencher: Bencher, len: usize) {
    let mut mat = OwnedMatrix::new(
        len,
        len,
        (0..len * len).map(|x| x as f64).collect::<Vec<_>>(),
        Some((0..len).map(|x| x.to_string()).collect::<Vec<_>>()),
    )
    .into_matrix();
    let file = File::new("mat.rkyv.gz", lmutils::FileType::Rkyv, true);
    bencher.bench(|| {
        file.write(&mut mat).unwrap();
    });
}

fn write_mat_float_gz(bencher: Bencher, len: usize) {
    let mut mat = OwnedMatrix::new(
        len,
        len,
        (0..len * len).map(|x| x as f64).collect::<Vec<_>>(),
        Some((0..len).map(|x| x.to_string()).collect::<Vec<_>>()),
    )
    .into_matrix();
    let file = File::new("mat.mat.gz", lmutils::FileType::Mat, true);
    bencher.bench(|| {
        file.write(&mut mat).unwrap();
    });
}

fn load_rkyv(bencher: Bencher, len: usize) {
    let mut mat = OwnedMatrix::new(
        len,
        len,
        (0..len * len).map(|x| x as f64).collect::<Vec<_>>(),
        Some((0..len).map(|x| x.to_string()).collect::<Vec<_>>()),
    )
    .into_matrix();
    let file = File::new("mat.rkyv", lmutils::FileType::Rkyv, false);
    file.write(&mut mat).unwrap();
    bencher.bench(|| {
        file.read().unwrap();
    });
}

fn load_mat_float(bencher: Bencher, len: usize) {
    let mut mat = OwnedMatrix::new(
        len,
        len,
        (0..len * len).map(|x| x as f64).collect::<Vec<_>>(),
        Some((0..len).map(|x| x.to_string()).collect::<Vec<_>>()),
    )
    .into_matrix();
    let file = File::new("mat.mat", lmutils::FileType::Mat, false);
    file.write(&mut mat).unwrap();
    bencher.bench(|| {
        file.read().unwrap();
    });
}

fn load_rkyv_gz(bencher: Bencher, len: usize) {
    let mut mat = OwnedMatrix::new(
        len,
        len,
        (0..len * len).map(|x| x as f64).collect::<Vec<_>>(),
        Some((0..len).map(|x| x.to_string()).collect::<Vec<_>>()),
    )
    .into_matrix();
    let file = File::new("mat.rkyv.gz", lmutils::FileType::Rkyv, true);
    file.write(&mut mat).unwrap();
    bencher.bench(|| {
        file.read().unwrap();
    });
}

fn load_mat_float_gz(bencher: Bencher, len: usize) {
    let mut mat = OwnedMatrix::new(
        len,
        len,
        (0..len * len).map(|x| x as f64).collect::<Vec<_>>(),
        Some((0..len).map(|x| x.to_string()).collect::<Vec<_>>()),
    )
    .into_matrix();
    let file = File::new("mat.mat.gz", lmutils::FileType::Mat, true);
    file.write(&mut mat).unwrap();
    bencher.bench(|| {
        file.read().unwrap();
    });
}

fn write_mat_binary(bencher: Bencher, len: usize) {
    let mut mat = OwnedMatrix::new(
        len,
        len,
        (0..len * len)
            .map(|x| if x % 2 == 0 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>(),
        Some((0..len).map(|x| x.to_string()).collect::<Vec<_>>()),
    )
    .into_matrix();
    let file = File::new("mat.mat", lmutils::FileType::Mat, false);
    bencher.bench(|| {
        file.write(&mut mat).unwrap();
    });
}

fn load_mat_binary(bencher: Bencher, len: usize) {
    let mut mat = OwnedMatrix::new(
        len,
        len,
        (0..len * len)
            .map(|x| if x % 2 == 0 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>(),
        Some((0..len).map(|x| x.to_string()).collect::<Vec<_>>()),
    )
    .into_matrix();
    let file = File::new("mat.mat", lmutils::FileType::Mat, false);
    file.write(&mut mat).unwrap();
    bencher.bench(|| {
        file.read().unwrap();
    });
}

fn write_mat_binary_gz(bencher: Bencher, len: usize) {
    let mut mat = OwnedMatrix::new(
        len,
        len,
        (0..len * len)
            .map(|x| if x % 2 == 0 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>(),
        Some((0..len).map(|x| x.to_string()).collect::<Vec<_>>()),
    )
    .into_matrix();
    let file = File::new("mat.mat.gz", lmutils::FileType::Mat, true);
    bencher.bench(|| {
        file.write(&mut mat).unwrap();
    });
}

fn load_mat_binary_gz(bencher: Bencher, len: usize) {
    let mut mat = OwnedMatrix::new(
        len,
        len,
        (0..len * len)
            .map(|x| if x % 2 == 0 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>(),
        Some((0..len).map(|x| x.to_string()).collect::<Vec<_>>()),
    )
    .into_matrix();
    let file = File::new("mat.mat.gz", lmutils::FileType::Mat, true);
    file.write(&mut mat).unwrap();
    bencher.bench(|| {
        file.read().unwrap();
    });
}

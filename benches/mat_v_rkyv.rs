use diol::prelude::*;
use lmutils::{File, IntoMatrix, OwnedMatrix};

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args());
    bench.register_many(
        list![
            write_rkyv,
            write_mat_float,
            write_mat_binary,
            load_rkyv,
            load_mat_float,
            load_mat_binary
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

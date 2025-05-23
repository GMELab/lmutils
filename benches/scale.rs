use diol::prelude::*;

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        list![scalar_sync, scalar, vector_sync, vector],
        [
            (10, 10),
            (100, 10),
            (1_000, 10),
            (10_000, 10),
            (100_000, 10),
            (10, 100),
            (100, 100),
            (1_000, 100),
            (10_000, 100),
            (100_000, 100),
            (10, 1_000),
            (100, 1_000),
            (1_000, 1_000),
            (10_000, 1_000),
            (100_000, 1_000),
            (10, 10_000),
            (100, 10_000),
            (1_000, 10_000),
            (10_000, 10_000),
            (100_000, 10_000),
        ],
    );
    bench.run()?;
    Ok(())
}

fn scalar_sync(bencher: Bencher, (nrows, ncols): (usize, usize)) {
    let mut data = (0..nrows * ncols).map(|_| 1.0).collect::<Vec<_>>();
    let mut mat = faer::MatMut::from_column_major_slice_mut(&mut data, nrows, ncols);
    let scale = 2.0;
    bencher.bench(|| {
        lmutils::scale_scalar_in_place_sync(mat.as_mut(), scale);
    });
}

fn scalar(bencher: Bencher, (nrows, ncols): (usize, usize)) {
    let mut data = (0..nrows * ncols).map(|_| 1.0).collect::<Vec<_>>();
    let mut mat = faer::MatMut::from_column_major_slice_mut(&mut data, nrows, ncols);
    let scale = 2.0;
    bencher.bench(|| lmutils::scale_scalar_in_place(mat.as_mut(), scale));
}

fn vector_sync(bencher: Bencher, (nrows, ncols): (usize, usize)) {
    let mut data = (0..nrows * ncols).map(|_| 1.0).collect::<Vec<_>>();
    let mut mat = faer::MatMut::from_column_major_slice_mut(&mut data, nrows, ncols);
    let scale = (0..ncols).map(|x| x as f64).collect::<Vec<_>>();
    bencher.bench(|| lmutils::scale_vector_in_place_sync(mat.as_mut(), &scale));
}

fn vector(bencher: Bencher, (nrows, ncols): (usize, usize)) {
    let mut data = (0..nrows * ncols).map(|_| 1.0).collect::<Vec<_>>();
    let mut mat = faer::MatMut::from_column_major_slice_mut(&mut data, nrows, ncols);
    let scale = (0..ncols).map(|x| x as f64).collect::<Vec<_>>();
    bencher.bench(|| lmutils::scale_vector_in_place(mat.as_mut(), &scale));
}

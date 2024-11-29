use std::{io::Read, mem::MaybeUninit};

use diol::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        list![
            expand_packing,
            expand_packing_preallocated,
            expand_packing_parallel
        ],
        [100, 1000, 10000, 100000, 1000000, 10000000, 100000000],
    );
    bench.run()?;
    Ok(())
}

fn expand_packing(bencher: Bencher, len: usize) {
    let data = (0..len)
        .map(|x| if x % 2 == 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let mut packed = Vec::with_capacity(len / 8 + (len % 8 != 0) as usize);
    let mut bits = 0u8;
    for chunk in data.chunks(8) {
        for (i, &val) in chunk.iter().enumerate() {
            bits |= (val as u8) << i;
        }
        packed.push(bits);
        bits = 0;
    }
    bencher.bench(|| {
        let mut reader = std::io::Cursor::new(&packed);
        let data = vec![MaybeUninit::<f64>::uninit(); len];
        let mut buf = [0; 1];
        for i in 0..(len / 8) {
            reader.read_exact(&mut buf).unwrap();
            for j in 0..8 {
                let val = (buf[0] >> j) & 1;
                unsafe {
                    *data.as_ptr().add(i + j).cast_mut().cast::<f64>() = val as f64;
                }
            }
        }
        if len % 8 != 0 {
            reader.read_exact(&mut buf).unwrap();
            for j in 0..(len % 8) {
                let val = (buf[0] >> j) & 1;
                unsafe {
                    *data.as_ptr().add(len / 8 + j).cast_mut().cast::<f64>() = val as f64;
                }
            }
        }
    });
}

fn expand_packing_preallocated(bencher: Bencher, len: usize) {
    let data = (0..len)
        .map(|x| if x % 2 == 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let mut packed = Vec::with_capacity(len / 8 + (len % 8 != 0) as usize);
    let mut bits = 0u8;
    for chunk in data.chunks(8) {
        for (i, &val) in chunk.iter().enumerate() {
            bits |= (val as u8) << i;
        }
        packed.push(bits);
        bits = 0;
    }
    let data = vec![MaybeUninit::<f64>::uninit(); len];
    bencher.bench(|| {
        let mut reader = std::io::Cursor::new(&packed);
        let mut buf = [0; 1];
        for i in 0..(len / 8) {
            reader.read_exact(&mut buf).unwrap();
            for j in 0..8 {
                let val = (buf[0] >> j) & 1;
                unsafe {
                    *data.as_ptr().add(i + j).cast_mut().cast::<f64>() = val as f64;
                }
            }
        }
        if len % 8 != 0 {
            reader.read_exact(&mut buf).unwrap();
            for j in 0..(len % 8) {
                let val = (buf[0] >> j) & 1;
                unsafe {
                    *data.as_ptr().add(len / 8 + j).cast_mut().cast::<f64>() = val as f64;
                }
            }
        }
    });
}

fn expand_packing_parallel(bencher: Bencher, len: usize) {
    let data = (0..len)
        .map(|x| if x % 2 == 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let mut packed = Vec::with_capacity(len / 8 + (len % 8 != 0) as usize);
    let mut bits = 0u8;
    for chunk in data.chunks(8) {
        for (i, &val) in chunk.iter().enumerate() {
            bits |= (val as u8) << i;
        }
        packed.push(bits);
        bits = 0;
    }
    let data = vec![MaybeUninit::<f64>::uninit(); len];
    bencher.bench(|| {
        let mut reader = std::io::Cursor::new(&packed);
        let mut buf = [0; 1];
        (0..(len / 8)).into_par_iter().for_each(|i| {
            for j in 0..8 {
                let val = (buf[0] >> j) & 1;
                unsafe {
                    *data.as_ptr().add(i + j).cast_mut().cast::<f64>() = val as f64;
                }
            }
        });
        if len % 8 != 0 {
            reader.read_exact(&mut buf).unwrap();
            for j in 0..(len % 8) {
                let val = (buf[0] >> j) & 1;
                unsafe {
                    *data.as_ptr().add(len / 8 + j).cast_mut().cast::<f64>() = val as f64;
                }
            }
        }
    });
}

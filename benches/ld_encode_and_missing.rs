use aligned_vec::AVec;
use diol::prelude::*;
use lmutils::ld::{
    encode_and_missing::{encode_and_missing_avx512, encode_and_missing_naive},
    LD_BLOCK_SIZE,
};

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        list![
            naive, // sse, avx2,
            avx512
        ],
        [64, 640, 6400, 64000, 640000, 6400000, 64000000],
    );
    bench.run()?;
    Ok(())
}

fn data(len: usize) -> AVec<u8> {
    let mut vec = AVec::with_capacity(LD_BLOCK_SIZE, len);
    vec.resize(len, 0b00011011);
    vec
}

fn naive(bencher: Bencher, len: usize) {
    let data = data(len);
    bencher.bench(|| {
        std::hint::black_box(encode_and_missing_naive(&mut data.clone()));
    });
}

// fn sse(bencher: Bencher, len: usize) {
//     let data = data(len);
//     bencher.bench(std::hint::black_box(|| {
//         if is_x86_feature_detected!("sse4.1") {
//             get_maf_sse4(&data, len as u64, ((len / 4) * 3) as u64);
//         }
//     }));
// }
//
// fn avx2(bencher: Bencher, len: usize) {
//     let data = data(len);
//     bencher.bench(|| {
//         if is_x86_feature_detected!("avx2") {
//             get_maf_avx2(&data, len as u64, ((len / 4) * 3) as u64);
//         }
//     });
// }

fn avx512(bencher: Bencher, len: usize) {
    let data = data(len);
    bencher.bench(|| {
        if is_x86_feature_detected!("avx512f") {
            encode_and_missing_avx512(&mut data.clone());
        }
    });
}

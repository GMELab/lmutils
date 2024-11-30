//! This module provides utilities for speedily packing and unpacking bit-packed
//! data. Much of the unpacking SIMD and inline assembly code is adapted from
//! MIT licensed code provided by sarah quiñones el kazdadi, massive thanks to
//! her for the help!

use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

const MIN_CHUNK_SIZE: usize = 16384;

// convert from bits to zero or one
pub fn unpack(out: &mut [f64], bytes: &[u8], zero: f64, one: f64) {
    if is_x86_feature_detected!("avx512f") {
        unpack_avx512(out, bytes, zero, one);
    } else if let Some(simd) = pulp::x86::V3::try_new() {
        unpack_avx2(simd, out, bytes, zero, one);
    } else {
        unpack_naive(out, bytes, zero, one);
    }
}

pub fn unpack_avx512(out: &mut [f64], bytes: &[u8], zero: f64, one: f64) {
    let threads = rayon::current_num_threads();
    let chunk_size = (bytes.len() / threads / 8 + 1) * 8;
    if chunk_size < MIN_CHUNK_SIZE {
        unpack_avx512(out, bytes, zero, one);
    } else {
        unpack_avx512_par(chunk_size, out, bytes, zero, one);
    }
}

pub fn unpack_avx512_sync(out: &mut [f64], bytes: &[u8], zero: f64, one: f64) {
    unsafe {
        core::arch::asm! {
            "vbroadcastsd zmm0, xmm0",
            "vbroadcastsd zmm1, xmm1",
            "test rax, rax",
            "jz 3f",
                "2:",
                "kmovb k1, byte ptr [rsi]",
                "vmovapd zmm2, zmm0",
                "vmovapd zmm2{{k1}}, zmm1",
                "vmovupd zmmword ptr [rdi], zmm2",

                "add rsi, 1",
                "add rdi, 64",
                "dec rax",
                "jnz 2b",
            "3:",
            "vzeroupper",

            inout("xmm0") zero => _,
            inout("xmm1") one => _,
            out("xmm2") _,
            inout("rax") out.len() / 8 => _,
            inout("rsi") bytes.as_ptr() => _,
            inout("rdi") out.as_mut_ptr() => _,
        }
    };
    let bits = out.len();
    unpack_naive_sync(
        out[(bits / 8 * 8)..].as_mut(),
        &bytes[(bits / 8)..],
        zero,
        one,
    );
}

pub fn unpack_avx512_par(chunk_size: usize, out: &mut [f64], bytes: &[u8], zero: f64, one: f64) {
    bytes
        .par_chunks(chunk_size)
        .zip(out.par_chunks_mut(8 * chunk_size))
        .for_each(|(chunk, out)| {
            unpack_avx512_sync(out, chunk, zero, one);
        });
}

pub fn unpack_avx2(simd: pulp::x86::V3, out: &mut [f64], bytes: &[u8], zero: f64, one: f64) {
    let threads = rayon::current_num_threads();
    let chunk_size = (bytes.len() / threads / 16 + 1) * 16;
    if chunk_size < MIN_CHUNK_SIZE {
        unpack_avx2_sync(simd, out, bytes, zero, one);
    } else {
        unpack_avx2_par(chunk_size, simd, out, bytes, zero, one);
    }
}

pub fn unpack_avx2_sync(simd: pulp::x86::V3, out: &mut [f64], bytes: &[u8], zero: f64, one: f64) {
    struct Impl<'a> {
        simd:  pulp::x86::V3,
        out:   &'a mut [f64],
        bytes: &'a [u8],
        zero:  f64,
        one:   f64,
    }
    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                out,
                bytes,
                zero,
                one,
            } = self;

            let (out128, out_tail) = pulp::as_arrays_mut::<128, _>(out);
            let (bytes16, bytes_tail) = pulp::as_arrays::<16, _>(bytes);

            for (out, bytes) in std::iter::zip(out128, bytes16) {
                let mut bytes = pulp::cast(*bytes);
                let (out0, out1) = out.split_at_mut(64);
                let out0 = pulp::as_arrays_mut::<2, _>(out0).0;
                let out1 = pulp::as_arrays_mut::<2, _>(out1).0;

                let zeros = simd.splat_f64x2(zero);
                let ones = simd.splat_f64x2(one);
                for (o0, o1) in std::iter::zip(out0.iter_mut().rev(), out1.iter_mut().rev()) {
                    let b0 = bytes;
                    let b1 = simd.shl_const_u64x2::<1>(b0);
                    bytes = simd.shl_const_u64x2::<1>(b1);

                    // out0[last], out1[last]
                    let b0 = unsafe { core::mem::transmute::<pulp::u64x2, pulp::m64x2>(b0) };
                    // out0[last-1], out1[last-1]
                    let b1 = unsafe { core::mem::transmute::<pulp::u64x2, pulp::m64x2>(b1) };

                    let f0 = simd.select_f64x2(b0, ones, zeros);
                    let f1 = simd.select_f64x2(b1, ones, zeros);

                    *o0 = pulp::cast(simd.sse2._mm_unpacklo_pd(pulp::cast(f1), pulp::cast(f0)));
                    *o1 = pulp::cast(simd.sse2._mm_unpackhi_pd(pulp::cast(f1), pulp::cast(f0)));
                }
            }

            if !out_tail.is_empty() {
                unpack_naive_sync(out_tail, bytes_tail, zero, one);
            }
        }
    }
    simd.vectorize(Impl {
        simd,
        out,
        bytes,
        zero,
        one,
    });
}

pub fn unpack_avx2_par(
    chunk_size: usize,
    simd: pulp::x86::V3,
    out: &mut [f64],
    bytes: &[u8],
    zero: f64,
    one: f64,
) {
    bytes
        .par_chunks(chunk_size)
        .zip(out.par_chunks_mut(8 * chunk_size))
        .for_each(|(chunk, out)| {
            unpack_avx2_sync(simd, out, chunk, zero, one);
        });
}

pub fn unpack_naive(out: &mut [f64], bytes: &[u8], zero: f64, one: f64) {
    let threads = rayon::current_num_threads();
    let chunk_size = bytes.len() / threads;
    if chunk_size < MIN_CHUNK_SIZE {
        unpack_naive_sync(out, bytes, zero, one);
    } else {
        unpack_naive_par(chunk_size, out, bytes, zero, one);
    }
}

pub fn unpack_naive_sync(out: &mut [f64], bytes: &[u8], zero: f64, one: f64) {
    for i in 0..out.len() {
        out[i] = if ((bytes[i / 8] >> (i % 8)) & 1) == 1 {
            one
        } else {
            zero
        };
    }
}

pub fn unpack_naive_par(chunk_size: usize, out: &mut [f64], bytes: &[u8], zero: f64, one: f64) {
    bytes
        .par_chunks(chunk_size)
        .zip(out.par_chunks_mut(8 * chunk_size))
        .for_each(|(chunk, out)| {
            unpack_naive_sync(out, chunk, zero, one);
        });
}

pub fn pack(out: &mut [u8], data: &[f64], zero: f64, one: f64) {
    if is_x86_feature_detected!("avx512f") {
        pack_avx512(out, data, zero, one);
    } else if let Some(simd) = pulp::x86::V3::try_new() {
        pack_avx2(simd, out, data, zero, one);
    } else {
        pack_naive(out, data, zero, one);
    }
}

pub fn pack_avx512(out: &mut [u8], data: &[f64], zero: f64, one: f64) {
    let threads = rayon::current_num_threads();
    let chunk_size = (data.len() / threads / 8 + 1) * 8;
    if chunk_size < MIN_CHUNK_SIZE {
        pack_avx512_sync(out, data, zero, one);
    } else {
        pack_avx512_par(chunk_size, out, data, zero, one);
    }
}

pub fn pack_avx512_sync(out: &mut [u8], data: &[f64], zero: f64, one: f64) {
    let bits = data.len();
    unsafe {
        core::arch::asm! {
            "vbroadcastsd zmm0, xmm0", // 8 copies of zero from xmm0 to zmm0
            "test rax, rax",
            "jz 3f",
                "2:",
                "vmovupd zmmword ptr [rsi], zmm1", // move the next 8 f64s into the input
                "vcmpneqpd k1, zmm1, zmm0", // compare the input to zero
                "kmovb byte ptr [rdi], k1", // move the next output byte into k1

                "add rsi, 64", // increment the input pointer
                "add rdi, 1", // increment the output pointer
                "dec rax", // decrement the counter
                "jz 3f", // if the counter is zero, jump to the
            "3:",

            inout("xmm0") zero => _,
            inout("xmm1") one => _,
            inout("rax") bits / 8 => _,
            inout("rsi") data.as_ptr() => _,
            inout("rdi") out.as_mut_ptr() => _,
        }
    };
    pack_naive_sync(
        out[(bits / 8)..].as_mut(),
        &data[(bits / 8 * 8)..],
        zero,
        one,
    );
}

pub fn pack_avx512_par(chunk_size: usize, out: &mut [u8], data: &[f64], zero: f64, one: f64) {
    data.par_chunks(chunk_size)
        .zip(out.par_chunks_mut(chunk_size / 8))
        .for_each(|(data, out)| {
            pack_avx512_sync(out, data, zero, one);
        });
}

pub fn pack_avx2(simd: pulp::x86::V3, out: &mut [u8], data: &[f64], zero: f64, one: f64) {
    let threads = rayon::current_num_threads();
    let chunk_size = (data.len() / threads / 8 + 1) * 8;
    if chunk_size < MIN_CHUNK_SIZE {
        pack_avx2_sync(simd, out, data, zero, one);
    } else {
        pack_avx2_par(chunk_size, simd, out, data, zero, one);
    }
}

pub fn pack_avx2_sync(simd: pulp::x86::V3, out: &mut [u8], data: &[f64], zero: f64, one: f64) {
    struct Impl<'a> {
        simd: pulp::x86::V3,
        out:  &'a mut [u8],
        data: &'a [f64],
        zero: f64,
        one:  f64,
    }
    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                out,
                data,
                zero,
                one,
            } = self;

            let (out16, out_tail) = pulp::as_arrays_mut::<16, _>(out);
            let (data128, data_tail) = pulp::as_arrays::<128, _>(data);

            for (out, data) in std::iter::zip(out16, data128) {
                let data = pulp::as_arrays::<8, _>(data).0;
                let zeros = simd.splat_f64x4(zero);
                let ones = simd.splat_f64x4(one);
                for (o, d) in std::iter::zip(out.iter_mut().rev(), data.iter().rev()) {
                    let d = pulp::as_arrays::<4, _>(d).0;
                    let d0 = pulp::cast(d[0]);
                    let d1 = pulp::cast(d[1]);

                    let f0 = simd.cmp_eq_f64x4(d0, ones);
                    let f1 = simd.cmp_eq_f64x4(d1, ones);

                    let f0 = simd.avx._mm256_movemask_pd(pulp::cast(f0));
                    let f1 = simd.avx._mm256_movemask_pd(pulp::cast(f1));
                    *o = (f0 as u8) | ((f1 as u8) << 4);
                }
            }

            if !out_tail.is_empty() {
                pack_naive_sync(out_tail, data_tail, zero, one);
            }
        }
    }
    simd.vectorize(Impl {
        simd,
        out,
        data,
        zero,
        one,
    });
}

pub fn pack_avx2_par(
    chunk_size: usize,
    simd: pulp::x86::V3,
    out: &mut [u8],
    data: &[f64],
    zero: f64,
    one: f64,
) {
    data.par_chunks(chunk_size)
        .zip(out.par_chunks_mut(chunk_size / 8))
        .for_each(|(data, out)| {
            pack_avx2_sync(simd, out, data, zero, one);
        });
}

pub fn pack_naive(out: &mut [u8], data: &[f64], zero: f64, one: f64) {
    let threads = rayon::current_num_threads();
    let chunk_size = data.len() / threads;
    if chunk_size < MIN_CHUNK_SIZE {
        pack_naive_sync(out, data, zero, one);
    } else {
        pack_naive_par(chunk_size, out, data, zero, one);
    }
}

pub fn pack_naive_sync(out: &mut [u8], data: &[f64], zero: f64, one: f64) {
    for i in 0..data.len() {
        out[i / 8] |= if data[i] == one { 1 << (i % 8) } else { 0 };
    }
}

pub fn pack_naive_par(chunk_size: usize, out: &mut [u8], data: &[f64], zero: f64, one: f64) {
    data.par_chunks(chunk_size)
        .zip(out.par_chunks_mut(chunk_size / 8))
        .for_each(|(data, out)| {
            pack_naive_sync(out, data, zero, one);
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    const BYTES: usize = 2101;

    fn bytes() -> Vec<u8> {
        let mut v: Vec<u8> = vec![0b10101010, 0b01010101]
            .into_iter()
            .cycle()
            .take(BYTES - 1)
            .collect();
        // we -5 bits to test partial bytes, this is that partial byte
        if BYTES % 2 == 0 {
            // the last byte is 0b10101010
            v.push(0b101);
        } else {
            // the last byte is 0b01010101
            v.push(0b010);
        }
        v
    }

    fn out() -> Vec<f64> {
        vec![0.0; bits() as usize]
    }

    fn bits() -> u64 {
        BYTES as u64 * 8 - 5
    }

    fn expected() -> Vec<f64> {
        vec![
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ]
        .into_iter()
        .cycle()
        .take(bits() as usize)
        .collect::<Vec<_>>()
    }

    #[test]
    fn test_unpack_naive_sync() {
        let mut out = out();
        unpack_naive_sync(&mut out, &bytes(), 0.0, 1.0);
        assert_eq!(out, expected(),);
    }

    #[test]
    fn test_unpack_naive_par() {
        let mut out = out();
        unpack_naive_par(128, &mut out, &bytes(), 0.0, 1.0);
        assert_eq!(out, expected());
    }

    #[test]
    fn test_unpack_avx2_sync() {
        if let Some(simd) = pulp::x86::V3::try_new() {
            let mut out = out();
            unpack_avx2_sync(simd, &mut out, &bytes(), 0.0, 1.0);
            assert_eq!(out, expected());
        }
    }

    #[test]
    fn test_unpack_avx2_par() {
        if let Some(simd) = pulp::x86::V3::try_new() {
            let mut out = out();
            unpack_avx2_par(128, simd, &mut out, &bytes(), 0.0, 1.0);
            assert_eq!(out, expected());
        }
    }

    #[test]
    fn test_unpack_avx512_sync() {
        if is_x86_feature_detected!("avx512f") {
            let mut out = out();
            unpack_avx512_sync(&mut out, &bytes(), 0.0, 1.0);
            assert_eq!(out, expected());
        }
    }

    #[test]
    fn test_unpack_avx512_par() {
        if is_x86_feature_detected!("avx512f") {
            let mut out = out();
            unpack_avx512_par(128, &mut out, &bytes(), 0.0, 1.0);
            assert_eq!(out, expected());
        }
    }

    #[test]
    fn test_pack_naive_sync() {
        let mut out = vec![0; bytes().len()];
        pack_naive_sync(&mut out, &expected(), 0.0, 1.0);
        assert_eq!(out, bytes());
    }

    #[test]
    fn test_pack_naive_par() {
        let mut out = vec![0; bytes().len()];
        pack_naive_par(128, &mut out, &expected(), 0.0, 1.0);
        assert_eq!(out, bytes());
    }

    #[test]
    fn test_pack_avx2_sync() {
        if let Some(simd) = pulp::x86::V3::try_new() {
            let mut out = vec![0; bytes().len()];
            pack_avx2_sync(simd, &mut out, &expected(), 0.0, 1.0);
            assert_eq!(out, bytes());
        }
    }

    #[test]
    fn test_pack_avx2_par() {
        if let Some(simd) = pulp::x86::V3::try_new() {
            let mut out = vec![0; bytes().len()];
            pack_avx2_par(128, simd, &mut out, &expected(), 0.0, 1.0);
            assert_eq!(out, bytes());
        }
    }

    #[test]
    fn test_pack_avx512_sync() {
        if is_x86_feature_detected!("avx512f") {
            let mut out = vec![0; bytes().len()];
            pack_avx512_sync(&mut out, &expected(), 0.0, 1.0);
            assert_eq!(out, bytes());
        }
    }

    // #[test]
    // fn test_pack_avx512_par() {
    //     if is_x86_feature_detected!("avx512f") {
    //         let mut out = vec![0; bytes().len()];
    //         pack_avx512_par(128, &mut out, &expected(), 0.0, 1.0);
    //         assert_eq!(out, bytes());
    //     }
    // }
}

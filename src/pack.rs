//! This module provides utilities for speedily packing and unpacking bit-packed
//! data. Much of the SIMD and inline assembly code is adapted from MIT licensed
//! code provided by sarah quiñones el kazdadi, massive thanks to her for the
//! help!

use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

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
    if chunk_size < 128 {
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
    if chunk_size < 128 {
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
                unpack_naive(out_tail, bytes_tail, zero, one);
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
    if chunk_size < 128 {
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
            for i in 0..out.len() {
                out[i] = if ((chunk[i / 8] >> (i % 8)) & 1) == 1 {
                    one
                } else {
                    zero
                };
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes() -> Vec<u8> {
        vec![0b10101010, 0b01010101]
            .into_iter()
            .cycle()
            .take(2101)
            .collect()
    }

    fn out() -> Vec<f64> {
        vec![0.0; bits() as usize]
    }

    fn bits() -> u64 {
        2101 * 8 - 5
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
}

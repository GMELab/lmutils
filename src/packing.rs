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
    } else if is_x86_feature_detected!("avx2") {
        unpack_avx2(out, bytes, zero, one);
    } else {
        unpack_naive(out, bytes, zero, one);
    }
}

pub fn unpack_avx512(out: &mut [f64], bytes: &[u8], zero: f64, one: f64) {
    let threads = rayon::current_num_threads();
    let chunk_size = (bytes.len() / threads / 8 + 1) * 8;
    if chunk_size < MIN_CHUNK_SIZE {
        unpack_avx512_sync(out, bytes, zero, one);
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
            out("k1") _,
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

pub fn unpack_avx2(out: &mut [f64], bytes: &[u8], zero: f64, one: f64) {
    let threads = rayon::current_num_threads();
    let chunk_size = (bytes.len() / threads / 16 + 1) * 16;
    if chunk_size < MIN_CHUNK_SIZE {
        unpack_avx2_sync(out, bytes, zero, one);
    } else {
        unpack_avx2_par(chunk_size, out, bytes, zero, one);
    }
}

pub fn unpack_avx2_sync(out: &mut [f64], bytes: &[u8], zero: f64, one: f64) {
    let bits = out.len();
    let mask1: [u64; 4] = [0x1, 0x2, 0x4, 0x8];
    let mask2: [u64; 4] = [0x10, 0x20, 0x40, 0x80];
    let zeroes: [u64; 4] = [0, 0, 0, 0];
    unsafe {
        core::arch::asm! {
            "vbroadcastsd ymm0, xmm0",
            "vbroadcastsd ymm1, xmm1",
            "test rax, rax",
            "jz 3f",
                "2:",
                // move the next byte from the input
                "movzx rcx, byte ptr [rsi]",
                // move to xmm2
                "vpbroadcastq ymm2, rcx",
                "vpand ymm2, ymm2, [{mask1}]",
                "vpcmpeqq ymm2, ymm2, [{zero}]",
                // other half to xmm3
                "vpbroadcastq ymm3, rcx",
                "vpand ymm3, ymm3, [{mask2}]",
                "vpcmpeqq ymm3, ymm3, [{zero}]",

                // select the correct f64s based on the bits in ecx
                "vblendvpd ymm4, ymm1, ymm0, ymm2",
                "vblendvpd ymm5, ymm1, ymm0, ymm3",

                // move the f64s into the output
                "vmovupd ymmword ptr [rdi], ymm4",
                "vmovupd ymmword ptr [rdi + 32], ymm5",

                "add rsi, 1",
                "add rdi, 32",
                "dec rax",
                "jnz 2b",
            "3:",
            "vzeroupper",

            mask1 = in(reg) &mask1,
            mask2 = in(reg) &mask2,
            zero = in(reg) &zeroes,
            inout("xmm0") zero => _,
            inout("xmm1") one => _,
            inout("rax") bits / 8 => _,
            inout("rsi") bytes.as_ptr() => _,
            inout("rdi") out.as_mut_ptr() => _,
            out("rcx") _,
            out("ymm2") _,
            out("ymm3") _,
            out("ymm4") _,
            out("ymm5") _,
            options(nostack, readonly),
        }
    };
    unpack_naive_sync(
        out[(bits / 8 * 8)..].as_mut(),
        &bytes[(bits / 8)..],
        zero,
        one,
    );
}

pub fn unpack_avx2_par(chunk_size: usize, out: &mut [f64], bytes: &[u8], zero: f64, one: f64) {
    bytes
        .par_chunks(chunk_size)
        .zip(out.par_chunks_mut(8 * chunk_size))
        .for_each(|(chunk, out)| {
            unpack_avx2_sync(out, chunk, zero, one);
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
    } else if is_x86_feature_detected!("avx2") {
        pack_avx2(out, data, zero, one);
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
            "vbroadcastsd zmm0, xmm0", // 8 copies of one from xmm0 to zmm0
            "test rax, rax",
            "jz 3f",
                "2:",
                "vmovupd zmm1, zmmword ptr [rsi]", // move the next 8 f64s into the input
                "vcmpeqpd k1, zmm1, zmm0", // compare the input to one
                "kmovb byte ptr [rdi], k1", // move the next output byte from k1

                "add rsi, 64", // increment the input pointer
                "add rdi, 1", // increment the output pointer
                "dec rax", // decrement the counter
                "jnz 2b", // if the counter is zero, jump to the start of the loop
            "3:",
            "vzeroupper",

            inout("xmm0") one => _,
            inout("rax") bits / 8 => _,
            inout("rsi") data.as_ptr() => _,
            inout("rdi") out.as_mut_ptr() => _,
            out("k1") _,
            out("zmm1") _,
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

pub fn pack_avx2(out: &mut [u8], data: &[f64], zero: f64, one: f64) {
    let threads = rayon::current_num_threads();
    let chunk_size = (data.len() / threads / 8 + 1) * 8;
    if chunk_size < MIN_CHUNK_SIZE {
        pack_avx2_sync(out, data, zero, one);
    } else {
        pack_avx2_par(chunk_size, out, data, zero, one);
    }
}

pub fn pack_avx2_sync(out: &mut [u8], data: &[f64], zero: f64, one: f64) {
    let bits = data.len();
    unsafe {
        core::arch::asm! {
            "vbroadcastsd ymm0, xmm0", // 4 copies of one from xmm0 to ymm0
            "test rax, rax",
            "jz 3f",
                "2:",
                // move the next 4 f64s from the input
                "vmovupd ymm1, ymmword ptr [rsi]",
                "vmovupd ymm2, ymmword ptr [rsi + 32]",

                // compare the input to one
                "vpcmpeqq ymm3, ymm1, ymm0",
                "vpcmpeqq ymm4, ymm2, ymm0",

                // combine the two masks
                "vmovmskpd ecx, ymm3",
                "vmovmskpd edx, ymm4",
                "shl edx, 4",
                "or ecx, edx",
                "mov byte ptr [rdi], cl",

                "add rsi, 64", // increment the input pointer
                "add rdi, 1", // increment the output pointer
                "dec rax", // decrement the counter
                "jnz 2b", // if the counter is not zero, jump to the start of the loop
            "3:",
            "vzeroupper",

            inout("xmm0") one => _,
            inout("rax") bits / 8 => _,
            inout("rsi") data.as_ptr() => _,
            inout("rdi") out.as_mut_ptr() => _,
            out("ymm1") _,
            out("ymm2") _,
            out("ymm3") _,
            out("ymm4") _,
            out("ecx") _,
            out("edx") _,
        }
    };
    pack_naive_sync(
        out[(bits / 8)..].as_mut(),
        &data[(bits / 8 * 8)..],
        zero,
        one,
    );
}

pub fn pack_avx2_par(chunk_size: usize, out: &mut [u8], data: &[f64], zero: f64, one: f64) {
    data.par_chunks(chunk_size)
        .zip(out.par_chunks_mut(chunk_size / 8))
        .for_each(|(data, out)| {
            pack_avx2_sync(out, data, zero, one);
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
    out.fill(0);
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

    const BYTES: usize = 2;

    fn bytes() -> Vec<u8> {
        let mut v: Vec<u8> = vec![0b10001010, 0b01000101]
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
            0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
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
        if is_x86_feature_detected!("avx2") {
            let mut out = out();
            unpack_avx2_sync(&mut out, &bytes(), 0.0, 1.0);
            assert_eq!(out, expected());
        }
    }

    #[test]
    fn test_unpack_avx2_par() {
        if is_x86_feature_detected!("avx2") {
            let mut out = out();
            unpack_avx2_par(128, &mut out, &bytes(), 0.0, 1.0);
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
        if is_x86_feature_detected!("avx2") {
            let mut out = vec![0; bytes().len()];
            pack_avx2_sync(&mut out, &expected(), 0.0, 1.0);
            assert_eq!(out, bytes());
        }
    }

    #[test]
    fn test_pack_avx2_par() {
        if is_x86_feature_detected!("avx2") {
            let mut out = vec![0; bytes().len()];
            pack_avx2_par(128, &mut out, &expected(), 0.0, 1.0);
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

    #[test]
    fn test_pack_avx512_par() {
        if is_x86_feature_detected!("avx512f") {
            let mut out = vec![0; bytes().len()];
            pack_avx512_par(128, &mut out, &expected(), 0.0, 1.0);
            assert_eq!(out, bytes());
        }
    }
}

use std::ffi::CStr;

fn main() {
    let libc_version = unsafe { CStr::from_ptr(libc::gnu_get_libc_version()) }
        .to_str()
        .unwrap();
    let (major, minor) = libc_version
        .split_once('.')
        .map(|(major, minor)| (major.parse::<u32>().unwrap(), minor.parse::<u32>().unwrap()))
        .unwrap();
    if major == 2 && minor >= 27 {
        println!("cargo:rustc-cfg=libc_2_27");
    }
    println!("cargo::rustc-check-cfg=cfg(libc_2_27)");
}

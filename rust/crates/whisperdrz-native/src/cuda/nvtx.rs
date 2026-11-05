//! Optional NVTX ranges for nsys, loaded at run time
//!
//! Set `WHISPERDRZ_NVTX_LIB` to an NVTX v2 library (`libnvToolsExt.so.1`, for example
//! the one in the venv's `nvidia/nvtx/lib`) to emit ranges; unset, every range is a
//! no-op. Loading at run time keeps the build free of a CUDA toolkit, like the rest of
//! the host layer

use std::ffi::{CString, c_char, c_int};
use std::sync::OnceLock;

type Push = unsafe extern "C" fn(*const c_char) -> c_int;
type Pop = unsafe extern "C" fn() -> c_int;

struct Nvtx {
    push: Push,
    pop: Pop,
    // keeps the symbols above valid
    _lib: libloading::Library,
}

fn nvtx() -> Option<&'static Nvtx> {
    static NVTX: OnceLock<Option<Nvtx>> = OnceLock::new();
    NVTX.get_or_init(|| {
        let path = std::env::var_os("WHISPERDRZ_NVTX_LIB")?;
        // SAFETY: loading the NVTX library runs no initializers beyond its own, and the
        // two symbols have the C signatures declared above
        unsafe {
            let lib = libloading::Library::new(&path).ok()?;
            let push = *lib.get::<Push>(b"nvtxRangePushA\0").ok()?;
            let pop = *lib.get::<Pop>(b"nvtxRangePop\0").ok()?;
            Some(Nvtx {
                push,
                pop,
                _lib: lib,
            })
        }
    })
    .as_ref()
}

/// A pushed NVTX range, popped on drop
#[must_use = "the range closes when this is dropped"]
pub struct Range(bool);

impl Range {
    /// Opens a range named `name` when NVTX is loaded
    pub fn new(name: &str) -> Self {
        let Some(n) = nvtx() else {
            return Self(false);
        };
        let Ok(name) = CString::new(name) else {
            return Self(false);
        };
        // SAFETY: `name` is a valid C string for the duration of the call
        unsafe { (n.push)(name.as_ptr()) };
        Self(true)
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        if let (true, Some(n)) = (self.0, nvtx()) {
            // SAFETY: pops the range this value pushed
            unsafe { (n.pop)() };
        }
    }
}

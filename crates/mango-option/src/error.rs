// SPDX-License-Identifier: MIT
use mango_option_sys as sys;

/// Error category mirroring the C ABI `MangoStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Validation,
    Arbitrage,
    NoConvergence,
    Bracketing,
    Solver,
}

/// A mango-option error: a category plus a (synthesized, possibly truncated)
/// diagnostic message from the C++ side.
#[derive(Debug, Clone)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
}

impl ErrorKind {
    pub(crate) fn from_status(status: i32) -> ErrorKind {
        match status {
            sys::MANGO_ERR_VALIDATION => ErrorKind::Validation,
            sys::MANGO_ERR_ARBITRAGE => ErrorKind::Arbitrage,
            sys::MANGO_ERR_NO_CONVERGENCE => ErrorKind::NoConvergence,
            sys::MANGO_ERR_BRACKETING => ErrorKind::Bracketing,
            _ => ErrorKind::Solver,
        }
    }
}

impl Error {
    /// Build an Error from a non-OK status and a populated MangoError.
    pub(crate) fn from_c(status: i32, err: &sys::MangoError) -> Error {
        // SAFETY: err.message is a 256-byte buffer we own by value; reinterpreting the
        // c_char array as bytes is sound. from_bytes_until_nul tolerates a missing
        // terminator (returns Err -> empty string) so this never reads out of bounds.
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(err.message.as_ptr().cast::<u8>(), err.message.len())
        };
        let msg = core::ffi::CStr::from_bytes_until_nul(bytes)
            .map(|c| c.to_string_lossy().into_owned())
            .unwrap_or_default();
        Error { kind: ErrorKind::from_status(status), message: msg }
    }
}

impl core::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            ErrorKind::Validation => "validation",
            ErrorKind::Arbitrage => "arbitrage",
            ErrorKind::NoConvergence => "no convergence",
            ErrorKind::Bracketing => "bracketing",
            ErrorKind::Solver => "solver",
        };
        f.write_str(s)
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl std::error::Error for Error {}

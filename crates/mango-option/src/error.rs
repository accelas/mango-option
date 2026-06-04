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
        // SAFETY: the C side always null-terminates message within 256 bytes.
        let msg = unsafe { core::ffi::CStr::from_ptr(err.message.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Error { kind: ErrorKind::from_status(status), message: msg }
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for Error {}

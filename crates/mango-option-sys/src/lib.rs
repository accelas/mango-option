// SPDX-License-Identifier: MIT
//! Raw FFI bindings to the mango-option C ABI (`src/ffi/mango_c_api.h`).
//! 1:1 with the C header; no safety. Use the `mango-option` crate instead.
#![allow(non_camel_case_types)]

pub type MangoStatus = i32;
pub const MANGO_OK: i32 = 0;
pub const MANGO_ERR_VALIDATION: i32 = 1;
pub const MANGO_ERR_ARBITRAGE: i32 = 2;
pub const MANGO_ERR_NO_CONVERGENCE: i32 = 3;
pub const MANGO_ERR_BRACKETING: i32 = 4;
pub const MANGO_ERR_SOLVER: i32 = 5;

pub type MangoOptionType = i32;
pub const MANGO_CALL: i32 = 0;
pub const MANGO_PUT: i32 = 1;

#[repr(C)]
pub struct MangoError {
    pub code: i32,
    pub message: [core::ffi::c_char; 256],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MangoDividend {
    pub calendar_time: f64,
    pub amount: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MangoTenorPoint {
    pub tenor: f64,
    pub log_discount: f64,
}

#[repr(C)]
pub struct MangoPricingParams {
    pub spot: f64,
    pub strike: f64,
    pub maturity: f64,
    pub dividend_yield: f64,
    pub volatility: f64,
    pub rate_const: f64,
    pub tenor_points: *const MangoTenorPoint,
    pub n_tenor_points: u64,
    pub dividends: *const MangoDividend,
    pub n_dividends: u64,
    pub option_type: MangoOptionType,
}

#[repr(C)]
pub struct MangoIvQuery {
    pub spot: f64,
    pub strike: f64,
    pub maturity: f64,
    pub dividend_yield: f64,
    pub market_price: f64,
    pub rate_const: f64,
    pub tenor_points: *const MangoTenorPoint,
    pub n_tenor_points: u64,
    pub dividends: *const MangoDividend,
    pub n_dividends: u64,
    pub option_type: MangoOptionType,
}

#[repr(C)]
pub struct MangoIvSuccess {
    pub implied_vol: f64,
    pub iterations: u64,
    pub final_error: f64,
    pub vega: f64,
    pub has_vega: i32,
    pub used_rate_approximation: i32,
}

#[repr(C)]
pub struct MangoIvConfig {
    pub brent_tol_abs: f64,
    pub max_iter: i32,
}

#[repr(C)]
pub struct MangoAmericanResult {
    _private: [u8; 0],
}

extern "C" {
    pub fn mango_price_american(
        params: *const MangoPricingParams,
        out_result: *mut *mut MangoAmericanResult,
        out_err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_american_value(r: *const MangoAmericanResult) -> f64;
    pub fn mango_american_delta(r: *const MangoAmericanResult) -> f64;
    pub fn mango_american_gamma(r: *const MangoAmericanResult) -> f64;
    pub fn mango_american_theta(r: *const MangoAmericanResult) -> f64;
    pub fn mango_american_value_at(
        r: *const MangoAmericanResult,
        spot: f64,
        out: *mut f64,
        out_err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_american_result_free(r: *mut MangoAmericanResult);
    pub fn mango_solve_iv(
        query: *const MangoIvQuery,
        config: *const MangoIvConfig,
        out_success: *mut MangoIvSuccess,
        out_err: *mut MangoError,
    ) -> MangoStatus;
}

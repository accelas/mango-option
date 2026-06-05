// SPDX-License-Identifier: MIT
//! Raw FFI bindings to the mango-option C ABI (`src/ffi/mango_c_api.h`).
//! 1:1 with the C header; no safety. Use the `mango-option` crate instead.
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

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
#[derive(Clone, Copy)]
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
#[derive(Clone, Copy)]
pub struct MangoInterpSolverConfig {
    pub max_iter: u64,
    pub tolerance: f64,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub vega_threshold: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MangoAdaptiveGridParams {
    pub target_iv_error: f64,
    pub max_iter: u64,
    pub max_points_per_dim: u64,
    pub min_moneyness_points: u64,
    pub validation_samples: u64,
    pub refinement_factor: f64,
    pub lhs_seed: u64,
    pub vega_floor: f64,
    pub max_failure_rate: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MangoMultiKRef {
    pub K_refs: *const f64,
    pub n_K_refs: u64,
    pub K_ref_count: i32,
    pub K_ref_span: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MangoDiscreteDividendConfig {
    pub maturity: f64,
    pub dividends: *const MangoDividend,
    pub n_dividends: u64,
    pub kref_config: MangoMultiKRef,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MangoIvFactoryConfig {
    pub option_type: MangoOptionType,
    pub spot: f64,
    pub dividend_yield: f64,
    pub moneyness: *const f64,
    pub n_moneyness: u64,
    pub vol: *const f64,
    pub n_vol: u64,
    pub rate: *const f64,
    pub n_rate: u64,
    pub maturity_grid: *const f64,
    pub n_maturity: u64,
    pub solver_config: MangoInterpSolverConfig,
    pub adaptive: *const MangoAdaptiveGridParams,
    pub discrete_dividends: *const MangoDiscreteDividendConfig,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MangoIvBatchSlot {
    pub status: i32,
    pub success: MangoIvSuccess,
}

#[repr(C)]
pub struct MangoInterpIvSolver {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MangoPriceTable {
    _private: [u8; 0],
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

    pub fn mango_make_interp_iv_solver(
        cfg: *const MangoIvFactoryConfig,
        out: *mut *mut MangoInterpIvSolver,
        err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_interp_iv_solve(
        s: *const MangoInterpIvSolver,
        q: *const MangoIvQuery,
        out: *mut MangoIvSuccess,
        err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_interp_iv_solve_batch(
        s: *const MangoInterpIvSolver,
        queries: *const MangoIvQuery,
        n: u64,
        out_slots: *mut MangoIvBatchSlot,
        out_failed_count: *mut u64,
        err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_interp_iv_solver_free(s: *mut MangoInterpIvSolver);

    pub fn mango_make_price_table(
        cfg: *const MangoIvFactoryConfig,
        out: *mut *mut MangoPriceTable,
        err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_price_table_validate(
        t: *const MangoPriceTable,
        p: *const MangoPricingParams,
        err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_price_table_price(t: *const MangoPriceTable, p: *const MangoPricingParams) -> f64;
    pub fn mango_price_table_vega(t: *const MangoPriceTable, p: *const MangoPricingParams) -> f64;
    pub fn mango_price_table_delta(
        t: *const MangoPriceTable,
        p: *const MangoPricingParams,
        out: *mut f64,
        err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_price_table_gamma(
        t: *const MangoPriceTable,
        p: *const MangoPricingParams,
        out: *mut f64,
        err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_price_table_theta(
        t: *const MangoPriceTable,
        p: *const MangoPricingParams,
        out: *mut f64,
        err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_price_table_rho(
        t: *const MangoPriceTable,
        p: *const MangoPricingParams,
        out: *mut f64,
        err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_price_table_option_type(t: *const MangoPriceTable) -> MangoOptionType;
    pub fn mango_price_table_dividend_yield(t: *const MangoPriceTable) -> f64;
    pub fn mango_price_table_make_iv_solver(
        t: *const MangoPriceTable,
        cfg: *const MangoInterpSolverConfig,
        out: *mut *mut MangoInterpIvSolver,
        err: *mut MangoError,
    ) -> MangoStatus;
    pub fn mango_price_table_free(t: *mut MangoPriceTable);
}

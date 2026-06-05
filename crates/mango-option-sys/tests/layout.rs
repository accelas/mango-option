// SPDX-License-Identifier: MIT
use core::mem::{align_of, offset_of, size_of};
use mango_option_sys::*;

#[test]
fn pricing_params_layout() {
    assert_eq!(size_of::<MangoPricingParams>(), 88);
    assert_eq!(align_of::<MangoPricingParams>(), 8);
    assert_eq!(offset_of!(MangoPricingParams, rate_const), 40);
    assert_eq!(offset_of!(MangoPricingParams, tenor_points), 48);
    assert_eq!(offset_of!(MangoPricingParams, n_dividends), 72);
    assert_eq!(offset_of!(MangoPricingParams, option_type), 80);
}

#[test]
fn iv_query_layout() {
    assert_eq!(size_of::<MangoIvQuery>(), 88);
    assert_eq!(offset_of!(MangoIvQuery, market_price), 32);
    assert_eq!(offset_of!(MangoIvQuery, tenor_points), 48);
    assert_eq!(offset_of!(MangoIvQuery, n_tenor_points), 56);
    assert_eq!(offset_of!(MangoIvQuery, dividends), 64);
    assert_eq!(offset_of!(MangoIvQuery, n_dividends), 72);
    assert_eq!(offset_of!(MangoIvQuery, option_type), 80);
}

#[test]
fn small_struct_layouts() {
    assert_eq!(size_of::<MangoDividend>(), 16);
    assert_eq!(size_of::<MangoTenorPoint>(), 16);
    assert_eq!(size_of::<MangoError>(), 260);
    assert_eq!(offset_of!(MangoError, message), 4);
    assert_eq!(size_of::<MangoIvSuccess>(), 40);
    assert_eq!(offset_of!(MangoIvSuccess, has_vega), 32);
    assert_eq!(offset_of!(MangoIvSuccess, used_rate_approximation), 36);
    assert_eq!(size_of::<MangoIvConfig>(), 16);
    assert_eq!(offset_of!(MangoIvConfig, max_iter), 8);
}

#[test]
fn interp_solver_config_layout() {
    assert_eq!(size_of::<MangoInterpSolverConfig>(), 40);
    assert_eq!(offset_of!(MangoInterpSolverConfig, max_iter), 0);
    assert_eq!(offset_of!(MangoInterpSolverConfig, tolerance), 8);
    assert_eq!(offset_of!(MangoInterpSolverConfig, sigma_min), 16);
    assert_eq!(offset_of!(MangoInterpSolverConfig, sigma_max), 24);
    assert_eq!(offset_of!(MangoInterpSolverConfig, vega_threshold), 32);
}

#[test]
fn adaptive_grid_params_layout() {
    assert_eq!(size_of::<MangoAdaptiveGridParams>(), 72);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, target_iv_error), 0);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, max_iter), 8);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, max_points_per_dim), 16);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, min_moneyness_points), 24);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, validation_samples), 32);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, refinement_factor), 40);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, lhs_seed), 48);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, vega_floor), 56);
    assert_eq!(offset_of!(MangoAdaptiveGridParams, max_failure_rate), 64);
}

#[test]
fn multi_kref_layout() {
    assert_eq!(size_of::<MangoMultiKRef>(), 32);
    assert_eq!(offset_of!(MangoMultiKRef, K_refs), 0);
    assert_eq!(offset_of!(MangoMultiKRef, n_K_refs), 8);
    assert_eq!(offset_of!(MangoMultiKRef, K_ref_count), 16);
    assert_eq!(offset_of!(MangoMultiKRef, K_ref_span), 24);
}

#[test]
fn discrete_dividend_config_layout() {
    assert_eq!(size_of::<MangoDiscreteDividendConfig>(), 56);
    assert_eq!(offset_of!(MangoDiscreteDividendConfig, maturity), 0);
    assert_eq!(offset_of!(MangoDiscreteDividendConfig, dividends), 8);
    assert_eq!(offset_of!(MangoDiscreteDividendConfig, n_dividends), 16);
    assert_eq!(offset_of!(MangoDiscreteDividendConfig, kref_config), 24);
}

#[test]
fn iv_factory_config_layout() {
    assert_eq!(size_of::<MangoIvFactoryConfig>(), 144);
    assert_eq!(offset_of!(MangoIvFactoryConfig, option_type), 0);
    assert_eq!(offset_of!(MangoIvFactoryConfig, spot), 8);
    assert_eq!(offset_of!(MangoIvFactoryConfig, dividend_yield), 16);
    assert_eq!(offset_of!(MangoIvFactoryConfig, moneyness), 24);
    assert_eq!(offset_of!(MangoIvFactoryConfig, n_moneyness), 32);
    assert_eq!(offset_of!(MangoIvFactoryConfig, vol), 40);
    assert_eq!(offset_of!(MangoIvFactoryConfig, n_vol), 48);
    assert_eq!(offset_of!(MangoIvFactoryConfig, rate), 56);
    assert_eq!(offset_of!(MangoIvFactoryConfig, n_rate), 64);
    assert_eq!(offset_of!(MangoIvFactoryConfig, maturity_grid), 72);
    assert_eq!(offset_of!(MangoIvFactoryConfig, n_maturity), 80);
    assert_eq!(offset_of!(MangoIvFactoryConfig, solver_config), 88);
    assert_eq!(offset_of!(MangoIvFactoryConfig, adaptive), 128);
    assert_eq!(offset_of!(MangoIvFactoryConfig, discrete_dividends), 136);
}

#[test]
fn iv_batch_slot_layout() {
    assert_eq!(size_of::<MangoIvBatchSlot>(), 48);
    assert_eq!(offset_of!(MangoIvBatchSlot, status), 0);
    assert_eq!(offset_of!(MangoIvBatchSlot, success), 8);
}

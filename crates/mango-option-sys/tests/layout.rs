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
    assert_eq!(size_of::<MangoIvConfig>(), 16);
    assert_eq!(offset_of!(MangoIvConfig, max_iter), 8);
}

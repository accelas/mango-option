// SPDX-License-Identifier: MIT
use mango_option::{
    price_american, solve_iv, Dividend, ErrorKind, IvConfig, IvQuery, OptionSpec,
    OptionType, PricingParams, Rate, TenorPoint,
};

fn put_spec() -> OptionSpec {
    OptionSpec {
        spot: 100.0,
        strike: 100.0,
        maturity: 1.0,
        dividend_yield: 0.0,
        rate: Rate::Const(0.05),
        discrete_dividends: vec![],
        option_type: OptionType::Put,
    }
}

#[test]
fn atm_put_price_and_greeks() {
    let r = price_american(&PricingParams { spec: put_spec(), volatility: 0.20 }).unwrap();
    // Reuse the loose tolerance from tests/american_option_test.cc (~6.35 +/- 0.5).
    assert!((r.value() - 6.35).abs() < 0.5, "value = {}", r.value());
    assert!(r.delta() < 0.0);           // put delta negative
    assert!(r.gamma() > 0.0);
    assert!(r.theta().is_finite());
    let deep = r.value_at(90.0).unwrap();
    assert!(deep > r.value());          // deeper ITM worth more
}

#[test]
fn iv_round_trip() {
    let priced = price_american(&PricingParams { spec: put_spec(), volatility: 0.25 }).unwrap();
    let market = priced.value();
    let s = solve_iv(
        &IvQuery { spec: put_spec(), market_price: market },
        &IvConfig::default(),
    )
    .unwrap();
    assert!((s.implied_vol - 0.25).abs() < 0.01, "iv = {}", s.implied_vol);
}

#[test]
fn discrete_dividend_iv_round_trip() {
    let mut spec = put_spec();
    spec.discrete_dividends = vec![Dividend { calendar_time: 0.5, amount: 2.0 }];
    let market = price_american(&PricingParams { spec: spec.clone(), volatility: 0.25 })
        .unwrap()
        .value();
    let s = solve_iv(
        &IvQuery { spec, market_price: market },
        &IvConfig::default(),
    )
    .unwrap();
    assert!((s.implied_vol - 0.25).abs() < 0.01, "iv = {}", s.implied_vol);
}

#[test]
fn yield_curve_prices() {
    let mut spec = put_spec();
    spec.rate = Rate::Curve(vec![
        TenorPoint { tenor: 0.0, log_discount: 0.0 },
        TenorPoint { tenor: 1.0, log_discount: -0.05 },
    ]);
    let r = price_american(&PricingParams { spec, volatility: 0.20 }).unwrap();
    assert!(r.value() > 0.0 && r.value().is_finite());
}

#[test]
fn invalid_yield_curve_is_validation_error() {
    let mut spec = put_spec();
    // Missing the required tenor=0 anchor point.
    spec.rate = Rate::Curve(vec![TenorPoint { tenor: 1.0, log_discount: -0.05 }]);
    let e = price_american(&PricingParams { spec, volatility: 0.20 }).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Validation);
}

#[test]
fn negative_spot_is_validation_error() {
    let mut spec = put_spec();
    spec.spot = -1.0;
    let e = price_american(&PricingParams { spec, volatility: 0.20 }).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Validation);
}

#[test]
fn arbitrage_violating_price_is_arbitrage_error() {
    // Market price above the strike upper bound for a put.
    let e = solve_iv(
        &IvQuery { spec: put_spec(), market_price: 1000.0 },
        &IvConfig::default(),
    )
    .unwrap_err();
    assert_eq!(e.kind, ErrorKind::Arbitrage);
}

#[test]
fn empty_yield_curve_is_validation_error() {
    let mut spec = put_spec();
    spec.rate = Rate::Curve(vec![]);
    let e = price_american(&PricingParams { spec, volatility: 0.20 }).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Validation);
}

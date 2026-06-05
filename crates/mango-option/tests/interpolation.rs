// SPDX-License-Identifier: MIT
use mango_option::{
    price_american, AdaptiveGridParams, DiscreteDividendConfig, Dividend, FactoryConfig,
    InterpIvSolver, InterpSolverConfig, IvGrid, IvQuery, MultiKRef, OptionSpec, OptionType, Rate,
};

fn base_config() -> FactoryConfig {
    FactoryConfig {
        option_type: OptionType::Put,
        spot: 100.0,
        dividend_yield: 0.0,
        grid: IvGrid {
            // Each axis needs >= 4 points for the cubic B-spline builder.
            moneyness: vec![0.8, 0.9, 1.0, 1.1, 1.2],
            vol: vec![0.10, 0.20, 0.30, 0.40],
            rate: vec![0.01, 0.03, 0.05, 0.07],
        },
        // >= 4 maturity points: the B-spline builder requires at least 4
        // control points per axis (axis 1 is the maturity grid).
        maturity_grid: vec![0.25, 0.5, 0.75, 1.0],
        solver: InterpSolverConfig::default(),
        adaptive: None,
        discrete_dividends: None,
    }
}

fn put_spec(sigma: f64) -> (OptionSpec, f64) {
    let spec = OptionSpec {
        spot: 100.0, strike: 100.0, maturity: 1.0, dividend_yield: 0.0,
        rate: Rate::Const(0.03), discrete_dividends: vec![], option_type: OptionType::Put,
    };
    (spec, sigma)
}

#[test]
fn iv_round_trip_continuous() {
    let solver = InterpIvSolver::new(&base_config()).expect("build solver");
    let (spec, sigma) = put_spec(0.25);
    // Reference price from the FDM path at the true sigma.
    let pp = mango_option::PricingParams { spec: spec.clone(), volatility: sigma };
    let price = price_american(&pp).unwrap().value();
    let q = IvQuery { spec, market_price: price };
    let r = solver.solve(&q).expect("solve iv");
    assert!((r.implied_vol - sigma).abs() < 1e-2, "iv={} sigma={}", r.implied_vol, sigma);
}

#[test]
fn batch_with_one_failure() {
    let solver = InterpIvSolver::new(&base_config()).unwrap();
    let (spec, sigma) = put_spec(0.25);
    let pp = mango_option::PricingParams { spec: spec.clone(), volatility: sigma };
    let good_price = price_american(&pp).unwrap().value();
    let good = IvQuery { spec: spec.clone(), market_price: good_price };
    // Arbitrage-violating price (above strike for a put) => solve failure.
    let bad = IvQuery { spec, market_price: 1_000.0 };
    let batch = solver.solve_batch(&[good, bad]);
    assert_eq!(batch.failed, 1);
    assert!(batch.results[0].is_ok());
    assert!(batch.results[1].is_err());
}

#[test]
fn adaptive_build_solves() {
    let mut cfg = base_config();
    cfg.adaptive = Some(AdaptiveGridParams {
        target_iv_error: 1e-3, max_iter: 2, validation_samples: 16,
        min_moneyness_points: 20, ..AdaptiveGridParams::default()
    });
    let solver = InterpIvSolver::new(&cfg).expect("adaptive build");
    let (spec, sigma) = put_spec(0.25);
    let pp = mango_option::PricingParams { spec: spec.clone(), volatility: sigma };
    let price = price_american(&pp).unwrap().value();
    let r = solver.solve(&IvQuery { spec, market_price: price }).expect("solve");
    assert!((r.implied_vol - sigma).abs() < 2e-2);
}

#[test]
fn discrete_dividend_build_solves() {
    let mut cfg = base_config();
    cfg.discrete_dividends = Some(DiscreteDividendConfig {
        maturity: 1.0,
        dividends: vec![Dividend { calendar_time: 0.5, amount: 2.0 }],
        kref_config: MultiKRef { k_refs: vec![90.0, 100.0, 110.0], ..MultiKRef::default() },
    });
    let solver = InterpIvSolver::new(&cfg).expect("discrete build");
    let spec = OptionSpec {
        spot: 100.0, strike: 100.0, maturity: 1.0, dividend_yield: 0.0,
        rate: Rate::Const(0.03),
        discrete_dividends: vec![Dividend { calendar_time: 0.5, amount: 2.0 }],
        option_type: OptionType::Put,
    };
    let pp = mango_option::PricingParams { spec: spec.clone(), volatility: 0.25 };
    let price = price_american(&pp).unwrap().value();
    let r = solver.solve(&IvQuery { spec, market_price: price }).expect("solve");
    assert!((r.implied_vol - 0.25).abs() < 3e-2, "iv={}", r.implied_vol);
}

#[test]
fn empty_maturity_grid_continuous_is_validation_error() {
    let mut cfg = base_config();
    cfg.maturity_grid = vec![];
    let err = InterpIvSolver::new(&cfg).unwrap_err();
    assert_eq!(err.kind, mango_option::ErrorKind::Validation);
}

#[test]
fn non_finite_spot_is_validation_error() {
    let mut cfg = base_config();
    cfg.spot = f64::NAN;
    let err = InterpIvSolver::new(&cfg).unwrap_err();
    assert_eq!(err.kind, mango_option::ErrorKind::Validation);
}

#[test]
fn solver_is_send_sync() {
    let solver = InterpIvSolver::new(&base_config()).unwrap();
    let (spec, sigma) = put_spec(0.25);
    let pp = mango_option::PricingParams { spec: spec.clone(), volatility: sigma };
    let price = price_american(&pp).unwrap().value();
    let s = std::sync::Arc::new(solver);
    let handles: Vec<_> = (0..2).map(|_| {
        let s = s.clone();
        let spec = spec.clone();
        std::thread::spawn(move || {
            s.solve(&IvQuery { spec, market_price: price }).map(|r| r.implied_vol)
        })
    }).collect();
    for h in handles { assert!(h.join().unwrap().is_ok()); }
}

use mango_option::{PriceTable, PricingParams};

#[test]
fn price_table_queries() {
    let table = PriceTable::new(&base_config()).expect("build table");
    let spec = OptionSpec {
        spot: 100.0, strike: 100.0, maturity: 1.0, dividend_yield: 0.0,
        rate: Rate::Const(0.03), discrete_dividends: vec![], option_type: OptionType::Put,
    };
    let pp = PricingParams { spec: spec.clone(), volatility: 0.25 };
    let price = table.price(&pp);
    assert!(price.is_finite() && price > 0.0);
    assert!(table.vega(&pp).is_finite());
    let delta = table.delta(&pp).expect("delta");
    assert!(delta < 0.0, "put delta should be negative: {}", delta);
    assert!(table.gamma(&pp).is_ok());
    assert!(table.theta(&pp).is_ok());
    assert!(table.rho(&pp).is_ok());
    assert_eq!(table.option_type(), OptionType::Put);

    // Reference FDM price for sanity (interpolation tolerance is loose).
    let fdm = price_american(&pp).unwrap().value();
    assert!((price - fdm).abs() < 0.5, "interp {} vs fdm {}", price, fdm);

    // Opt-in validation.
    assert!(table.validate(&pp).is_ok());
    let oob = PricingParams { spec, volatility: 5.0 }; // far outside vol grid
    assert!(table.validate(&oob).is_err());
}

#[test]
fn price_table_derives_iv_solver() {
    let table = PriceTable::new(&base_config()).unwrap();
    let solver = table.iv_solver(None).expect("derive solver");
    let (spec, sigma) = put_spec(0.25);
    let pp = PricingParams { spec: spec.clone(), volatility: sigma };
    let price = price_american(&pp).unwrap().value();
    let r = solver.solve(&IvQuery { spec, market_price: price }).expect("solve");
    assert!((r.implied_vol - sigma).abs() < 1e-2);
}

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Tests for mango-option Python bindings
"""

import sys
import os
import tempfile
import numpy as np
import mango_option


def test_option_types():
    """Test OptionType enum"""
    print("Testing OptionType enum...")
    assert mango_option.OptionType.CALL is not None
    assert mango_option.OptionType.PUT is not None
    print("✓ OptionType enum works")


def test_yield_curve():
    """Test YieldCurve creation and methods"""
    print("Testing YieldCurve...")

    # Flat curve
    flat = mango_option.YieldCurve.flat(0.05)
    assert abs(flat.rate(1.0) - 0.05) < 1e-10
    assert abs(flat.zero_rate(1.0) - 0.05) < 1e-10
    print("✓ Flat YieldCurve works")

    # From discounts
    tenors = [0.0, 0.25, 0.5, 1.0, 2.0]
    discounts = [1.0, 0.9876, 0.9753, 0.9512, 0.9048]
    curve = mango_option.YieldCurve.from_discounts(tenors, discounts)
    assert curve.discount(1.0) > 0
    print("✓ YieldCurve from discounts works")


def test_iv_query():
    """Test IVQuery construction"""
    print("Testing IVQuery...")

    query = mango_option.IVQuery()
    query.spot = 100.0
    query.strike = 100.0
    query.maturity = 1.0
    query.rate = 0.05
    query.dividend_yield = 0.02
    query.option_type = mango_option.OptionType.PUT
    query.market_price = 10.0

    assert query.spot == 100.0
    assert query.strike == 100.0
    print("✓ IVQuery works")


def test_iv_solver_fdm():
    """Test IVSolver (FDM-based)"""
    print("Testing IVSolver...")

    config = mango_option.IVSolverConfig()
    solver = mango_option.IVSolver(config)

    query = mango_option.IVQuery()
    query.spot = 100.0
    query.strike = 100.0
    query.maturity = 1.0
    query.rate = 0.05
    query.dividend_yield = 0.02
    query.option_type = mango_option.OptionType.PUT
    query.market_price = 10.0

    success, result, error = solver.solve(query)
    if success:
        print(f"✓ IVSolver solved: IV = {result.implied_vol:.4f}")
    else:
        print(f"✓ IVSolver ran (error: {error.message})")


def test_american_option_price():
    """Test american_option_price function"""
    print("Testing american_option_price...")

    params = mango_option.PricingParams()
    params.strike = 100.0
    params.spot = 100.0
    params.maturity = 1.0
    params.volatility = 0.20
    params.rate = 0.05
    params.dividend_yield = 0.02
    params.option_type = mango_option.OptionType.PUT

    result = mango_option.american_option_price(params)
    delta = result.delta()
    print(f"✓ American option price computed, delta = {delta:.4f}")


def test_american_option_price_with_accuracy():
    """Test american_option_price with accuracy profile"""
    print("Testing american_option_price with accuracy profile...")

    params = mango_option.PricingParams()
    params.strike = 100.0
    params.spot = 100.0
    params.maturity = 1.0
    params.volatility = 0.20
    params.rate = 0.05
    params.dividend_yield = 0.02
    params.option_type = mango_option.OptionType.PUT

    result = mango_option.american_option_price(
        params, accuracy=mango_option.GridAccuracyProfile.HIGH)
    price = result.value_at(100.0)
    assert price > 0, f"Expected positive price, got {price}"
    print(f"  Price (HIGH accuracy): {price:.6f}")
    print(f"  Delta: {result.delta():.4f}")
    print(f"  Gamma: {result.gamma():.4f}")
    print(f"  Theta: {result.theta():.4f}")
    print("✓ Accuracy profile works")


def test_american_option_discrete_dividends():
    """Test american_option_price with discrete dividends"""
    print("Testing american_option_price with discrete dividends...")

    params = mango_option.PricingParams()
    params.strike = 100.0
    params.spot = 100.0
    params.maturity = 1.0
    params.volatility = 0.20
    params.rate = 0.05
    params.dividend_yield = 0.0
    params.option_type = mango_option.OptionType.PUT
    params.discrete_dividends = [mango_option.Dividend(0.25, 2.0), mango_option.Dividend(0.75, 2.0)]

    result = mango_option.american_option_price(params)
    price_div = result.value_at(100.0)
    assert price_div > 0, f"Expected positive price, got {price_div}"

    # Compare with no dividends
    params.discrete_dividends = []
    result_no_div = mango_option.american_option_price(params)
    price_no_div = result_no_div.value_at(100.0)

    print(f"  Price with dividends: {price_div:.6f}")
    print(f"  Price without dividends: {price_no_div:.6f}")
    print("✓ Discrete dividends work")


def test_american_option_yield_curve():
    """Test american_option_price with yield curve rate"""
    print("Testing american_option_price with yield curve...")

    params = mango_option.PricingParams()
    params.strike = 100.0
    params.spot = 100.0
    params.maturity = 1.0
    params.volatility = 0.20
    params.rate = mango_option.YieldCurve.flat(0.05)
    params.dividend_yield = 0.02
    params.option_type = mango_option.OptionType.PUT

    result = mango_option.american_option_price(params)
    price = result.value_at(100.0)
    assert price > 0, f"Expected positive price, got {price}"
    print(f"  Price with YieldCurve: {price:.6f}")
    print("✓ Yield curve pricing works")


def test_batch_solver():
    """Test BatchAmericanOptionSolver"""
    print("Testing BatchAmericanOptionSolver...")

    batch = []
    for K in [90.0, 95.0, 100.0, 105.0, 110.0]:
        p = mango_option.PricingParams()
        p.spot = 100.0
        p.strike = K
        p.maturity = 1.0
        p.volatility = 0.20
        p.rate = 0.05
        p.dividend_yield = 0.02
        p.option_type = mango_option.OptionType.PUT
        batch.append(p)

    solver = mango_option.BatchAmericanOptionSolver()
    solver.set_grid_accuracy(mango_option.GridAccuracyProfile.LOW)

    results, failed_count = solver.solve_batch(batch, use_shared_grid=True)
    assert failed_count == 0, f"Expected 0 failures, got {failed_count}"
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    for i, (success, result, error) in enumerate(results):
        assert success, f"Option {i} failed: {error}"
        price = result.value_at(100.0)
        assert price > 0, f"Option {i}: expected positive price, got {price}"
        print(f"  K={batch[i].strike}: price={price:.4f}, delta={result.delta():.4f}")

    print(f"✓ Batch solver works ({len(results)} options, {failed_count} failed)")


def test_batch_solver_per_option_grids():
    """Test BatchAmericanOptionSolver with per-option grid estimation"""
    print("Testing BatchAmericanOptionSolver with per-option grids...")

    batch = []
    for T in [0.25, 0.5, 1.0]:
        p = mango_option.PricingParams()
        p.spot = 100.0
        p.strike = 100.0
        p.maturity = T
        p.volatility = 0.20
        p.rate = 0.05
        p.dividend_yield = 0.02
        p.option_type = mango_option.OptionType.PUT
        batch.append(p)

    solver = mango_option.BatchAmericanOptionSolver()
    results, failed_count = solver.solve_batch(batch, use_shared_grid=False)
    assert failed_count == 0
    assert len(results) == 3

    for i, (success, result, error) in enumerate(results):
        assert success
        price = result.value_at(100.0)
        print(f"  T={batch[i].maturity}: price={price:.4f}")

    print("✓ Per-option grid batch works")


def test_iv_solver_interpolated():
    """Test InterpolatedIVSolver via make_interpolated_iv_solver factory"""
    print("Testing InterpolatedIVSolver...")

    config = mango_option.IVSolverFactoryConfig()
    config.option_type = mango_option.OptionType.PUT
    config.spot = 100.0
    config.dividend_yield = 0.02
    config.grid.moneyness = [0.8, 0.9, 1.0, 1.1, 1.2]
    config.grid.vol = [0.10, 0.20, 0.30, 0.40]
    config.grid.rate = [0.01, 0.03, 0.05, 0.07]

    backend = mango_option.BSplineBackend()
    backend.maturity_grid = [0.1, 0.25, 0.5, 1.0]
    config.backend = backend

    solver = mango_option.make_interpolated_iv_solver(config)
    print("✓ Created InterpolatedIVSolver via factory")

    query = mango_option.IVQuery()
    query.spot = 100.0
    query.strike = 100.0
    query.maturity = 0.5
    query.rate = 0.05
    query.dividend_yield = 0.02
    query.option_type = mango_option.OptionType.PUT
    query.market_price = 5.0

    success, result, error = solver.solve(query)
    print(f"✓ solve ran: success={success}")

    # Test batch
    queries = [query, query, query]
    results, failed_count = solver.solve_batch(queries)
    print(f"✓ solve_batch ran: {len(results)} results, {failed_count} failed")


def test_error_handling():
    """Test error conditions are properly raised"""
    print("Testing error handling...")

    # PriceTableAxes with wrong number of grids
    axes = mango_option.PriceTableAxes()
    try:
        axes.grids = [np.array([1.0, 2.0, 3.0, 4.0])]  # Only 1 grid instead of 4
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "exactly 4" in str(e)
        print("✓ Wrong number of grids raises ValueError")


def test_iv_solver_config_defaults():
    """Verify IVSolverFactoryConfig field defaults"""
    print("Testing IVSolverFactoryConfig defaults...")

    config = mango_option.IVSolverFactoryConfig()
    assert config.option_type == mango_option.OptionType.PUT
    assert config.spot == 100.0
    assert config.dividend_yield == 0.0
    assert config.adaptive is None
    print("✓ Config defaults correct")

    # Setting adaptive
    adaptive = mango_option.AdaptiveGridParams()
    assert adaptive.target_iv_error > 0
    config.adaptive = adaptive
    assert config.adaptive is not None
    config.adaptive = None
    assert config.adaptive is None
    print("✓ Adaptive optional works")

    # Backend variant
    bspline = mango_option.BSplineBackend()
    config.backend = bspline
    cheb = mango_option.ChebyshevBackend()
    config.backend = cheb
    print("✓ Backend variant setter works")

    # Discrete dividends optional
    div_config = mango_option.DiscreteDividendConfig()
    config.discrete_dividends = div_config
    assert config.discrete_dividends is not None
    config.discrete_dividends = None
    assert config.discrete_dividends is None
    print("✓ Discrete dividends optional works")


if __name__ == "__main__":
    tests = [
        test_option_types,
        test_yield_curve,
        test_iv_query,
        test_iv_solver_fdm,
        test_american_option_price,
        test_american_option_price_with_accuracy,
        test_american_option_discrete_dividends,
        test_american_option_yield_curve,
        test_batch_solver,
        test_batch_solver_per_option_grids,
        test_iv_solver_interpolated,
        test_error_handling,
        test_iv_solver_config_defaults,
    ]

    failed = 0
    for test in tests:
        try:
            test()
            print()
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    if failed == 0:
        print("🎉 All Python binding tests passed!")
    else:
        print(f"❌ {failed}/{len(tests)} tests failed")
        sys.exit(1)

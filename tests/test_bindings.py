#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

import math
import pathlib
import tempfile

import mango_option as mo


def make_pricing_params(rate=0.05):
    p = mo.PricingParams()
    p.spot = 100.0
    p.strike = 100.0
    p.maturity = 0.5
    p.volatility = 0.20
    p.rate = rate
    p.dividend_yield = 0.02
    p.option_type = mo.OptionType.PUT
    return p


def make_bspline_4d_off_grid_params():
    p = make_pricing_params()
    p.strike = 97.0
    p.maturity = 0.37
    p.volatility = 0.23
    p.rate = 0.037
    return p


def make_iv_query(price, params=None, rate=0.05):
    q = mo.IVQuery()
    if params is None:
        q.spot = 100.0
        q.strike = 100.0
        q.maturity = 0.5
        q.rate = rate
        q.dividend_yield = 0.02
        q.option_type = mo.OptionType.PUT
    else:
        q.spot = params.spot
        q.strike = params.strike
        q.maturity = params.maturity
        q.rate = params.rate
        q.dividend_yield = params.dividend_yield
        q.option_type = params.option_type
    q.market_price = price
    return q


def make_price_table_config():
    config = mo.PriceTableConfig()
    config.option_type = mo.OptionType.PUT
    config.spot = 100.0
    config.dividend_yield = 0.02
    config.grid.moneyness = [0.8, 0.9, 1.0, 1.1, 1.2]
    config.grid.vol = [0.10, 0.20, 0.30, 0.40]
    config.grid.rate = [0.01, 0.03, 0.05, 0.07]
    backend = mo.BSplineBackend()
    backend.maturity_grid = [0.1, 0.25, 0.5, 1.0]
    config.backend = backend
    return config


def assert_finite_number(value):
    assert isinstance(value, float)
    assert math.isfinite(value)


def test_rate_spec_conversions():
    p = make_pricing_params(rate=1)
    assert p.rate == 1.0
    p.rate = 0.05
    assert p.rate == 0.05
    curve = mo.YieldCurve.flat(0.04)
    p.rate = curve
    assert isinstance(p.rate, mo.YieldCurve)

    try:
        p.rate = "0.05"
        raise AssertionError("string rate should fail")
    except mo.TypeConversionError:
        pass


def test_sequence_conversions_for_vectors_and_axes():
    config = make_price_table_config()
    config.grid.moneyness = (0.8, 0.9, 1.0, 1.1, 1.2)
    assert list(config.grid.moneyness) == [0.8, 0.9, 1.0, 1.1, 1.2]

    axes = mo.PriceTableAxes()
    axes.grids = [
        [0.8, 1.0, 1.2, 1.4],
        (0.1, 0.5, 1.0, 1.5),
        [0.1, 0.2, 0.3, 0.4],
        (0.01, 0.03, 0.05, 0.07),
    ]
    assert axes.shape() == (4, 4, 4, 4)
    assert axes.total_points() == 256
    assert list(axes.grids[0]) == [0.8, 1.0, 1.2, 1.4]
    axes.names = ("moneyness", "maturity", "vol", "rate")
    assert list(axes.names) == ["moneyness", "maturity", "vol", "rate"]


def test_optional_and_backend_variant_conversions():
    config = make_price_table_config()
    assert config.adaptive is None
    adaptive = mo.AdaptiveGridParams()
    adaptive.target_iv_error = 0.001
    config.adaptive = adaptive
    assert isinstance(config.adaptive, mo.AdaptiveGridParams)
    config.adaptive = None
    assert config.adaptive is None

    bspline = mo.BSplineBackend()
    bspline.maturity_grid = [0.25, 0.5, 1.0]
    config.backend = bspline
    assert isinstance(config.backend, mo.BSplineBackend)

    cheb = mo.ChebyshevBackend()
    cheb.maturity = 1.0
    cheb.num_pts = [8, 6, 6, 4]
    config.backend = cheb
    assert isinstance(config.backend, mo.ChebyshevBackend)

    dim = mo.DimensionlessBackend()
    dim.maturity = 1.0
    dim.interpolant = mo.DimensionlessInterpolant.BSPLINE
    config.backend = dim
    assert isinstance(config.backend, mo.DimensionlessBackend)

    try:
        config.backend = object()
        raise AssertionError("invalid backend should fail")
    except mo.TypeConversionError:
        pass


def test_dividend_conversions():
    p = make_pricing_params()
    p.discrete_dividends = [mo.Dividend(0.25, 1.0), mo.Dividend(0.75, 1.0)]
    assert len(p.discrete_dividends) == 2
    p.discrete_dividends = [(0.25, 1.0), (0.75, 1.0)]
    assert len(p.discrete_dividends) == 2
    assert p.discrete_dividends[0].calendar_time == 0.25

    config = make_price_table_config()
    divs = mo.DiscreteDividendConfig()
    divs.maturity = 1.0
    divs.discrete_dividends = [(0.25, 1.0), (0.75, 1.0)]
    config.discrete_dividends = divs
    assert config.discrete_dividends is not None
    config.discrete_dividends = None
    assert config.discrete_dividends is None


def test_bspline_4d_price_table_workflow_and_persistence_paths():
    table = mo.make_price_table(make_price_table_config())
    assert table.surface_type == "bspline_4d"

    p = make_bspline_4d_off_grid_params()
    price = table.price(p)
    assert_finite_number(price)
    assert_finite_number(table.vega(p))
    assert_finite_number(table.delta(p))
    assert_finite_number(table.gamma(p))
    assert_finite_number(table.theta(p))
    assert_finite_number(table.rho(p))

    q = make_iv_query(price, p)
    iv = table.solve_iv(q)
    assert isinstance(iv, mo.IVSuccess)

    solver = table.make_iv_solver()
    success, result, error = solver.solve(q)
    assert success
    assert isinstance(result, mo.IVSuccess)

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "surface.parquet"
        table.save(path)
        loaded = mo.PriceTable.load(path)
        assert loaded.surface_type == table.surface_type
        assert_finite_number(loaded.price(p))


def test_legacy_interpolated_iv_solver_factory_still_works():
    solver = mo.make_interpolated_iv_solver(make_price_table_config())
    table = mo.make_price_table(make_price_table_config())
    price = table.price(make_pricing_params())
    success, result, error = solver.solve(make_iv_query(price))
    assert success
    assert isinstance(result, mo.IVSuccess)


def test_typed_exceptions_for_validation_and_persistence():
    config = make_price_table_config()
    config.grid.moneyness = [-1.0, 0.9, 1.0, 1.1]
    try:
        mo.make_price_table(config)
        raise AssertionError("invalid moneyness should fail")
    except mo.ValidationError as e:
        assert hasattr(e, "code")

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "missing.parquet"
        try:
            mo.PriceTable.load(path)
            raise AssertionError("missing file should fail")
        except mo.PriceTableError as e:
            assert hasattr(e, "code")


def main():
    tests = [
        test_rate_spec_conversions,
        test_sequence_conversions_for_vectors_and_axes,
        test_optional_and_backend_variant_conversions,
        test_dividend_conversions,
        test_bspline_4d_price_table_workflow_and_persistence_paths,
        test_legacy_interpolated_iv_solver_factory_still_works,
        test_typed_exceptions_for_validation_and_persistence,
    ]
    for test in tests:
        test()


if __name__ == "__main__":
    main()

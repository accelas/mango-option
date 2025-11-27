"""IV computation service using mango_iv C++ bindings."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add bazel-bin to path for mango_iv module
BAZEL_BIN = Path(__file__).parents[5] / "bazel-bin" / "python"
if BAZEL_BIN.exists():
    sys.path.insert(0, str(BAZEL_BIN))

# Try to import mango_iv, fall back to mock for development
try:
    import mango_iv

    HAS_MANGO_IV = True
except ImportError:
    HAS_MANGO_IV = False
    print("Warning: mango_iv not found, using fallback IV computation")


def compute_iv_surface(
    chain_data: dict,
    option_type: str = "put",
    rate: float = 0.05,
) -> dict:
    """
    Compute IV surface from option chain data.

    Args:
        chain_data: Option chain from yfinance_service
        option_type: 'put' or 'call'
        rate: Risk-free rate (default 5%)

    Returns:
        Surface data for Plotly visualization
    """
    if "error" in chain_data:
        return chain_data

    spot = chain_data["spot"]
    dividend_yield = chain_data.get("dividend_yield", 0.0)

    # Collect all options across expirations
    strikes: list[float] = []
    maturities: list[float] = []
    ivs: list[float] = []
    prices: list[float] = []

    now = datetime.now()

    for expiry_str, chain in chain_data["chains"].items():
        # Parse expiry and compute time to maturity
        expiry = datetime.strptime(expiry_str, "%Y-%m-%d")
        tau = (expiry - now).days / 365.0

        if tau <= 0:
            continue  # Skip expired options

        options = chain["puts"] if option_type == "put" else chain["calls"]

        for opt in options:
            strike = opt["strike"]
            bid = opt["bid"]
            ask = opt["ask"]

            # Use mid price
            if bid <= 0 or ask <= 0:
                continue
            mid_price = (bid + ask) / 2.0

            # Compute IV
            if HAS_MANGO_IV:
                iv = _compute_iv_mango(
                    spot=spot,
                    strike=strike,
                    maturity=tau,
                    rate=rate,
                    dividend_yield=dividend_yield,
                    market_price=mid_price,
                    option_type=option_type,
                )
            else:
                # Fallback: use yfinance's implied vol if available
                iv = opt.get("implied_vol", 0.0)
                if iv <= 0:
                    iv = _estimate_iv_bisection(
                        spot=spot,
                        strike=strike,
                        maturity=tau,
                        rate=rate,
                        dividend_yield=dividend_yield,
                        market_price=mid_price,
                        option_type=option_type,
                    )

            if iv is not None and 0.01 < iv < 3.0:  # Reasonable IV range
                strikes.append(strike)
                maturities.append(tau)
                ivs.append(iv)
                prices.append(mid_price)

    if not ivs:
        return {"error": "Could not compute any valid IVs"}

    # Compute moneyness for better visualization
    moneyness = [s / spot for s in strikes]

    return {
        "symbol": chain_data["symbol"],
        "spot": spot,
        "option_type": option_type,
        "rate": rate,
        "dividend_yield": dividend_yield,
        "quote_time": chain_data["quote_time"],
        "is_eod": chain_data["is_eod"],
        "surface": {
            "strikes": strikes,
            "moneyness": moneyness,
            "maturities": maturities,
            "ivs": ivs,
            "prices": prices,
        },
        "stats": {
            "n_points": len(ivs),
            "iv_min": min(ivs),
            "iv_max": max(ivs),
            "iv_mean": sum(ivs) / len(ivs),
        },
    }


def _compute_iv_mango(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    market_price: float,
    option_type: str,
) -> float | None:
    """Compute IV using mango_iv C++ solver."""
    try:
        # Create query
        query = mango_iv.IVQuery()
        query.spot = spot
        query.strike = strike
        query.maturity = maturity
        query.rate = rate
        query.dividend_yield = dividend_yield
        query.market_price = market_price
        query.type = (
            mango_iv.OptionType.PUT
            if option_type == "put"
            else mango_iv.OptionType.CALL
        )

        # Create solver with default config
        config = mango_iv.IVSolverFDMConfig()
        solver = mango_iv.IVSolverFDM(config)

        # Solve
        success, result, error = solver.solve_impl(query)

        if success:
            return result.implied_vol
        return None

    except Exception:
        return None


def _estimate_iv_bisection(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    market_price: float,
    option_type: str,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """
    Fallback IV estimation using bisection on Black-Scholes.

    This is a simplified European approximation for development/testing.
    """
    from math import exp, log, sqrt

    from scipy.stats import norm

    def bs_price(sigma: float) -> float:
        if sigma <= 0:
            return 0.0
        d1 = (log(spot / strike) + (rate - dividend_yield + 0.5 * sigma**2) * maturity) / (
            sigma * sqrt(maturity)
        )
        d2 = d1 - sigma * sqrt(maturity)

        if option_type == "call":
            return spot * exp(-dividend_yield * maturity) * norm.cdf(d1) - strike * exp(
                -rate * maturity
            ) * norm.cdf(d2)
        else:
            return strike * exp(-rate * maturity) * norm.cdf(-d2) - spot * exp(
                -dividend_yield * maturity
            ) * norm.cdf(-d1)

    # Bisection
    low, high = 0.001, 3.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        price = bs_price(mid)
        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            low = mid
        else:
            high = mid

    return (low + high) / 2

#!/usr/bin/env python3
"""
Analyze option chain dimensions for price table sizing.
Fetches real market data from Yahoo Finance.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def analyze_option_chain(ticker_symbol: str = "QQQ"):
    """Fetch and analyze option chain for price table sizing."""

    print(f"Fetching option chain for {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)

    # Get current price
    hist = ticker.history(period="1d")
    spot = hist['Close'].iloc[-1]
    print(f"\nSpot price: ${spot:.2f}")

    # Get all expiration dates
    expirations = ticker.options
    print(f"\nExpirations available: {len(expirations)}")

    # Collect all option data
    all_strikes = set()
    all_ivs = []
    maturities_days = []

    today = datetime.now()

    for exp_date in expirations:
        try:
            chain = ticker.option_chain(exp_date)
            calls = chain.calls
            puts = chain.puts

            # Calculate days to expiry
            exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
            days_to_exp = (exp_dt - today).days
            if days_to_exp <= 0:
                continue

            maturities_days.append(days_to_exp)

            # Collect strikes
            all_strikes.update(calls['strike'].tolist())
            all_strikes.update(puts['strike'].tolist())

            # Collect IVs (filter valid ones)
            for iv in calls['impliedVolatility'].dropna():
                if 0.01 < iv < 3.0:  # Filter outliers
                    all_ivs.append(iv)
            for iv in puts['impliedVolatility'].dropna():
                if 0.01 < iv < 3.0:
                    all_ivs.append(iv)

        except Exception as e:
            print(f"  Skipping {exp_date}: {e}")
            continue

    # Convert to sorted arrays
    strikes = np.array(sorted(all_strikes))
    maturities_days = np.array(sorted(set(maturities_days)))
    ivs = np.array(all_ivs)

    # Compute moneyness = spot/strike
    moneyness = spot / strikes

    print("\n" + "="*60)
    print("OPTION CHAIN ANALYSIS")
    print("="*60)

    # Strikes / Moneyness
    print(f"\n--- STRIKES (Nm = {len(strikes)}) ---")
    print(f"  Range: ${strikes.min():.2f} to ${strikes.max():.2f}")
    print(f"  Moneyness range: {moneyness.min():.3f} to {moneyness.max():.3f}")
    print(f"  ATM region (0.95-1.05): {np.sum((moneyness > 0.95) & (moneyness < 1.05))} strikes")

    # Maturities
    print(f"\n--- MATURITIES (Nt = {len(maturities_days)}) ---")
    print(f"  Range: {maturities_days.min()} to {maturities_days.max()} days")
    print(f"  In years: {maturities_days.min()/252:.4f} to {maturities_days.max()/252:.2f}")
    print(f"  Short-dated (< 7 days): {np.sum(maturities_days < 7)}")
    print(f"  Weekly (7-30 days): {np.sum((maturities_days >= 7) & (maturities_days < 30))}")
    print(f"  Monthly (30-90 days): {np.sum((maturities_days >= 30) & (maturities_days < 90))}")
    print(f"  Quarterly+ (90+ days): {np.sum(maturities_days >= 90)}")

    # Implied Volatilities
    print(f"\n--- IMPLIED VOLATILITIES ---")
    print(f"  Samples: {len(ivs)}")
    print(f"  Range: {ivs.min()*100:.1f}% to {ivs.max()*100:.1f}%")
    print(f"  Mean: {ivs.mean()*100:.1f}%")
    print(f"  Std: {ivs.std()*100:.1f}%")
    print(f"  Percentiles: 5th={np.percentile(ivs, 5)*100:.1f}%, "
          f"50th={np.percentile(ivs, 50)*100:.1f}%, "
          f"95th={np.percentile(ivs, 95)*100:.1f}%")

    # Estimate grid sizing for price table
    print("\n" + "="*60)
    print("RECOMMENDED PRICE TABLE GRID")
    print("="*60)

    # Moneyness grid - cover the actual range with some buffer
    m_min = max(0.5, moneyness.min() * 0.95)
    m_max = min(2.0, moneyness.max() * 1.05)

    # Volatility grid - cover observed range with buffer for regime changes
    iv_min = max(0.05, np.percentile(ivs, 1) * 0.8)
    iv_max = min(1.5, np.percentile(ivs, 99) * 1.5)

    # For IV precision of 1e-4 to 1e-3, we need sufficient grid density
    # B-spline interpolation error ~ O(h^4) for cubic splines
    # For 1e-4 precision: need ~20-30 points per dimension in active region

    print(f"\n--- For 1e-4 to 1e-3 IV precision ---")

    # Moneyness
    Nm = 50  # Dense enough for cubic B-spline accuracy
    print(f"\nMoneyness (Nm = {Nm}):")
    print(f"  Range: [{m_min:.3f}, {m_max:.3f}]")
    print(f"  Suggested: non-uniform, denser near ATM (m=1.0)")

    # Maturity - non-uniform grid
    # Short-dated need fine resolution for theta/gamma
    mat_grid = [
        1, 2, 3, 5, 7,           # Daily for first week
        10, 14, 21,              # ~weekly for first month
        30, 45, 60,              # bi-weekly for 1-2 months
        90, 120, 180, 252, 365   # monthly+ for longer dates
    ]
    mat_grid = [d for d in mat_grid if d <= maturities_days.max()]
    Nt = len(mat_grid)
    print(f"\nMaturity (Nt = {Nt}):")
    print(f"  Days: {mat_grid}")
    print(f"  Years: {[f'{d/252:.4f}' for d in mat_grid[:5]]}...")

    # Volatility
    Nv = 25  # Sufficient for smooth IV surface
    print(f"\nVolatility (Nσ = {Nv}):")
    print(f"  Range: [{iv_min*100:.1f}%, {iv_max*100:.1f}%]")

    # Rate - minimal, doesn't change intraday
    Nr = 3
    print(f"\nRate (Nr = {Nr}):")
    print(f"  Range: [-0.01, 0.03, 0.07] (covers most scenarios)")

    # Memory calculation
    total_points = Nm * Nt * Nv * Nr
    memory_bytes = total_points * 8  # doubles
    memory_mb = memory_bytes / (1024 * 1024)

    print(f"\n--- MEMORY ESTIMATE ---")
    print(f"  Grid points: {Nm} × {Nt} × {Nv} × {Nr} = {total_points:,}")
    print(f"  B-spline coefficients: {memory_mb:.2f} MB")
    print(f"  With PUT + CALL tables: {memory_mb * 2:.2f} MB")

    # Build time estimate
    n_pde_solves = Nv * Nr
    time_per_solve_ms = 15  # typical
    build_time_s = (n_pde_solves * time_per_solve_ms) / 1000
    print(f"\n--- BUILD TIME ESTIMATE ---")
    print(f"  PDE solves: {Nv} × {Nr} = {n_pde_solves}")
    print(f"  Serial time: ~{build_time_s:.1f}s")
    print(f"  With 8-core parallel: ~{build_time_s/6:.1f}s")

    # Precision analysis
    print(f"\n--- INTERPOLATION PRECISION ---")
    print(f"  Cubic B-spline error: O(h^4)")
    print(f"  With Nm={Nm}, Nv={Nv}: expect ~1e-4 relative error")
    print(f"  For IV in [{iv_min*100:.0f}%, {iv_max*100:.0f}%]:")
    print(f"    Absolute IV error: ~{iv_min * 1e-4 * 100:.4f}% to {iv_max * 1e-4 * 100:.4f}%")

    return {
        'spot': spot,
        'strikes': strikes,
        'maturities_days': maturities_days,
        'ivs': ivs,
        'moneyness': moneyness,
        'recommended': {
            'Nm': Nm, 'Nt': Nt, 'Nv': Nv, 'Nr': Nr,
            'm_range': (m_min, m_max),
            'iv_range': (iv_min, iv_max),
            'mat_grid': mat_grid
        }
    }


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
    analyze_option_chain(ticker)

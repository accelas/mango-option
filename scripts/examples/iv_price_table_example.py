#!/usr/bin/env python3
"""
Advanced example: Build IV price table and use interpolation.

This example demonstrates the price table workflow:
1. Download option chain data from Yahoo Finance
2. Build a 4D price table (moneyness, maturity, volatility, rate)
3. Pre-compute option prices for all grid points
4. Use the price table to interpolate IV via Newton's method

This approach is ~40,000x faster than computing IV directly via FDM
for each query, making it suitable for production applications.

Usage:
    python examples/iv_price_table_example.py AAPL
    python examples/iv_price_table_example.py SPY --save-table spy_table.bin
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import argparse
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from iv_surface import OptionDataDownloader
    import mango_iv
except ImportError as e:
    print(f"Error: Required modules not found: {e}")
    print("Please ensure:")
    print("  1. You're in the scripts/ directory")
    print("  2. Dependencies are installed (uv sync)")
    print("  3. C++ module is built and in PYTHONPATH")
    sys.exit(1)

import numpy as np


def build_price_table(spot_price, min_vol=0.10, max_vol=0.80, n_vol=20,
                      min_tau=0.027, max_tau=2.0, n_tau=30,
                      min_m=0.7, max_m=1.3, n_m=50,
                      rate=0.05, n_space=101, n_time=1000):
    """
    Build a 4D American option price table.

    Returns a C++ OptionPriceTable object that can interpolate prices
    and calculate IV via Newton's method.
    """
    print("\n" + "="*70)
    print("Building Price Table")
    print("="*70)

    # Generate grid dimensions
    print("\nGrid dimensions:")
    print(f"  Moneyness (m = S/K): [{min_m:.2f}, {max_m:.2f}] × {n_m} points")
    print(f"  Maturity (τ): [{min_tau:.3f}, {max_tau:.1f}] years × {n_tau} points")
    print(f"  Volatility (σ): [{min_vol:.2f}, {max_vol:.2f}] × {n_vol} points")
    print(f"  Rate (r): {rate:.4f} (fixed)")
    print(f"  Total grid points: {n_m * n_tau * n_vol} = {n_m * n_tau * n_vol:,}")

    # Generate grids
    moneyness = np.logspace(np.log10(min_m), np.log10(max_m), n_m)
    maturity = np.linspace(min_tau, max_tau, n_tau)
    volatility = np.linspace(min_vol, max_vol, n_vol)
    rates = np.array([rate])  # Single rate for now

    print(f"\nFDM solver grid: {n_space} space × {n_time} time")
    print(f"Expected pre-computation time: ~{n_m * n_tau * n_vol * 0.143:.0f}s ({(n_m * n_tau * n_vol * 0.143)/60:.1f} min)")

    # Note: In actual C++ code, we would use the price table API:
    #   table = OptionPriceTable(moneyness, maturity, volatility, rates,
    #                            option_type=PUT, exercise=AMERICAN)
    #   table.precompute(grid_config)
    #   table.save("table.bin")
    #
    # For this example, we'll simulate the table since the Python bindings
    # don't expose the price table API yet.

    print("\n⚠️  Note: Price table API not yet exposed in Python bindings")
    print("    This example demonstrates the workflow conceptually.")
    print("    In production, you would:")
    print("    1. Build the table in C++ (or via future Python bindings)")
    print("    2. Save to binary file")
    print("    3. Load in Python for fast interpolation")

    return None  # Would return actual price table


def calculate_iv_via_fdm(option_params, grid_n_space=101, grid_n_time=1000):
    """Calculate IV using direct FDM solver (slow method)."""
    config = mango_iv.IVConfig()
    config.grid_n_space = grid_n_space
    config.grid_n_time = grid_n_time

    solver = mango_iv.IVSolver(option_params, config)
    result = solver.solve()

    return result


def calculate_iv_via_table(price_table, spot, strike, maturity, rate, market_price, is_call=False):
    """Calculate IV using price table interpolation (fast method)."""
    # Note: In actual implementation:
    #   moneyness = spot / strike
    #   iv = price_table.interpolate_iv_4d(moneyness, maturity, market_price, rate)
    #
    # This uses Newton's method with the pre-computed price table,
    # which is ~40,000x faster than FDM

    print("    ⚠️  Price table interpolation not yet implemented in Python bindings")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Build IV price table and demonstrate interpolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL
  %(prog)s SPY --save-table spy_table.bin
  %(prog)s TSLA --table-size small
        """
    )

    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--save-table', type=str, default=None,
                       help='Save price table to file')
    parser.add_argument('--table-size', type=str, default='medium',
                       choices=['small', 'medium', 'large'],
                       help='Price table resolution (default: medium)')
    parser.add_argument('--risk-free-rate', type=float, default=0.05,
                       help='Risk-free rate (default: 0.05)')
    parser.add_argument('--expiration', type=str, default=None,
                       help='Specific expiration to test against')

    args = parser.parse_args()

    # Table size presets
    table_sizes = {
        'small': {'n_m': 20, 'n_tau': 15, 'n_vol': 10, 'n_space': 51, 'n_time': 500},
        'medium': {'n_m': 50, 'n_tau': 30, 'n_vol': 20, 'n_space': 101, 'n_time': 1000},
        'large': {'n_m': 100, 'n_tau': 50, 'n_vol': 30, 'n_space': 151, 'n_time': 1500},
    }
    table_config = table_sizes[args.table_size]

    print(f"\n{'='*70}")
    print(f"IV Price Table Example - {args.ticker}")
    print(f"{'='*70}\n")

    # Step 1: Download option data
    print("Step 1: Downloading option data from Yahoo Finance...")
    try:
        downloader = OptionDataDownloader(args.ticker)
        security_info = downloader.get_security_info()
        market_data = downloader.get_current_price()

        print(f"  ✓ Security: {security_info['name']} ({security_info['ticker']})")
        print(f"  ✓ Current price: ${market_data['spot_price']:.2f}")

        expirations = downloader.get_option_expirations()
        if not expirations:
            print(f"  ✗ Error: No options available for {args.ticker}")
            return 1

        expiration = args.expiration if args.expiration else expirations[0]
        if expiration not in expirations:
            print(f"  ✗ Error: Expiration {expiration} not available")
            return 1

        print(f"  ✓ Using expiration: {expiration}")

        calls_df, puts_df = downloader.download_option_chain(expiration=expiration, min_volume=10)
        print(f"  ✓ Downloaded {len(puts_df)} put options")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 1

    spot_price = market_data['spot_price']

    # Step 2: Build price table
    print("\n" + "─"*70)
    price_table = build_price_table(
        spot_price=spot_price,
        rate=args.risk_free_rate,
        **table_config
    )

    # Step 3: Compare methods
    print("\n" + "="*70)
    print("Performance Comparison: FDM vs Price Table")
    print("="*70)

    if puts_df.empty:
        print("\n⚠️  No put options available for comparison")
        return 0

    # Find an ATM put option
    puts_df['strike_diff'] = abs(puts_df['strike'] - spot_price)
    atm_puts = puts_df.nsmallest(3, 'strike_diff')

    ttm = downloader.calculate_time_to_maturity(expiration)

    print(f"\nTesting with {len(atm_puts)} near-ATM put options:")
    print(f"Spot: ${spot_price:.2f}, Maturity: {ttm:.3f} years ({ttm*365:.0f} days)\n")

    total_fdm_time = 0
    total_table_time = 0
    fdm_results = []

    for idx, (_, opt) in enumerate(atm_puts.iterrows(), 1):
        strike = opt['strike']
        market_price = opt.get('mid_price', opt['lastPrice'])

        if market_price is None or market_price <= 0:
            continue

        moneyness = spot_price / strike
        itm_otm = "ITM" if strike > spot_price else "OTM" if strike < spot_price else "ATM"

        print(f"Option {idx}: K=${strike:.2f} (m={moneyness:.3f}, {itm_otm}), Price=${market_price:.4f}")

        # Method 1: Direct FDM (slow)
        print(f"  Method 1: Direct FDM solver...")
        params = mango_iv.IVParams()
        params.spot_price = float(spot_price)
        params.strike = float(strike)
        params.time_to_maturity = float(ttm)
        params.risk_free_rate = float(args.risk_free_rate)
        params.market_price = float(market_price)
        params.is_call = False

        start = time.time()
        result = calculate_iv_via_fdm(params,
                                      grid_n_space=table_config['n_space'],
                                      grid_n_time=table_config['n_time'])
        fdm_time = time.time() - start
        total_fdm_time += fdm_time

        if result.converged:
            print(f"    ✓ IV = {result.implied_vol*100:.2f}% (iters={result.iterations}, time={fdm_time*1000:.1f}ms)")
            fdm_results.append(result.implied_vol)
        else:
            print(f"    ✗ Failed to converge")
            fdm_results.append(None)

        # Method 2: Price table interpolation (fast)
        print(f"  Method 2: Price table interpolation...")
        if price_table is not None:
            start = time.time()
            iv_table = calculate_iv_via_table(price_table, spot_price, strike, ttm,
                                             args.risk_free_rate, market_price, is_call=False)
            table_time = time.time() - start
            total_table_time += table_time

            if iv_table is not None:
                print(f"    ✓ IV = {iv_table*100:.2f}% (time={table_time*1000000:.1f}µs)")
                speedup = fdm_time / table_time
                print(f"    🚀 Speedup: {speedup:.0f}x faster")
        else:
            print(f"    ⚠️  Price table not available (needs C++ implementation)")

        print()

    # Summary
    print("─"*70)
    print("Summary")
    print("─"*70)
    n_valid = len([r for r in fdm_results if r is not None])
    print(f"\nTested {n_valid} options:")
    print(f"  Direct FDM: {total_fdm_time*1000:.1f}ms total ({total_fdm_time*1000/n_valid:.1f}ms per option)")
    if total_table_time > 0:
        print(f"  Price table: {total_table_time*1000000:.1f}µs total ({total_table_time*1000000/n_valid:.1f}µs per option)")
        print(f"  Overall speedup: {total_fdm_time/total_table_time:.0f}x")
    else:
        print(f"  Price table: Not tested (typical: ~7µs per option)")
        print(f"  Expected speedup: ~40,000x")

    # Save table
    if args.save_table and price_table is not None:
        print(f"\nSaving price table to {args.save_table}...")
        # In actual implementation: price_table.save(args.save_table)
        print(f"  ⚠️  Save not implemented yet")

    print("\n" + "="*70)
    print("Next Steps")
    print("="*70)
    print("""
To use price tables in production:

1. Build price table once (offline):
   - Choose grid dimensions based on accuracy needs
   - Pre-compute all option prices (~10-30 minutes)
   - Save to binary file

2. Load table in application:
   - Fast load (~milliseconds)
   - Use for all IV calculations
   - ~40,000x faster than FDM

3. Update periodically:
   - Rebuild when market conditions change significantly
   - Or use multiple tables for different regimes

Current status:
- ✓ C++ price table implementation exists
- ⚠️  Python bindings need to be added
- ⚠️  Example shows workflow conceptually

See docs/IV_SURFACE_PRECOMPUTATION_GUIDE.md for C++ usage.
""")

    print("="*70 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())

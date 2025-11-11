#!/usr/bin/env python3
"""
Simple example: Download option data and calculate implied volatility.

This example demonstrates the complete workflow:
1. Download option chain data from Yahoo Finance
2. Calculate implied volatility using the C++ solver
3. Display results

Usage:
    python examples/simple_iv_calculation.py AAPL
    python examples/simple_iv_calculation.py SPY --expiration 2025-12-19
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from iv_surface import OptionDataDownloader, IVCalculator, validate_option_price
except ImportError:
    print("Error: iv_surface module not found.")
    print("Please ensure you're in the scripts/ directory and dependencies are installed.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download option data and calculate implied volatility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL
  %(prog)s SPY --expiration 2025-12-19
  %(prog)s TSLA --min-volume 50
        """
    )

    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL, SPY)')
    parser.add_argument('--expiration', type=str, default=None,
                       help='Specific expiration date (YYYY-MM-DD). If not provided, uses nearest expiration.')
    parser.add_argument('--min-volume', type=int, default=10,
                       help='Minimum option volume filter (default: 10)')
    parser.add_argument('--risk-free-rate', type=float, default=0.05,
                       help='Risk-free rate (default: 0.05 = 5%%)')
    parser.add_argument('--grid-n-space', type=int, default=101,
                       help='Spatial grid points for PDE solver (default: 101)')
    parser.add_argument('--grid-n-time', type=int, default=1000,
                       help='Time steps for PDE solver (default: 1000)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Implied Volatility Calculator - {args.ticker}")
    print(f"{'='*70}\n")

    # Step 1: Download option data
    print("Step 1: Downloading option data from Yahoo Finance...")
    try:
        downloader = OptionDataDownloader(args.ticker)
        security_info = downloader.get_security_info()
        market_data = downloader.get_current_price()

        print(f"  ✓ Security: {security_info['name']} ({security_info['ticker']})")
        print(f"  ✓ Current price: ${market_data['spot_price']:.2f}")
        print(f"  ✓ Exchange: {security_info.get('exchange', 'N/A')}")

        # Get expiration dates
        expirations = downloader.get_option_expirations()
        if not expirations:
            print(f"  ✗ Error: No options available for {args.ticker}")
            return 1

        # Use specified expiration or nearest
        expiration = args.expiration if args.expiration else expirations[0]
        if expiration not in expirations:
            print(f"  ✗ Error: Expiration {expiration} not available")
            print(f"  Available expirations: {', '.join(expirations[:5])}...")
            return 1

        print(f"  ✓ Using expiration: {expiration}")

        # Download option chain
        calls_df, puts_df = downloader.download_option_chain(
            expiration=expiration,
            min_volume=args.min_volume
        )

        print(f"  ✓ Found {len(calls_df)} calls and {len(puts_df)} puts")

    except Exception as e:
        print(f"  ✗ Error downloading data: {e}")
        return 1

    # Step 2: Calculate implied volatility
    print(f"\nStep 2: Calculating implied volatility...")
    print(f"  Grid: {args.grid_n_space} space × {args.grid_n_time} time")

    try:
        calculator = IVCalculator(
            grid_n_space=args.grid_n_space,
            grid_n_time=args.grid_n_time
        )

        spot_price = market_data['spot_price']
        ttm = downloader.calculate_time_to_maturity(expiration)

        print(f"  Time to maturity: {ttm:.3f} years ({ttm*365:.0f} days)")

        # Calculate IV for calls
        if not calls_df.empty:
            print(f"\n  Processing {len(calls_df)} call options...")
            calls_with_iv = calculator.calculate_chain_iv(
                calls_df, spot_price, args.risk_free_rate, ttm, is_call=True
            )
            n_converged_calls = calls_with_iv['converged'].sum()
            print(f"    ✓ Converged: {n_converged_calls}/{len(calls_df)} ({100*n_converged_calls/len(calls_df):.1f}%)")
        else:
            calls_with_iv = calls_df
            n_converged_calls = 0

        # Calculate IV for puts
        if not puts_df.empty:
            print(f"  Processing {len(puts_df)} put options...")
            puts_with_iv = calculator.calculate_chain_iv(
                puts_df, spot_price, args.risk_free_rate, ttm, is_call=False
            )
            n_converged_puts = puts_with_iv['converged'].sum()
            print(f"    ✓ Converged: {n_converged_puts}/{len(puts_df)} ({100*n_converged_puts/len(puts_df):.1f}%)")
        else:
            puts_with_iv = puts_df
            n_converged_puts = 0

    except Exception as e:
        print(f"  ✗ Error calculating IV: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Display results
    print(f"\n{'='*70}")
    print("Results Summary")
    print(f"{'='*70}\n")

    total_converged = n_converged_calls + n_converged_puts
    total_options = len(calls_df) + len(puts_df)
    print(f"Total options: {total_options}")
    print(f"Successful IV calculations: {total_converged} ({100*total_converged/total_options:.1f}%)")

    # Display sample results for ATM options
    print(f"\n{'─'*70}")
    print("Sample Results (near ATM)")
    print(f"{'─'*70}")

    if not calls_with_iv.empty and n_converged_calls > 0:
        # Find ATM call (strike closest to spot)
        calls_converged = calls_with_iv[calls_with_iv['converged'] == True].copy()
        calls_converged['strike_diff'] = abs(calls_converged['strike'] - spot_price)
        atm_call = calls_converged.nsmallest(1, 'strike_diff').iloc[0]

        print("\n📈 ATM Call Option:")
        print(f"  Strike: ${atm_call['strike']:.2f}")
        print(f"  Market Price: ${atm_call.get('mid_price', atm_call['lastPrice']):.4f}")
        print(f"  Implied Volatility: {atm_call['calculated_iv']*100:.2f}%")
        if 'vega' in atm_call and atm_call['vega'] is not None:
            print(f"  Vega: {atm_call['vega']:.4f}")
        print(f"  Iterations: {int(atm_call['iterations'])}")
        print(f"  Final Error: ${atm_call['final_error']:.6f}")

        # Show a few more strikes
        print("\n  Other Strikes:")
        sample_calls = calls_converged.nsmallest(5, 'strike_diff')
        for _, opt in sample_calls.iterrows():
            moneyness = opt['strike'] / spot_price
            itm_otm = "ITM" if moneyness < 1.0 else "OTM" if moneyness > 1.0 else "ATM"
            print(f"    K=${opt['strike']:>7.2f}  IV={opt['calculated_iv']*100:>5.2f}%  "
                  f"Price=${opt.get('mid_price', opt['lastPrice']):>6.4f}  {itm_otm}")

    if not puts_with_iv.empty and n_converged_puts > 0:
        # Find ATM put (strike closest to spot)
        puts_converged = puts_with_iv[puts_with_iv['converged'] == True].copy()
        puts_converged['strike_diff'] = abs(puts_converged['strike'] - spot_price)
        atm_put = puts_converged.nsmallest(1, 'strike_diff').iloc[0]

        print("\n📉 ATM Put Option:")
        print(f"  Strike: ${atm_put['strike']:.2f}")
        print(f"  Market Price: ${atm_put.get('mid_price', atm_put['lastPrice']):.4f}")
        print(f"  Implied Volatility: {atm_put['calculated_iv']*100:.2f}%")
        if 'vega' in atm_put and atm_put['vega'] is not None:
            print(f"  Vega: {atm_put['vega']:.4f}")
        print(f"  Iterations: {int(atm_put['iterations'])}")
        print(f"  Final Error: ${atm_put['final_error']:.6f}")

        # Show a few more strikes
        print("\n  Other Strikes:")
        sample_puts = puts_converged.nsmallest(5, 'strike_diff')
        for _, opt in sample_puts.iterrows():
            moneyness = opt['strike'] / spot_price
            itm_otm = "ITM" if moneyness > 1.0 else "OTM" if moneyness < 1.0 else "ATM"
            print(f"    K=${opt['strike']:>7.2f}  IV={opt['calculated_iv']*100:>5.2f}%  "
                  f"Price=${opt.get('mid_price', opt['lastPrice']):>6.4f}  {itm_otm}")

    # Performance summary
    if total_converged > 0:
        avg_iters_calls = calls_with_iv[calls_with_iv['converged'] == True]['iterations'].mean()
        avg_iters_puts = puts_with_iv[puts_with_iv['converged'] == True]['iterations'].mean()

        print(f"\n{'─'*70}")
        print("Performance")
        print(f"{'─'*70}")
        print(f"  Average iterations (calls): {avg_iters_calls:.1f}")
        print(f"  Average iterations (puts): {avg_iters_puts:.1f}")
        print(f"  Estimated time per option: ~143ms (101×1000 grid)")
        print(f"  Total estimated time: ~{total_converged * 0.143:.1f}s")

    print(f"\n{'='*70}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())

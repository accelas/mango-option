#!/usr/bin/env python3
"""
Main script to download option data and calculate IV surface.

This script:
1. Downloads option chain data from Yahoo Finance
2. Calculates implied volatility using mango-option C++ solver
3. Stores results and raw data in SQLite3 database

Usage:
    python calculate_iv_surface.py AAPL
    python calculate_iv_surface.py SPY --max-expirations 6
    python calculate_iv_surface.py TSLA --db-path my_options.db --risk-free-rate 0.045
"""

import argparse
import sys
from datetime import datetime
from typing import Optional
import pandas as pd

# Support both package and standalone usage
try:
    from .database import OptionDatabase
    from .data_downloader import OptionDataDownloader, download_and_format_options
    from .iv_calculator import IVCalculator, validate_option_price
except ImportError:
    from database import OptionDatabase
    from data_downloader import OptionDataDownloader, download_and_format_options
    from iv_calculator import IVCalculator, validate_option_price


def process_ticker(ticker: str, db_path: str = "options.db",
                   max_expirations: Optional[int] = None,
                   risk_free_rate: float = 0.05,
                   min_volume: int = 10,
                   grid_n_space: int = 101,
                   grid_n_time: int = 1000,
                   verbose: bool = True) -> bool:
    """
    Download and calculate IV surface for a ticker.

    Args:
        ticker: Stock ticker symbol
        db_path: Path to SQLite database
        max_expirations: Maximum number of expirations to process
        risk_free_rate: Risk-free rate (as decimal)
        min_volume: Minimum option volume filter
        grid_n_space: Spatial grid points for PDE solver
        grid_n_time: Time steps for PDE solver
        verbose: Print progress messages

    Returns:
        True if successful, False otherwise
    """
    try:
        # 1. Download option data
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing {ticker}")
            print(f"{'='*60}")
            print(f"Step 1: Downloading option data from Yahoo Finance...")

        data = download_and_format_options(
            ticker,
            max_expirations=max_expirations,
            min_volume=min_volume
        )

        security_info = data['security_info']
        market_data = data['market_data']
        option_chains = data['option_chains']
        downloader = data['downloader']

        if verbose:
            print(f"  Security: {security_info['name']} ({security_info['ticker']})")
            print(f"  ISIN: {security_info.get('isin', 'N/A')}")
            print(f"  Exchange: {security_info.get('exchange', 'N/A')}")
            print(f"  Current price: ${market_data['spot_price']:.2f}")
            print(f"  Found {len(option_chains)} expiration dates")

        if not option_chains:
            print(f"Error: No option chains found for {ticker}")
            return False

        # 2. Initialize database
        if verbose:
            print(f"\nStep 2: Initializing database...")

        with OptionDatabase(db_path) as db:
            # Add security
            security_id = db.add_security(
                ticker=security_info['ticker'],
                name=security_info['name'],
                isin=security_info.get('isin'),
                cusip=security_info.get('cusip'),
                exchange=security_info.get('exchange'),
                sector=security_info.get('sector')
            )

            if verbose:
                print(f"  Security ID: {security_id}")

            # Add market data
            data_id = db.add_market_data(
                security_id=security_id,
                timestamp=market_data['timestamp'],
                spot_price=market_data['spot_price'],
                bid=market_data['bid'],
                ask=market_data['ask'],
                volume=market_data['volume'],
                dividend_yield=market_data['dividend_yield']
            )

            if verbose:
                print(f"  Market data ID: {data_id}")

            # 3. Calculate IV surface
            if verbose:
                print(f"\nStep 3: Calculating IV surface...")
                print(f"  Using FDM grid: {grid_n_space} space × {grid_n_time} time")

            calculator = IVCalculator(
                grid_n_space=grid_n_space,
                grid_n_time=grid_n_time
            )

            spot_price = market_data['spot_price']
            total_options = 0
            successful_ivs = 0
            failed_ivs = 0

            # Process each expiration
            for expiration, (calls_df, puts_df) in option_chains.items():
                ttm = downloader.calculate_time_to_maturity(expiration)

                if verbose:
                    n_calls = len(calls_df)
                    n_puts = len(puts_df)
                    print(f"\n  Expiration: {expiration} (T={ttm:.3f} years)")
                    print(f"    Calls: {n_calls}, Puts: {n_puts}")

                # Process calls
                if not calls_df.empty:
                    calls_with_iv = calculator.calculate_chain_iv(
                        calls_df, spot_price, risk_free_rate, ttm, is_call=True
                    )

                    # Store in database
                    for _, row in calls_with_iv.iterrows():
                        total_options += 1

                        # Add contract
                        contract_id = db.add_option_contract(
                            security_id=security_id,
                            contract_symbol=row['contractSymbol'],
                            option_type='CALL',
                            strike=row['strike'],
                            expiration=expiration
                        )

                        # Add price
                        mid_price = row.get('mid_price')
                        if pd.isna(mid_price) and 'bid' in row and 'ask' in row:
                            if not pd.isna(row['bid']) and not pd.isna(row['ask']):
                                mid_price = (row['bid'] + row['ask']) / 2.0

                        price_id = db.add_option_price(
                            contract_id=contract_id,
                            timestamp=market_data['timestamp'],
                            bid=row.get('bid'),
                            ask=row.get('ask'),
                            last=row.get('lastPrice'),
                            volume=int(row['volume']) if not pd.isna(row['volume']) else None,
                            open_interest=int(row['openInterest']) if not pd.isna(row['openInterest']) else None
                        )

                        # Add IV calculation
                        market_price = mid_price if not pd.isna(mid_price) else row.get('lastPrice')

                        if not pd.isna(market_price):
                            converged = bool(row.get('converged', False))
                            if converged:
                                successful_ivs += 1
                            else:
                                failed_ivs += 1

                            db.add_iv_calculation(
                                contract_id=contract_id,
                                data_id=data_id,
                                price_id=price_id,
                                market_price=float(market_price),
                                spot_price=spot_price,
                                time_to_maturity=ttm,
                                risk_free_rate=risk_free_rate,
                                implied_volatility=float(row['calculated_iv']) if not pd.isna(row['calculated_iv']) else None,
                                converged=converged,
                                iterations=int(row.get('iterations', 0)),
                                final_error=float(row.get('final_error', 0.0)) if not pd.isna(row.get('final_error')) else 0.0,
                                vega=float(row['vega']) if not pd.isna(row.get('vega')) else None,
                                solver_config=calculator.get_config_dict()
                            )

                # Process puts (similar to calls)
                if not puts_df.empty:
                    puts_with_iv = calculator.calculate_chain_iv(
                        puts_df, spot_price, risk_free_rate, ttm, is_call=False
                    )

                    for _, row in puts_with_iv.iterrows():
                        total_options += 1

                        contract_id = db.add_option_contract(
                            security_id=security_id,
                            contract_symbol=row['contractSymbol'],
                            option_type='PUT',
                            strike=row['strike'],
                            expiration=expiration
                        )

                        mid_price = row.get('mid_price')
                        if pd.isna(mid_price) and 'bid' in row and 'ask' in row:
                            if not pd.isna(row['bid']) and not pd.isna(row['ask']):
                                mid_price = (row['bid'] + row['ask']) / 2.0

                        price_id = db.add_option_price(
                            contract_id=contract_id,
                            timestamp=market_data['timestamp'],
                            bid=row.get('bid'),
                            ask=row.get('ask'),
                            last=row.get('lastPrice'),
                            volume=int(row['volume']) if not pd.isna(row['volume']) else None,
                            open_interest=int(row['openInterest']) if not pd.isna(row['openInterest']) else None
                        )

                        market_price = mid_price if not pd.isna(mid_price) else row.get('lastPrice')

                        if not pd.isna(market_price):
                            converged = bool(row.get('converged', False))
                            if converged:
                                successful_ivs += 1
                            else:
                                failed_ivs += 1

                            db.add_iv_calculation(
                                contract_id=contract_id,
                                data_id=data_id,
                                price_id=price_id,
                                market_price=float(market_price),
                                spot_price=spot_price,
                                time_to_maturity=ttm,
                                risk_free_rate=risk_free_rate,
                                implied_volatility=float(row['calculated_iv']) if not pd.isna(row['calculated_iv']) else None,
                                converged=converged,
                                iterations=int(row.get('iterations', 0)),
                                final_error=float(row.get('final_error', 0.0)) if not pd.isna(row.get('final_error')) else 0.0,
                                vega=float(row['vega']) if not pd.isna(row.get('vega')) else None,
                                solver_config=calculator.get_config_dict()
                            )

        # 4. Summary
        if verbose:
            print(f"\n{'='*60}")
            print(f"Summary for {ticker}")
            print(f"{'='*60}")
            print(f"Total options processed: {total_options}")
            print(f"Successful IV calculations: {successful_ivs} ({100*successful_ivs/total_options:.1f}%)")
            print(f"Failed IV calculations: {failed_ivs} ({100*failed_ivs/total_options:.1f}%)")
            print(f"\nResults saved to: {db_path}")

        return True

    except Exception as e:
        print(f"Error processing {ticker}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download option data and calculate IV surface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL
  %(prog)s SPY --max-expirations 6 --db-path spy_options.db
  %(prog)s TSLA --risk-free-rate 0.045 --grid-n-space 201
        """
    )

    parser.add_argument('ticker', type=str,
                       help='Stock ticker symbol (e.g., AAPL, SPY)')

    parser.add_argument('--db-path', type=str, default='options.db',
                       help='Path to SQLite database (default: options.db)')

    parser.add_argument('--max-expirations', type=int, default=None,
                       help='Maximum number of expirations to process (default: all)')

    parser.add_argument('--risk-free-rate', type=float, default=0.05,
                       help='Risk-free interest rate as decimal (default: 0.05)')

    parser.add_argument('--min-volume', type=int, default=10,
                       help='Minimum option volume filter (default: 10)')

    parser.add_argument('--grid-n-space', type=int, default=101,
                       help='Spatial grid points for PDE solver (default: 101)')

    parser.add_argument('--grid-n-time', type=int, default=1000,
                       help='Time steps for PDE solver (default: 1000)')

    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress progress messages')

    args = parser.parse_args()

    success = process_ticker(
        ticker=args.ticker,
        db_path=args.db_path,
        max_expirations=args.max_expirations,
        risk_free_rate=args.risk_free_rate,
        min_volume=args.min_volume,
        grid_n_space=args.grid_n_space,
        grid_n_time=args.grid_n_time,
        verbose=not args.quiet
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

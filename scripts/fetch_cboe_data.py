#!/usr/bin/env python3
"""
Fetch option chain data from CBOE for top equities by open interest.

This script:
1. Identifies top 10 equities by option open interest
2. Downloads current option chain data (calls and puts)
3. Fetches underlying price, interest rate, dividends
4. Saves data in JSON format for C++ benchmarking

Dependencies:
    pip install yfinance pandas requests
"""

import yfinance as yf
import pandas as pd
import json
import datetime
from typing import List, Dict, Any
import argparse


def get_risk_free_rate() -> float:
    """
    Fetch current risk-free rate (10-year Treasury yield).

    Returns:
        Annual risk-free rate as decimal (e.g., 0.045 for 4.5%)
    """
    try:
        # Fetch 10-year Treasury yield (^TNX gives percentage, divide by 100)
        tnx = yf.Ticker("^TNX")
        rate_pct = tnx.history(period="1d")['Close'].iloc[-1]
        return rate_pct / 100.0
    except Exception as e:
        print(f"Warning: Could not fetch risk-free rate: {e}")
        print("Using default rate: 4.5%")
        return 0.045


def get_top_equities_by_oi(limit: int = 10) -> List[str]:
    """
    Get top equities by option open interest.

    For now, uses a curated list of highly liquid equity options.
    In production, this would query CBOE's options hub API.

    Args:
        limit: Number of tickers to return

    Returns:
        List of ticker symbols
    """
    # High-volume option tickers (typically top by open interest)
    # Source: CBOE's most active equity options
    popular_tickers = [
        "SPY",   # SPDR S&P 500 ETF
        "AAPL",  # Apple
        "TSLA",  # Tesla
        "NVDA",  # NVIDIA
        "AMD",   # AMD
        "AMZN",  # Amazon
        "MSFT",  # Microsoft
        "QQQ",   # Invesco QQQ ETF
        "IWM",   # iShares Russell 2000 ETF
        "META",  # Meta Platforms
        "GOOGL", # Alphabet
        "NFLX",  # Netflix
    ]

    return popular_tickers[:limit]


def fetch_option_chain(ticker: str, risk_free_rate: float) -> Dict[str, Any]:
    """
    Fetch complete option chain data for a ticker.

    Args:
        ticker: Stock ticker symbol
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary containing:
        - ticker: Symbol
        - spot: Current underlying price
        - rate: Risk-free rate
        - dividend_yield: Continuous dividend yield
        - timestamp: Data fetch timestamp
        - expirations: List of expiration dates
        - chains: Dict mapping expiration -> {calls: [...], puts: [...]}
    """
    print(f"Fetching data for {ticker}...")

    stock = yf.Ticker(ticker)

    # Get current price
    try:
        spot_price = stock.history(period="1d")['Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching spot price for {ticker}: {e}")
        return None

    # Get dividend yield
    try:
        info = stock.info
        dividend_yield = info.get('dividendYield', 0.0) or 0.0
    except Exception as e:
        print(f"Warning: Could not fetch dividend for {ticker}: {e}")
        dividend_yield = 0.0

    # Get option expirations
    try:
        expirations = stock.options
    except Exception as e:
        print(f"Error fetching options for {ticker}: {e}")
        return None

    if not expirations:
        print(f"No options available for {ticker}")
        return None

    # Limit to next 6 expirations (avoid too much data)
    expirations = expirations[:6]

    chains = {}
    today = datetime.date.today()

    for exp_date in expirations:
        try:
            opt_chain = stock.option_chain(exp_date)

            # Parse expiration date
            exp_dt = datetime.datetime.strptime(exp_date, "%Y-%m-%d").date()
            days_to_exp = (exp_dt - today).days
            years_to_exp = days_to_exp / 365.0

            if years_to_exp <= 0:
                continue  # Skip expired or today-expiring options

            # Process calls
            calls = opt_chain.calls
            calls_data = []
            for _, row in calls.iterrows():
                if row['openInterest'] > 0 and row['lastPrice'] > 0:
                    calls_data.append({
                        'strike': float(row['strike']),
                        'last_price': float(row['lastPrice']),
                        'bid': float(row['bid']) if row['bid'] > 0 else None,
                        'ask': float(row['ask']) if row['ask'] > 0 else None,
                        'volume': int(row['volume']) if pd.notna(row['volume']) else 0,
                        'open_interest': int(row['openInterest']),
                        'implied_volatility': float(row['impliedVolatility']) if pd.notna(row['impliedVolatility']) else None
                    })

            # Process puts
            puts = opt_chain.puts
            puts_data = []
            for _, row in puts.iterrows():
                if row['openInterest'] > 0 and row['lastPrice'] > 0:
                    puts_data.append({
                        'strike': float(row['strike']),
                        'last_price': float(row['lastPrice']),
                        'bid': float(row['bid']) if row['bid'] > 0 else None,
                        'ask': float(row['ask']) if row['ask'] > 0 else None,
                        'volume': int(row['volume']) if pd.notna(row['volume']) else 0,
                        'open_interest': int(row['openInterest']),
                        'implied_volatility': float(row['impliedVolatility']) if pd.notna(row['impliedVolatility']) else None
                    })

            if calls_data or puts_data:
                chains[exp_date] = {
                    'days_to_expiration': days_to_exp,
                    'years_to_expiration': years_to_exp,
                    'calls': calls_data,
                    'puts': puts_data
                }

        except Exception as e:
            print(f"  Warning: Could not fetch chain for {exp_date}: {e}")
            continue

    if not chains:
        print(f"No valid option chains for {ticker}")
        return None

    result = {
        'ticker': ticker,
        'spot': float(spot_price),
        'rate': float(risk_free_rate),
        'dividend_yield': float(dividend_yield),
        'timestamp': datetime.datetime.now().isoformat(),
        'expirations': list(chains.keys()),
        'chains': chains
    }

    print(f"  Spot: ${spot_price:.2f}, Dividend: {dividend_yield*100:.2f}%, "
          f"Expirations: {len(chains)}, Total options: "
          f"{sum(len(c['calls']) + len(c['puts']) for c in chains.values())}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch CBOE option chain data")
    parser.add_argument("--tickers", type=int, default=10,
                       help="Number of top tickers to fetch (default: 10)")
    parser.add_argument("--output", type=str, default="option_data.json",
                       help="Output JSON file (default: option_data.json)")
    args = parser.parse_args()

    print("=" * 70)
    print("CBOE Option Data Fetcher")
    print("=" * 70)

    # Get risk-free rate
    print("\nFetching risk-free rate...")
    risk_free_rate = get_risk_free_rate()
    print(f"Using risk-free rate: {risk_free_rate*100:.2f}%")

    # Get top tickers
    print(f"\nFetching top {args.tickers} equities by option volume...")
    tickers = get_top_equities_by_oi(args.tickers)
    print(f"Tickers: {', '.join(tickers)}")

    # Fetch data for each ticker
    print("\nDownloading option chains...")
    all_data = []

    for ticker in tickers:
        data = fetch_option_chain(ticker, risk_free_rate)
        if data:
            all_data.append(data)

    # Save to JSON
    output_path = args.output
    print(f"\nSaving data to {output_path}...")

    summary = {
        'fetch_timestamp': datetime.datetime.now().isoformat(),
        'risk_free_rate': risk_free_rate,
        'num_tickers': len(all_data),
        'tickers': [d['ticker'] for d in all_data],
        'data': all_data
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Summary:")
    print(f"  Tickers fetched: {len(all_data)}")
    print(f"  Total expirations: {sum(len(d['expirations']) for d in all_data)}")
    print(f"  Total options: {sum(sum(len(c['calls']) + len(c['puts']) for c in d['chains'].values()) for d in all_data)}")
    print(f"  Output file: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

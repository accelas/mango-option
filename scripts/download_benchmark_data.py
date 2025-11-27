#!/usr/bin/env python3
"""
Download real option chain data from yfinance and generate C++ benchmark data header.

Usage:
    python scripts/download_benchmark_data.py [SYMBOL]

Example:
    python scripts/download_benchmark_data.py SPY

Output:
    benchmarks/real_market_data.hpp - C++ header with real option data
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import numpy as np


def get_treasury_rate() -> float:
    """Get approximate risk-free rate from 10Y Treasury."""
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="1d")
        if not hist.empty:
            return hist["Close"].iloc[-1] / 100.0
    except Exception:
        pass
    return 0.045  # Default fallback


def download_option_chain(symbol: str) -> dict:
    """Download complete option chain for a symbol."""
    ticker = yf.Ticker(symbol)

    # Get current price
    info = ticker.info
    spot = info.get("regularMarketPrice") or info.get("currentPrice")
    if spot is None:
        raise ValueError(f"Could not get spot price for {symbol}")

    # Dividend yield from yfinance is already in decimal form (e.g., 0.0109 for 1.09%)
    dividend_yield = info.get("dividendYield", 0.0) or 0.0
    # But sometimes it returns as percentage, so cap at reasonable value
    if dividend_yield > 0.20:  # If > 20%, assume it's in percentage form
        dividend_yield = dividend_yield / 100.0

    # Get available expirations
    expirations = ticker.options
    if not expirations:
        raise ValueError(f"No options available for {symbol}")

    risk_free_rate = get_treasury_rate()
    today = datetime.now()

    options = []
    for expiry_str in expirations[:12]:  # First 12 expirations for more variety
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
            days_to_expiry = (expiry_date - today).days
            if days_to_expiry < 7:  # Skip options expiring in < 1 week
                continue
            maturity = days_to_expiry / 365.0

            opt_chain = ticker.option_chain(expiry_str)

            # Process puts (more interesting for American options)
            for _, row in opt_chain.puts.iterrows():
                strike = row["strike"]
                bid = row.get("bid", 0) or 0
                ask = row.get("ask", 0) or 0

                # Skip illiquid options
                if bid <= 0 or ask <= 0:
                    continue
                if ask - bid > 0.5 * (bid + ask):  # Spread > 50% of mid
                    continue

                mid_price = (bid + ask) / 2.0
                moneyness = spot / strike

                # Focus on near-the-money options (0.85 to 1.15 moneyness)
                if 0.85 <= moneyness <= 1.15:
                    options.append({
                        "strike": strike,
                        "maturity": maturity,
                        "market_price": mid_price,
                        "is_call": False,
                        "expiry": expiry_str
                    })

            # Process calls
            for _, row in opt_chain.calls.iterrows():
                strike = row["strike"]
                bid = row.get("bid", 0) or 0
                ask = row.get("ask", 0) or 0

                if bid <= 0 or ask <= 0:
                    continue
                if ask - bid > 0.5 * (bid + ask):
                    continue

                mid_price = (bid + ask) / 2.0
                moneyness = spot / strike

                if 0.85 <= moneyness <= 1.15:
                    options.append({
                        "strike": strike,
                        "maturity": maturity,
                        "market_price": mid_price,
                        "is_call": True,
                        "expiry": expiry_str
                    })

        except Exception as e:
            print(f"Warning: Failed to process {expiry_str}: {e}", file=sys.stderr)
            continue

    return {
        "symbol": symbol,
        "spot": spot,
        "risk_free_rate": risk_free_rate,
        "dividend_yield": dividend_yield,
        "timestamp": datetime.now().isoformat(),
        "options": options
    }


def generate_cpp_header(data: dict, output_path: Path) -> None:
    """Generate C++ header file with real market data."""

    # Select a diverse subset for benchmarks
    puts = [o for o in data["options"] if not o["is_call"]]
    calls = [o for o in data["options"] if o["is_call"]]

    # Sort by maturity then strike
    puts.sort(key=lambda x: (x["maturity"], x["strike"]))
    calls.sort(key=lambda x: (x["maturity"], x["strike"]))

    # Take up to 64 puts for batch benchmark, diverse selection
    selected_puts = []
    maturities = sorted(set(p["maturity"] for p in puts))
    for mat in maturities[:4]:  # Up to 4 maturities
        mat_puts = [p for p in puts if abs(p["maturity"] - mat) < 0.01]
        # Take every Nth strike to get ~16 per maturity
        step = max(1, len(mat_puts) // 16)
        selected_puts.extend(mat_puts[::step][:16])

    selected_puts = selected_puts[:64]

    header = f'''// Auto-generated real market data for benchmarks
// Generated: {data["timestamp"]}
// Symbol: {data["symbol"]}
// DO NOT EDIT - regenerate with: python scripts/download_benchmark_data.py {data["symbol"]}

#pragma once

#include <array>
#include <cstddef>

namespace mango::benchmark_data {{

// Market snapshot
constexpr const char* SYMBOL = "{data["symbol"]}";
constexpr double SPOT = {data["spot"]:.2f};
constexpr double RISK_FREE_RATE = {data["risk_free_rate"]:.4f};
constexpr double DIVIDEND_YIELD = {data["dividend_yield"]:.4f};

// Option data structure
struct RealOptionData {{
    double strike;
    double maturity;
    double market_price;
    bool is_call;
}};

// Real put options for batch benchmarks ({len(selected_puts)} options)
constexpr std::array<RealOptionData, {len(selected_puts)}> REAL_PUTS = {{{{
'''

    for i, opt in enumerate(selected_puts):
        comma = "," if i < len(selected_puts) - 1 else ""
        header += f'    {{{opt["strike"]:.2f}, {opt["maturity"]:.6f}, {opt["market_price"]:.4f}, false}}{comma}\n'

    header += '''}}};

// Sample ATM put for single option benchmark
'''

    # Find ATM put with ~1 year maturity
    atm_puts = [p for p in puts if abs(p["strike"] - data["spot"]) < 5]
    if atm_puts:
        # Sort by maturity and pick one near 1 year if available
        atm_puts.sort(key=lambda x: abs(x["maturity"] - 1.0))
        atm = atm_puts[0]
        header += f'''constexpr RealOptionData ATM_PUT = {{{atm["strike"]:.2f}, {atm["maturity"]:.6f}, {atm["market_price"]:.4f}, false}};
'''
    else:
        header += f'''constexpr RealOptionData ATM_PUT = {{{data["spot"]:.2f}, 1.0, 0.0, false}};  // Fallback - no ATM data
'''

    # Add some calls for diversification
    selected_calls = []
    for mat in maturities[:2]:
        mat_calls = [c for c in calls if abs(c["maturity"] - mat) < 0.01]
        step = max(1, len(mat_calls) // 8)
        selected_calls.extend(mat_calls[::step][:8])
    selected_calls = selected_calls[:16]

    header += f'''
// Real call options ({len(selected_calls)} options)
constexpr std::array<RealOptionData, {len(selected_calls)}> REAL_CALLS = {{{{
'''

    for i, opt in enumerate(selected_calls):
        comma = "," if i < len(selected_calls) - 1 else ""
        header += f'    {{{opt["strike"]:.2f}, {opt["maturity"]:.6f}, {opt["market_price"]:.4f}, true}}{comma}\n'

    header += '''}}};

}}  // namespace mango::benchmark_data
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header)
    print(f"Generated {output_path}")
    print(f"  - {len(selected_puts)} put options")
    print(f"  - {len(selected_calls)} call options")
    print(f"  - Spot: ${data['spot']:.2f}")
    print(f"  - Rate: {data['risk_free_rate']*100:.2f}%")


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"

    print(f"Downloading option chain for {symbol}...")
    data = download_option_chain(symbol)

    print(f"Found {len(data['options'])} options")

    output_path = Path(__file__).parent.parent / "benchmarks" / "real_market_data.hpp"
    generate_cpp_header(data, output_path)


if __name__ == "__main__":
    main()

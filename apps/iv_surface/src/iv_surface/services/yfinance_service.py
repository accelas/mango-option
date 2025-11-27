"""YFinance data fetching service."""

import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


def is_market_open() -> bool:
    """Check if US market is currently open."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    # Weekday and between 9:30 AM and 4:00 PM ET
    if now_et.weekday() >= 5:  # Weekend
        return False
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close


def is_after_market_close() -> bool:
    """Check if we're after market close (for EOD data)."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:  # Weekend - use Friday's EOD
        return True
    return now_et.hour >= 16


async def fetch_option_chain(symbol: str) -> dict:
    """
    Fetch complete option chain for a symbol.

    Returns dict with:
    - spot: current spot price
    - quote_time: timestamp of quote
    - is_eod: whether this is EOD data
    - expirations: list of expiry dates
    - chains: dict[expiry] -> {calls: [...], puts: [...]}
    """

    def _fetch() -> dict:
        try:
            ticker = yf.Ticker(symbol)

            # Get current price
            info = ticker.info
            spot = info.get("regularMarketPrice") or info.get("currentPrice")
            if spot is None:
                return {"error": f"Could not get spot price for {symbol}"}

            # Get available expirations
            expirations = ticker.options
            if not expirations:
                return {"error": f"No options available for {symbol}"}

            chains: dict[str, dict] = {}
            for expiry in expirations:
                try:
                    opt = ticker.option_chain(expiry)
                    chains[expiry] = {
                        "calls": _process_options(opt.calls, expiry),
                        "puts": _process_options(opt.puts, expiry),
                    }
                except Exception:
                    continue  # Skip problematic expirations

            if not chains:
                return {"error": f"Could not fetch any option chains for {symbol}"}

            return {
                "symbol": symbol,
                "spot": float(spot),
                "quote_time": datetime.now(timezone.utc).isoformat(),
                "is_eod": is_after_market_close(),
                "is_market_open": is_market_open(),
                "expirations": list(chains.keys()),
                "chains": chains,
                "dividend_yield": info.get("dividendYield", 0.0) or 0.0,
            }

        except Exception as e:
            return {"error": str(e)}

    # Run in thread pool to not block async
    return await asyncio.get_event_loop().run_in_executor(None, _fetch)


def _process_options(df: pd.DataFrame, expiry: str) -> list[dict]:
    """Process options DataFrame to list of dicts."""
    options = []
    for _, row in df.iterrows():
        bid = row.get("bid", 0) or 0
        ask = row.get("ask", 0) or 0

        # Skip options with no market
        if bid <= 0 and ask <= 0:
            continue

        options.append(
            {
                "strike": float(row["strike"]),
                "bid": float(bid),
                "ask": float(ask),
                "last": float(row.get("lastPrice", 0) or 0),
                "volume": int(row.get("volume", 0) or 0),
                "open_interest": int(row.get("openInterest", 0) or 0),
                "implied_vol": float(row.get("impliedVolatility", 0) or 0),
                "expiry": expiry,
            }
        )

    return options


async def fetch_equity_history(symbol: str, period: str = "1y") -> dict:
    """
    Fetch historical equity data.

    Returns dict with OHLCV data for charting.
    """

    def _fetch() -> dict:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty:
                return {"error": f"No historical data for {symbol}"}

            # Convert to lists for JSON serialization
            dates = hist.index.strftime("%Y-%m-%d").tolist()
            return {
                "symbol": symbol,
                "dates": dates,
                "open": hist["Open"].tolist(),
                "high": hist["High"].tolist(),
                "low": hist["Low"].tolist(),
                "close": hist["Close"].tolist(),
                "volume": hist["Volume"].tolist(),
                "adj_close": (
                    hist["Close"].tolist()
                ),  # yfinance now returns adjusted by default
            }

        except Exception as e:
            return {"error": str(e)}

    return await asyncio.get_event_loop().run_in_executor(None, _fetch)


async def fetch_dividends(symbol: str) -> dict:
    """Fetch dividend history for a symbol."""

    def _fetch() -> dict:
        try:
            ticker = yf.Ticker(symbol)
            divs = ticker.dividends

            if divs.empty:
                return {"symbol": symbol, "dividends": []}

            return {
                "symbol": symbol,
                "dividends": [
                    {"date": d.strftime("%Y-%m-%d"), "amount": float(a)}
                    for d, a in divs.items()
                ],
            }

        except Exception as e:
            return {"error": str(e)}

    return await asyncio.get_event_loop().run_in_executor(None, _fetch)

#!/usr/bin/env python3
"""
Download real market option data, price it with the mango_option Python binding,
and persist the snapshot as an Arrow IPC (Feather) file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import List

import pyarrow as pa
import pyarrow.ipc as ipc
import yfinance as yf

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BAZEL_PYTHON_DIR = REPO_ROOT / "bazel-bin" / "python"
if BAZEL_PYTHON_DIR.exists():
    sys.path.append(str(BAZEL_PYTHON_DIR))

try:
    import mango_option
except ImportError as exc:  # pragma: no cover - handled in CLI
    raise SystemExit(
        "mango_option module not available. Build it with: bazel build //python:mango_option"
    ) from exc


@dataclass
class OptionSample:
    ticker: str
    contract_symbol: str
    option_type: int  # 0 = PUT, 1 = CALL
    spot: float
    strike: float
    time_to_maturity: float
    rate: float
    dividend_yield: float
    implied_volatility: float
    market_price: float
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    model_price: float
    asof: dt.datetime


def _mid(bid: float | None, ask: float | None) -> float | None:
    if bid and ask and bid > 0 and ask > 0:
        return 0.5 * (bid + ask)
    return None


def _time_to_maturity(expiration: str) -> float:
    exp_date = dt.datetime.strptime(expiration, "%Y-%m-%d").date()
    today = dt.date.today()
    days = max((exp_date - today).days, 1)
    return days / 365.25


def _estimate_rate() -> float:
    """Use 13-week T-Bill (^IRX) as a crude proxy."""
    try:
        history = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        if not history.empty:
            return float(history.iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.045


def _build_params(
    sample: OptionSample, option_kind: mango_option.OptionType
) -> mango_option.AmericanOptionParams:
    params = mango_option.AmericanOptionParams()
    params.spot = sample.spot
    params.strike = sample.strike
    params.maturity = sample.time_to_maturity
    params.rate = sample.rate
    params.dividend_yield = sample.dividend_yield
    params.volatility = sample.implied_volatility
    params.type = option_kind
    return params


def _price_option(sample: OptionSample) -> float:
    option_kind = mango_option.OptionType.CALL if sample.option_type == 1 else mango_option.OptionType.PUT
    params = _build_params(sample, option_kind)
    result = mango_option.american_option_price(params)
    return float(result.value)


def collect_samples(
    ticker: str,
    max_options: int,
    expiration_index: int,
    min_volume: int,
) -> List[OptionSample]:
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info or {}
    dividend_yield = float(info.get("dividendYield") or 0.0)

    history = ticker_obj.history(period="5d")["Close"].dropna()
    if history.empty:
        raise RuntimeError(f"Failed to fetch history for {ticker}")
    spot = float(history.iloc[-1])

    expirations = ticker_obj.options
    if not expirations:
        raise RuntimeError(f"No options found for {ticker}")
    expiration_index = min(expiration_index, len(expirations) - 1)
    expiration = expirations[expiration_index]

    option_chain = ticker_obj.option_chain(expiration)
    snapshot_time = dt.datetime.now(dt.timezone.utc)
    rate = _estimate_rate()
    tau = _time_to_maturity(expiration)

    def to_samples(frame, opt_type: int) -> List[OptionSample]:
        samples = []
        sorted_frame = frame.sort_values("volume", ascending=False)
        for _, row in sorted_frame.iterrows():
            if row.get("volume", 0) < min_volume:
                continue
            iv = float(row.get("impliedVolatility") or 0.0)
            if not math.isfinite(iv) or iv <= 0:
                continue
            strike = float(row["strike"])
            bid = float(row.get("bid") or 0.0)
            ask = float(row.get("ask") or 0.0)
            last = float(row.get("lastPrice") or 0.0)
            market_price = _mid(bid, ask) or (last if last > 0 else None)
            if market_price is None:
                continue
            samples.append(
                OptionSample(
                    ticker=ticker,
                    contract_symbol=str(row.get("contractSymbol")),
                    option_type=opt_type,
                    spot=spot,
                    strike=strike,
                    time_to_maturity=tau,
                    rate=rate,
                    dividend_yield=dividend_yield,
                    implied_volatility=iv,
                    market_price=float(market_price),
                    bid=bid,
                    ask=ask,
                    last_price=last,
                    volume=int(row.get("volume") or 0),
                    open_interest=int(row.get("openInterest") or 0),
                    model_price=0.0,  # placeholder, updated later
                    asof=snapshot_time,
                )
            )
            if len(samples) >= max_options:
                break
        return samples

    call_samples = to_samples(option_chain.calls, opt_type=1)
    put_samples = to_samples(option_chain.puts, opt_type=0)

    samples = (call_samples + put_samples)[:max_options]
    if not samples:
        raise RuntimeError("No option samples met the filtering criteria")

    for sample in samples:
        sample.model_price = _price_option(sample)
    return samples


def write_arrow(samples: List[OptionSample], output: pathlib.Path) -> None:
    schema = pa.schema(
        [
            pa.field("ticker", pa.string()),
            pa.field("contract_symbol", pa.string()),
            pa.field("option_type", pa.uint8()),
            pa.field("spot", pa.float64()),
            pa.field("strike", pa.float64()),
            pa.field("time_to_maturity", pa.float64()),
            pa.field("rate", pa.float64()),
            pa.field("dividend_yield", pa.float64()),
            pa.field("implied_volatility", pa.float64()),
            pa.field("market_price", pa.float64()),
            pa.field("bid", pa.float64()),
            pa.field("ask", pa.float64()),
            pa.field("last_price", pa.float64()),
            pa.field("volume", pa.int64()),
            pa.field("open_interest", pa.int64()),
            pa.field("model_price", pa.float64()),
            pa.field("timestamp", pa.timestamp("us")),
        ]
    )

    records = [
        {
            "ticker": s.ticker,
            "contract_symbol": s.contract_symbol,
            "option_type": s.option_type,
            "spot": s.spot,
            "strike": s.strike,
            "time_to_maturity": s.time_to_maturity,
            "rate": s.rate,
            "dividend_yield": s.dividend_yield,
            "implied_volatility": s.implied_volatility,
            "market_price": s.market_price,
            "bid": s.bid,
            "ask": s.ask,
            "last_price": s.last_price,
            "volume": s.volume,
            "open_interest": s.open_interest,
            "model_price": s.model_price,
            "timestamp": s.asof,
        }
        for s in samples
    ]

    table = pa.Table.from_pylist(records, schema=schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ipc.new_file(output, schema) as writer:
        writer.write_table(table)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol to download (default: AAPL)")
    parser.add_argument("--max-options", type=int, default=12, help="Maximum samples to store")
    parser.add_argument(
        "--expiration-index",
        type=int,
        default=0,
        help="Index of expiration to fetch (0 = nearest)",
    )
    parser.add_argument(
        "--min-volume",
        type=int,
        default=50,
        help="Minimum daily volume required for an option to be included",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("data/real_option_chain.arrow"),
        help="Output Arrow IPC file",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    samples = collect_samples(
        ticker=args.ticker,
        max_options=args.max_options,
        expiration_index=args.expiration_index,
        min_volume=args.min_volume,
    )
    write_arrow(samples, args.output)
    print(f"Wrote {len(samples)} samples to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main(sys.argv[1:]))

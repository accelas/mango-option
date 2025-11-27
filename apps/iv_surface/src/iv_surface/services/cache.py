"""DuckDB cache layer for IV Surface."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

# Global connection
_conn: duckdb.DuckDBPyConnection | None = None


def init_db(db_path: Path) -> None:
    """Initialize database with schema."""
    global _conn

    db_path.parent.mkdir(parents=True, exist_ok=True)
    _conn = duckdb.connect(str(db_path))

    # Create tables
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS symbols (
            symbol VARCHAR PRIMARY KEY,
            name VARCHAR,
            status VARCHAR DEFAULT 'active',
            has_options BOOLEAN DEFAULT true,
            last_quote_fetch TIMESTAMP,
            last_options_fetch TIMESTAMP
        )
    """)

    _conn.execute("""
        CREATE TABLE IF NOT EXISTS eod_quotes (
            symbol VARCHAR,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            adj_close DOUBLE,
            volume BIGINT,
            PRIMARY KEY (symbol, date)
        )
    """)

    _conn.execute("""
        CREATE TABLE IF NOT EXISTS options_snapshots (
            symbol VARCHAR,
            fetched_at TIMESTAMP,
            is_eod BOOLEAN,
            expiry DATE,
            strike DOUBLE,
            option_type VARCHAR,
            bid DOUBLE,
            ask DOUBLE,
            last_price DOUBLE,
            volume INTEGER,
            open_interest INTEGER,
            computed_iv DOUBLE,
            PRIMARY KEY (symbol, fetched_at, expiry, strike, option_type)
        )
    """)

    _conn.execute("""
        CREATE TABLE IF NOT EXISTS price_tables (
            symbol VARCHAR,
            option_type VARCHAR,
            arrow_path VARCHAR,
            created_at TIMESTAMP,
            spot_at_build DOUBLE,
            rate_at_build DOUBLE,
            UNIQUE(symbol, option_type)
        )
    """)


def get_conn() -> duckdb.DuckDBPyConnection:
    """Get database connection."""
    if _conn is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _conn


def get_cached_symbols() -> list[dict]:
    """Get list of cached symbols with freshness info."""
    conn = get_conn()
    result = conn.execute("""
        SELECT
            symbol,
            last_options_fetch,
            EXTRACT(EPOCH FROM (NOW() - last_options_fetch)) / 60 as age_minutes
        FROM symbols
        WHERE last_options_fetch IS NOT NULL
        ORDER BY last_options_fetch DESC
        LIMIT 20
    """).fetchall()

    return [
        {
            "symbol": row[0],
            "last_fetch": row[1].isoformat() if row[1] else None,
            "age_minutes": int(row[2]) if row[2] else None,
        }
        for row in result
    ]


def store_options_snapshot(symbol: str, chain_data: dict) -> None:
    """Store options snapshot to database."""
    conn = get_conn()
    now = datetime.now(timezone.utc)
    is_eod = chain_data.get("is_eod", False)

    # Update symbol record
    conn.execute(
        """
        INSERT INTO symbols (symbol, last_options_fetch)
        VALUES (?, ?)
        ON CONFLICT (symbol) DO UPDATE SET last_options_fetch = ?
    """,
        [symbol, now, now],
    )

    # Store each option
    for expiry_str, chain in chain_data.get("chains", {}).items():
        expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()

        for opt_type, options in [("call", chain.get("calls", [])), ("put", chain.get("puts", []))]:
            for opt in options:
                try:
                    conn.execute(
                        """
                        INSERT INTO options_snapshots
                        (symbol, fetched_at, is_eod, expiry, strike, option_type,
                         bid, ask, last_price, volume, open_interest, computed_iv)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT DO NOTHING
                    """,
                        [
                            symbol,
                            now,
                            is_eod,
                            expiry,
                            opt["strike"],
                            opt_type,
                            opt.get("bid", 0),
                            opt.get("ask", 0),
                            opt.get("last", 0),
                            opt.get("volume", 0),
                            opt.get("open_interest", 0),
                            opt.get("implied_vol", 0),
                        ],
                    )
                except Exception:
                    continue  # Skip problematic records


def store_eod_quotes(symbol: str, equity_data: dict) -> None:
    """Store EOD quotes to database."""
    conn = get_conn()

    dates = equity_data.get("dates", [])
    opens = equity_data.get("open", [])
    highs = equity_data.get("high", [])
    lows = equity_data.get("low", [])
    closes = equity_data.get("close", [])
    volumes = equity_data.get("volume", [])
    adj_closes = equity_data.get("adj_close", closes)

    for i, date_str in enumerate(dates):
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
            conn.execute(
                """
                INSERT INTO eod_quotes
                (symbol, date, open, high, low, close, adj_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, date) DO UPDATE SET
                    open = ?, high = ?, low = ?, close = ?, adj_close = ?, volume = ?
            """,
                [
                    symbol,
                    date,
                    opens[i],
                    highs[i],
                    lows[i],
                    closes[i],
                    adj_closes[i],
                    int(volumes[i]),
                    opens[i],
                    highs[i],
                    lows[i],
                    closes[i],
                    adj_closes[i],
                    int(volumes[i]),
                ],
            )
        except Exception:
            continue


def get_eod_quotes(symbol: str, days: int = 252) -> list[dict]:
    """Get recent EOD quotes for a symbol."""
    conn = get_conn()
    result = conn.execute(
        """
        SELECT date, open, high, low, close, adj_close, volume
        FROM eod_quotes
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT ?
    """,
        [symbol, days],
    ).fetchall()

    return [
        {
            "date": row[0].isoformat(),
            "open": row[1],
            "high": row[2],
            "low": row[3],
            "close": row[4],
            "adj_close": row[5],
            "volume": row[6],
        }
        for row in result
    ]

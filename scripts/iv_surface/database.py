"""
Database schema and utilities for storing option data and IV calculations.

This module provides:
- SQLite3 database schema for securities, option chains, and IV surfaces
- Standard security identification (ticker, ISIN, CUSIP)
- Data access functions
"""

import sqlite3
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import json


class OptionDatabase:
    """SQLite3 database for option data and IV calculations."""

    def __init__(self, db_path: str = "options.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dicts
        self._create_schema()

    def _create_schema(self):
        """Create database schema for securities, options, and IV data."""
        cursor = self.conn.cursor()

        # Securities table - stores underlying securities with standard identifiers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS securities (
                security_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL UNIQUE,
                name TEXT,
                isin TEXT,
                cusip TEXT,
                exchange TEXT,
                sector TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index on standard identifiers
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_securities_ticker
            ON securities(ticker)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_securities_isin
            ON securities(isin)
        """)

        # Market data snapshots - stores underlying price data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                data_id INTEGER PRIMARY KEY AUTOINCREMENT,
                security_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                spot_price REAL NOT NULL,
                bid REAL,
                ask REAL,
                volume INTEGER,
                dividend_yield REAL,
                FOREIGN KEY (security_id) REFERENCES securities(security_id),
                UNIQUE(security_id, timestamp)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_security_time
            ON market_data(security_id, timestamp DESC)
        """)

        # Option contracts table - stores option contract specifications
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS option_contracts (
                contract_id INTEGER PRIMARY KEY AUTOINCREMENT,
                security_id INTEGER NOT NULL,
                contract_symbol TEXT NOT NULL UNIQUE,
                option_type TEXT NOT NULL CHECK(option_type IN ('CALL', 'PUT')),
                strike REAL NOT NULL,
                expiration DATE NOT NULL,
                exercise_style TEXT DEFAULT 'AMERICAN' CHECK(exercise_style IN ('AMERICAN', 'EUROPEAN')),
                contract_size INTEGER DEFAULT 100,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (security_id) REFERENCES securities(security_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_contracts_security_exp
            ON option_contracts(security_id, expiration)
        """)

        # Option prices table - stores observed market prices
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS option_prices (
                price_id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                bid REAL,
                ask REAL,
                last REAL,
                mid_price REAL,
                volume INTEGER,
                open_interest INTEGER,
                FOREIGN KEY (contract_id) REFERENCES option_contracts(contract_id),
                UNIQUE(contract_id, timestamp)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_option_prices_contract_time
            ON option_prices(contract_id, timestamp DESC)
        """)

        # IV calculations table - stores calculated implied volatilities
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS iv_calculations (
                calculation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id INTEGER NOT NULL,
                data_id INTEGER NOT NULL,
                price_id INTEGER NOT NULL,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                -- Market inputs
                market_price REAL NOT NULL,
                spot_price REAL NOT NULL,
                time_to_maturity REAL NOT NULL,
                risk_free_rate REAL NOT NULL,

                -- IV result
                implied_volatility REAL,
                converged BOOLEAN NOT NULL,
                iterations INTEGER,
                final_error REAL,
                failure_reason TEXT,
                vega REAL,

                -- Solver configuration
                solver_config TEXT,  -- JSON with grid params, tolerances

                FOREIGN KEY (contract_id) REFERENCES option_contracts(contract_id),
                FOREIGN KEY (data_id) REFERENCES market_data(data_id),
                FOREIGN KEY (price_id) REFERENCES option_prices(price_id),
                UNIQUE(contract_id, data_id, price_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_iv_contract_time
            ON iv_calculations(contract_id, calculated_at DESC)
        """)

        # IV surface snapshots - stores complete IV surface at a point in time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS iv_surfaces (
                surface_id INTEGER PRIMARY KEY AUTOINCREMENT,
                security_id INTEGER NOT NULL,
                data_id INTEGER NOT NULL,
                snapshot_time TIMESTAMP NOT NULL,
                num_strikes INTEGER,
                num_expirations INTEGER,
                num_points INTEGER,  -- Total number of IV points
                metadata TEXT,  -- JSON with surface construction details
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (security_id) REFERENCES securities(security_id),
                FOREIGN KEY (data_id) REFERENCES market_data(data_id),
                UNIQUE(security_id, snapshot_time)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_surfaces_security_time
            ON iv_surfaces(security_id, snapshot_time DESC)
        """)

        self.conn.commit()

    def add_security(self, ticker: str, name: Optional[str] = None,
                     isin: Optional[str] = None, cusip: Optional[str] = None,
                     exchange: Optional[str] = None, sector: Optional[str] = None) -> int:
        """
        Add or update a security.

        Args:
            ticker: Stock ticker symbol (required)
            name: Full company name
            isin: International Securities Identification Number
            cusip: Committee on Uniform Securities Identification Procedures number
            exchange: Exchange where security is traded
            sector: Business sector

        Returns:
            security_id: Database ID of the security
        """
        cursor = self.conn.cursor()

        # Try to insert, update on conflict
        cursor.execute("""
            INSERT INTO securities (ticker, name, isin, cusip, exchange, sector)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                name = COALESCE(excluded.name, name),
                isin = COALESCE(excluded.isin, isin),
                cusip = COALESCE(excluded.cusip, cusip),
                exchange = COALESCE(excluded.exchange, exchange),
                sector = COALESCE(excluded.sector, sector),
                updated_at = CURRENT_TIMESTAMP
        """, (ticker, name, isin, cusip, exchange, sector))

        self.conn.commit()

        # Get the security_id
        cursor.execute("SELECT security_id FROM securities WHERE ticker = ?", (ticker,))
        return cursor.fetchone()[0]

    def get_security(self, ticker: str) -> Optional[Dict]:
        """Get security information by ticker."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM securities WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def add_market_data(self, security_id: int, timestamp: datetime,
                       spot_price: float, bid: Optional[float] = None,
                       ask: Optional[float] = None, volume: Optional[int] = None,
                       dividend_yield: Optional[float] = None) -> int:
        """Add market data snapshot for underlying security."""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO market_data
            (security_id, timestamp, spot_price, bid, ask, volume, dividend_yield)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(security_id, timestamp) DO UPDATE SET
                spot_price = excluded.spot_price,
                bid = excluded.bid,
                ask = excluded.ask,
                volume = excluded.volume,
                dividend_yield = excluded.dividend_yield
        """, (security_id, timestamp, spot_price, bid, ask, volume, dividend_yield))

        self.conn.commit()
        return cursor.lastrowid

    def add_option_contract(self, security_id: int, contract_symbol: str,
                           option_type: str, strike: float, expiration: str,
                           exercise_style: str = "AMERICAN") -> int:
        """Add option contract specification."""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO option_contracts
            (security_id, contract_symbol, option_type, strike, expiration, exercise_style)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(contract_symbol) DO NOTHING
        """, (security_id, contract_symbol, option_type.upper(), strike, expiration, exercise_style))

        self.conn.commit()

        # Get contract_id
        cursor.execute("SELECT contract_id FROM option_contracts WHERE contract_symbol = ?",
                      (contract_symbol,))
        return cursor.fetchone()[0]

    def add_option_price(self, contract_id: int, timestamp: datetime,
                        bid: Optional[float] = None, ask: Optional[float] = None,
                        last: Optional[float] = None, volume: Optional[int] = None,
                        open_interest: Optional[int] = None) -> int:
        """Add option price data."""
        cursor = self.conn.cursor()

        # Calculate mid price if bid and ask available
        mid_price = None
        if bid is not None and ask is not None:
            mid_price = (bid + ask) / 2.0

        cursor.execute("""
            INSERT INTO option_prices
            (contract_id, timestamp, bid, ask, last, mid_price, volume, open_interest)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(contract_id, timestamp) DO UPDATE SET
                bid = excluded.bid,
                ask = excluded.ask,
                last = excluded.last,
                mid_price = excluded.mid_price,
                volume = excluded.volume,
                open_interest = excluded.open_interest
        """, (contract_id, timestamp, bid, ask, last, mid_price, volume, open_interest))

        self.conn.commit()
        return cursor.lastrowid

    def add_iv_calculation(self, contract_id: int, data_id: int, price_id: int,
                          market_price: float, spot_price: float, time_to_maturity: float,
                          risk_free_rate: float, implied_volatility: Optional[float],
                          converged: bool, iterations: int, final_error: float,
                          failure_reason: Optional[str] = None, vega: Optional[float] = None,
                          solver_config: Optional[Dict] = None) -> int:
        """Add IV calculation result."""
        cursor = self.conn.cursor()

        config_json = json.dumps(solver_config) if solver_config else None

        cursor.execute("""
            INSERT INTO iv_calculations
            (contract_id, data_id, price_id, market_price, spot_price, time_to_maturity,
             risk_free_rate, implied_volatility, converged, iterations, final_error,
             failure_reason, vega, solver_config)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(contract_id, data_id, price_id) DO UPDATE SET
                market_price = excluded.market_price,
                implied_volatility = excluded.implied_volatility,
                converged = excluded.converged,
                iterations = excluded.iterations,
                final_error = excluded.final_error,
                failure_reason = excluded.failure_reason,
                vega = excluded.vega,
                calculated_at = CURRENT_TIMESTAMP
        """, (contract_id, data_id, price_id, market_price, spot_price, time_to_maturity,
              risk_free_rate, implied_volatility, converged, iterations, final_error,
              failure_reason, vega, config_json))

        self.conn.commit()
        return cursor.lastrowid

    def get_iv_surface(self, security_id: int, limit: int = 100) -> List[Dict]:
        """
        Get recent IV calculations for a security.

        Returns list of dicts with IV data sorted by expiration and strike.
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                oc.strike,
                oc.expiration,
                oc.option_type,
                iv.time_to_maturity,
                iv.implied_volatility,
                iv.converged,
                iv.market_price,
                iv.spot_price,
                iv.calculated_at,
                iv.vega
            FROM iv_calculations iv
            JOIN option_contracts oc ON iv.contract_id = oc.contract_id
            WHERE oc.security_id = ?
              AND iv.converged = 1
            ORDER BY oc.expiration, oc.strike
            LIMIT ?
        """, (security_id, limit))

        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

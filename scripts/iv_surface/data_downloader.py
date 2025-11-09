"""
Download option chain data from Yahoo Finance.

This module provides functions to:
- Download current option chains for a ticker
- Extract security identification (ticker, ISIN, etc.)
- Format data for database storage
"""

import yfinance as yf
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


class OptionDataDownloader:
    """Downloads and processes option chain data from Yahoo Finance."""

    def __init__(self, ticker: str):
        """
        Initialize downloader for a specific ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'SPY')
        """
        self.ticker = ticker.upper()
        self.yf_ticker = yf.Ticker(self.ticker)
        self._info = None

    @property
    def info(self) -> Dict:
        """Get ticker information (cached)."""
        if self._info is None:
            self._info = self.yf_ticker.info
        return self._info

    def get_security_info(self) -> Dict:
        """
        Extract security identification and metadata.

        Returns:
            Dict with security information including:
            - ticker: Stock ticker
            - name: Company name
            - isin: ISIN code (if available)
            - cusip: CUSIP code (if available)
            - exchange: Exchange code
            - sector: Business sector
        """
        info = self.info

        return {
            'ticker': self.ticker,
            'name': info.get('longName') or info.get('shortName'),
            'isin': info.get('isin'),
            'cusip': info.get('cusip'),
            'exchange': info.get('exchange'),
            'sector': info.get('sector'),
        }

    def get_current_price(self) -> Dict:
        """
        Get current market data for the underlying.

        Returns:
            Dict with:
            - spot_price: Current price
            - bid: Bid price
            - ask: Ask price
            - volume: Trading volume
            - dividend_yield: Annual dividend yield
            - timestamp: Data timestamp
        """
        info = self.info
        history = self.yf_ticker.history(period='1d')

        # Get most recent price
        if not history.empty:
            spot_price = history['Close'].iloc[-1]
        else:
            spot_price = info.get('currentPrice') or info.get('regularMarketPrice')

        # Get bid/ask
        bid = info.get('bid')
        ask = info.get('ask')

        # Get volume
        if not history.empty:
            volume = int(history['Volume'].iloc[-1])
        else:
            volume = info.get('volume')

        # Get dividend yield (as decimal, e.g., 0.02 for 2%)
        dividend_yield = info.get('dividendYield')

        return {
            'spot_price': float(spot_price),
            'bid': float(bid) if bid else None,
            'ask': float(ask) if ask else None,
            'volume': volume,
            'dividend_yield': dividend_yield,
            'timestamp': datetime.now(),
        }

    def get_option_expirations(self) -> List[str]:
        """
        Get available option expiration dates.

        Returns:
            List of expiration dates in YYYY-MM-DD format
        """
        return list(self.yf_ticker.options)

    def download_option_chain(self, expiration: Optional[str] = None,
                             min_volume: int = 10,
                             exclude_itm_by: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download option chain for a specific expiration.

        Args:
            expiration: Expiration date (YYYY-MM-DD). If None, uses nearest expiration.
            min_volume: Minimum volume filter (exclude illiquid options)
            exclude_itm_by: Exclude options ITM by more than this fraction
                           (e.g., 0.2 = exclude options >20% ITM)

        Returns:
            Tuple of (calls_df, puts_df) with columns:
            - contractSymbol: Option contract identifier
            - strike: Strike price
            - lastPrice: Last traded price
            - bid: Bid price
            - ask: Ask price
            - volume: Trading volume
            - openInterest: Open interest
            - impliedVolatility: Exchange-reported IV (for comparison)
        """
        if expiration is None:
            expirations = self.get_option_expirations()
            if not expirations:
                raise ValueError(f"No options available for {self.ticker}")
            expiration = expirations[0]

        # Get option chain
        opt_chain = self.yf_ticker.option_chain(expiration)
        calls = opt_chain.calls.copy()
        puts = opt_chain.puts.copy()

        # Get current spot price for filtering
        spot = self.get_current_price()['spot_price']

        # Filter by volume
        if min_volume > 0:
            calls = calls[calls['volume'] >= min_volume]
            puts = puts[puts['volume'] >= min_volume]

        # Filter out deep ITM options
        if exclude_itm_by > 0:
            # For calls, ITM means strike < spot
            call_threshold = spot * (1 - exclude_itm_by)
            calls = calls[calls['strike'] >= call_threshold]

            # For puts, ITM means strike > spot
            put_threshold = spot * (1 + exclude_itm_by)
            puts = puts[puts['strike'] <= put_threshold]

        # Select relevant columns
        columns = ['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask',
                  'volume', 'openInterest', 'impliedVolatility']

        calls = calls[columns].reset_index(drop=True)
        puts = puts[columns].reset_index(drop=True)

        return calls, puts

    def download_all_expirations(self, max_expirations: Optional[int] = None,
                                 **kwargs) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Download option chains for all available expirations.

        Args:
            max_expirations: Maximum number of expirations to download (None = all)
            **kwargs: Additional arguments passed to download_option_chain()

        Returns:
            Dict mapping expiration date to (calls_df, puts_df)
        """
        expirations = self.get_option_expirations()

        if max_expirations:
            expirations = expirations[:max_expirations]

        result = {}
        for exp in expirations:
            try:
                calls, puts = self.download_option_chain(exp, **kwargs)
                if not calls.empty or not puts.empty:
                    result[exp] = (calls, puts)
            except Exception as e:
                print(f"Warning: Failed to download options for {exp}: {e}")
                continue

        return result

    def calculate_time_to_maturity(self, expiration: str) -> float:
        """
        Calculate time to maturity in years.

        Args:
            expiration: Expiration date (YYYY-MM-DD)

        Returns:
            Time to maturity in years (e.g., 0.25 for 3 months)
        """
        exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
        today = date.today()

        days_to_exp = (exp_date - today).days
        return max(days_to_exp / 365.25, 1/365.25)  # At least 1 day

    def estimate_risk_free_rate(self) -> float:
        """
        Estimate risk-free rate from Treasury data.

        For simplicity, uses a proxy or allows manual override.
        In production, you'd fetch current Treasury rates.

        Returns:
            Estimated risk-free rate (e.g., 0.05 for 5%)
        """
        # Simple proxy: try to get from market data
        # In practice, you'd query current 3-month or 10-year Treasury rates
        try:
            # Try to get from ticker info if available
            info = self.info
            # Some tickers may have this, but most won't
            return 0.05  # Default 5% - should be updated with real data
        except:
            return 0.05


def download_and_format_options(ticker: str, max_expirations: Optional[int] = None,
                                min_volume: int = 10) -> Dict:
    """
    Convenience function to download all option data for a ticker.

    Args:
        ticker: Stock ticker symbol
        max_expirations: Max number of expirations to download
        min_volume: Minimum volume filter

    Returns:
        Dict with:
        - security_info: Security metadata
        - market_data: Current market data
        - option_chains: Dict mapping expiration to (calls, puts)
    """
    downloader = OptionDataDownloader(ticker)

    security_info = downloader.get_security_info()
    market_data = downloader.get_current_price()
    option_chains = downloader.download_all_expirations(
        max_expirations=max_expirations,
        min_volume=min_volume
    )

    return {
        'security_info': security_info,
        'market_data': market_data,
        'option_chains': option_chains,
        'downloader': downloader,  # Keep reference for rate/maturity calculations
    }

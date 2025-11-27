"""
IV Surface Calculator - Python package for option data and implied volatility.

This package provides tools to download option chain data from Yahoo Finance
and calculate implied volatility surfaces using the mango-option C++ library.

Modules:
    database: SQLite3 database schema and utilities
    data_downloader: Yahoo Finance data download
    iv_calculator: IV calculation using C++ bindings
    calculate_iv_surface: Main orchestration script

Example:
    from iv_surface import OptionDatabase, OptionDataDownloader, IVCalculator

    # Download data
    downloader = OptionDataDownloader("AAPL")
    security_info = downloader.get_security_info()
    market_data = downloader.get_current_price()

    # Calculate IV
    calculator = IVCalculator()
    result = calculator.calculate_iv(
        spot_price=150.0,
        strike=155.0,
        time_to_maturity=0.25,
        risk_free_rate=0.05,
        market_price=2.50,
        is_call=True
    )
    print(f"Implied Volatility: {result['implied_vol']}")
"""

from .database import OptionDatabase
from .data_downloader import OptionDataDownloader, download_and_format_options
from .iv_calculator import IVCalculator, validate_option_price

__version__ = "1.0.0"
__all__ = [
    "OptionDatabase",
    "OptionDataDownloader",
    "download_and_format_options",
    "IVCalculator",
    "validate_option_price",
]

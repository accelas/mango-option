"""
IV surface calculation using mango-iv C++ library.

This module provides functions to:
- Calculate implied volatility for individual options
- Calculate IV surface for entire option chains
- Batch processing with parallelization
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Import the C++ bindings (will be built with Bazel)
try:
    import mango_iv
except ImportError:
    print("Warning: mango_iv module not found. Please build with: bazel build //python:mango_iv")
    print("Using mock implementation for development")
    mango_iv = None


class IVCalculator:
    """Calculate implied volatility using mango-iv C++ solver."""

    def __init__(self, grid_n_space: int = 101, grid_n_time: int = 1000,
                 max_iter: int = 100, tolerance: float = 1e-6):
        """
        Initialize IV calculator with solver configuration.

        Args:
            grid_n_space: Number of spatial grid points for PDE solver
            grid_n_time: Number of time steps for PDE solver
            max_iter: Maximum iterations for root-finding
            tolerance: Convergence tolerance
        """
        if mango_iv is None:
            raise ImportError("mango_iv module not available. Build with Bazel first.")

        # Create solver configuration
        self.config = mango_iv.IVConfig()
        self.config.grid_n_space = grid_n_space
        self.config.grid_n_time = grid_n_time
        self.config.root_config.max_iter = max_iter
        self.config.root_config.tolerance = tolerance

    def calculate_iv(self, spot_price: float, strike: float, time_to_maturity: float,
                     risk_free_rate: float, market_price: float,
                     is_call: bool = True) -> Dict:
        """
        Calculate implied volatility for a single option.

        Args:
            spot_price: Current underlying price
            strike: Option strike price
            time_to_maturity: Time to expiration in years
            risk_free_rate: Risk-free interest rate (as decimal)
            market_price: Observed market price
            is_call: True for call, False for put

        Returns:
            Dict with:
            - implied_vol: Calculated IV (if converged)
            - converged: Boolean convergence status
            - iterations: Number of iterations
            - final_error: Final pricing error
            - failure_reason: Error message if failed
            - vega: Vega at the solution (if available)
        """
        # Create IV parameters
        params = mango_iv.IVParams()
        params.spot_price = float(spot_price)
        params.strike = float(strike)
        params.time_to_maturity = float(time_to_maturity)
        params.risk_free_rate = float(risk_free_rate)
        params.market_price = float(market_price)
        params.is_call = bool(is_call)

        # Create solver and solve
        solver = mango_iv.IVSolver(params, self.config)
        result = solver.solve()

        # Convert result to dict
        return {
            'implied_vol': result.implied_vol if result.converged else None,
            'converged': result.converged,
            'iterations': result.iterations,
            'final_error': result.final_error,
            'failure_reason': result.failure_reason if hasattr(result, 'failure_reason') else None,
            'vega': result.vega if hasattr(result, 'vega') else None,
        }

    def calculate_iv_batch(self, params_list: List[Dict]) -> List[Dict]:
        """
        Calculate IV for a batch of options in parallel.

        Args:
            params_list: List of dicts, each containing:
                - spot_price, strike, time_to_maturity, risk_free_rate,
                  market_price, is_call

        Returns:
            List of result dicts (same order as input)
        """
        # Create list of IVParams
        iv_params = []
        for p in params_list:
            params = mango_iv.IVParams()
            params.spot_price = float(p['spot_price'])
            params.strike = float(p['strike'])
            params.time_to_maturity = float(p['time_to_maturity'])
            params.risk_free_rate = float(p['risk_free_rate'])
            params.market_price = float(p['market_price'])
            params.is_call = bool(p.get('is_call', True))
            iv_params.append(params)

        # Solve batch in parallel
        results = mango_iv.solve_implied_vol_batch(iv_params, self.config)

        # Convert results to dicts
        return [
            {
                'implied_vol': r.implied_vol if r.converged else None,
                'converged': r.converged,
                'iterations': r.iterations,
                'final_error': r.final_error,
                'failure_reason': r.failure_reason if hasattr(r, 'failure_reason') else None,
                'vega': r.vega if hasattr(r, 'vega') else None,
            }
            for r in results
        ]

    def calculate_chain_iv(self, options_df: pd.DataFrame, spot_price: float,
                          risk_free_rate: float, time_to_maturity: float,
                          is_call: bool = True,
                          price_column: str = 'mid_price') -> pd.DataFrame:
        """
        Calculate IV for an entire option chain.

        Args:
            options_df: DataFrame with columns: strike, bid, ask, lastPrice, etc.
            spot_price: Current underlying price
            risk_free_rate: Risk-free rate
            time_to_maturity: Time to expiration in years
            is_call: True for calls, False for puts
            price_column: Column to use for market price ('mid_price', 'lastPrice', etc.)

        Returns:
            DataFrame with original columns plus IV results
        """
        df = options_df.copy()

        # Calculate mid price if needed
        if price_column == 'mid_price' and 'mid_price' not in df.columns:
            if 'bid' in df.columns and 'ask' in df.columns:
                df['mid_price'] = (df['bid'] + df['ask']) / 2.0
            else:
                price_column = 'lastPrice'

        # Filter out rows with missing prices
        valid_mask = df[price_column].notna() & (df[price_column] > 0)
        valid_df = df[valid_mask].copy()

        if valid_df.empty:
            # No valid prices, return empty results
            df['calculated_iv'] = np.nan
            df['converged'] = False
            df['iterations'] = 0
            df['final_error'] = np.nan
            return df

        # Build parameter list for batch calculation
        params_list = [
            {
                'spot_price': spot_price,
                'strike': row['strike'],
                'time_to_maturity': time_to_maturity,
                'risk_free_rate': risk_free_rate,
                'market_price': row[price_column],
                'is_call': is_call,
            }
            for _, row in valid_df.iterrows()
        ]

        # Calculate IV in batch
        results = self.calculate_iv_batch(params_list)

        # Add results to dataframe
        valid_df['calculated_iv'] = [r['implied_vol'] for r in results]
        valid_df['converged'] = [r['converged'] for r in results]
        valid_df['iterations'] = [r['iterations'] for r in results]
        valid_df['final_error'] = [r['final_error'] for r in results]
        valid_df['vega'] = [r['vega'] for r in results]

        # Merge back to original dataframe
        df = df.join(valid_df[['calculated_iv', 'converged', 'iterations', 'final_error', 'vega']],
                     how='left')

        return df

    def calculate_surface(self, option_chains: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
                         spot_price: float, risk_free_rate: float,
                         time_calculator_fn) -> Dict[str, Dict]:
        """
        Calculate IV surface for all expirations and strikes.

        Args:
            option_chains: Dict mapping expiration to (calls_df, puts_df)
            spot_price: Current underlying price
            risk_free_rate: Risk-free rate
            time_calculator_fn: Function(expiration_str) -> time_to_maturity

        Returns:
            Dict mapping expiration to {'calls': df_with_iv, 'puts': df_with_iv}
        """
        surface = {}

        for expiration, (calls_df, puts_df) in option_chains.items():
            ttm = time_calculator_fn(expiration)

            result = {}

            if not calls_df.empty:
                calls_with_iv = self.calculate_chain_iv(
                    calls_df, spot_price, risk_free_rate, ttm, is_call=True
                )
                result['calls'] = calls_with_iv

            if not puts_df.empty:
                puts_with_iv = self.calculate_chain_iv(
                    puts_df, spot_price, risk_free_rate, ttm, is_call=False
                )
                result['puts'] = puts_with_iv

            surface[expiration] = result

        return surface

    def get_config_dict(self) -> Dict:
        """Get solver configuration as dict for database storage."""
        return {
            'grid_n_space': self.config.grid_n_space,
            'grid_n_time': self.config.grid_n_time,
            'max_iter': self.config.root_config.max_iter,
            'tolerance': self.config.root_config.tolerance,
        }


def validate_option_price(spot: float, strike: float, market_price: float,
                         is_call: bool) -> Tuple[bool, Optional[str]]:
    """
    Validate option price for arbitrage violations.

    Returns:
        (is_valid, error_message)
    """
    if market_price <= 0:
        return False, "Market price must be positive"

    # Check arbitrage bounds
    if is_call:
        # Call value <= spot (otherwise arbitrage)
        if market_price > spot:
            return False, f"Call price ({market_price}) exceeds spot ({spot})"

        # Call value >= intrinsic = max(spot - strike, 0)
        intrinsic = max(spot - strike, 0)
        if market_price < intrinsic * 0.99:  # Small tolerance for rounding
            return False, f"Call price ({market_price}) below intrinsic ({intrinsic})"
    else:
        # Put value <= strike
        if market_price > strike:
            return False, f"Put price ({market_price}) exceeds strike ({strike})"

        # Put value >= intrinsic = max(strike - spot, 0)
        intrinsic = max(strike - spot, 0)
        if market_price < intrinsic * 0.99:
            return False, f"Put price ({market_price}) below intrinsic ({intrinsic})"

    return True, None

"""Volatility models: realized vol and GARCH."""

import numpy as np


def analyze_volatility(equity_data: dict, model_type: str = "gjr") -> dict:
    """
    Analyze volatility for equity data.

    Args:
        equity_data: Dict with 'close' prices
        model_type: 'garch', 'gjr', or 'egarch'

    Returns:
        Dict with realized vol, GARCH forecast, and model params
    """
    if "error" in equity_data:
        return equity_data

    closes = np.array(equity_data.get("close", []))
    if len(closes) < 30:
        return {"error": "Insufficient data for volatility analysis"}

    # Compute log returns (percentage for GARCH stability)
    log_returns = 100 * np.diff(np.log(closes))

    # Realized volatility (different windows)
    rv_20 = _realized_vol(log_returns, 20)
    rv_60 = _realized_vol(log_returns, 60)

    # GARCH forecast
    garch_result = _fit_garch(log_returns, model_type)

    return {
        "realized_vol": {
            "rv_20": rv_20,
            "rv_60": rv_60,
            "current_return": float(log_returns[-1]) if len(log_returns) > 0 else 0,
        },
        "garch": garch_result,
        "model_type": model_type,
    }


def _realized_vol(returns: np.ndarray, window: int) -> float:
    """Compute annualized realized volatility."""
    if len(returns) < window:
        window = len(returns)
    recent = returns[-window:]
    # Returns are in %, so divide by 100 before annualizing
    return float(np.std(recent) * np.sqrt(252) / 100)


def _fit_garch(returns: np.ndarray, model_type: str) -> dict:
    """
    Fit GARCH model and forecast.

    Returns dict with current vol, forecast, and params.
    """
    try:
        from arch import arch_model

        # Select model
        if model_type == "garch":
            model = arch_model(returns, vol="Garch", p=1, q=1, rescale=True)
        elif model_type == "gjr":
            model = arch_model(returns, vol="Garch", p=1, o=1, q=1, rescale=True)
        elif model_type == "egarch":
            model = arch_model(returns, vol="EGARCH", p=1, q=1, rescale=True)
        else:
            return {"error": f"Unknown model type: {model_type}"}

        # Fit
        result = model.fit(disp="off", show_warning=False)

        if not result.convergence_flag == 0:
            return {
                "error": "GARCH did not converge",
                "fallback_vol": _realized_vol(returns, 20),
            }

        # Forecast
        forecast = result.forecast(horizon=5)
        forecast_var = forecast.variance.values[-1, :]

        # Annualize: returns are in %, variance is in %^2
        # sqrt(var) gives % vol, then annualize and convert to decimal
        annualized = np.sqrt(forecast_var * 252) / 100

        # Current conditional volatility
        current_vol = float(result.conditional_volatility[-1] * np.sqrt(252) / 100)

        return {
            "current_vol": current_vol,
            "forecast": annualized.tolist(),
            "forecast_days": [1, 2, 3, 4, 5],
            "params": {
                "omega": float(result.params.get("omega", 0)),
                "alpha": float(result.params.get("alpha[1]", 0)),
                "beta": float(result.params.get("beta[1]", 0)),
                "gamma": float(result.params.get("gamma[1]", 0))
                if "gamma[1]" in result.params
                else None,
            },
            "log_likelihood": float(result.loglikelihood),
            "aic": float(result.aic),
            "bic": float(result.bic),
        }

    except ImportError:
        return {
            "error": "arch library not installed",
            "fallback_vol": _realized_vol(returns, 20),
        }
    except Exception as e:
        return {
            "error": str(e),
            "fallback_vol": _realized_vol(returns, 20),
        }

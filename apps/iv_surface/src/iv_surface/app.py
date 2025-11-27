"""FastAPI application for IV surface visualization."""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from iv_surface.services import cache, yfinance_service, iv_compute

# Paths
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR.parent.parent.parent / "data"

app = FastAPI(
    title="IV Surface",
    description="Interactive implied volatility surface visualization",
    version="0.1.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.on_event("startup")
async def startup() -> None:
    """Initialize database on startup."""
    cache.init_db(DATA_DIR / "iv_surface.duckdb")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render main page."""
    cached_symbols = cache.get_cached_symbols()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "cached_symbols": cached_symbols,
            "current_symbol": None,
        },
    )


@app.get("/api/surface/{symbol}")
async def get_surface(symbol: str, option_type: str = "put") -> dict:
    """
    Compute IV surface for a symbol.

    Returns surface data for Plotly 3D visualization.
    Now returns BOTH puts and calls to avoid redundant fetches.
    """
    symbol = symbol.upper()

    # Fetch option chain from yfinance
    chain_data = await yfinance_service.fetch_option_chain(symbol)
    if "error" in chain_data:
        return chain_data

    # Compute IVs for BOTH puts and calls
    put_surface = iv_compute.compute_iv_surface(chain_data, option_type="put")
    call_surface = iv_compute.compute_iv_surface(chain_data, option_type="call")

    # Cache the result
    cache.store_options_snapshot(symbol, chain_data)

    # Return both surfaces - client can switch without re-fetching
    return {
        "symbol": symbol,
        "spot": chain_data["spot"],
        "quote_time": chain_data["quote_time"],
        "is_eod": chain_data.get("is_eod", False),
        "dividend_yield": chain_data.get("dividend_yield", 0.0),
        "rate": 0.05,  # Default rate
        "surfaces": {
            "put": put_surface.get("surface") if "surface" in put_surface else None,
            "call": call_surface.get("surface") if "surface" in call_surface else None,
        },
        "stats": {
            "put": put_surface.get("stats") if "stats" in put_surface else None,
            "call": call_surface.get("stats") if "stats" in call_surface else None,
        },
    }


@app.get("/api/surface/{symbol}/stream")
async def stream_surface(symbol: str):
    """
    Stream IV surface data as it's computed, expiration by expiration.

    Uses Server-Sent Events for progressive rendering.
    """
    symbol = symbol.upper()

    async def generate():
        # First, fetch the option chain
        chain_data = await yfinance_service.fetch_option_chain(symbol)

        if "error" in chain_data:
            yield {"event": "error", "data": json.dumps(chain_data)}
            return

        # Send metadata first
        yield {
            "event": "metadata",
            "data": json.dumps({
                "symbol": symbol,
                "spot": chain_data["spot"],
                "quote_time": chain_data["quote_time"],
                "is_eod": chain_data.get("is_eod", False),
                "dividend_yield": chain_data.get("dividend_yield", 0.0),
                "expirations": chain_data["expirations"],
                "total_expirations": len(chain_data["expirations"]),
            })
        }

        # Stream IV data by expiration
        all_put_data = {"strikes": [], "moneyness": [], "maturities": [], "maturities_days": [], "ivs": [], "prices": []}
        all_call_data = {"strikes": [], "moneyness": [], "maturities": [], "maturities_days": [], "ivs": [], "prices": []}

        for i, expiry in enumerate(chain_data["expirations"]):
            # Compute IVs for this expiration only
            single_chain = {
                "symbol": symbol,
                "spot": chain_data["spot"],
                "dividend_yield": chain_data.get("dividend_yield", 0.0),
                "quote_time": chain_data["quote_time"],
                "is_eod": chain_data.get("is_eod", False),
                "chains": {expiry: chain_data["chains"][expiry]},
            }

            put_result = iv_compute.compute_iv_surface(single_chain, option_type="put")
            call_result = iv_compute.compute_iv_surface(single_chain, option_type="call")

            # Accumulate data
            if "surface" in put_result:
                for key in all_put_data:
                    all_put_data[key].extend(put_result["surface"].get(key, []))
            if "surface" in call_result:
                for key in all_call_data:
                    all_call_data[key].extend(call_result["surface"].get(key, []))

            # Send progress update with current data
            yield {
                "event": "progress",
                "data": json.dumps({
                    "expiration": expiry,
                    "index": i + 1,
                    "total": len(chain_data["expirations"]),
                    "surfaces": {
                        "put": all_put_data if all_put_data["ivs"] else None,
                        "call": all_call_data if all_call_data["ivs"] else None,
                    },
                })
            }

            # Small yield to allow other tasks
            await asyncio.sleep(0)

        # Send final complete message
        yield {
            "event": "complete",
            "data": json.dumps({
                "symbol": symbol,
                "spot": chain_data["spot"],
                "surfaces": {
                    "put": all_put_data if all_put_data["ivs"] else None,
                    "call": all_call_data if all_call_data["ivs"] else None,
                },
                "stats": {
                    "put": {
                        "n_points": len(all_put_data["ivs"]),
                        "iv_min": min(all_put_data["ivs"]) if all_put_data["ivs"] else 0,
                        "iv_max": max(all_put_data["ivs"]) if all_put_data["ivs"] else 0,
                        "iv_mean": sum(all_put_data["ivs"]) / len(all_put_data["ivs"]) if all_put_data["ivs"] else 0,
                    } if all_put_data["ivs"] else None,
                    "call": {
                        "n_points": len(all_call_data["ivs"]),
                        "iv_min": min(all_call_data["ivs"]) if all_call_data["ivs"] else 0,
                        "iv_max": max(all_call_data["ivs"]) if all_call_data["ivs"] else 0,
                        "iv_mean": sum(all_call_data["ivs"]) / len(all_call_data["ivs"]) if all_call_data["ivs"] else 0,
                    } if all_call_data["ivs"] else None,
                },
            })
        }

        # Cache the result
        cache.store_options_snapshot(symbol, chain_data)

    return EventSourceResponse(generate())


@app.get("/api/equity/{symbol}")
async def get_equity(symbol: str, period: str = "1y") -> dict:
    """
    Get equity price history for a symbol.

    Returns OHLCV data for charting.
    """
    symbol = symbol.upper()
    return await yfinance_service.fetch_equity_history(symbol, period)


@app.get("/api/vol_analysis/{symbol}")
async def get_vol_analysis(symbol: str, model: str = "gjr") -> dict:
    """
    Get volatility analysis for a symbol.

    Returns realized vol, GARCH forecast, and ATM IV.
    """
    symbol = symbol.upper()

    # Get historical data
    equity_data = await yfinance_service.fetch_equity_history(symbol, "1y")
    if "error" in equity_data:
        return equity_data

    # Compute volatility metrics
    from iv_surface.services import vol_models

    return vol_models.analyze_volatility(equity_data, model_type=model)


@app.get("/api/cached")
async def get_cached() -> dict:
    """Get list of cached symbols with freshness info."""
    return {"symbols": cache.get_cached_symbols()}

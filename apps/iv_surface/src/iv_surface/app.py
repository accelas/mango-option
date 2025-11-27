"""FastAPI application for IV surface visualization."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
    """
    symbol = symbol.upper()

    # Fetch option chain from yfinance
    chain_data = await yfinance_service.fetch_option_chain(symbol)
    if "error" in chain_data:
        return chain_data

    # Compute IVs
    surface_data = iv_compute.compute_iv_surface(
        chain_data,
        option_type=option_type,
    )

    # Cache the result
    cache.store_options_snapshot(symbol, chain_data)

    return surface_data


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

# IV Surface

Interactive implied volatility surface visualization using mango-iv.

## Quick Start

```bash
cd apps/iv_surface
uv sync
uv run iv-surface
```

Open http://127.0.0.1:8000 in your browser.

## Features

- 3D interactive IV surface (Plotly)
- Real-time option chain fetching (yfinance)
- Equity price charts
- Volatility analysis (realized vol, GARCH)
- Mobile-responsive design

## Development

```bash
# Run with auto-reload
uv run iv-surface

# Or directly with uvicorn
uv run uvicorn iv_surface.app:app --reload
```

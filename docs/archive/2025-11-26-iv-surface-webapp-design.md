<!-- SPDX-License-Identifier: MIT -->
# IV Surface Web App Design

## Overview

A FastAPI web application that fetches option chain data from yfinance, computes implied volatility surfaces using mango-iv, and displays interactive 3D visualizations. Includes equity analysis with realized volatility and GARCH forecasting.

## Architecture

```
apps/iv_surface/
├── pyproject.toml          # uv-based packaging
├── src/
│   └── iv_surface/
│       ├── __init__.py
│       ├── __main__.py     # Entry: `uv run iv-surface`
│       ├── app.py          # FastAPI app
│       ├── services/
│       │   ├── yfinance.py     # Fetch option chain, EOD quotes
│       │   ├── iv_compute.py   # Build price table, solve IVs
│       │   ├── cache.py        # DuckDB + Arrow file management
│       │   └── vol_models.py   # GARCH, realized vol
│       ├── templates/
│       │   └── index.html      # Tailwind + HTMX + Plotly
│       └── static/
└── data/                   # Runtime data directory
    ├── iv_surface.duckdb
    └── tables/             # Arrow files: {symbol}_{put|call}.arrow
```

## Data Flow

1. User enters symbol → POST to `/api/surface`
2. `cache.py` checks DuckDB for cached price table
3. Cache miss:
   - `yfinance.py` fetches current option chain + EOD quotes
   - `iv_compute.py` builds price table (~10-30s)
   - Save to Arrow file, register in DuckDB
4. Cache hit → load Arrow directly
5. Solve IVs for all strikes/expiries
6. Return JSON with surface data
7. Frontend renders 3D Plotly surface

## DuckDB Schema

```sql
-- Symbol registry with corporate action tracking
CREATE TABLE symbols (
    symbol VARCHAR PRIMARY KEY,
    name VARCHAR,
    status VARCHAR DEFAULT 'active',  -- active, delisted, renamed, merged
    renamed_to VARCHAR,
    merged_into VARCHAR,
    has_options BOOLEAN DEFAULT true,
    last_quote_fetch TIMESTAMP,
    last_options_fetch TIMESTAMP
);

-- EOD equity data (historical, for realized vol)
CREATE TABLE eod_quotes (
    symbol VARCHAR,
    date DATE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    adj_close DOUBLE,
    volume BIGINT,
    PRIMARY KEY (symbol, date)
);

-- Dividend history
CREATE TABLE dividends (
    symbol VARCHAR,
    ex_date DATE,
    amount DOUBLE,
    PRIMARY KEY (symbol, ex_date)
);

-- Options snapshots (accumulated over time)
CREATE TABLE options_snapshots (
    symbol VARCHAR,
    fetched_at TIMESTAMP,
    is_eod BOOLEAN,             -- True if after 4pm ET
    expiry DATE,
    strike DOUBLE,
    option_type VARCHAR,        -- 'call' or 'put'
    bid DOUBLE,
    ask DOUBLE,
    last DOUBLE,
    volume INTEGER,
    open_interest INTEGER,
    computed_iv DOUBLE,
    PRIMARY KEY (symbol, fetched_at, expiry, strike, option_type)
);

-- Cached price tables (Arrow files)
CREATE TABLE price_tables (
    symbol VARCHAR,
    option_type VARCHAR,
    arrow_path VARCHAR,
    created_at TIMESTAMP,
    spot_at_build DOUBLE,
    rate_at_build DOUBLE,
    UNIQUE(symbol, option_type)
);
```

## EOD vs Intraday Data

- EOD data: fetched after 4pm ET, stored in `eod_quotes` and `options_snapshots` with `is_eod=true`
- Intraday data: timestamped snapshots for research, `is_eod=false`
- UI indicates data freshness: "EOD 2024-11-26" vs "2:35 PM snapshot (market open)"

## UI Design

### Layout

```
┌──────────┬──────────────────────────────────────────┐
│  ◆ IV    │   [Search symbol...]          [Refresh]  │
│  Surface │                                          │
├──────────┼──────────────────────────────────────────┤
│ CACHED   │  [IV Surface]  [Equity]  [Vol Analysis]  │
│ ──────── ├──────────────────────────────────────────┤
│ AAPL  ●  │                                          │
│ SPY  15m │      Tab content (3D surface / charts)   │
│ TSLA  1h │                                          │
│          │                                          │
│          ├──────────────────────────────────────────┤
│          │  Spot: $189.45  |  Rate: 5.2%  |  Puts   │
└──────────┴──────────────────────────────────────────┘
```

### Landing State

When no symbol is loaded, show welcome message guiding user to enter a symbol or select from cached list.

### Tabs

1. **IV Surface** - 3D Plotly surface (strike × maturity × IV), rotatable/zoomable
2. **Equity** - Price chart with volume, period selectors (1M/3M/1Y)
3. **Vol Analysis** - Realized vol, GARCH forecast, ATM IV comparison

### Visual Style

- Dark mode default (slate-900 background, slate-800 cards)
- Blue accent (#3b82f6) for interactive elements
- Inter/system fonts, clear hierarchy
- Professional financial-app aesthetic

## Vol Analysis

### Realized Volatility

```python
def realized_vol(symbol: str, window: int = 20) -> float:
    """Annualized realized volatility from log returns."""
    closes = fetch_adj_close(symbol, days=window + 1)
    log_returns = np.diff(np.log(closes))
    return np.std(log_returns) * np.sqrt(252)
```

### GARCH Models

Default: GJR-GARCH (captures leverage effect - down moves increase vol)

```python
from arch import arch_model

def fit_vol_model(symbol: str, model_type: str = 'gjr') -> dict:
    returns = get_returns(symbol)

    if model_type == 'garch':
        model = arch_model(returns, vol='Garch', p=1, q=1)
    elif model_type == 'gjr':
        model = arch_model(returns, vol='Garch', p=1, o=1, q=1)
    elif model_type == 'egarch':
        model = arch_model(returns, vol='EGARCH', p=1, q=1)

    result = model.fit(disp='off')
    forecast = result.forecast(horizon=5)

    return {
        'current': annualize(forecast.variance.values[-1, 0]),
        'forecast': [annualize(v) for v in forecast.variance.values[-1, :]],
        'params': dict(result.params),
    }
```

UI toggle: `[GARCH(1,1)] [GJR-GARCH ●] [EGARCH]`

## Packaging

### pyproject.toml

```toml
[project]
name = "iv-surface"
version = "0.1.0"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]",
    "yfinance",
    "duckdb",
    "plotly",
    "jinja2",
    "arch",           # GARCH models
    "numpy",
    "pyarrow",
]

[project.scripts]
iv-surface = "iv_surface:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Usage

```bash
# Development
cd apps/iv_surface
uv run iv-surface

# Build wheel (includes mango_iv.so from Bazel)
uv build

# Install and run
uv pip install dist/iv_surface-0.1.0-*.whl
iv-surface  # Starts on localhost:8000
```

### Bazel Integration

Bazel builds `mango_iv.so`, which gets bundled into the wheel. Future: systemd service file for production deployment.

## Cache Policy

- Price tables: stale after 24 hours (spot/rate drift)
- EOD quotes: refresh daily after market close
- Options snapshots: refresh on explicit user action or if > 1 hour stale
- Symbol resolution: follow rename/merge chains automatically

## Dependencies

- **FastAPI** - Web framework
- **uvicorn** - ASGI server
- **yfinance** - Market data
- **DuckDB** - Embedded analytics database
- **PyArrow** - Arrow IPC for price tables
- **Plotly** - Interactive charts
- **arch** - GARCH volatility models
- **Tailwind CSS** - Styling (CDN)
- **HTMX** - Dynamic updates (CDN)
- **mango_iv** - C++ IV solver (pybind11)

## Databento Migration Path

### Overview

Phase 1 uses yfinance for rapid prototyping. Phase 2 migrates to Databento for production-grade data with historical options support. Key architectural decision: **all data processing stays in C++**, Python is UI-only.

### Why Databento

| Feature | yfinance | Databento |
|---------|----------|-----------|
| Historical options chain | No | Yes (full tick history) |
| Data quality | Screen-scraped | Exchange-direct |
| Latency | ~1s | ~10ms |
| Price format | float64 | Fixed-point int64 (1e-9 precision) |
| C++ client | No | Yes (databento-cpp) |

### Databento Data Model

**Key schemas for options:**
- `definition` - Instrument definitions (strike, expiry, underlying_id)
- `mbp-1` - Top-of-book bid/ask (MBP = Market By Price)
- `trades` - Tick-by-tick trades
- `statistics` - Open interest, settlement prices
- `ohlcv-1d` - Daily OHLCV bars

**DBN format (Databento Binary Encoding):**
```cpp
// Fixed-point price: 1 unit = 1e-9
// Example: price = 3750500000000 → $3750.50
int64_t price;

// Timestamps: nanoseconds since Unix epoch
uint64_t ts_event;   // Event time
uint64_t ts_recv;    // Gateway receive time
```

### C++ Integration Architecture

```
src/
├── data/
│   ├── provider.hpp          # Abstract data provider interface
│   ├── yfinance_provider.cpp # Phase 1: Python bridge via pybind11
│   └── databento_provider.cpp # Phase 2: Native C++ client
├── simple/                   # Existing converters
│   └── sources/
│       └── databento.hpp     # Already implemented!
```

**Provider interface:**
```cpp
namespace mango::data {

struct OptionQuote {
    int64_t bid_px;      // Fixed-point (1e-9)
    int64_t ask_px;
    uint32_t bid_sz;
    uint32_t ask_sz;
    uint64_t ts_event;   // Nanoseconds
};

struct OptionDefinition {
    std::string symbol;
    int64_t strike_price;  // Fixed-point
    uint64_t expiration;   // Unix nanos
    char instrument_class; // 'C' or 'P'
    uint32_t underlying_id;
};

class DataProvider {
public:
    virtual ~DataProvider() = default;

    // Fetch option chain for symbol
    virtual std::expected<std::vector<OptionQuote>, DataError>
    fetch_options(std::string_view symbol,
                  std::chrono::system_clock::time_point as_of) = 0;

    // Fetch definitions
    virtual std::expected<std::vector<OptionDefinition>, DataError>
    fetch_definitions(std::string_view parent_symbol) = 0;

    // Fetch historical EOD quotes
    virtual std::expected<std::vector<OHLCVBar>, DataError>
    fetch_eod(std::string_view symbol,
              std::chrono::year_month_day start,
              std::chrono::year_month_day end) = 0;
};

// Phase 2 implementation
class DatabentоProvider : public DataProvider {
    databento::Historical client_;
public:
    explicit DatabentoProvider(std::string_view api_key);
    // ... implementations using databento-cpp
};

}
```

### Databento-cpp Integration

**Bazel dependency (MODULE.bazel):**
```starlark
bazel_dep(name = "databento-cpp", version = "0.20.0")
```

**Fetching options chain:**
```cpp
#include <databento/historical.hpp>
#include <databento/symbol_map.hpp>

auto client = databento::HistoricalBuilder{}
    .SetKeyFromEnv()
    .Build();

// Fetch option definitions for AAPL
databento::TsSymbolMap symbol_map;
client.TimeseriesGetRange(
    "OPRA.PILLAR",                          // Dataset
    {"2024-11-26T09:30", "2024-11-26T16:00"}, // Time range
    {"AAPL.OPT"},                            // Parent symbol
    databento::Schema::Definition,
    databento::SType::Parent,
    databento::SType::InstrumentId,
    {},
    [&](const databento::Metadata& m) { symbol_map = m.CreateSymbolMap(); },
    [](const databento::Record& r) {
        const auto& def = r.Get<databento::InstrumentDefMsg>();
        // Process definition...
        return databento::KeepGoing::Continue;
    }
);
```

### Storage: DuckDB + Arrow (Native C++)

DuckDB has a C++ API. All storage operations move to C++:

```cpp
#include <duckdb.hpp>

duckdb::DuckDB db("iv_surface.duckdb");
duckdb::Connection conn(db);

// Store options snapshot
conn.Query(R"(
    INSERT INTO options_snapshots
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
)", symbol, ts, is_eod, expiry, strike, ...);

// Query with Arrow result
auto result = conn.Query("SELECT * FROM eod_quotes WHERE symbol = ?", symbol);
auto arrow_array = result->Fetch();  // Returns Arrow RecordBatch
```

### Migration Phases

**Phase 1: yfinance (Current Design)**
- Python fetches data via yfinance
- Passes to C++ via pybind11 for IV computation
- Results returned to Python for display

**Phase 2: Databento C++ Client**
- C++ fetches data directly via databento-cpp
- C++ stores to DuckDB
- C++ computes IVs
- Python only handles HTTP/WebSocket and rendering

**Phase 3: Full C++ Backend (Optional)**
- Replace FastAPI with C++ HTTP server (e.g., Drogon, Crow)
- Python completely eliminated
- Single binary deployment

### Data Format Alignment

The `mango::simple::DatabentоSource` converter already handles:
- Fixed-point int64 → Price (deferred conversion)
- Nanosecond timestamps → Timestamp
- Instrument class → OptionType

```cpp
// Already in src/simple/sources/databento.hpp
namespace mango::simple {

template<>
struct Converter<DatabentoSource> {
    static Price to_price(int64_t fixed_point) {
        return Price::from_fixed_point(fixed_point);
    }

    static Timestamp to_timestamp(uint64_t nanos) {
        return Timestamp::from_nanos(nanos);
    }
};

}
```

### API Key Management

```cpp
// Environment variable (preferred)
auto client = databento::HistoricalBuilder{}
    .SetKeyFromEnv()  // Reads DATABENTO_API_KEY
    .Build();

// Or config file
// ~/.config/iv_surface/config.toml
// [databento]
// api_key = "db-..."
```

### Cost Considerations

Databento pricing is per-record. Strategies to minimize cost:
1. Cache aggressively (DuckDB stores all fetched data)
2. Use `definition` schema for chain structure (cheap)
3. Fetch `mbp-1` only for strikes near ATM
4. Batch historical requests by date range

## Future Enhancements

- Systemd service management
- Multiple vol model comparison overlay
- Historical IV surface playback (from accumulated snapshots)
- Implied vs realized spread alerts
- Term structure analysis
- **Databento live streaming** (WebSocket for real-time surface updates)
- **Full C++ backend** (eliminate Python entirely)

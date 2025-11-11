# Option IV Surface Calculator Scripts

Python scripts for downloading option data from Yahoo Finance and calculating implied volatility surfaces using the mango-iv C++ library.

## Overview

This collection of scripts provides an end-to-end workflow for:

1. **Downloading** option chain data from Yahoo Finance with yfinance
2. **Calculating** implied volatility using high-performance C++ FDM solver
3. **Storing** results and raw data in SQLite3 database with standard security identifiers

## Files

### Scripts
- **`download_iv_surface.py`** - Standalone entry point script

### iv_surface/ Package
- **`calculate_iv_surface.py`** - Main orchestration module
- **`data_downloader.py`** - Yahoo Finance data download module
- **`iv_calculator.py`** - IV calculation using C++ bindings
- **`database.py`** - SQLite3 database schema and utilities
- **`__init__.py`** - Package initialization

### Configuration
- **`pyproject.toml`** - Python project configuration (uv/pip)
- **`requirements.txt`** - Python dependencies (pip-compatible)
- **`.python-version`** - Python version specification for uv

### Examples
- **`examples/simple_iv_calculation.py`** - Download and calculate IV for a single ticker
- **`examples/README.md`** - Example documentation and usage guide

## Setup

### 1. Build the C++ Python Bindings

First, build the mango-iv Python module:

```bash
# From repository root
bazel build //python:mango_iv
```

This creates a Python extension module that wraps the C++ IV solver.

### 2. Install Python Dependencies

**Recommended: Use uv (fast, modern package manager)**

```bash
cd scripts

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows
```

**Alternative: Use pip**

```bash
cd scripts
pip install -r requirements.txt
```

### 3. Set Python Path

Add the Bazel build output to your Python path:

```bash
# Find the built module
export BAZEL_BIN=$(bazel info bazel-bin)

# Add to PYTHONPATH (adjust path based on your platform)
export PYTHONPATH="${BAZEL_BIN}/python:${PYTHONPATH}"
```

Or install the module in development mode (recommended):

```bash
# Create a symlink in your Python environment
ln -s ${BAZEL_BIN}/python/mango_iv.so $(python -c "import site; print(site.getsitepackages()[0])")/
```

## Quick Start

### Example: Calculate IV for a single ticker

For a quick demonstration, see the example script:

```bash
# From scripts/ directory
python examples/simple_iv_calculation.py AAPL
```

This will download option data and calculate IV for the nearest expiration, displaying results for ATM options.

**See [examples/README.md](examples/README.md) for more examples and detailed usage.**

## Usage

### Basic Usage

Calculate IV surface for a single ticker:

```bash
# From scripts/ directory
python download_iv_surface.py AAPL

# Or as a module
python -m iv_surface.calculate_iv_surface AAPL
```

This will:
- Download all available option expirations for AAPL
- Calculate IV for each option using the C++ FDM solver
- Store results in `options.db` (SQLite3)

### Advanced Options

```bash
# Specify database path
python download_iv_surface.py SPY --db-path spy_options.db

# Limit number of expirations
python download_iv_surface.py TSLA --max-expirations 6

# Custom risk-free rate
python download_iv_surface.py AAPL --risk-free-rate 0.045

# Higher resolution grid (slower but more accurate)
python download_iv_surface.py AAPL --grid-n-space 201 --grid-n-time 2000

# Filter low-volume options
python download_iv_surface.py AAPL --min-volume 50

# Quiet mode (suppress progress messages)
python download_iv_surface.py AAPL --quiet
```

### Help

```bash
python download_iv_surface.py --help
```

## Database Schema

The SQLite3 database stores comprehensive option data with standard identifiers:

### Securities Table
- `ticker` - Stock ticker symbol (primary identifier)
- `name` - Company name
- `isin` - International Securities Identification Number
- `cusip` - CUSIP identifier
- `exchange` - Trading exchange
- `sector` - Business sector

### Option Contracts Table
- `contract_symbol` - Unique option contract identifier
- `option_type` - CALL or PUT
- `strike` - Strike price
- `expiration` - Expiration date
- `exercise_style` - AMERICAN or EUROPEAN

### IV Calculations Table
- `implied_volatility` - Calculated IV
- `converged` - Solver convergence status
- `iterations` - Number of iterations
- `final_error` - Pricing error
- `vega` - Option vega (if available)
- `solver_config` - JSON with grid parameters

### Accessing Data

```python
from iv_surface import OptionDatabase

# Open database
with OptionDatabase("options.db") as db:
    # Get security info
    security = db.get_security("AAPL")
    print(security)

    # Get IV surface
    iv_surface = db.get_iv_surface(security['security_id'])
    for row in iv_surface:
        print(f"Strike: {row['strike']}, IV: {row['implied_volatility']}")
```

## Performance

### Typical Performance (101×1000 grid)

- **Per option**: ~143ms (FDM-based IV calculation)
- **Batch processing**: Parallelized with OpenMP
- **Full chain** (100 options): ~2-3 minutes on modern CPU

### Tuning Performance

**Faster (lower accuracy):**
```bash
python download_iv_surface.py AAPL --grid-n-space 51 --grid-n-time 500
```

**Slower (higher accuracy):**
```bash
python download_iv_surface.py AAPL --grid-n-space 201 --grid-n-time 2000
```

## Security Identification

The scripts use multiple standard identifiers to ensure correct security matching:

1. **Ticker Symbol** - Primary identifier (e.g., "AAPL")
2. **ISIN** - International standard (e.g., "US0378331005")
3. **CUSIP** - US/Canada identifier (e.g., "037833100")
4. **Exchange** - Trading venue (e.g., "NMS", "NYQ")

Yahoo Finance provides ISIN and CUSIP when available. The database schema supports all standard identifiers for robust security matching.

## Example Workflow

### Download and Calculate

```bash
# Download SPY options and calculate IV
python download_iv_surface.py SPY --max-expirations 4 --db-path spy.db
```

### Query Results

```python
from iv_surface import OptionDatabase
import pandas as pd

with OptionDatabase("spy.db") as db:
    # Get security
    spy = db.get_security("SPY")
    print(f"Security: {spy['name']} ({spy['ticker']})")
    print(f"ISIN: {spy['isin']}")

    # Get IV surface
    surface = db.get_iv_surface(spy['security_id'], limit=1000)
    df = pd.DataFrame(surface)

    # Analyze
    print(f"\nIV Statistics:")
    print(df['implied_volatility'].describe())

    # Plot (requires matplotlib)
    import matplotlib.pyplot as plt

    # Separate calls and puts
    calls = df[df['option_type'] == 'CALL']
    puts = df[df['option_type'] == 'PUT']

    # Get nearest expiration
    nearest_exp = df['expiration'].min()
    calls_near = calls[calls['expiration'] == nearest_exp]
    puts_near = puts[puts['expiration'] == nearest_exp]

    # Plot IV smile
    plt.figure(figsize=(10, 6))
    plt.plot(calls_near['strike'], calls_near['implied_volatility'], 'b-', label='Calls')
    plt.plot(puts_near['strike'], puts_near['implied_volatility'], 'r-', label='Puts')
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.title(f'IV Smile - {spy["ticker"]} ({nearest_exp})')
    plt.legend()
    plt.grid(True)
    plt.show()
```

## Troubleshooting

### Import Error: mango_iv not found

Make sure you've built the Python module and set PYTHONPATH:

```bash
bazel build //python:mango_iv
export PYTHONPATH="$(bazel info bazel-bin)/python:${PYTHONPATH}"
```

### No options available for ticker

Some tickers may not have listed options. Verify on Yahoo Finance website first.

### IV calculation failed

Common reasons:
- Option price below intrinsic value (arbitrage violation)
- Extremely illiquid options with stale prices
- Very short time to expiration (<1 day)

Check the `failure_reason` field in the database for diagnostics.

## Related Documentation

- [CLAUDE.md](../CLAUDE.md) - Main project documentation
- [Python Bindings](../python/README.md) - C++ bindings documentation
- [IV Solver](../docs/plans/IV_IMPLEMENTATION_SUMMARY.md) - IV implementation details

## License

See [LICENSE](../LICENSE) for details.

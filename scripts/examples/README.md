# IV Surface Calculator Examples

This directory contains example scripts demonstrating how to use the mango-iv option data downloader and IV calculator.

## Prerequisites

1. **Build the C++ bindings:**
   ```bash
   # From repository root
   bazel build //python:mango_iv.so
   ```

2. **Install Python dependencies:**
   ```bash
   cd scripts
   uv sync  # or: pip install -r requirements.txt
   source .venv/bin/activate
   ```

3. **Set PYTHONPATH:**
   ```bash
   export BAZEL_BIN=$(bazel info bazel-bin)
   export PYTHONPATH="${BAZEL_BIN}/python:${PYTHONPATH}"
   ```

## Examples

### 1. Simple IV Calculation (`simple_iv_calculation.py`)

Download option data for a ticker and calculate implied volatility for a single expiration.

**Basic usage:**
```bash
python examples/simple_iv_calculation.py AAPL
```

**With options:**
```bash
# Specific expiration date
python examples/simple_iv_calculation.py SPY --expiration 2025-12-19

# Higher volume filter (more liquid options only)
python examples/simple_iv_calculation.py AAPL --min-volume 50

# Higher resolution grid (slower but more accurate)
python examples/simple_iv_calculation.py TSLA --grid-n-space 201 --grid-n-time 2000

# Custom risk-free rate
python examples/simple_iv_calculation.py MSFT --risk-free-rate 0.045
```

**Output:**
```
======================================================================
Implied Volatility Calculator - AAPL
======================================================================

Step 1: Downloading option data from Yahoo Finance...
  ✓ Security: Apple Inc. (AAPL)
  ✓ Current price: $150.23
  ✓ Exchange: NMS
  ✓ Using expiration: 2025-12-19
  ✓ Found 45 calls and 43 puts

Step 2: Calculating implied volatility...
  Grid: 101 space × 1000 time
  Time to maturity: 0.247 years (90 days)

  Processing 45 call options...
    ✓ Converged: 42/45 (93.3%)
  Processing 43 put options...
    ✓ Converged: 40/43 (93.0%)

======================================================================
Results Summary
======================================================================

Total options: 88
Successful IV calculations: 82 (93.2%)

──────────────────────────────────────────────────────────────────────
Sample Results (near ATM)
──────────────────────────────────────────────────────────────────────

📈 ATM Call Option:
  Strike: $150.00
  Market Price: $8.2500
  Implied Volatility: 24.35%
  Vega: 0.3421
  Iterations: 12
  Final Error: $0.000023

  Other Strikes:
    K=$150.00  IV=24.35%  Price=$8.2500  ATM
    K=$145.00  IV=23.89%  Price=$11.450  ITM
    K=$155.00  IV=25.12%  Price=$5.7500  OTM
    K=$140.00  IV=23.45%  Price=$14.750  ITM
    K=$160.00  IV=26.01%  Price=$3.8750  OTM

📉 ATM Put Option:
  Strike: $150.00
  Market Price: $7.9500
  Implied Volatility: 24.28%
  Vega: 0.3398
  Iterations: 11
  Final Error: $0.000019

  Other Strikes:
    K=$150.00  IV=24.28%  Price=$7.9500  ATM
    K=$155.00  IV=25.34%  Price=$11.250  ITM
    K=$145.00  IV=23.76%  Price=$5.4500  OTM
    K=$160.00  IV=26.45%  Price=$14.500  ITM
    K=$140.00  IV=22.89%  Price=$3.5500  OTM

──────────────────────────────────────────────────────────────────────
Performance
──────────────────────────────────────────────────────────────────────
  Average iterations (calls): 11.8
  Average iterations (puts): 11.3
  Estimated time per option: ~143ms (101×1000 grid)
  Total estimated time: ~11.7s

======================================================================
```

**What it does:**
1. Downloads option chain from Yahoo Finance
2. Filters options by volume (default: min 10 contracts)
3. Calculates implied volatility using American option PDE solver
4. Shows results for ATM options and nearby strikes
5. Displays convergence statistics and performance metrics

**Use cases:**
- Quick IV calculation for specific ticker and expiration
- Testing the solver on real market data
- Understanding IV surface characteristics
- Comparing calculated IV with exchange-reported IV
- Educational purposes (learning how IV calculation works)

## Tips

### Choosing Grid Resolution

| Grid Size | Accuracy | Speed | Use Case |
|-----------|----------|-------|----------|
| 51×500 | Low | Fast (~35ms) | Quick estimates |
| 101×1000 | Good | Medium (~143ms) | Default, balanced |
| 201×2000 | High | Slow (~570ms) | High-accuracy needs |

### Volume Filtering

- `--min-volume 10`: Default, includes most options
- `--min-volume 50`: More liquid options only
- `--min-volume 100`: Very liquid (major indices, popular stocks)

### Common Issues

**Import Error: mango_iv not found**
```bash
# Make sure you built the C++ module and set PYTHONPATH
bazel build //python:mango_iv.so
export PYTHONPATH="$(bazel info bazel-bin)/python:${PYTHONPATH}"
```

**No options available**
- Check if the ticker has listed options
- Try a different ticker (SPY, AAPL, TSLA usually work)
- Verify market is open or use after-hours data

**Many failures to converge**
- Increase grid resolution: `--grid-n-space 201 --grid-n-time 2000`
- Some deep OTM options may have stale prices
- Options with bid=0 cannot be priced

## See Also

- [Main README](../README.md) - Full documentation
- [Database API](../iv_surface/database.py) - For storing results
- [IV Calculator](../iv_surface/iv_calculator.py) - Batch processing

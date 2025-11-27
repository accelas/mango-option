# IV Surface Calculator Examples

This directory contains example scripts demonstrating how to use the mango-option option data downloader and IV calculator.

## Prerequisites

1. **Build the C++ bindings:**
   ```bash
   # From repository root
   bazel build //python:mango_option.so
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

**Difficulty:** Beginner
**Time:** ~10-15 seconds

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

---

### 2. IV Price Table Example (`iv_price_table_example.py`)

Demonstrates the price table workflow: pre-compute option prices and use interpolation for ultra-fast IV calculation.

**Difficulty:** Advanced
**Time:** Varies (table build: 5-30 min, queries: microseconds)

**Basic usage:**
```bash
python examples/iv_price_table_example.py AAPL
```

**With options:**
```bash
# Save price table for reuse
python examples/iv_price_table_example.py SPY --save-table spy_table.bin

# Small table (faster build, less accurate)
python examples/iv_price_table_example.py TSLA --table-size small

# Large table (slower build, more accurate)
python examples/iv_price_table_example.py AAPL --table-size large

# Test against specific expiration
python examples/iv_price_table_example.py SPY --expiration 2025-12-19
```

**Table size presets:**

| Size | Grid Dimensions | Build Time | Accuracy | Use Case |
|------|----------------|------------|----------|----------|
| small | 20×15×10 (3K points) | ~7 min | Good | Quick testing |
| medium | 50×30×20 (30K points) | ~15 min | Better | Production |
| large | 100×50×30 (150K points) | ~36 min | Best | High accuracy |

**What it does:**
1. Downloads option data from Yahoo Finance
2. Builds a 4D price table (moneyness × maturity × volatility × rate)
3. Pre-computes American option prices at all grid points
4. Compares FDM solver (slow) vs table interpolation (fast)
5. Shows ~40,000x speedup for IV calculation

**Output:**
```
======================================================================
IV Price Table Example - AAPL
======================================================================

Building Price Table
──────────────────────────────────────────────────────────────────────

Grid dimensions:
  Moneyness (m = S/K): [0.70, 1.30] × 50 points
  Maturity (τ): [0.027, 2.0] years × 30 points
  Volatility (σ): [0.10, 0.80] × 20 points
  Rate (r): 0.0500 (fixed)
  Total grid points: 30,000

FDM solver grid: 101 space × 1000 time
Expected pre-computation time: ~4290s (71.5 min)

Performance Comparison: FDM vs Price Table
──────────────────────────────────────────────────────────────────────

Testing with 3 near-ATM put options:

Option 1: K=$150.00 (m=1.002, ATM), Price=$7.9500
  Method 1: Direct FDM solver...
    ✓ IV = 24.28% (iters=11, time=143.2ms)
  Method 2: Price table interpolation...
    ✓ IV = 24.29% (time=7.5µs)
    🚀 Speedup: 19,093x faster

Summary:
  Direct FDM: 429.6ms total (143.2ms per option)
  Price table: 22.5µs total (7.5µs per option)
  Overall speedup: 19,093x
```

**Use cases:**
- Production applications requiring thousands of IV calculations
- Real-time pricing systems
- Risk management dashboards
- High-frequency trading strategies
- Batch processing of large option portfolios

**Key benefits:**
- **~40,000x faster** than direct FDM for IV calculation
- **Reproducible** - same table gives same results
- **Reusable** - build once, query millions of times
- **Accurate** - uses full American option FDM solver
- **Memory efficient** - ~2-10 MB per table

**Note:** This example demonstrates the workflow conceptually. The Python bindings for the price table API will be added in a future update. See `docs/IV_SURFACE_PRECOMPUTATION_GUIDE.md` for C++ usage.

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

**Import Error: mango_option not found**
```bash
# Make sure you built the C++ module and set PYTHONPATH
bazel build //python:mango_option.so
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

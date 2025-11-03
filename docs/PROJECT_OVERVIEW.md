# Project Overview: mango-iv

## The Problem: Option Pricing and Implied Volatility

### What is an Option?

An **option** is a financial derivative that gives the holder the right (but not obligation) to buy or sell an underlying asset at a specified price (strike) on or before a specified date (expiration).

**Types:**
- **Call option**: Right to buy the asset
- **Put option**: Right to sell the asset

**Exercise styles:**
- **European**: Can only be exercised at expiration
- **American**: Can be exercised any time before expiration

### The Pricing Problem

**Given:** Option parameters (spot price, strike, time to maturity, interest rate, volatility, dividends)
**Need:** Fair market price of the option

**Why it's hard:**
1. **European options** have closed-form solutions (Black-Scholes formula) - fast but limited
2. **American options** can be exercised early, requiring numerical PDE solvers - slow but realistic
3. **Path-dependent options** (barriers, Asians, etc.) need simulation or PDEs

**mango-iv focuses on American options** - the more challenging and practically relevant case for equity options.

### The Inverse Problem: Implied Volatility

**Given:** Market price of an option
**Need:** The volatility that would produce that price (implied volatility)

**Why it matters:**
- **Trading**: Compare options across strikes/maturities
- **Risk management**: Volatility is the key risk factor
- **Market sentiment**: IV surfaces reveal market expectations

**Why it's hard:**
- Black-Scholes formula is not analytically invertible
- Requires root-finding (Brent's method, Newton's method)
- **American options need PDE solver in each iteration** (~145ms per IV calculation, very expensive!)
- Each Brent iteration solves a full PDE (~21ms) to price the option with guessed volatility

### Real-World Use Cases

1. **Options Trading**
   - Price thousands of options per second
   - Calculate Greeks for hedging (delta, gamma, vega, theta)
   - Build volatility surfaces from market data

2. **Risk Management**
   - Portfolio VaR (Value at Risk) calculation
   - Stress testing under different volatility scenarios
   - Margin calculation for options positions

3. **Market Making**
   - Quote bid/ask spreads in real-time
   - Hedge delta exposure continuously
   - Manage inventory risk

4. **Research & Development**
   - Test new pricing models
   - Calibrate volatility models (SABR, SVI, etc.)
   - Backtest trading strategies

### Performance Requirements

**Trading systems need:**
- **Latency**: <1µs per query (sub-microsecond)
- **Throughput**: >100,000 prices per second
- **Accuracy**: <0.5% relative error
- **Stability**: No crashes, memory leaks, or numerical issues

**Research systems need:**
- **Flexibility**: Easy to test new models
- **Correctness**: Mathematically rigorous implementations
- **Transparency**: Clear code, comprehensive tests
- **Reproducibility**: Validated against industry benchmarks

---

## The Solution: mango-iv

**mango-iv** is a research-grade numerical library for option pricing and implied volatility calculation, designed to balance **performance**, **flexibility**, and **correctness**.

### What mango-iv Provides

#### 1. **Let's Be Rational (European IV Estimation)**

```c
LBRResult result = lbr_implied_volatility(spot, strike, maturity, rate, market_price, is_call);
```

- **Purpose**: Fast European IV estimation for American IV upper bounds
- **Performance**: ~781ns per calculation (20-30 bisection iterations)
- **Accuracy**: Sufficient for bound calculation
- **Use case**: Provides tight bracketing interval for American IV search

#### 2. **American Implied Volatility Calculation**

```c
IVResult result = calculate_iv_simple(&params);
// result.implied_vol, result.iterations
```

- **FDM-based**: Each Brent iteration solves full PDE (~21ms)
- **Performance**: ~145ms per calculation (5-8 Brent iterations)
- **Convergence**: Robust with Let's Be Rational bounds
- **Validation**: 9 American IV test cases + 4 Let's Be Rational test cases

#### 3. **American Option Pricing (PDE Solver)**

```c
double price = american_option_price(
    S, K, T, r, sigma, dividend, is_call,
    n_space_points, n_time_steps
);
```

- **TR-BDF2 method**: L-stable implicit time-stepping
- **Log-space transformation**: Constant PDE coefficients
- **Obstacle conditions**: Enforces early exercise constraint
- **Discrete dividends**: Handled via temporal event system
- **Performance**: 21.7ms per option (with AVX-512)
- **Validation**: Tested against QuantLib (0.5% relative error)

#### 4. **General PDE Solver (Research Tool)**

```c
PDESolver *solver = pde_solver_create(&grid, &time, &bc, &trbdf2, &callbacks);
pde_solver_initialize(solver);
pde_solver_solve(solver);
```

- **Callback-based**: User defines initial/boundary conditions, spatial operators
- **Vectorized**: OpenMP SIMD pragmas for automatic vectorization
- **Flexible**: Solve any parabolic PDE (heat equation, diffusion, etc.)
- **Memory-efficient**: Single contiguous workspace (12n doubles)
- **Use case**: Research, custom derivatives, exotic options

### Key Design Principles

1. **Modularity**
   - Clear separation: IV calculation ← American pricing ← PDE solver
   - Each component usable independently
   - Easy to test, debug, and extend

2. **Performance-Conscious**
   - Vectorized callbacks (entire arrays, not point-by-point)
   - SIMD-friendly memory layout (64-byte alignment)
   - Single-allocation workspace (no malloc in hot loops)
   - Batch APIs for parallel processing (OpenMP)

3. **Correctness First**
   - Well-established methods (Black-Scholes, TR-BDF2, Brent's)
   - Comprehensive test suite (80+ test cases)
   - Validated against QuantLib benchmarks
   - Clear documentation of limitations

4. **Zero-Overhead Diagnostics**
   - USDT (User Statically-Defined Tracing) probes
   - No printf/fprintf in library code
   - Dynamic tracing with bpftrace (sub-nanosecond overhead)
   - Production-safe instrumentation

5. **Research-Friendly**
   - C23 with modern features (nullptr, designated initializers)
   - Clear, readable code (not over-optimized)
   - Extensive documentation (CLAUDE.md, TRACING.md, etc.)
   - Example programs for common patterns

---

## Performance Comparison

### Current Performance (FDM-based)

| Operation | Time | Notes |
|-----------|------|-------|
| Let's Be Rational (European IV) | ~781ns | Fast bound estimation, 20-30 iterations |
| American option (single) | 21.7ms | TR-BDF2, 141 points × 1000 steps |
| American IV (single) | ~145ms | 5-8 Brent iterations × 21.7ms per FDM |
| American option (batch 64) | ~1.5ms wall | OpenMP parallelization |
| vs QuantLib | 2.1x slower | Reasonable for research code |

### Achieved Performance (Interpolation-based)

| Operation | FDM-based | Table-based | Speedup |
|-----------|-----------|-------------|---------|
| American option price | 21.7ms | ~500ns | **43,400× achieved** |
| American IV calculation | ~145ms | ~11.8ms | **22.5× achieved** |
| Greeks (vega, gamma) | ~65ms | ~500ns | **130,000× achieved** |
| Batch (1000 options) | 21.7s | ~0.5s | **43× achieved** |

**Approach:** Pre-compute option prices during downtime, use multi-dimensional cubic spline interpolation for real-time queries. Greeks computed via finite differences during table generation.

---

## Project Structure

```
mango-iv/
├── src/                           # Core library
│   ├── implied_volatility.{h,c}   # American IV calculation (FDM + Brent)
│   ├── lets_be_rational.{h,c}     # European IV estimation (bounds)
│   ├── american_option.{h,c}      # American option pricing
│   ├── pde_solver.{h,c}           # General PDE solver (FDM)
│   ├── cubic_spline.{h,c}         # Interpolation
│   ├── brent.h                    # Root-finding
│   └── mango_trace.h             # USDT tracing probes
│
├── examples/                      # Demonstration programs
│   ├── example_implied_volatility.c
│   ├── example_american_option.c
│   └── example_heat_equation.c
│
├── tests/                         # Comprehensive test suite
│   ├── implied_volatility_test.cc # 9 American IV test cases
│   ├── lets_be_rational_test.cc   # 4 European IV test cases
│   ├── american_option_test.cc    # 42 test cases
│   └── pde_solver_test.cc         # Core solver tests
│
├── benchmarks/                    # Performance benchmarks (not run in CI)
│   ├── batch_benchmark.cc         # Batch processing benchmarks
│   └── quantlib_benchmark.cc      # QuantLib comparison
│
├── docs/                          # Design documentation
│   ├── PROJECT_OVERVIEW.md        # This file (problem & solution)
│   ├── ARCHITECTURE.md            # Detailed technical architecture
│   └── QUICK_REFERENCE.md         # Developer quick-start
│
├── scripts/                       # Utilities
│   ├── mango-trace               # USDT tracing helper
│   └── tracing/                   # bpftrace scripts
│
├── CLAUDE.md                      # Instructions for Claude Code
├── TRACING.md                     # USDT tracing guide
├── TRACING_QUICKSTART.md          # 5-minute tracing tutorial
└── MODULE.bazel                   # Build configuration
```

---

## Who Should Use mango-iv?

### ✅ Good Fit

- **Researchers** building new pricing models
- **Students** learning computational finance
- **Developers** needing flexible option pricing
- **Quants** prototyping trading strategies
- **Engineers** requiring transparent, well-tested code

### ⚠️ Consider Alternatives

- **High-frequency trading**: Use optimized libraries (QuantLib, proprietary)
- **Production systems**: Consider mature, battle-tested solutions
- **GPU acceleration needed**: mango-iv is CPU-only (for now)
- **Exotic options**: Limited support (barriers, Asians not yet implemented)

---

## Getting Started

### Build and Run

```bash
# Build everything
bazel build //...

# Run tests
bazel test //...

# Run example
bazel run //examples:example_implied_volatility
```

### Quick Example: Calculate American Implied Volatility

```c
#include "src/implied_volatility.h"

IVParams params = {
    .spot_price = 100.0,
    .strike = 100.0,
    .time_to_maturity = 1.0,
    .risk_free_rate = 0.05,
    .market_price = 6.08,  // American put market price
    .is_call = false
};

// Simple API: uses default grid and Let's Be Rational for bounds
IVResult result = calculate_iv_simple(&params);

if (result.converged) {
    printf("American IV: %.4f (%.1f%%)\n",
           result.implied_vol, result.implied_vol * 100);
    printf("Iterations: %d\n", result.iterations);
} else {
    printf("Failed: %s\n", result.error);
}
```

### Quick Example: Price American Option

```c
#include "src/american_option.h"

double price = american_option_price(
    100.0,  // spot
    100.0,  // strike
    1.0,    // maturity
    0.05,   // rate
    0.25,   // volatility
    0.0,    // dividend yield
    false,  // is_call (false = put)
    141,    // spatial points
    1000    // time steps
);

printf("American put price: %.4f\n", price);
```

---

## Next Steps

1. **Read the architecture**: `docs/ARCHITECTURE.md`
2. **Quick reference**: `docs/QUICK_REFERENCE.md`
3. **Try examples**: `examples/example_*.c`
4. **Run benchmarks**: `bazel run //benchmarks:quantlib_benchmark`
5. **Learn tracing**: `TRACING_QUICKSTART.md`

---

## Comparison to Other Libraries

| Feature | mango-iv | QuantLib | PyQL | Bloomberg API |
|---------|---------|----------|------|---------------|
| **Language** | C23 | C++17 | Python | C++/Python |
| **License** | Open-source | BSD | BSD | Proprietary |
| **American options** | ✅ PDE (TR-BDF2) | ✅ Multiple methods | ✅ Via QuantLib | ✅ |
| **Implied volatility** | ✅ Brent's | ✅ Newton/Brent | ✅ | ✅ |
| **Custom PDEs** | ✅ Callback-based | ❌ | ❌ | ❌ |
| **USDT tracing** | ✅ Zero-overhead | ❌ | ❌ | ❌ |
| **Batch API** | ✅ OpenMP | ❌ | ❌ | ❌ |
| **Documentation** | ✅ Extensive | ⚠️ Patchy | ⚠️ Limited | ✅ Commercial |
| **Performance** | 2.1x slower | Baseline | Slow (Python) | Fast (optimized) |
| **Exotic options** | ❌ Limited | ✅ Extensive | ✅ Via QuantLib | ✅ Extensive |

**mango-iv's niche:** Research-grade flexibility with production-conscious design, optimized for learning and prototyping rather than maximum performance.

---

## Roadmap

### Current State (v0.1)
- ✅ American option pricing (PDE-based, TR-BDF2)
- ✅ American option implied volatility (FDM + Brent's method)
- ✅ Let's Be Rational (European IV for bound estimation)
- ✅ USDT tracing system
- ✅ Comprehensive test suite
- ✅ QuantLib benchmarks
- ✅ Cubic spline interpolation (C² continuous, accurate Greeks)
- ✅ Coordinate transformation support (log-sqrt, log-variance)

### Near-Term (v0.2-0.3)
- ✅ Interpolation-based pricing engine (43,400× speedup achieved)
- ✅ Table-based IV calculation (22.5× speedup achieved)
- ✅ CPU optimizations (AVX-512, FMA, restrict)
- ✅ Greeks calculation (vega, gamma via precomputed derivatives)
- ✅ Adaptive grid refinement (<1bp IV error for 95% of points)
- ✅ Unified grid architecture (20,000× memcpy reduction)
- 🚧 Volatility surface calibration

### Future (v1.0+)
- ⭐ GPU acceleration (CUDA/OpenCL)
- ⭐ Exotic options (barriers, Asians, lookbacks)
- ⭐ Monte Carlo simulation
- ⭐ Stochastic volatility models (Heston, SABR)
- ⭐ Python bindings

---

## Contributing

This project welcomes contributions! Areas of interest:

1. **Performance**: Implement optimizations (SIMD, cache-blocking, algorithm improvements)
2. **Features**: Add exotic options, new PDE schemes
3. **Testing**: More test cases, edge cases, stress tests
4. **Documentation**: Tutorials, explanations, examples
5. **Benchmarks**: Compare against other libraries

See the main repository for contribution guidelines.

---

## License

[Add your license here]

---

## References

### Books
- **Hull, J.** (2021). *Options, Futures, and Other Derivatives* (10th ed.)
- **Wilmott, P.** (2006). *Paul Wilmott on Quantitative Finance*
- **Gatheral, J.** (2006). *The Volatility Surface: A Practitioner's Guide*

### Papers
- **Black, F., & Scholes, M.** (1973). *The Pricing of Options and Corporate Liabilities*
- **Merton, R. C.** (1973). *Theory of Rational Option Pricing*
- **Brent, R. P.** (1973). *Algorithms for Minimization without Derivatives*
- **Ascher, U. M., Ruuth, S. J., & Wetton, B. T. R.** (1995). *Implicit-Explicit Methods for Time-Dependent PDEs*

### Software
- **QuantLib**: https://www.quantlib.org/ (C++ library for quantitative finance)
- **FastVol**: https://github.com/vgalanti/fastvol (High-performance option pricing)
- **PyQL**: https://github.com/enthought/pyql (Python bindings for QuantLib)

---

**For detailed technical architecture, see:** `docs/ARCHITECTURE.md`

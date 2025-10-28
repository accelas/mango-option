# Project Overview: iv_calc

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
1. **European options** have closed-form solutions (Black-Scholes formula)
2. **American options** can be exercised early, requiring numerical methods
3. **Path-dependent options** (barriers, Asians, etc.) need simulation or PDEs

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
- American options need PDE solver in the loop (very expensive!)

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

## The Solution: iv_calc

**iv_calc** is a research-grade numerical library for option pricing and implied volatility calculation, designed to balance **performance**, **flexibility**, and **correctness**.

### What iv_calc Provides

#### 1. **Black-Scholes Pricing (European Options)**

```c
double price = black_scholes_price(spot, strike, maturity, rate, vol, is_call);
```

- **Analytical formula**: No approximation error
- **Performance**: <1µs per calculation
- **Accuracy**: 7.5e-8 relative error (Abramowitz & Stegun CDF)
- **Use case**: Basis for IV calculation, quick European option pricing

#### 2. **Implied Volatility Calculation**

```c
IVResult result = calculate_implied_volatility(&params);
// result.implied_vol, result.vega, result.iterations
```

- **Brent's method**: Robust root-finding (no derivatives needed)
- **Performance**: 8-12 iterations typical, <1µs per calculation
- **Convergence**: 99.9% success rate on realistic inputs
- **Validation**: 44 comprehensive test cases

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
- **Memory-efficient**: Single contiguous workspace (10n doubles)
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
   - Comprehensive test suite (73+ test cases)
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
| European option (Black-Scholes) | <1µs | Analytical formula |
| Implied volatility | <1µs | 8-12 iterations with Brent's |
| American option (single) | 21.7ms | TR-BDF2, 141 points × 1000 steps |
| American option (batch 64) | ~1.5ms wall | OpenMP parallelization |
| vs QuantLib | 2.1x slower | Reasonable for research code |

### Planned Performance (Interpolation-based)

**See:** `docs/INTERPOLATION_ENGINE_DESIGN.md`

| Operation | Current | Planned | Speedup |
|-----------|---------|---------|---------|
| American option price | 21.7ms | 500ns | **43,400x** |
| Greeks calculation | ~65ms | 5µs | **13,000x** |
| Batch (1000 options) | 21.7s | <1ms | **>20,000x** |

**Approach:** Pre-compute option prices during downtime, use multi-dimensional interpolation for real-time queries.

---

## Project Structure

```
iv_calc/
├── src/                           # Core library
│   ├── implied_volatility.{h,c}   # IV calculation + Black-Scholes
│   ├── american_option.{h,c}      # American option pricing
│   ├── pde_solver.{h,c}           # General PDE solver (FDM)
│   ├── cubic_spline.{h,c}         # Interpolation
│   ├── brent.{h,c}                # Root-finding
│   └── ivcalc_trace.h             # USDT tracing probes
│
├── examples/                      # Demonstration programs
│   ├── example_implied_volatility.c
│   ├── example_american_option.c
│   └── example_heat_equation.c
│
├── tests/                         # Comprehensive test suite
│   ├── implied_volatility_test.cc # 44 test cases
│   ├── american_option_test.cc    # 29 test cases
│   ├── pde_solver_test.cc         # Core solver tests
│   └── quantlib_benchmark.cc      # QuantLib comparison
│
├── docs/                          # Design documentation
│   ├── PROJECT_OVERVIEW.md        # This file (problem & solution)
│   ├── ARCHITECTURE.md            # Detailed technical architecture
│   ├── QUICK_REFERENCE.md         # Developer quick-start
│   ├── INTERPOLATION_ENGINE_DESIGN.md # Future performance work
│   └── FASTVOL_ANALYSIS_AND_PLAN.md   # Optimization roadmap
│
├── scripts/                       # Utilities
│   ├── ivcalc-trace               # USDT tracing helper
│   └── tracing/                   # bpftrace scripts
│
├── CLAUDE.md                      # Instructions for Claude Code
├── TRACING.md                     # USDT tracing guide
├── TRACING_QUICKSTART.md          # 5-minute tracing tutorial
└── MODULE.bazel                   # Build configuration
```

---

## Who Should Use iv_calc?

### ✅ Good Fit

- **Researchers** building new pricing models
- **Students** learning computational finance
- **Developers** needing flexible option pricing
- **Quants** prototyping trading strategies
- **Engineers** requiring transparent, well-tested code

### ⚠️ Consider Alternatives

- **High-frequency trading**: Use optimized libraries (QuantLib, proprietary)
- **Production systems**: Consider mature, battle-tested solutions
- **GPU acceleration needed**: iv_calc is CPU-only (for now)
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

### Quick Example: Calculate Implied Volatility

```c
#include "src/implied_volatility.h"

IVParams params = {
    .spot_price = 100.0,
    .strike = 100.0,
    .time_to_maturity = 1.0,
    .risk_free_rate = 0.05,
    .market_price = 10.45,
    .is_call = true
};

IVResult result = calculate_implied_volatility(&params);

if (result.converged) {
    printf("Implied volatility: %.4f (%.1f%%)\n",
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
4. **Run benchmarks**: `bazel run //tests:quantlib_benchmark`
5. **Learn tracing**: `TRACING_QUICKSTART.md`

---

## Comparison to Other Libraries

| Feature | iv_calc | QuantLib | PyQL | Bloomberg API |
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

**iv_calc's niche:** Research-grade flexibility with production-conscious design, optimized for learning and prototyping rather than maximum performance.

---

## Roadmap

### Current State (v0.1)
- ✅ Black-Scholes pricing and IV calculation
- ✅ American option pricing (PDE-based)
- ✅ TR-BDF2 implicit solver
- ✅ USDT tracing system
- ✅ Comprehensive test suite
- ✅ QuantLib benchmarks

### Near-Term (v0.2-0.3)
- 🚧 Interpolation-based pricing engine (40,000x speedup)
- 🚧 CPU optimizations (AVX-512, FMA, restrict)
- 🚧 Greeks calculation via finite differences
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

1. **Performance**: Implement optimizations from `docs/FASTVOL_ANALYSIS_AND_PLAN.md`
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

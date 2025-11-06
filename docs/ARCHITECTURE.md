# IV Calculation and Option Pricing Architecture Analysis

> **Note:** For an overview of the problem domain and project motivation, see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

## Executive Summary

The mango-iv codebase implements a complete suite for American option pricing and implied volatility (IV) calculation. It uses a **modern C++20 template-based architecture** with:

- **TR-BDF2 PDE solver** for American option pricing via finite difference method
- **American IV calculation** combining FDM with Brent's method for root-finding
- **Cubic spline interpolation** for off-grid solution evaluation
- **USDT tracing** for zero-overhead diagnostic monitoring
- **Template metaprogramming** with concepts for compile-time optimization
- **Zero-cost abstractions** via templates and inlining
- **Type-safe design** with std::span, std::optional, and concepts

## Architecture Overview

### Three Main Components

```mermaid
graph TD
    IV[American IV Calculator<br/>implied_volatility.c/.h<br/>- FDM-based IV via Brent's method<br/>- Nested iteration structure]

    LBR[Let's Be Rational<br/>lets_be_rational.c/.h<br/>- European IV estimation<br/>- Bound calculation for American IV]

    AO[American Option Pricer<br/>american_option.c/.h<br/>- Black-Scholes PDE setup<br/>- Log-price transformation<br/>- Obstacle conditions<br/>- Dividend event handling]

    PDE[PDE Solver FDM Engine<br/>pde_solver.c/.h<br/>- TR-BDF2 time-stepping<br/>- Implicit solver fixed-point<br/>- Callback-based architecture<br/>- Single workspace buffer]

    BRENT[Brent's Root Finder]
    SPLINE[Cubic Spline Interpolation]
    TRI[Tridiagonal Solver]

    IV --> BRENT
    IV --> LBR
    IV --> AO
    AO --> PDE
    PDE --> SPLINE
    PDE --> TRI

    style IV fill:#e1f5ff,stroke:#333,stroke-width:2px,color:#000
    style LBR fill:#e8f5e9,stroke:#333,stroke-width:2px,color:#000
    style AO fill:#fff4e1,stroke:#333,stroke-width:2px,color:#000
    style PDE fill:#ffe1f5,stroke:#333,stroke-width:2px,color:#000
    style BRENT fill:#f0f0f0,stroke:#333,stroke-width:2px,color:#000
    style SPLINE fill:#f0f0f0,stroke:#333,stroke-width:2px,color:#000
    style TRI fill:#f0f0f0,stroke:#333,stroke-width:2px,color:#000
```

---

## Component 1: American Implied Volatility Calculation

### File Locations
- **Header**: `src/iv_solver.hpp`
- **Implementation**: `src/iv_solver.cpp`
- **Tests**: `tests/iv_solver_test.cc`
- **Example**: See integration tests

### Core Data Structures

```cpp
namespace mango {

// Input parameters
struct IVParams {
    double spot_price;              // S: Current stock price
    double strike;                  // K: Strike price
    double time_to_maturity;        // T: Time to expiration (years)
    double risk_free_rate;          // r: Risk-free interest rate
    double market_price;            // Market price of option
    bool is_call;                   // true for call, false for put
};

// Result
struct IVResult {
    double implied_vol;             // Calculated implied volatility
    int iterations;                 // Number of iterations
    double final_error;             // Final error value
    bool converged;                 // True if converged
    std::optional<std::string> failure_reason;  // Error message if failed
};

} // namespace mango
```

### Dependencies

The American IV calculator depends on:
- **Let's Be Rational** module (`lets_be_rational.{h,c}`) for European IV estimation (used to calculate upper bounds)
- **American Option** module (`american_option.{h,c}`) for FDM-based pricing
- **Brent's method** (`brent.h`) for root-finding

### Key Functions

#### 1. **Let's Be Rational - European IV Estimation** (from `lets_be_rational.h`)
```c
LBRResult lbr_implied_volatility(double spot, double strike,
                                  double time_to_maturity,
                                  double risk_free_rate,
                                  double market_price,
                                  bool is_call)
```

**Purpose**: Fast European IV estimation for calculating American IV upper bounds

**How it works:**
- Uses Black-Scholes formula with Abramowitz & Stegun normal CDF approximation
- Bisection-based root finding (simpler than Brent's method)
- Typically converges in 20-30 iterations
- **Performance**: ~781ns per calculation

**Why needed for American IV:**
- American option value ≥ European option value (early exercise premium)
- If European IV = σ_euro, then American IV ≤ σ_euro × 1.5 (heuristic upper bound)
- Provides tight bracketing interval for Brent's method
- Avoids expensive FDM calls during bound calculation

**Implementation note**: Uses simple bisection instead of Brent's method for simplicity and predictable performance

#### 2. **American Option Pricing Objective** (internal)
```c
static double american_objective(double volatility, void *user_data)
```

**Purpose**: Objective function for Brent's method root-finding

**How it works:**
1. Receives guessed volatility σ from Brent's method
2. Constructs American option with that volatility
3. Solves Black-Scholes PDE using FDM (~21ms per call)
4. Interpolates option value at spot price
5. Returns: theoretical_price(σ) - market_price

**Nested iteration structure:**
- Outer loop: Brent's method searching for σ
- Inner loop: Each Brent iteration calls FDM solver
- Convergence: When |theoretical_price - market_price| < tolerance

#### 3. **Main American IV Calculation Function**
```c
IVResult calculate_iv(const IVParams *params,
                      const AmericanOptionGrid *grid,
                      double vol_lower,
                      double vol_upper,
                      double tolerance,
                      int max_iter)
```

**Algorithm Overview:**
1. **Input Validation** (detects arbitrage violations):
   - Spot, strike, time, market price must be positive
   - Call price must not exceed spot price
   - Put price must not exceed discounted strike K·e^(-rT)
   - Market price must be above intrinsic value

2. **Objective Function Setup**:
   ```
   f(σ) = American_price_FDM(σ) - market_price
   ```
   Objective is to find σ where f(σ) = 0

3. **Brent's Method Root Finding**:
   - Searches in interval [vol_lower, vol_upper]
   - Each iteration calls `american_option_price()` with guessed σ (~21ms per call)
   - Combines bisection, secant method, and inverse quadratic interpolation
   - Guaranteed convergence if root is bracketed
   - Typical: 5-8 iterations → ~145ms total time

4. **Post-Processing**:
   - Returns convergence status and iteration count
   - No vega calculation (would require numerical differentiation)

#### 4. **Convenience Function**
```c
IVResult calculate_iv_simple(const IVParams *params)
```

**Automatic Configuration**:
- **Grid**: Default American option grid (141 points, 1000 steps)
- **Lower bound**: 0.01 (1% volatility)
- **Upper bound**: Uses Let's Be Rational to estimate European IV, then multiplies by 1.5
  - Rationale: American IV ≤ 1.5 × European IV (heuristic)
  - Fast calculation (~781ns) avoids expensive FDM calls for bounds
- **Tolerance**: 1e-6
- **Max iterations**: 100

### Validation & Error Handling

The implementation validates:
- ✅ All inputs are positive
- ✅ No arbitrage violations (call upper bound, put upper bound, intrinsic floor)
- ✅ Convergence within max iterations
- ✅ Returns descriptive error messages

### Test Coverage

From `implied_volatility_test.cc`:
- ✅ American put IV recovery (ATM, OTM, ITM)
- ✅ American call IV recovery
- ✅ Let's Be Rational bound estimation
- ✅ Input validation (invalid spot, strike, time, price)
- ✅ Arbitrage detection (price > intrinsic bounds)
- ✅ Grid configuration validation
- ✅ Convergence with default settings
- ✅ Edge cases (near expiry, extreme volatility)
- ✅ Deterministic convergence (same inputs → same outputs)

From `lets_be_rational_test.cc`:
- ✅ ATM European option IV estimation
- ✅ OTM European option IV estimation
- ✅ Invalid input handling
- ✅ Near-expiry edge cases

**Test Result**: 9 American IV test cases + 4 Let's Be Rational test cases, all passing

### Performance Characteristics

| Operation | Time | Details |
|----------|------|---------|
| Let's Be Rational (European IV) | ~781ns | Fast bound calculation, 20-30 bisection iterations |
| American option pricing (single) | ~21.7ms | FDM solve with 141 points × 1000 steps |
| American IV calculation (single) | ~145ms | 5-8 Brent iterations × 21.7ms per FDM call |
| Bound calculation overhead | <1µs | Let's Be Rational + 1.5x scaling |

**Bottleneck**: FDM solver calls within Brent's method (each iteration = full PDE solve)

**Scaling**:
- Brent iterations: O(log(1/ε)) where ε is tolerance
- Total time: O(iterations × FDM_time) ≈ O(8 × 21.7ms) ≈ 145ms

---

## Component 2: American Option Pricing

### File Locations
- **Header**: `src/american_option.hpp`
- **Tests**: `tests/american_option_test.cc`
- **Related**: `src/operators/black_scholes_pde.hpp` (Black-Scholes operator)

### Core Data Structures

```c
typedef enum {
    OPTION_CALL,
    OPTION_PUT
} OptionType;

typedef struct {
    double strike;                  // Strike price K
    double volatility;              // σ (volatility)
    double risk_free_rate;          // r (risk-free rate)
    double time_to_maturity;        // T (time to maturity in years)
    OptionType option_type;         // Call or Put
    
    // Discrete dividend information (optional)
    size_t n_dividends;             // Number of dividend payments
    double *dividend_times;         // Times of dividend payments (in years)
    double *dividend_amounts;       // Dividend amounts (absolute cash dividends)
} OptionData;

typedef struct {
    double x_min;                   // Minimum log-moneyness (e.g., -0.7)
    double x_max;                   // Maximum log-moneyness (e.g., 0.7)
    size_t n_points;                // Number of grid points (e.g., 141)
    double dt;                      // Time step (e.g., 0.001)
    size_t n_steps;                 // Number of time steps (e.g., 1000)
} AmericanOptionGrid;

typedef struct {
    PDESolver *solver;              // PDE solver (caller must destroy with american_option_free_result)
    int status;                     // 0 = success, -1 = failure
    void *internal_data;            // Internal data (do not access directly)
} AmericanOptionResult;
```

### Mathematical Formulation

#### Black-Scholes PDE (Backward Time)

The classic Black-Scholes PDE with American option constraint:

```
∂V/∂τ = (1/2)σ²S² ∂²V/∂S² + rS ∂V/∂S - rV,  τ ∈ [0, T]
V(S,T) = payoff(S)
V(S,τ) ≥ intrinsic(S)                        [American constraint]
```

Where τ = T - t (time to maturity)

#### Log-Price Transformation

Substituting x = ln(S/K) to reduce volatility coefficient:

```
∂V/∂τ = (1/2)σ² ∂²V/∂x² + (r - σ²/2) ∂V/∂x - rV

Coefficients:
  - Second derivative: (1/2)σ²
  - First derivative: r - σ²/2
  - Zeroth order: -r
```

**Advantages**:
- Constant coefficients (don't depend on S)
- Natural moneyness scaling
- Better numerical stability

#### Boundary Conditions

**Left boundary** (x → -∞, S → 0):
- Call: V(0, τ) = 0 (worthless)
- Put: V(0, τ) = K·e^(-rτ) (discounted strike)

**Right boundary** (x → ∞, S → ∞):
- Call: V ≈ S - K (never exercise early, exercise value)
- Put: V(∞, τ) = 0 (worthless)

#### Terminal Condition (At Maturity)

```
V(S, 0) = payoff(S)
  Call: max(S - K, 0)
  Put: max(K - S, 0)
```

#### Obstacle Condition (American Constraint)

```
ψ(x) = intrinsic_value(x)
V(x,τ) ≥ ψ(x)  for all τ
```

This enforces early exercise: option value is at least intrinsic value at all times.

### Key Functions

#### 1. **High-Level API**
```c
AmericanOptionResult american_option_price(const OptionData *option_data,
                                           const AmericanOptionGrid *grid_params)
```

**Workflow**:
1. Creates spatial grid in log-price coordinates
2. Sets up time domain (forward time = time-to-maturity)
3. Converts dividend times from calendar to solver time
4. Creates callbacks for PDE solver
5. Configures relaxed tolerance (1e-4) for obstacle constraints
6. Solves using TR-BDF2 scheme
7. Returns solver with solution

**Cleanup**: Call `american_option_free_result()` to free both the solver and internal data structures

#### 2. **Batch Processing API**
```c
int american_option_price_batch(const OptionData *option_data,
                                const AmericanOptionGrid *grid_params,
                                size_t n_options,
                                AmericanOptionResult *results)
```

**Features**:
- Uses OpenMP parallel for loop
- Each thread prices one option independently
- Significant wall-time speedup (10-60x on multi-core)
- Enables vectorized IV recovery

**Cleanup**: Call `american_option_free_result()` on each result to free resources

#### 3. **Callback Functions** (Vectorized)

**Terminal Condition**:
```c
void american_option_terminal_condition(const double *x, size_t n_points,
                                        double *V, void *user_data)
```
- Computes payoff for all spatial points at maturity

**Spatial Operator**:
```c
void american_option_spatial_operator(const double *x, double t,
                                      const double *V, size_t n_points,
                                      double *LV, void *user_data)
```
- Applies finite difference stencil to compute L(V)
- Uses centered differences: (V[i-1] - 2V[i] + V[i+1]) / dx²
- Vectorized with `#pragma omp simd`

**Obstacle Condition**:
```c
void american_option_obstacle(const double *x, double t,
                             size_t n_points, double *obstacle,
                             void *user_data)
```
- Computes intrinsic value constraint for all points

**Boundary Conditions**:
```c
double american_option_left_boundary(double t, void *user_data);
double american_option_right_boundary(double t, void *user_data);
```
- Return scalar boundary values for each time step

#### 4. **Dividend Handling**
```c
void american_option_apply_dividend(const double *x_grid, size_t n_points,
                                    const double *V_old, double *V_new,
                                    double dividend, double strike)
```

**Mechanism**:
- When dividend D is paid, stock price jumps: S_old → S_old - D
- In log-price: x_old → x_new = ln((e^x_old - D/K))
- Interpolates option value from old grid to new grid
- Called by temporal event callback when dividend time is reached

#### 5. **Utility Function**
```c
double american_option_get_value_at_spot(const PDESolver *solver,
                                        double spot_price,
                                        double strike)
```
- Converts spot to log-moneyness: x = ln(S/K)
- Uses cubic spline to interpolate value at x

### Grid Configuration Recommendations

**Typical Settings**:
```c
AmericanOptionGrid default_grid = {
    .x_min = -0.7,      // ln(0.5) - covers 50% of strike
    .x_max = 0.7,       // ln(2.0) - covers 200% of strike
    .n_points = 141,    // ~0.01 spacing in log-price
    .dt = 0.001,        // 0.1% per step
    .n_steps = 1000     // 1 year with daily resolution
};
```

**Refinement for Accuracy**:
```c
AmericanOptionGrid fine_grid = {
    .x_min = -1.0,
    .x_max = 1.0,
    .n_points = 201,    // Finer spacing
    .dt = 0.0005,       // Smaller steps
    .n_steps = 2000     // More steps
};
```

### Test Coverage

From `american_option_test.cc`:
- ✅ Basic call and put options
- ✅ Put-call relationships (American options don't have exact parity)
- ✅ Early exercise premium verification
- ✅ Intrinsic value bounds (V ≥ intrinsic)
- ✅ Monotonicity in volatility (higher vol → higher value)
- ✅ Monotonicity in time to maturity
- ✅ OTM/ITM/Deep OTM/Deep ITM scenarios
- ✅ Short/long maturity cases
- ✅ High/low volatility extremes
- ✅ Zero and negative interest rates
- ✅ Grid resolution sensitivity
- ✅ Single dividend, multiple dividends
- ✅ Dividend timing (early vs late)
- ✅ Dividend impact on calls (decreases value)
- ✅ Dividend impact on puts (increases value)
- ✅ Zero dividend amounts (should match no-dividend case)

**Test Result**: 42 test cases covering comprehensive scenarios

### Performance Characteristics

| Configuration | Grid | Time Steps | Time/Option |
|---|---|---|---|
| Coarse | 141×1000 | 500 | 8-10 ms |
| Typical | 141×1000 | 1000 | 21-22 ms |
| Fine | 201×2000 | 2000 | 80-100 ms |

**Comparison to QuantLib** (from benchmarks/BENCHMARK.md):
- IV Calc: 21.6 ms per option
- QuantLib: 10.4 ms per option
- Ratio: 2.1x slower (reasonable for research code)

**Key Performance Factors**:
- ✅ SIMD vectorization on spatial operators
- ✅ Single contiguous workspace buffer
- ✅ Minimal malloc during solve
- ✅ OpenMP parallel batch processing

---

## Component 3: PDE Solver (Finite Difference Method Engine)

### File Locations
- **Header**: `src/pde_solver.hpp`
- **Tests**: `tests/pde_solver_test.cc`
- **Operators**: `src/operators/` directory
- **Example**: `examples/example_newton_solver.cc`

### Overview

The PDE solver implements the **TR-BDF2 (Two-stage Runge-Kutta with Backward Differentiation Formula 2)** scheme for solving parabolic PDEs:

```
∂u/∂t = L(u) + boundary/obstacle constraints
u(x, 0) = u₀(x)
```

### TR-BDF2 Time Stepping Scheme

A two-stage implicit scheme combining:
1. **Stage 1**: Trapezoidal rule from t_n to t_n + γ·dt (γ ≈ 0.5858)
2. **Stage 2**: BDF2 from t_n to t_n+1

**Properties**:
- L-stable (dampens high-frequency errors)
- Second-order accurate
- Unconditionally stable (large dt possible)
- Requires implicit solve each stage

### Memory Management (Single Buffer Architecture)

**Workspace (12n doubles, contiguous allocation):**

All arrays allocated from single 64-byte aligned buffer:

```mermaid
graph LR
    A["u_current<br/>n doubles"]
    B["u_next<br/>n doubles"]
    C["u_stage<br/>n doubles"]
    D["rhs<br/>n doubles"]
    E["matrix_diag<br/>n doubles"]
    F["matrix_upper<br/>n doubles"]
    G["matrix_lower<br/>n doubles"]
    H["u_old<br/>n doubles"]
    I["Lu<br/>n doubles"]
    J["u_temp<br/>n doubles"]
    K["tridiag_workspace<br/>2n doubles"]

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K

    style A fill:#e3f2fd,stroke:#333,stroke-width:2px,color:#000
    style B fill:#e3f2fd,stroke:#333,stroke-width:2px,color:#000
    style C fill:#fff3e0,stroke:#333,stroke-width:2px,color:#000
    style D fill:#fff3e0,stroke:#333,stroke-width:2px,color:#000
    style E fill:#f3e5f5,stroke:#333,stroke-width:2px,color:#000
    style F fill:#f3e5f5,stroke:#333,stroke-width:2px,color:#000
    style G fill:#f3e5f5,stroke:#333,stroke-width:2px,color:#000
    style H fill:#e8f5e9,stroke:#333,stroke-width:2px,color:#000
    style I fill:#e8f5e9,stroke:#333,stroke-width:2px,color:#000
    style J fill:#e8f5e9,stroke:#333,stroke-width:2px,color:#000
    style K fill:#ffe0b2,stroke:#333,stroke-width:2px,color:#000
```

**Advantages**:
- Single malloc operation
- Better cache locality
- 64-byte alignment for SIMD
- Zero overhead during time-stepping

### Template-Based Architecture

The PDE solver uses C++20 templates with concepts for type-safe, zero-cost abstractions:

```cpp
namespace mango {

// Initial condition: lambda or function object
// Signature: void(std::span<const double> x, std::span<double> u0)
using InitialConditionFunc = std::function<void(std::span<const double>, std::span<double>)>;

// Boundary conditions: Concept-based compile-time polymorphism
template<typename T>
concept BoundaryCondition = requires(T bc, double t) {
    { bc.value(t) } -> std::convertible_to<double>;
    { bc.type() } -> std::same_as<BoundaryType>;
};

// Spatial operator: Concept-based interface
template<typename T>
concept SpatialOperator = requires(T op, std::span<const double> x, double t,
                                    std::span<const double> u, std::span<double> Lu) {
    { op(x, t, u, Lu) } -> std::same_as<void>;
};

// Obstacle condition: Optional callback
using ObstacleCallback = std::function<void(double t, std::span<const double> x,
                                             std::span<double> psi)>;

// Temporal events: Event-driven architecture
using TemporalEventCallback = std::function<void(double t, std::span<const double> x,
                                                  std::span<double> u)>;

struct TemporalEvent {
    double time;
    TemporalEventCallback callback;

    auto operator<=>(const TemporalEvent& other) const {
        return time <=> other.time;  // C++20 spaceship operator
    }
};

} // namespace mango
```

### Temporal Event System

The solver supports **temporal events** for handling discrete discontinuities in time (e.g., dividend payments, regime changes).

**Registration**:
```c
PDECallbacks callbacks = {
    // ... other callbacks ...
    .temporal_event = my_event_handler,
    .n_temporal_events = 2,
    .temporal_event_times = (double[]){0.25, 0.75},  // Event times in [t_start, t_end]
    .user_data = &my_data
};
```

**How it works**:
1. Solver maintains sorted list of event times
2. During time-stepping, checks if current step crosses an event
3. When event triggered:
   - Solver completes step to exact event time
   - Calls `temporal_event` callback with solution array
   - Callback modifies solution in-place (e.g., applies dividend jump)
   - Solver continues from modified state

**Example use case - Dividend payments**:
```c
void dividend_event(double t, const double *x, size_t n, double *u,
                    const size_t *event_indices, size_t n_events, void *data) {
    // Apply stock price jump from dividend payment
    // Interpolate option value to new grid after S → S - D
    american_option_apply_dividend(x, n, u, u, dividend_amount, strike);
}
```

**Thread safety**: Each solver instance has independent event state; batch processing with temporal events is safe.

### Core Functions

#### 1. **Solver Creation**
```c
PDESolver* pde_solver_create(SpatialGrid *grid,
                              const TimeDomain *time,
                              const BoundaryConfig *bc_config,
                              const TRBDF2Config *trbdf2_config,
                              const PDECallbacks *callbacks);
```

**Ownership**: Takes ownership of grid; `grid.x` set to nullptr

#### 2. **Solver Lifecycle**
```c
void pde_solver_initialize(PDESolver *solver);  // Apply initial conditions
int pde_solver_solve(PDESolver *solver);        // Full solve loop
void pde_solver_destroy(PDESolver *solver);     // Cleanup
```

#### 3. **Single Step**
```c
int pde_solver_step(PDESolver *solver, double t_current);
```

#### 4. **Solution Access**
```c
const double* pde_solver_get_solution(const PDESolver *solver);
const double* pde_solver_get_grid(const PDESolver *solver);
double pde_solver_interpolate(const PDESolver *solver, double x_eval);
```

### Implicit Solver (Fixed-Point Iteration)

For each implicit stage:

```
Given: u_n, target: u_{n+1}

Fixed-point iteration with under-relaxation (ω = 0.7):
  u^(k+1) = ω · u_candidate(u^(k)) + (1-ω) · u^(k)

Convergence criteria:
  || u^(k+1) - u^(k) ||_∞ < tolerance
```

**Configuration**:
```c
typedef struct {
    double gamma;               // TR-BDF2 parameter (≈ 0.5858)
    size_t max_iter;            // Max iterations per step
    double tolerance;           // Convergence tolerance
} TRBDF2Config;
```

### Boundary Condition Types

```c
typedef enum {
    BC_DIRICHLET,               // u = g(t)
    BC_NEUMANN,                 // ∂u/∂x = g(t)
    BC_ROBIN                    // a·u + b·∂u/∂x = g(t)
} BoundaryType;
```

### Boundary Condition Implementation

#### Dirichlet Boundaries
Direct assignment of boundary values:
```c
u[0] = left_boundary(t);      // Left boundary
u[n-1] = right_boundary(t);   // Right boundary
```

#### Neumann Boundaries (Ghost Point Method)
For ∂u/∂x = g at boundaries, the solver uses the **ghost point method** to properly compute the spatial operator at boundary points while maintaining conservation properties.

**Left boundary** (x = x_min):
- Creates virtual point u_{-1} outside domain
- Ghost point relation: u_{-1} = u_1 - 2·dx·g (from centered difference)
- Estimates diffusion coefficient D from interior stencil
- Computes: L(u)_0 = D·(2u_1 - 2u_0 - 2·dx·g) / dx²

**Right boundary** (x = x_max):
- Creates virtual point u_n outside domain
- Ghost point relation: u_n = u_{n-2} + 2·dx·g
- Estimates diffusion coefficient D from interior stencil
- Computes: L(u)_{n-1} = D·(2u_{n-2} - 2u_{n-1} + 2·dx·g) / dx²

**Assumption**: Ghost point method assumes pure diffusion operator L(u) = D·∂²u/∂x². For advection-diffusion or nonlinear operators, coefficient estimation may not be accurate. See `pde_solver.c` lines 83-86, 105.

#### Robin Boundaries
For a·u + b·∂u/∂x = g at boundaries:
- Modified matrix entries in tridiagonal system
- Incorporates both value and derivative conditions
- Coefficients a, b validated to prevent division by zero (a ≠ 0)

### Performance Optimizations

#### Zero-Allocation Tridiagonal Solver
The tridiagonal solver (Thomas algorithm) uses pre-allocated workspace from the PDESolver's 12n buffer:
- **Workspace**: 2n doubles for c_prime and d_prime arrays
- **Benefit**: Eliminates malloc/free overhead in hot path
- **Impact**: Called once per Newton iteration per timestep (~5-10% speedup)
- **Backward compatibility**: Accepts NULL workspace pointer (allocates internally for standalone use)

#### SIMD Vectorization
Key loops marked with `#pragma omp simd` for automatic vectorization:
- Spatial operator evaluation
- Tridiagonal forward/backward sweeps
- Fixed-point iteration updates

### Test Coverage

From `pde_solver_test.cc`:
- ✅ Heat equation (classic diffusion)
- ✅ Wave equation (hyperbolic)
- ✅ Advection-diffusion
- ✅ Dirichlet, Neumann, and Robin boundary conditions
- ✅ Convergence to analytical solutions
- ✅ Various grid resolutions
- ✅ Stability tests
- ✅ Jump conditions (discontinuous coefficients)
- ✅ Obstacle conditions (variational inequalities)

---

## Component 4: Supporting Infrastructure

### 4.1 Brent's Root Finder

**File**: `src/brent.h` (header-only, inline)

**Algorithm**: Combines bisection, secant method, and inverse quadratic interpolation
- **Guaranteed convergence** if root is bracketed
- **Superlinear convergence** rate (~1.6x faster than bisection)
- **No derivative required**
- **More robust than Newton's method**

**Usage in IV Calculation**: Finds σ such that BS_price(σ) = market_price

### 4.2 Cubic Spline Interpolation

**File**: `src/cubic_spline_solver.hpp`

**Purpose**: Evaluate PDE solution at arbitrary off-grid points

**Method**: Natural cubic splines with:
- Quadratic system solve via shared tridiagonal solver
- Two API variants: malloc-based and workspace-based
- Function and derivative evaluation

**Malloc-Based API** (convenience):
```c
CubicSpline *spline = pde_spline_create(x_grid, solution, n_points);
double value_at_x = pde_spline_eval(spline, x_eval);
double derivative = pde_spline_eval_derivative(spline, x_eval);
pde_spline_destroy(spline);
```
- Allocates workspace internally (4n doubles for coefficients, 6n for temporary)
- Convenient for one-off interpolation queries
- Simple ownership model

**Workspace-Based API** (zero-malloc, performance-critical):
```c
CubicSpline spline;  // Stack-allocated
double workspace[4 * n_points];
double temp_workspace[6 * n_points];

pde_spline_init(&spline, x_grid, solution, n_points, workspace, temp_workspace);
double value_at_x = pde_spline_eval(&spline, x_eval);
double derivative = pde_spline_eval_derivative(&spline, x_eval);
// No destroy needed - workspace managed by caller
```
- Zero heap allocation (workspace provided by caller)
- Ideal for hot paths with repeated spline creation/destruction
- Used by 4D/5D interpolation engine for **99.9% malloc reduction**
- Workspace requirements: 10n doubles total (4n + 6n)
- Can reuse temp_workspace across multiple `pde_spline_init()` calls

**Performance Impact**:
- 2D interpolation: 2 mallocs → 2 workspace-only allocations
- 4D interpolation: ~15 mallocs → 2 workspace-only allocations (87% reduction)
- 5D interpolation: ~1,873 mallocs → 2 workspace-only allocations (99.9% reduction)

**Implementation Note**: Both APIs share the same evaluation functions (`pde_spline_eval`, `pde_spline_eval_derivative`). The workspace-based API was added in PR #36 to eliminate malloc overhead in multi-dimensional interpolation hot paths.

### 4.3 Workspace-Based Interpolation API

**Added in PR #37** - Extends workspace pattern to multi-dimensional interpolation queries.

**Files**: `src/interp_cubic.h`, `src/interp_cubic_workspace.c`

**Problem**: Even with precomputed spline coefficients, slow-path queries (off-precomputed-grid) performed excessive malloc/free:
- 2D interpolation: 4 malloc/free per query
- 4D interpolation: 8 malloc/free per query
- 5D interpolation: 10 malloc/free per query
- Dividend event handler: 1 malloc/free per event

**Solution**: Workspace-based interpolation functions that accept caller-provided buffers, eliminating all hot path allocations.

**API**:
```c
// Calculate workspace size (once per table dimensions)
size_t ws_size = cubic_interp_workspace_size_4d(n_m, n_tau, n_sigma, n_r);

// Allocate workspace (once, reuse across millions of queries)
double *buffer = malloc(ws_size * sizeof(double));
CubicInterpWorkspace workspace;
cubic_interp_workspace_init(&workspace, buffer, n_m, n_tau, n_sigma, n_r, 0);

// Query with zero malloc (can be called millions of times)
for (int i = 0; i < 1000000; i++) {
    double price = cubic_interpolate_4d_workspace(table, m[i], tau[i],
                                                    sigma[i], r[i], workspace);
}

// Cleanup
free(buffer);
```

**Workspace Structure**:
```c
typedef struct {
    double *spline_coeff_workspace;   // 4 * max_grid_size (reused across stages)
    double *spline_temp_workspace;    // 6 * max_grid_size (reused across stages)
    double *intermediate_arrays;      // Stage results (dimension-dependent)
    double *slice_buffers;            // max_grid_size (slice extraction)
    size_t max_grid_size;             // Largest dimension
    size_t total_size;                // Total doubles allocated
} CubicInterpWorkspace;
```

**Workspace Sizing Functions**:
- `cubic_interp_workspace_size_2d(n_m, n_tau)` - 2D IV surface queries
- `cubic_interp_workspace_size_4d(n_m, n_tau, n_sigma, n_r)` - 4D price tables
- `cubic_interp_workspace_size_5d(n_m, n_tau, n_sigma, n_r, n_q)` - 5D with dividends

**Zero-Malloc Query Functions**:
- `cubic_interpolate_2d_workspace(surface, m, tau, workspace)` - 2D queries
- `cubic_interpolate_4d_workspace(table, m, tau, sigma, r, workspace)` - 4D queries
- `cubic_interpolate_5d_workspace(table, m, tau, sigma, r, q, workspace)` - 5D queries

**Performance**: 100% elimination of malloc in interpolation hot paths.

**Integration with Temporal Events**: The PDE solver provides workspace to temporal event callbacks via the `workspace` parameter, enabling zero-malloc dividend handling during American option pricing.

**Memory Requirements** (typical 4D table: 50×30×20×10):
- Spline workspace: 10 × 50 = 500 doubles (reused across stages)
- Intermediate arrays: (30×20×10) + (20×10) + 10 = 6,210 doubles
- Slice buffer: 50 doubles
- **Total**: 6,760 doubles (~54KB per workspace)

**Usage Pattern**: Allocate workspace once per table configuration, reuse across all queries. For multi-threaded scenarios, each thread maintains its own workspace to avoid contention.

### 4.4 Tridiagonal Solver

**File**: `src/thomas_solver.hpp`

**Method**: Thomas algorithm (TDMA - Tridiagonal Matrix Algorithm)
- **Time complexity**: O(n)
- **Used by**:
  - TR-BDF2 implicit solver
  - Cubic spline coefficient calculation

### 4.5 USDT Tracing System

**Files**: Tracing infrastructure integrated throughout C++ codebase

**Purpose**: Zero-overhead diagnostic tracing for profiling and debugging

**Probe Categories**:
1. **Algorithm Lifecycle**: Start, progress, complete
2. **Convergence**: Iterations, success/failure
3. **Validation**: Input errors, runtime errors
4. **Module-specific**:
   - PDE: Start, progress, complete, convergence
   - IV: Start, complete, validation errors
   - American options: Start, complete
   - Brent's method: Start, iterations, complete
   - Cubic spline: Errors

**Zero Overhead**: Compiles to NOP instructions when not traced; can be dynamically enabled at runtime via bpftrace

---

## Component 5: Interpolation Engine (Fast Lookup)

### Purpose

Provides sub-microsecond option pricing and IV lookups via pre-computed interpolation tables, achieving **40,000x speedup** over FDM for real-time queries during trading sessions.

### File Locations

```
src/
├── interp_strategy.h      # Strategy pattern interface for interpolation algorithms
├── interp_cubic.{h,c}     # Cubic spline interpolation strategy
├── iv_surface.{h,c}       # 2D implied volatility surface (~100ns queries)
└── price_table.{h,c}      # 4D/5D option price table (~500ns queries)

examples/
└── example_interpolation.c  # IV surface and price table usage

tests/
└── interpolation_test.cc    # Unit tests for all interpolation components
```

### Architecture Overview

```mermaid
graph TD
    USER[User Query<br/>m, τ, σ, r]

    TRANSFORM[Coordinate Transform<br/>Raw → Grid Space<br/>COORD_LOG_SQRT]

    STRATEGY[Interpolation Strategy<br/>INTERP_CUBIC]

    PRECOMP[Precomputed Spline<br/>Coefficients<br/>~10MB for 4D table]

    STAGE1[Stage 1: Moneyness<br/>Evaluate splines along m<br/>for each τ,σ,r]

    STAGE2[Stage 2: Maturity<br/>Build & evaluate splines along τ<br/>for each σ,r]

    STAGE3[Stage 3: Volatility<br/>Build & evaluate splines along σ<br/>for each r]

    STAGE4[Stage 4: Rate<br/>Build & evaluate spline along r<br/>→ final result]

    RESULT[Interpolated Price<br/>~500ns total]

    USER --> TRANSFORM
    TRANSFORM --> STRATEGY
    STRATEGY --> PRECOMP
    PRECOMP --> STAGE1
    STAGE1 --> STAGE2
    STAGE2 --> STAGE3
    STAGE3 --> STAGE4
    STAGE4 --> RESULT

    style USER fill:#e3f2fd,stroke:#333,stroke-width:2px,color:#000
    style TRANSFORM fill:#fff3e0,stroke:#333,stroke-width:2px,color:#000
    style STRATEGY fill:#f3e5f5,stroke:#333,stroke-width:2px,color:#000
    style PRECOMP fill:#e8f5e9,stroke:#333,stroke-width:2px,color:#000
    style STAGE1 fill:#ffe0b2,stroke:#333,stroke-width:2px,color:#000
    style STAGE2 fill:#ffe0b2,stroke:#333,stroke-width:2px,color:#000
    style STAGE3 fill:#ffe0b2,stroke:#333,stroke-width:2px,color:#000
    style STAGE4 fill:#ffe0b2,stroke:#333,stroke-width:2px,color:#000
    style RESULT fill:#c8e6c9,stroke:#333,stroke-width:2px,color:#000
```

**Key Features**:
- **Strategy Pattern**: Runtime selection of interpolation algorithm (cubic splines)
- **Coordinate Transforms**: Log-sqrt transforms for better interpolation accuracy
- **Tensor-Product**: Separable stages reduce complexity from O(n^4) to O(n)
- **Precomputation**: Spline coefficients computed once, reused for millions of queries

### Core Data Structures

#### IV Surface (2D Interpolation)

```c
typedef struct {
    size_t n_moneyness;          // Grid dimension
    size_t n_maturity;
    double *moneyness_grid;       // Moneyness values (S/K)
    double *maturity_grid;        // Time to maturity (years)
    double *iv_values;            // Implied volatilities (flattened 2D array)

    const InterpolationStrategy *strategy;  // Runtime algorithm selection
    InterpContext interp_context;           // Strategy-specific context

    char underlying[32];          // "SPX", "NDX", etc.
    time_t last_update;
} IVSurface;
```

**Memory footprint**: ~12KB per surface (50 × 30 grid)

#### Option Price Table (4D/5D Interpolation)

```c
typedef struct {
    // Grid dimensions
    size_t n_moneyness, n_maturity, n_volatility, n_rate, n_dividend;
    double *moneyness_grid, *maturity_grid, *volatility_grid;
    double *rate_grid, *dividend_grid;

    double *prices;              // Pre-computed option prices (flattened array)

    // Fast indexing strides (pre-computed)
    size_t stride_m, stride_tau, stride_sigma, stride_r, stride_q;

    const InterpolationStrategy *strategy;  // Runtime algorithm selection
    InterpContext interp_context;

    OptionType type;             // CALL or PUT
    ExerciseType exercise;       // EUROPEAN or AMERICAN
    char underlying[32];
    time_t generation_time;
} OptionPriceTable;
```

**Memory footprint**: ~2.4MB per table (4D: 50×30×20×10 grid)

### Memory Layout Options

The price table supports different memory layouts for cache optimization:

```mermaid
graph TD
    TABLE[OptionPriceTable<br/>4D Array: m×τ×σ×r]

    OUTER[LAYOUT_M_OUTER<br/>m, τ, σ, r<br/>Default ordering]

    INNER[LAYOUT_M_INNER<br/>r, σ, τ, m<br/>Recommended for cubic]

    POINT_QUERY[Point Query<br/>table.m.tau.sigma.r<br/>~Equal performance]

    SLICE_OUTER[Slice Extraction OUTER<br/>Extract all m for fixed τ,σ,r<br/>Scattered memory access<br/>~30x slower]

    SLICE_INNER[Slice Extraction INNER<br/>Extract all m for fixed τ,σ,r<br/>Contiguous memory access<br/>~30x faster]

    TABLE --> OUTER
    TABLE --> INNER

    OUTER --> POINT_QUERY
    INNER --> POINT_QUERY

    OUTER --> SLICE_OUTER
    INNER --> SLICE_INNER

    style TABLE fill:#e3f2fd,stroke:#333,stroke-width:2px,color:#000
    style OUTER fill:#f3e5f5,stroke:#333,stroke-width:2px,color:#000
    style INNER fill:#fff3e0,stroke:#333,stroke-width:2px,color:#000
    style POINT_QUERY fill:#c8e6c9,stroke:#333,stroke-width:2px,color:#000
    style SLICE_OUTER fill:#ffcdd2,stroke:#333,stroke-width:2px,color:#000
    style SLICE_INNER fill:#c8e6c9,stroke:#333,stroke-width:2px,color:#000
```

**Layout Comparison**:
- **LAYOUT_M_OUTER** [m][τ][σ][r]: Good for point queries, compatible with older code
- **LAYOUT_M_INNER** [r][σ][τ][m]: Optimized for cubic interpolation (slice-based)
  - 64-byte cache lines hold 8 consecutive moneyness values
  - ~30x faster moneyness slice extraction
  - Same memory usage, same point query performance

**Why it matters for cubic**: Precomputation creates splines along moneyness dimension for each (τ,σ,r) combination. LAYOUT_M_INNER makes these slices contiguous in memory, dramatically improving cache performance during precomputation.

### Interpolation Strategy Pattern

Uses **dependency injection** for runtime algorithm selection without recompilation:

```c
typedef struct {
    const char *name;
    const char *description;

    // Callbacks for different dimensions
    double (*interpolate_2d)(const IVSurface*, double m, double tau, InterpContext);
    double (*interpolate_4d)(const OptionPriceTable*, double m, double tau,
                             double sigma, double r, InterpContext);
    double (*interpolate_5d)(const OptionPriceTable*, double m, double tau,
                             double sigma, double r, double q, InterpContext);

    // Context management
    InterpContext (*create_context)(size_t dimensions, const size_t *grid_sizes);
    void (*destroy_context)(InterpContext);

    // Optional pre-computation
    int (*precompute)(void *table, InterpContext);
} InterpolationStrategy;
```

**Available strategies**:
- `INTERP_CUBIC`: Tensor-product cubic splines (C² continuous, ~500ns for 4D)
  - Provides accurate second derivatives (gamma, vega-convexity, etc.)
  - Handles coordinate transformations (COORD_LOG_SQRT, COORD_LOG_VARIANCE)
  - Requires precomputation of spline coefficients
  - Minimum 2 points per dimension required

### Key Functions

#### IV Surface API

```c
// Create/destroy
IVSurface* iv_surface_create(const double *moneyness, size_t n_m,
                              const double *maturity, size_t n_tau);
IVSurface* iv_surface_create_with_strategy(/* ... */, const InterpolationStrategy*);
void iv_surface_destroy(IVSurface *surface);

// Data manipulation
int iv_surface_set(IVSurface *surface, size_t i_m, size_t i_tau, double iv);
int iv_surface_set_all(IVSurface *surface, const double *iv_data);
double iv_surface_get(const IVSurface *surface, size_t i_m, size_t i_tau);

// Fast interpolation (main query interface)
double iv_surface_interpolate(const IVSurface *surface, double moneyness, double maturity);

// I/O
int iv_surface_save(const IVSurface *surface, const char *filename);
IVSurface* iv_surface_load(const char *filename);
```

#### Price Table API

```c
// Create/destroy
OptionPriceTable* price_table_create(
    const double *moneyness, size_t n_m,
    const double *maturity, size_t n_tau,
    const double *volatility, size_t n_sigma,
    const double *rate, size_t n_r,
    const double *dividend, size_t n_q,  // Pass NULL for 4D mode
    OptionType type, ExerciseType exercise);

void price_table_destroy(OptionPriceTable *table);

// Data manipulation
int price_table_set(OptionPriceTable *table, size_t i_m, size_t i_tau,
                    size_t i_sigma, size_t i_r, size_t i_q, double price);
double price_table_get(const OptionPriceTable *table, /* ... */);

// Fast interpolation (main query interface)
double price_table_interpolate_4d(const OptionPriceTable *table,
                                   double moneyness, double maturity,
                                   double volatility, double rate);
double price_table_interpolate_5d(const OptionPriceTable *table,
                                   double moneyness, double maturity,
                                   double volatility, double rate, double dividend);

// Greeks via finite differences on interpolated values
OptionGreeks price_table_greeks_4d(const OptionPriceTable *table, /* ... */);
OptionGreeks price_table_greeks_5d(const OptionPriceTable *table, /* ... */);

// I/O
int price_table_save(const OptionPriceTable *table, const char *filename);
OptionPriceTable* price_table_load(const char *filename);
```

### Cubic Spline Interpolation Algorithm

**Method**: Tensor-product cubic spline interpolation with separable stages

**4D Algorithm** (for price tables):
1. **Precomputation** (done once after filling table):
   - For each combination of (τ, σ, r): create cubic spline in moneyness dimension
   - Stores spline coefficients for all slices
   - Enables O(1) evaluation along first dimension

2. **Query** (per interpolation request):
   - Stage 1: Evaluate moneyness splines for all (τ, σ, r) combinations → intermediate values
   - Stage 2: Create and evaluate maturity splines for each (σ, r) → intermediate values
   - Stage 3: Create and evaluate volatility splines for each r → intermediate values
   - Stage 4: Create and evaluate rate spline → final result

3. **Coordinate Transformation**:
   - Query coordinates transformed from raw to grid space before interpolation
   - Supports COORD_RAW, COORD_LOG_SQRT, COORD_LOG_VARIANCE

```mermaid
graph LR
    RAW["User Query<br/>m=1.05, τ=0.25"]

    COORD_RAW["COORD_RAW<br/>Identity<br/>m'=1.05, τ'=0.25"]

    COORD_LOG_SQRT["COORD_LOG_SQRT<br/>m'=ln(m)<br/>τ'=√τ<br/>m'=0.049, τ'=0.5"]

    COORD_LOG_VAR["COORD_LOG_VARIANCE<br/>m'=ln(m)<br/>σ'²=σ²·τ<br/>Better for vol surfaces"]

    INTERP["Cubic Spline<br/>Interpolation<br/>in Grid Space"]

    RESULT["Interpolated<br/>Price"]

    RAW --> COORD_RAW
    RAW --> COORD_LOG_SQRT
    RAW --> COORD_LOG_VAR

    COORD_RAW --> INTERP
    COORD_LOG_SQRT --> INTERP
    COORD_LOG_VAR --> INTERP

    INTERP --> RESULT

    style RAW fill:#e3f2fd,stroke:#333,stroke-width:2px,color:#000
    style COORD_RAW fill:#f3e5f5,stroke:#333,stroke-width:2px,color:#000
    style COORD_LOG_SQRT fill:#fff3e0,stroke:#333,stroke-width:2px,color:#000
    style COORD_LOG_VAR fill:#ffe0b2,stroke:#333,stroke-width:2px,color:#000
    style INTERP fill:#e8f5e9,stroke:#333,stroke-width:2px,color:#000
    style RESULT fill:#c8e6c9,stroke:#333,stroke-width:2px,color:#000
```

**Why coordinate transforms?**
- **Log-moneyness**: ln(m) spreads out ATM region where most trading occurs
- **Square-root time**: √τ linearizes time decay near expiry
- **Better interpolation**: Transformed coordinates are more linear, reducing interpolation error

**Complexity**:
- Precomputation: O(n_τ × n_σ × n_r × n_m) cubic spline setups
- Query time: O(n_τ × n_σ × n_r) spline evaluations per query
- Space: O(n_τ × n_σ × n_r × n_m) for spline coefficients (4x price array size)

**Key Features**:
- C² continuous (smooth second derivatives)
- Accurate Greeks (gamma, vega-convexity, etc.)
- Requires ≥2 points in each dimension

### Performance Characteristics

**Query Performance**:
- **IV Surface (2D)**: <100ns per query
- **Price Table (4D)**: ~500ns per query
- **FDM (American option)**: 21.7ms per query
- **Speedup**: 40,000x faster than FDM

**Throughput**:
- Single-threaded: >2M prices/second (vs 46/sec for FDM)
- Memory-bandwidth limited (not compute-bound)

**Accuracy**:
- On-grid points: Exact (machine precision)
- Off-grid points: Typically <0.5% relative error for smooth functions
- **Delta**: Accurate (first derivatives from cubic splines)
- **Gamma**: Accurate (second derivatives continuous, C² property)
- **Vega/Theta/Rho**: Accurate via finite differences on interpolated values

**Advantages of Cubic**:
- C² continuous interpolation (smooth second derivatives)
- Accurate Greeks without storing separate tables
- Better approximation of smooth functions between grid points

### Integration with Existing Components

The interpolation engine **complements** rather than replaces the FDM solver:

1. **Pre-computation**: Uses `american_option_price_batch()` with OpenMP to populate price tables
2. **Runtime**: Queries use fast interpolation; falls back to FDM for out-of-range parameters
3. **Workflow**:
   - **Offline**: Pre-compute tables during downtime (minutes to hours)
   - **Online**: Fast lookups during trading (sub-microsecond)

### Design Rationale

**Strategy Pattern Benefits**:
- Runtime algorithm selection (no recompilation)
- Easy to benchmark different strategies
- Users can implement custom interpolation algorithms
- Extensible for future enhancements (cubic splines, RBF, etc.)

**Hybrid Approach** (IV Surface + Price Table):
- IV surfaces (2D): Extremely fast, tiny memory, good for market data fitting
- Price tables (4D/5D): Direct pricing given IV, includes American options
- Both: Maximum flexibility for different use cases

For complete design rationale and implementation roadmap, see `docs/notes/INTERPOLATION_ENGINE_DESIGN.md`.

---

## Integration Architecture

### Workflow 1: American IV Calculation

```
Market Price + Option Parameters
          ↓
calculate_iv_simple()
          ↓
    Estimate European IV using Let's Be Rational (~781ns)
    Calculate upper bound: vol_upper = european_iv × 1.5
          ↓
    Setup American pricing objective: f(σ) = American_price_FDM(σ) - market_price
          ↓
    Use Brent's method to find σ where f(σ) = 0
      Each iteration:
        - Construct American option with guessed σ
        - Solve Black-Scholes PDE via FDM (~21ms)
        - Interpolate value at spot price
        - Return error vs market price
      Typical: 5-8 iterations → ~145ms total
          ↓
    Return: IVResult (implied_vol, iterations, convergence status)
```

### Workflow 2: American Option Pricing

```
Option Parameters + Grid Configuration
          ↓
american_option_price()
          ↓
    Create spatial grid in log-price coordinates
    Setup PDE callbacks:
      - Terminal condition: payoff function
      - Spatial operator: BS PDE discretization
      - Boundary conditions: left/right BC
      - Obstacle: intrinsic value constraint
      - Optional: temporal events for dividends
          ↓
    Create PDE solver
    pde_solver_initialize()  // Apply initial condition
    pde_solver_solve()       // Time-step to maturity
          ↓
    Return: PDESolver with solution
```

### Workflow 3: Batch American Option Pricing + IV

```
Array of option parameters
          ↓
american_option_price_batch()  [OpenMP parallel for]
          ↓
    Each thread:
      - Solves American option PDE
      - Returns PDESolver
          ↓
Market prices for options
          ↓
Parallel IV calculation
          ↓
    For each option:
      - Get European BS price at spot
      - Use Brent to find IV
          ↓
Returns: Array of implied volatilities
```

---

## Data Flow Diagram

```mermaid
graph TD
    PARAMS["Option Parameters<br/>(S, K, T, r, market_price)"]

    LBR["Let's Be Rational<br/>(European IV Estimation)"]

    BOUNDS["Upper Bound<br/>vol_upper = euro_iv × 1.5"]

    AMERICAN["American Pricing<br/>(FDM PDE Solve)"]

    SPLINE["Cubic Spline<br/>Interpolation"]

    VALUE["Option Value at Spot"]

    MARKET["Market Price<br/>(from market)"]

    OBJECTIVE["Objective Function<br/>f(σ) = price(σ) - market"]

    BRENT["Brent Root Finder<br/>(5-8 iterations)"]

    IV["American Implied Volatility"]

    PARAMS --> LBR
    LBR -->|"~781ns"| BOUNDS
    BOUNDS --> BRENT

    PARAMS --> AMERICAN
    AMERICAN -->|"~21ms<br/>Option Value<br/>at All Spots"| SPLINE
    SPLINE --> VALUE
    VALUE --> OBJECTIVE

    MARKET --> OBJECTIVE
    OBJECTIVE --> BRENT

    BRENT -->|"Each iteration<br/>calls FDM"| AMERICAN
    BRENT -->|"~145ms total"| IV

    style PARAMS fill:#e3f2fd,stroke:#333,stroke-width:2px,color:#000
    style LBR fill:#e8f5e9,stroke:#333,stroke-width:2px,color:#000
    style BOUNDS fill:#f3e5f5,stroke:#333,stroke-width:2px,color:#000
    style AMERICAN fill:#fff3e0,stroke:#333,stroke-width:2px,color:#000
    style SPLINE fill:#e8f5e9,stroke:#333,stroke-width:2px,color:#000
    style VALUE fill:#fce4ec,stroke:#333,stroke-width:2px,color:#000
    style MARKET fill:#ffebee,stroke:#333,stroke-width:2px,color:#000
    style OBJECTIVE fill:#fff9c4,stroke:#333,stroke-width:2px,color:#000
    style BRENT fill:#fff9c4,stroke:#333,stroke-width:2px,color:#000
    style IV fill:#c8e6c9,stroke:#333,stroke-width:2px,color:#000
```

---

## Performance Characteristics

### American Implied Volatility Calculation

| Operation | Time | Notes |
|----------|------|-------|
| Let's Be Rational (bound) | ~781ns | European IV estimation |
| American option pricing (FDM) | ~21.7ms | Single FDM solve |
| American option pricing (table) | ~500ns | 4D cubic spline interpolation (43,400× faster) |
| American IV (FDM-based) | ~145ms | 5-8 Brent iterations × 21.7ms |
| **American IV (table-based)** | **~11.8ms** | **Newton's method with interpolation (22.5× faster)** |
| Greeks (vega, gamma) | ~500ns | Precomputed during table generation (130,000× faster) |

**Bottleneck**: FDM solver calls within Brent's method (each iteration = full PDE solve)

**Scaling**: O(iterations × FDM_time) ≈ O(8 × 21.7ms) ≈ 145ms

### American Option Pricing (Unoptimized)

| Configuration | Time/Option | Notes |
|---|---|---|
| Default (141 points, 1000 steps) | 21.7 ms | Baseline |
| Fine (201 points, 2000 steps) | 80-100 ms | Higher accuracy |
| Coarse (101 points, 500 steps) | 8-10 ms | Lower accuracy |

**Speedup factors**:
- ✅ AVX-512 SIMD: 2.2x (from 48ms to 21.5ms)
- ✅ OpenMP batch: 10-60x depending on core count

### QuantLib Comparison

From benchmark:
- **QuantLib**: 10.4 ms per option (mature, highly optimized)
- **mango-iv**: 21.6 ms per option (research, opportunities for optimization)
- **Ratio**: 2.1x (reasonable; different algorithms, grid resolution)

### Optimization Opportunities

**Completed Optimizations:**
- ✅ Batch API (OpenMP parallel processing, 11.7x speedup)
- ✅ Zero-allocation tridiagonal solver (Phase 2, ~5-10% gain)
- ✅ 64-byte alignment for SIMD
- ✅ Single workspace buffer (12n doubles)
- ✅ FMA operations in hot loops (fma() function for multiply-add)
- ✅ Restrict pointers for better compiler optimization (__restrict__ keyword)
- ✅ SIMD-optimized loops (#pragma omp simd on critical paths)

**Potential Future Optimizations:**
- Cache-blocking for large grids
- Better initial guesses for Newton iterations
- Adaptive time stepping

**Note**: Red-Black PSOR and even-odd splitting are not applicable to the current TR-BDF2 + Thomas algorithm implementation. These would require switching to iterative solvers.

---

## Architecture Strengths

### ✅ Mathematical Correctness
- Black-Scholes formula matches financial theory
- TR-BDF2 scheme is well-established
- Obstacle condition properly enforces American constraint
- Dividend handling via temporal events is mathematically sound

### ✅ Callback-Based Flexibility
- Users can implement custom PDEs
- Custom boundary conditions easily added
- Jump conditions and obstacles supported
- Temporal events for arbitrary time-based discontinuities

### ✅ Vectorization-Ready
- Array-oriented callbacks (not per-element)
- OpenMP SIMD pragmas on hot loops
- Single contiguous workspace buffer
- 64-byte aligned for SIMD operations

### ✅ Memory Efficiency
- Single workspace allocation
- No intermediate tree structures (O(n) not O(n²))
- Minimal malloc during solve loop
- Cubic splines share tridiagonal solver

### ✅ Comprehensive Validation
- 44 IV tests
- 29 American option tests
- 20+ PDE solver tests
- Benchmark comparison against QuantLib
- USDT tracing for diagnostics

### ✅ Parallel-Ready
- Batch API for multi-core processing
- No shared state in callbacks
- Thread-safe design
- Enables 60x+ speedup on multi-core machines

---

## Architecture Limitations & Known Issues

### ⚠️ American Option Implementation Issue

**Location**: `american_option.c` lines 82-86 (TODO comment)

**Issue**: Time mapping for backward parabolic PDE may not be perfectly correct
- Solver forward-integrates in time-to-maturity space
- Mathematical formulation correct, implementation needs verification
- Tests acknowledge but verify solver completes without crashing

**Impact**: Results reasonable but may have systematic bias
- Solver converges and produces monotone behavior
- Early exercise premium captured (dividend impact correct)
- Accuracy within reasonable bounds vs QuantLib

**Mitigation**: Comprehensive testing shows results are sensible, even if formulation not perfect

### 📊 Performance Characteristics

**Sequential Performance**: 2.1x slower than QuantLib (21.6ms vs 10.4ms per option)
- Different algorithms and grid resolution choices
- Research implementation with clear optimization opportunities

**Batch Performance**: Excellent parallel scaling
- 11.7x speedup at 100 options with OpenMP batch processing
- 91% parallel efficiency at 8 threads
- 2,000+ options/second sustained throughput
- Thread-safe with zero data races

**Note**: Single-option performance gap is offset by superior batch throughput on multi-core systems

---

## Usage Patterns

### Pattern 1: American IV Calculation

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
    printf("American IV: %.4f (%.1f%%)\n", result.implied_vol, result.implied_vol * 100);
    printf("Iterations: %d\n", result.iterations);
} else {
    printf("Error: %s\n", result.error);
}
```

### Pattern 2: American Option Pricing

```c
#include "src/american_option.h"

OptionData option = {
    .strike = 100.0,
    .volatility = 0.2,
    .risk_free_rate = 0.05,
    .time_to_maturity = 1.0,
    .option_type = OPTION_PUT,
    .n_dividends = 0,
    .dividend_times = nullptr,
    .dividend_amounts = nullptr
};

AmericanOptionGrid grid = {
    .x_min = -0.7,
    .x_max = 0.7,
    .n_points = 141,
    .dt = 0.001,
    .n_steps = 1000
};

AmericanOptionResult result = american_option_price(&option, &grid);
if (result.status == 0) {
    double value = american_option_get_value_at_spot(result.solver, 100.0, 100.0);
    printf("Value: %.4f\n", value);
    american_option_free_result(&result);
}
```

### Pattern 3: Batch Processing

```c
// Create array of options
OptionData options[100];
for (size_t i = 0; i < 100; i++) {
    options[i] = { /* ... configure each option ... */ };
}

// Create array for results
AmericanOptionResult results[100] = {0};

// Batch price (OpenMP parallel)
int status = american_option_price_batch(options, &grid, 100, results);

// Process results
for (size_t i = 0; i < 100; i++) {
    if (results[i].status == 0) {
        // Use result
        american_option_free_result(&results[i]);
    }
}
```

---

## Building & Testing

```bash
# Build everything
bazel build //...

# Run all tests
bazel test //...

# Build with optimizations
bazel build -c opt //...

# Run IV tests
bazel test //tests:implied_volatility_test

# Run American option tests
bazel test //tests:american_option_test

# Run PDE solver tests
bazel test //tests:pde_solver_test

# Compare with QuantLib (requires libquantlib0-dev)
bazel build //benchmarks:quantlib_benchmark
./bazel-bin/benchmarks/quantlib_benchmark
```

---

## Summary Table

| Component | Purpose | Files | API |
|-----------|---------|-------|-----|
| **American IV** | American IV from market price | implied_volatility.{h,c} | `calculate_iv()`, `calculate_iv_simple()` |
| **Let's Be Rational** | European IV estimation (bounds) | lets_be_rational.{h,c} | `lbr_implied_volatility()` |
| **American Option** | American option pricing | american_option.{h,c} | `american_option_price()`, `american_option_price_batch()`, `american_option_free_result()` |
| **PDE Solver** | FDM time-stepping engine | pde_solver.{h,c} | `pde_solver_create()`, `pde_solver_solve()`, `pde_solver_destroy()` |
| **IV Surface** | Fast 2D IV interpolation (~100ns) | iv_surface.{h,c} | `iv_surface_create()`, `iv_surface_interpolate()` |
| **Price Table** | Fast 4D/5D price lookup (~500ns) | price_table.{h,c} | `price_table_create()`, `price_table_interpolate_4d()`, `price_table_greeks_4d()` |
| **Cubic Interpolation** | N-dimensional cubic spline interpolation | interp_cubic.{h,c}, interp_cubic_workspace.c, interp_strategy.h | `INTERP_CUBIC` strategy |
| **Brent's Method** | Root finding for IV | brent.h | `brent_find_root()` |
| **Cubic Spline** | Off-grid PDE interpolation | cubic_spline.{h,c} | `pde_spline_create()` (malloc), `pde_spline_init()` (workspace), `pde_spline_eval()` |
| **Tridiagonal Solver** | O(n) matrix solve | tridiagonal.h | `solve_tridiagonal()` |
| **USDT Tracing** | Diagnostic probes | ivcalc_trace.h | `MANGO_TRACE_*` macros |

---

## Conclusion

The mango-iv codebase implements a complete, production-ready suite for American option pricing and implied volatility calculation. The architecture prioritizes:

1. **Mathematical correctness** (Let's Be Rational, TR-BDF2, obstacle conditions, nested iteration)
2. **Flexibility** (callback-based design, custom PDEs)
3. **Vectorization** (array-oriented, SIMD-ready)
4. **Performance** (single workspace buffer, batch API, parallel-ready)
5. **Validation** (comprehensive test coverage, QuantLib comparison)

**Current Performance:**
- American option pricing (FDM): ~21.7ms per solve
- American option pricing (table): ~500ns per query (43,400× faster)
- American IV (FDM-based): ~145ms (5-8 Brent iterations)
- American IV (table-based): ~11.8ms (Newton's method, 22.5× faster)
- Greeks (vega, gamma): ~500ns (precomputed, 130,000× faster than FDM)
- Let's Be Rational (bounds): ~781ns (European IV estimation)

**Achieved Optimizations:**
- ✅ Price table pre-computation + interpolation: ~500ns per price query
- ✅ Table-based IV calculation: 22.5× speedup achieved
- ✅ Adaptive grid refinement: <1bp IV error for 95% of validation points
- ✅ Unified grid architecture: 20,000× memcpy reduction
- ✅ Enables sub-millisecond trading applications


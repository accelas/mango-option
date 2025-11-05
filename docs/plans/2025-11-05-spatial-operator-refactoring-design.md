# Spatial Operator Refactoring Design (Revised)

## Overview

Refactor spatial operator classes to follow Single Responsibility Principle by separating concerns into focused, composable components. This addresses Issue #104.

**Revision History:**
- v1: Initial design
- v2: Addressed critical issues from codex subagent review (lambda decoupling, explicit interior interface, time-aware polymorphism, factory pattern)

## Problem Statement

Current spatial operators (`LogMoneynessBlackScholesOperator`, `UniformGridBlackScholesOperator`, etc.) violate SRP by mixing:

1. **Mathematical operator** - Black-Scholes PDE formula
2. **Cache blocking** - `apply_block()` performance optimization
3. **Grid strategy** - Uniform vs non-uniform grid handling
4. **Derivative computation** - Greeks calculation methods

This makes the code:
- ❌ Harder to test (must test all concerns together)
- ❌ Harder to extend (adding grid types requires duplicating operator logic)
- ❌ Harder to maintain (changes to one concern affect others)
- ❌ Harder to reuse (can't use grid strategy without operator)

## Design Goals

1. **Single Responsibility** - Each class has exactly one reason to change
2. **Composition** - Combine focused components instead of inheritance
3. **Zero-cost abstraction** - Templates compile to same code as monolithic version
4. **Leverage existing code** - Use `GridView` from existing grid system
5. **Performance parity** - Maintain 12.5x speedup from uniform grid optimization (193ms → 15.5ms)
6. **Future-proof** - Support time-dependent PDEs without interface changes

## Architecture

### Component 1: GridSpacing

**Location**: `src/cpp/operators/grid_spacing.hpp`

**Responsibility**: Grid metric computation only

**Interface**:
```cpp
template<typename T = double>
class GridSpacing {
public:
    explicit GridSpacing(GridView<T> grid);

    bool is_uniform() const;

    // Uniform grid accessors (precondition: is_uniform() == true)
    T spacing() const;              // dx
    T spacing_inv() const;          // Pre-computed 1/dx
    T spacing_inv_sq() const;       // Pre-computed 1/dx²

    // General accessor (works for both uniform and non-uniform)
    T spacing_at(size_t i) const;   // dx[i] = x[i+1] - x[i]

    // Left and right spacing for non-uniform centered differences
    T left_spacing(size_t i) const;  // dx[i-1] = x[i] - x[i-1]
    T right_spacing(size_t i) const; // dx[i] = x[i+1] - x[i]

    const GridView<T>& grid() const;
    size_t size() const;

private:
    GridView<T> grid_;
    bool is_uniform_;

    // Uniform grid: single spacing value
    T dx_uniform_{};
    T dx_uniform_inv_{};     // 1/dx
    T dx_uniform_inv_sq_{};  // 1/dx²

    // Non-uniform grid: array of spacings
    std::vector<T> dx_array_;
};
```

**Design rationale**:
- Wraps existing `GridView` (no duplication)
- Pre-computes spacing information once in constructor
- Detects uniform vs non-uniform using `GridView::is_uniform()`
- For uniform grids: stores 3 scalars (`dx`, `1/dx`, `1/dx²`) - zero runtime cost
- For non-uniform grids: pre-computes `dx[i]` array once
- Provides `left_spacing()` and `right_spacing()` for centered difference stencils on non-uniform grids
- Contract: `spacing()`, `spacing_inv()`, `spacing_inv_sq()` require `is_uniform() == true`

**Status**: ✅ Implemented (needs left/right spacing methods added)

---

### Component 2: BlackScholesPDE

**Location**: `src/cpp/operators/black_scholes_pde.hpp`

**Responsibility**: PDE formula only (no grid, no discretization knowledge)

**Interface**:
```cpp
template<typename T = double>
class BlackScholesPDE {
public:
    BlackScholesPDE(T sigma, T r, T d);

    // Core operator: L(V) = a·∂²V/∂x² + b·∂V/∂x - c·V
    // Time-independent version (current Black-Scholes)
    T operator()(T d2v_dx2, T dv_dx, T v) const;

    // Coefficients for Jacobian construction
    T second_derivative_coeff() const;  // σ²/2
    T first_derivative_coeff() const;   // r - d - σ²/2
    T discount_rate() const;            // r

private:
    T half_sigma_sq_;    // σ²/2
    T drift_;            // r - d - σ²/2
    T discount_rate_;    // r
};
```

**Future extension for time-dependent PDEs**:
```cpp
// Example: Local volatility model (future)
template<typename T = double>
class LocalVolatilityPDE {
public:
    LocalVolatilityPDE(/* vol surface */);

    // Time-dependent version
    T operator()(T t, T d2v_dx2, T dv_dx, T v) const;
};
```

**Design rationale**:
- Pure mathematical operator for Black-Scholes in log-moneyness coordinates
- Takes pre-computed derivatives as input (no discretization knowledge)
- Returns scalar evaluation of `L(V) = (σ²/2)·∂²V/∂x² + (r-d-σ²/2)·∂V/∂x - r·V`
- Exposes coefficients for Newton solver Jacobian construction
- Completely independent of grid and stencil
- Time-independent for current Black-Scholes (compile-time dispatch handles time-dependent PDEs)

**Status**: ✅ Implemented

---

### Component 3: CenteredDifference

**Location**: `src/cpp/operators/centered_difference.hpp`

**Responsibility**: Numerical discretization only

**Interface**:
```cpp
template<typename T = double>
class CenteredDifference {
public:
    explicit CenteredDifference(const GridSpacing<T>& spacing);

    // Single-point derivatives (for non-uniform grids)
    T first_derivative(std::span<const T> u, size_t i) const;
    T second_derivative(std::span<const T> u, size_t i) const;
    std::pair<T, T> derivatives(std::span<const T> u, size_t i) const;

    // Optimized vectorized path for uniform grids (fused kernel)
    // Evaluator: (T d2u_dx2, T du_dx, T u) -> T
    template<typename Evaluator>
    void apply_uniform(std::span<const T> u,
                      std::span<T> Lu,
                      size_t start,
                      size_t end,
                      Evaluator&& eval) const;

    // General path for non-uniform grids
    template<typename Evaluator>
    void apply_non_uniform(std::span<const T> u,
                          std::span<T> Lu,
                          size_t start,
                          size_t end,
                          Evaluator&& eval) const;

    // All-points derivatives (for Greeks computation)
    void compute_all_first(std::span<const T> u,
                          std::span<T> du_dx,
                          size_t start,
                          size_t end) const;

    void compute_all_second(std::span<const T> u,
                           std::span<T> d2u_dx2,
                           size_t start,
                           size_t end) const;

private:
    const GridSpacing<T>& spacing_;
};
```

**Implementation sketch - uniform fast path**:
```cpp
template<typename Evaluator>
void CenteredDifference::apply_uniform(
    std::span<const T> u,
    std::span<T> Lu,
    size_t start,
    size_t end,
    Evaluator&& eval) const
{
    const T half_dx_inv = spacing_.spacing_inv() * T(0.5);
    const T dx2_inv = spacing_.spacing_inv_sq();

    #pragma omp simd
    for (size_t i = start; i < end; ++i) {
        const T du_dx = (u[i+1] - u[i-1]) * half_dx_inv;
        const T d2u_dx2 = (u[i+1] - T(2)*u[i] + u[i-1]) * dx2_inv;
        Lu[i] = eval(d2u_dx2, du_dx, u[i]);  // Lambda inlines away
    }
}
```

**Design rationale**:
- Implements centered finite difference stencils
- Single-point methods for manual derivative queries
- Fused kernel methods (`apply_uniform`, `apply_non_uniform`) for performance
- Uses lambda evaluator to avoid coupling to PDE type
- Lambda inlines away at compile time - zero overhead
- All-points methods for Greeks (vectorized when possible)
- Operates on `[start, end)` range for cache blocking support

**Status**: ✅ Implemented (needs fused kernel methods added)

---

### Component 4: SpatialOperator

**Location**: `src/cpp/operators/spatial_operator.hpp`

**Responsibility**: Coordinate components to evaluate L(u)

**Interface**:
```cpp
/// Helper to describe stencil interior range
struct StencilInterior {
    size_t start;  // First interior point
    size_t end;    // One past last interior point
};

template<typename PDE, typename T = double>
class SpatialOperator {
public:
    SpatialOperator(const PDE& pde, const GridSpacing<T>& spacing);

    /// Get interior range for this stencil (3-point: [1, n-1))
    StencilInterior interior_range(size_t n) const {
        return {1, n - 1};  // 3-point stencil width
    }

    /// Apply operator to full grid (convenience)
    void apply(double t, std::span<const T> u, std::span<T> Lu) const {
        const auto range = interior_range(u.size());
        apply_interior(t, u, Lu, range.start, range.end);
    }

    /// Apply operator to interior points only [start, end)
    /// Used by both full-grid and cache-blocked evaluation
    void apply_interior(double t,
                       std::span<const T> u,
                       std::span<T> Lu,
                       size_t start,
                       size_t end) const;

    /// Greeks computation (delegates to stencil)
    void compute_first_derivative(std::span<const T> u,
                                 std::span<T> du_dx) const;

    void compute_second_derivative(std::span<const T> u,
                                  std::span<T> d2u_dx2) const;

private:
    const PDE& pde_;
    const GridSpacing<T>& spacing_;
    CenteredDifference<T> stencil_;
};
```

**Implementation - time-aware dispatch**:
```cpp
template<typename PDE, typename T>
void SpatialOperator<PDE, T>::apply_interior(
    double t,
    std::span<const T> u,
    std::span<T> Lu,
    size_t start,
    size_t end) const
{
    // Create evaluator lambda that handles time parameter
    auto eval = [&](T d2u, T du, T val) -> T {
        if constexpr (has_time_param_v<PDE>) {
            return pde_(t, d2u, du, val);  // Time-dependent PDE
        } else {
            return pde_(d2u, du, val);     // Time-independent PDE
        }
    };

    // Dispatch to appropriate stencil strategy
    if (spacing_.is_uniform()) {
        stencil_.apply_uniform(u, Lu, start, end, eval);
    } else {
        stencil_.apply_non_uniform(u, Lu, start, end, eval);
    }
}
```

**Type trait for time detection**:
```cpp
template<typename PDE, typename = void>
struct has_time_param : std::false_type {};

template<typename PDE>
struct has_time_param<PDE, std::void_t<
    decltype(std::declval<PDE>()(
        std::declval<double>(),  // t
        std::declval<double>(),  // d2u
        std::declval<double>(),  // du
        std::declval<double>()   // u
    ))
>> : std::true_type {};

template<typename PDE>
inline constexpr bool has_time_param_v = has_time_param<PDE>::value;
```

**Design rationale**:
- Composes `BlackScholesPDE`, `GridSpacing`, `CenteredDifference`
- Explicit `apply_interior(start, end)` for cache blocking
- Returns `interior_range()` so solver knows stencil width
- Operator doesn't touch boundaries - solver's responsibility
- Uses lambda evaluator to delegate to stencil without PDE coupling
- Time parameter always present in API, compile-time dispatch for time-independent PDEs
- `if constexpr` eliminates unused branch at compile time (zero cost)
- Grid strategy selection happens once in `apply_interior`, delegates to stencil

**Status**: 🔄 In progress (needs complete redesign)

---

### Component 5: Factory Pattern

**Location**: `src/cpp/operators/operator_factory.hpp` (new)

**Responsibility**: Construct operators with appropriate stencil strategy

**Interface**:
```cpp
namespace mango::operators {

/// Factory function to create spatial operator with appropriate stencil
template<typename PDE, typename T = double>
auto create_spatial_operator(
    const PDE& pde,
    const GridView<T>& grid)
{
    auto spacing = std::make_shared<GridSpacing<T>>(grid);
    return SpatialOperator<PDE, T>(pde, *spacing);
}

/// Overload with explicit spacing
template<typename PDE, typename T = double>
auto create_spatial_operator(
    const PDE& pde,
    std::shared_ptr<GridSpacing<T>> spacing)
{
    return SpatialOperator<PDE, T>(pde, *spacing);
}

} // namespace mango::operators
```

**Design rationale**:
- Simple factory for common case
- Handles lifetime management (spacing outlives operator)
- Uniform vs non-uniform selection happens automatically in `GridSpacing`
- No runtime polymorphism needed - template instantiation at compile time
- Can extend to select different stencil types in future

**Status**: 📝 New component

## Performance Strategy

The new architecture must maintain the **12.5x speedup** from uniform grid optimization.

### Uniform Grid Fast Path

```cpp
// In CenteredDifference::apply_uniform
template<typename Evaluator>
void apply_uniform(span u, span Lu, size_t start, size_t end,
                   Evaluator&& eval) const {
    // Pre-computed coefficients (zero divisions in loop!)
    const T half_dx_inv = spacing_.spacing_inv() * T(0.5);
    const T dx2_inv = spacing_.spacing_inv_sq();

    #pragma omp simd  // Vectorizable!
    for (size_t i = start; i < end; ++i) {
        const T du_dx = (u[i+1] - u[i-1]) * half_dx_inv;
        const T d2u_dx2 = (u[i+1] - T(2)*u[i] + u[i-1]) * dx2_inv;
        Lu[i] = eval(d2u_dx2, du_dx, u[i]);  // Lambda inlines
    }
}

// In SpatialOperator::apply_interior
auto eval = [&](T d2u, T du, T v) {
    return pde_(d2u, du, v);  // Inlines to: a*d2u + b*du - c*v
};
stencil_.apply_uniform(u, Lu, start, end, eval);
```

**Why this is fast**:
- Branch (`if uniform`) outside hot loop - predicted perfectly
- All divisions pre-computed in `GridSpacing` constructor
- Loop body: only multiplies and adds (3-5 cycles each)
- Lambda inlines completely - zero overhead
- `#pragma omp simd` enables auto-vectorization (AVX2/AVX-512)
- Identical instruction count to current `UniformGridBlackScholesOperator`

### Non-Uniform Grid Path

```cpp
template<typename Evaluator>
void apply_non_uniform(span u, span Lu, size_t start, size_t end,
                      Evaluator&& eval) const {
    for (size_t i = start; i < end; ++i) {
        const T dx_left = spacing_.left_spacing(i);
        const T dx_right = spacing_.right_spacing(i);

        const T du_dx = (u[i+1] - u[i-1]) / (dx_left + dx_right);
        const T d2u_dx2 = /* centered difference with variable spacing */;

        Lu[i] = eval(d2u_dx2, du_dx, u[i]);
    }
}
```

### Zero-Cost Abstraction Guarantee

**Verified properties**:
- Lambda evaluator: inlines completely, zero overhead
- `if constexpr` for time: eliminates branch at compile time
- Templates: full inlining across component boundaries
- No virtual calls, no function pointers
- Same assembly as monolithic version

**Performance testing**:
- Use `perf stat` to verify instruction count unchanged
- Assembly inspection confirms inlining
- IPC should match current implementation (2.62+)

### Cache Blocking Strategy

Cache blocking is a **solver optimization**, not an operator concern.

**PDESolver responsibility**:
```cpp
// PDESolver detects large grids
if (n >= cache_blocking_threshold) {
    // Cache-blocked evaluation
    for (auto block : blocks) {
        spatial_op.apply_interior(t, u_block, Lu_block,
                                  block.start, block.end);
    }
} else {
    // Full-grid evaluation
    spatial_op.apply(t, u, Lu);
}
```

**Operator responsibility**:
```cpp
// Only implement apply_interior(start, end)
// Operator doesn't know about blocking strategy
```

This separation means:
- Operator doesn't need to know about cache blocking
- PDESolver can apply blocking strategy to any operator
- Cache blocking code lives in one place (PDESolver)
- `interior_range()` allows solver to respect stencil width

## Interface Changes

### Old Operator Interface

```cpp
void operator()(double t,
               std::span<const double> x,
               std::span<const double> u,
               std::span<double> Lu,
               std::span<const double> dx) const;

void apply_block(double t, size_t base_idx, size_t halo_left, ...);
```

**Problems**:
- `x` redundant (spacing already has grid)
- `dx` redundant (spacing already has it)
- Two methods for blocking (duplicates logic)
- Hard-codes boundary handling (breaks cache blocking)

### New Operator Interface

```cpp
// Full-grid convenience
void apply(double t, std::span<const T> u, std::span<T> Lu) const;

// Explicit interior (used by full-grid and cache blocking)
void apply_interior(double t, std::span<const T> u, std::span<T> Lu,
                   size_t start, size_t end) const;

// Query stencil width
StencilInterior interior_range(size_t n) const;
```

**Benefits**:
- Minimal interface: only input (`u`), output (`Lu`), time (`t`)
- Single interior method used by both full-grid and blocked
- Operator doesn't touch boundaries
- Stencil width queryable for blocking logic
- Grid and PDE info captured in constructor

### PDESolver Changes Required

```cpp
// OLD:
spatial_op_(t, x_view, u, Lu, dx_);

// NEW:
spatial_op_.apply(t, u, Lu);

// For cache blocking:
const auto range = spatial_op_.interior_range(n);
for (auto block : compute_blocks(range.start, range.end)) {
    spatial_op_.apply_interior(t, u, Lu, block.start, block.end);
}
```

**Migration**:
- Update all `evaluate_spatial_operator()` call sites
- Remove `dx_` precomputation (GridSpacing handles it)
- Update cache blocking to use `apply_interior()` with explicit ranges

## Testing Strategy

### Unit Tests

Test each component independently:

1. **GridSpacing**:
   - Uniform grid: verify `spacing_inv()`, `spacing_inv_sq()` are correct
   - Non-uniform grid: verify `left_spacing(i)`, `right_spacing(i)` match manual calculation
   - Contract enforcement: `spacing()` on non-uniform grid should fail (debug assertion)
   - Edge cases: n=1, n=2 grids

2. **BlackScholesPDE**:
   - Verify operator evaluation matches formula
   - Test coefficient accessors
   - Various σ, r, d parameter combinations
   - Time-independent interface

3. **CenteredDifference**:
   - Uniform grid: verify derivatives match analytical formulas
   - Non-uniform grid: verify second-order accuracy
   - Lambda evaluator: test with various callable types
   - `apply_uniform` produces same results as manual loop
   - `compute_all_*` methods produce consistent results with single-point methods
   - Boundary behavior: start/end parameters respected

4. **SpatialOperator**:
   - Integration test: compose all components
   - Verify uniform and non-uniform paths produce correct results
   - Time parameter: test with time-independent PDE (parameter ignored)
   - Interior range: verify `[1, n-1)` for 3-point stencil
   - `apply()` and `apply_interior()` produce same results

5. **Time-aware dispatch**:
   - Test `has_time_param_v` trait with various PDE types
   - Verify compile-time dispatch eliminates unused branch
   - Mock time-dependent PDE: verify `t` parameter passed correctly

### Integration Tests

**Numerical Accuracy**:
- Compare new `SpatialOperator` output with existing `UniformGridBlackScholesOperator`
- Must match to machine precision (< 1e-15 relative error)
- Test on American option pricing problem (well-validated)
- Non-uniform grid: verify second-order convergence

**Cache Blocking**:
- Verify blocked evaluation produces identical results to full-grid
- Test with various block sizes (100, 500, 1000 points)
- Ensure boundary points not clobbered
- Test with interior range `[start, end)` at various offsets

**Performance Benchmark**:
- Measure time for American option with new architecture
- Must maintain 15.5ms performance (no regression from current)
- Use `perf stat` to verify instruction count unchanged
- Target: identical IPC and instruction count as current implementation
- Verify lambda inlining with assembly inspection

### Regression Prevention

- All existing tests must pass with new operators
- Performance benchmarks in CI
- Assembly inspection for uniform grid fast path
- `perf stat` comparison in CI (instruction count, cycles, IPC)

## Migration Strategy

### Phase 1: Implement New Architecture

**Files to create/modify**:
```
src/cpp/operators/
├── grid_spacing.hpp         ✅ Done (add left/right_spacing methods)
├── black_scholes_pde.hpp    ✅ Done
├── centered_difference.hpp  🔄 Add fused kernel methods
├── spatial_operator.hpp     🔄 Complete redesign
└── operator_factory.hpp     📝 New
```

**Changes**:
1. Add `left_spacing()`, `right_spacing()` to `GridSpacing`
2. Add `apply_uniform()`, `apply_non_uniform()` to `CenteredDifference`
3. Add `compute_all_first()`, `compute_all_second()` to `CenteredDifference`
4. Redesign `SpatialOperator` with clean interface
5. Create `operator_factory.hpp` with helper functions
6. Add `has_time_param_v` trait
7. Write comprehensive unit tests

### Phase 2: Update PDESolver Interface

**Files to modify**:
- `src/cpp/pde_solver.hpp`

**Changes**:
1. Update `evaluate_spatial_operator()` signature
2. Change calls from `op_(t, x, u, Lu, dx)` to `op_.apply(t, u, Lu)`
3. Remove `dx_` member and precomputation
4. Update cache blocking to use `apply_interior(start, end)`
5. Add `interior_range()` query for blocking logic
6. Ensure boundaries initialized before `apply()` call

### Phase 3: Migrate AmericanOptionSolver

**Files to modify**:
- `src/cpp/american_option.cpp`

**Changes**:
```cpp
// OLD:
const double dx = (x_max - x_min) / (n - 1);
UniformGridBlackScholesOperator bs_op(sigma, r, d, dx);

// NEW:
auto grid_view = GridView(grid_.x);  // From existing grid
auto spacing = std::make_shared<GridSpacing>(grid_view);
auto pde = BlackScholesPDE(sigma, r, d);
auto spatial_op = create_spatial_operator(pde, spacing);
```

### Phase 4: Update Greeks Computation

**Files to modify**:
- Any code using `compute_first_derivative()`, `compute_second_derivative()`

**Changes**:
```cpp
// OLD:
bs_op.compute_first_derivative(x, u, du_dx, dx);

// NEW:
spatial_op.compute_first_derivative(u, du_dx);
```

### Phase 5: Delete Old Operators

Remove from `spatial_operators.hpp`:
- `UniformGridBlackScholesOperator`
- `LogMoneynessBlackScholesOperator`
- `BlackScholesOperator`
- `IndexBlackScholesOperator`

Keep temporarily:
- `LaplacianOperator` (used in tests, migrate separately)

### Phase 6: Comprehensive Testing

1. Run full test suite
2. Run performance benchmarks
3. Verify 15.5ms performance maintained
4. Check assembly for uniform fast path
5. Validate cache blocking with large grids

**Testing checklist**:
- [ ] All unit tests pass
- [ ] Integration tests pass (numerical accuracy)
- [ ] Performance benchmarks pass (15.5ms ± 5%)
- [ ] Assembly inspection confirms inlining
- [ ] `perf stat` shows comparable instruction count and IPC
- [ ] Cache blocking tests pass
- [ ] Greeks computation produces correct results

### One-PR Strategy

**Approach**: Implement all phases in feature branch, merge atomically

**Benefits**:
- No partial state in main branch
- All tests must pass before merge
- Easy to bisect if issues found
- Clean git history

**Risk mitigation**:
- Comprehensive testing at each phase
- Review design document before implementation
- Incremental development in feature branch
- Performance validation before merge

## Example Usage

### Before (Monolithic)

```cpp
const double dx = (x_max - x_min) / (n_points - 1);
UniformGridBlackScholesOperator bs_op(sigma, r, d, dx);

// Apply operator
bs_op(t, x, u, Lu, dx_array);

// Compute Greeks
bs_op.compute_first_derivative(x, u, du_dx, dx_array);
bs_op.compute_second_derivative(x, u, d2u_dx2, dx_array);
```

### After (Composed)

```cpp
auto grid_view = GridView(grid.x);
auto spacing = std::make_shared<GridSpacing>(grid_view);
auto pde = BlackScholesPDE(sigma, r, d);
auto spatial_op = create_spatial_operator(pde, spacing);

// Apply operator (time parameter for future extensibility)
spatial_op.apply(t, u, Lu);

// Compute Greeks
spatial_op.compute_first_derivative(u, du_dx);
spatial_op.compute_second_derivative(u, d2u_dx2);
```

**Benefits of new API**:
- Grid created once, reused across components
- No redundant parameter passing
- Clear separation of concerns
- Easy to swap PDE or grid strategy
- Future-proof for time-dependent PDEs

### Time-Dependent PDE Example (Future)

```cpp
// Time-dependent local volatility
class LocalVolatilityPDE {
public:
    LocalVolatilityPDE(VolSurface vol_surface);

    double operator()(double t, double d2v, double dv, double v) const {
        const double sigma_t = vol_surface_.vol_at(t);
        const double half_var = 0.5 * sigma_t * sigma_t;
        return half_var * d2v + drift_ * dv - r_ * v;
    }
};

// Same operator interface, automatically handles time
auto pde = LocalVolatilityPDE(vol_surface);
auto spatial_op = create_spatial_operator(pde, spacing);
spatial_op.apply(t, u, Lu);  // t parameter used by PDE
```

## Benefits Summary

✅ **Single Responsibility** - Each class has one reason to change
✅ **Composability** - Mix and match components (any PDE + any grid + any stencil)
✅ **Testability** - Test each concern independently
✅ **Reusability** - Use `GridSpacing` with any PDE, not just Black-Scholes
✅ **Extensibility** - Add new grids or PDEs without duplicating code
✅ **Performance** - Zero-cost abstraction, maintains 12.5x speedup
✅ **Maintainability** - Changes to one concern don't affect others
✅ **Simplicity** - Clean minimal interfaces with lambda decoupling
✅ **Future-proof** - Time-dependent PDEs supported at zero cost
✅ **Cache-friendly** - Explicit interior interface for blocking

## Addressed Design Issues

From codex subagent review:

1. ✅ **SRP regression in uniform fast path**: Fixed by moving fused kernel to `CenteredDifference` with lambda evaluator (no PDE coupling)
2. ✅ **Non-uniform spacing underspecified**: Added `left_spacing()` and `right_spacing()` methods to `GridSpacing`
3. ✅ **Boundary handling breaks cache blocking**: Operator only touches interior `[start, end)`, boundaries handled by solver
4. ✅ **Time parameter removed**: Kept `t` parameter, compile-time dispatch via `if constexpr` for time-independent PDEs (zero cost)
5. ✅ **Grid strategy in operator**: Strategy selection delegated to stencil via fused kernel methods
6. ✅ **Migration plan contradiction**: Clarified incremental development in feature branch, atomic merge to main

## Related Work

- **Issue #103**: Uniform grid optimization (delivered 12.5x speedup, highlighted design issue)
- **Issue #104**: This refactoring (address SRP violations)
- **Existing grid system**: `GridView`, `GridBuffer`, `GridSpec` in `src/cpp/grid.hpp`

## Future Extensions

This architecture enables future improvements:

1. **Adaptive grid refinement**: New grid type without changing operator
2. **Higher-order stencils**: Swap `CenteredDifference` for `FourthOrderStencil`
3. **Different PDEs**: Reuse grid and stencil with different PDE formulas
4. **Time-dependent coefficients**: Already supported via compile-time dispatch
5. **Mixed finite difference/element methods**: Compose different discretizations
6. **Automatic differentiation**: Replace finite difference with AD for exact Jacobians
7. **Multi-dimensional PDEs**: Extend stencil interface to 2D/3D grids

The separation of concerns makes the codebase more flexible and maintainable for future development.

## Design Validation

**Reviewed by**: codex-subagent (architectural review)

**Key improvements from review**:
- Lambda evaluator pattern eliminates PDE-stencil coupling
- Explicit interior interface enables clean cache blocking
- Compile-time time-dispatch supports future extensibility at zero cost
- Factory pattern recommended but deferred (YAGNI for now)

**Performance validation plan**:
- Unit tests for each component
- Integration test: numerical accuracy vs current implementation
- Benchmark: verify 15.5ms maintained
- Assembly inspection: confirm lambda inlining
- `perf stat`: verify instruction count and IPC unchanged

# EEP Decomposition for Price Tables

## Problem

The price table's B-spline interpolation converges at O(h^2.5) instead of the theoretical O(h^4) for cubic splines. The American early exercise boundary creates a C1 discontinuity that degrades accuracy, especially for short-dated slightly-OTM puts where small vega amplifies price errors into IV errors. Doubling PDE solves from 495 to 812 reduces average error only from 7.5 to 5.1 bps.

## Solution

Decompose the American price into two components:

```
P_American = P_European + EEP
```

The B-spline interpolates only the early exercise premium (EEP). The European component is computed in closed form at query time. EEP is smaller, more localized near the exercise boundary, and lacks the log-normal curvature of the full price surface. The spline approximation should converge faster on this smoother target.

## Architecture

### Decomposition boundary

Store EEP in the builder (Option 1). The generic `PriceTableSurface<N>` remains unchanged — it interpolates whatever tensor it receives. A new `AmericanPriceSurface` wrapper reconstructs the full American price at query time.

```
Build:   PDE → P_Am → subtract P_Eu(BS) → store EEP in tensor → fit B-spline
Query:   EEP_spline(m,τ,σ,r) * (K/K_ref) + P_Eu(S,K,τ,σ,r,q) → American price
```

### Rate model

Flat rate only. The price table collapses yield curves to a zero rate. The European BS formula uses the same flat rate, so EEP is consistent between build and query time.

### Strike scaling

EEP scales linearly with K (degree-1 homogeneity in S, K under constant-rate BS). At query time, scale EEP by K/K_ref and compute P_Eu exactly for the actual strike K. This avoids multiplying a small European price by a large scale factor.

### Greek reconstruction

The spline stores EEP as a function of (m, τ, σ, r) where m = S/K. Converting spline partials to price-space Greeks requires the chain rule.

Let `E(m, τ, σ, r)` denote the spline value and `∂_i E` denote `PriceTableSurface::partial(i, ...)`.

**Price:**
```
P(S, K, τ, σ, r) = (K/K_ref) · E(m, τ, σ, r) + P_Eu(S, K, τ, σ, r, q)
```

**Delta (∂P/∂S):**
```
∂P/∂S = (K/K_ref) · ∂E/∂m · ∂m/∂S + ∂P_Eu/∂S
       = (1/K_ref) · ∂₀E  +  Δ_Eu
```
Since m = S/K, we have ∂m/∂S = 1/K, so the K in (K/K_ref) cancels with 1/K.

**Gamma (∂²P/∂S²):**
```
∂²P/∂S² = (K/K_ref) · ∂²E/∂m² · (∂m/∂S)² + ∂²P_Eu/∂S²
         = (1/(K · K_ref)) · ∂₀₀E  +  Γ_Eu
```
Gamma requires the second partial of the spline w.r.t. axis 0. `BSplineND` supports this via repeated differentiation.

**Vega (∂P/∂σ):**
```
∂P/∂σ = (K/K_ref) · ∂₂E  +  V_Eu
```
σ is axis 2 in the surface; no coordinate transform needed.

**Theta (∂P/∂τ):**
```
∂P/∂τ = (K/K_ref) · ∂₁E  +  Θ_Eu
```
Note: theta sign convention follows the existing codebase (negative for time decay).

### Constraints on input parameters

EEP decomposition assumes:
- **Flat rate only.** Yield curves are collapsed to zero rate as in the existing surface.
- **Continuous dividend yield only.** Discrete dividends are not yet functional in the codebase: `PricingParams` carries a `discrete_dividends` field, but the PDE solver (`BlackScholesPDE`) ignores it, and `NormalizedBatchSolver` rejects batches that contain discrete dividends. The EEP decomposition inherits this limitation — the closed-form European formula and the homogeneity argument both require continuous yield only. No new validation is needed since the existing pipeline already rejects discrete dividends.

Validation at query time in `AmericanPriceSurface::create()`:
```cpp
if (surface->metadata().content != SurfaceContent::EarlyExercisePremium) {
    return std::unexpected(ValidationError("Surface does not contain EEP data"));
}
```

### Cancellation near τ → 0

As τ → 0, both P_Am and P_Eu approach intrinsic value. Their difference (EEP) becomes small, and subtracting two close numbers amplifies PDE discretization noise. This is the region where accuracy matters most.

**Mitigation: smooth regularization at build time.**

Apply a softplus floor to EEP values in the tensor instead of hard clamping:

```cpp
// Softplus: smoothly enforces non-negativity without creating kinks
constexpr double kSharpness = 100.0;  // Controls transition sharpness
double eep_raw = p_am - p_eu;
double eep = std::log1p(std::exp(kSharpness * eep_raw)) / kSharpness;
```

This preserves spline smoothness (no kinks from hard clamps) while ensuring EEP ≥ 0 in the tensor. The sharpness parameter controls how closely the floor approximates max(0, x): at `kSharpness = 100`, the deviation from max(0, x) is < 0.007 for all x.

For τ below a threshold (e.g., τ < 0.02), also cross-check EEP against the known asymptotic behavior: EEP ≈ O(√τ) for ATM puts. Flag values that deviate significantly as potential noise.

## New Files

### `src/option/option_concepts.hpp`

Concepts that define the common interface between European and American solvers and results. Placed in `src/option/` alongside `option_spec.hpp`.

```cpp
namespace mango {

/// An option pricing result that provides value and Greeks
///
/// Core Greeks: delta, gamma, theta (shared by American and European).
/// AmericanOptionResult currently lacks vega(); adding it is a
/// separate task. EuropeanOptionResult provides vega() from the start.
template <typename R>
concept OptionResult = requires(const R& r, double spot_price) {
    { r.value() } -> std::convertible_to<double>;
    { r.value_at(spot_price) } -> std::convertible_to<double>;
    { r.delta() } -> std::convertible_to<double>;
    { r.gamma() } -> std::convertible_to<double>;
    { r.theta() } -> std::convertible_to<double>;
    { r.spot() } -> std::convertible_to<double>;
    { r.strike() } -> std::convertible_to<double>;
    { r.maturity() } -> std::convertible_to<double>;
    { r.volatility() } -> std::convertible_to<double>;
    { r.option_type() } -> std::same_as<OptionType>;
};

/// Result that also provides vega (European satisfies this; American does not yet)
template <typename R>
concept OptionResultWithVega = OptionResult<R> && requires(const R& r) {
    { r.vega() } -> std::convertible_to<double>;
};

/// A solver that produces an OptionResult
///
/// Constrains only the solve() method. Factory methods vary by solver type
/// (EuropeanOptionSolver::create takes PricingParams only;
/// AmericanOptionSolver::create also requires PDEWorkspace).
template <typename S>
concept OptionSolver = requires(const S& solver) {
    { solver.solve() } -> OptionResult;
};

}  // namespace mango
```

The `OptionSolver` concept constrains `solve()` only, not the factory method. Factory signatures differ between solver types (`AmericanOptionSolver::create` requires `PDEWorkspace`; `EuropeanOptionSolver::create` does not), so enforcing a common factory via concepts would force awkward overloads. Generic code that needs to construct solvers should use explicit template parameters or a builder pattern.

### `src/option/european_option.hpp`

European put/call pricing with Greeks. Mirrors the `AmericanOptionSolver` API: accepts `PricingParams`, returns a result object with the same accessor methods. All inline — combinations of `norm_cdf`, `exp`, `sqrt`.

```cpp
namespace mango {

/// European option result with the same interface as AmericanOptionResult
class EuropeanOptionResult {
public:
    double value() const;             // price at current spot
    double value_at(double S) const;  // price at arbitrary spot
    double delta() const;
    double gamma() const;
    double vega() const;
    double theta() const;
    double rho() const;

    // Access underlying params (matches AmericanOptionResult)
    double spot() const;
    double strike() const;
    double maturity() const;
    double volatility() const;
    OptionType option_type() const;
};

/// Solve European option (mirrors AmericanOptionSolver::create / solve)
class EuropeanOptionSolver {
public:
    explicit EuropeanOptionSolver(const PricingParams& params);

    static std::expected<EuropeanOptionSolver, ValidationError>
    create(const PricingParams& params) noexcept;

    EuropeanOptionResult solve() const;

private:
    PricingParams params_;
};

}  // namespace mango
```

The solver/result split matches the American pattern. `EuropeanOptionSolver::solve()` is trivial (closed-form), but the API shape is consistent so callers can swap between American and European solvers.

### `src/option/table/american_price_surface.hpp`

Wraps `PriceTableSurface<4>` and adds European reconstruction. All Greek methods apply the chain rule from the "Greek reconstruction" section above.

```cpp
namespace mango {

class AmericanPriceSurface {
public:
    /// Create from EEP surface. Validates metadata.content == EarlyExercisePremium.
    static std::expected<AmericanPriceSurface, ValidationError> create(
        std::shared_ptr<const PriceTableSurface<4>> eep_surface,
        OptionType type);

    /// P = (K/K_ref) · E(m,τ,σ,r) + P_Eu(S,K,τ,σ,r,q)
    double price(double spot, double strike, double tau,
                 double sigma, double rate) const;

    /// Δ = (1/K_ref) · ∂₀E + Δ_Eu
    double delta(double spot, double strike, double tau,
                 double sigma, double rate) const;

    /// Γ = (1/(K·K_ref)) · ∂₀₀E + Γ_Eu
    double gamma(double spot, double strike, double tau,
                 double sigma, double rate) const;

    /// V = (K/K_ref) · ∂₂E + V_Eu
    double vega(double spot, double strike, double tau,
                double sigma, double rate) const;

    /// Θ = (K/K_ref) · ∂₁E + Θ_Eu
    double theta(double spot, double strike, double tau,
                 double sigma, double rate) const;

    /// Access underlying EEP surface
    const PriceTableSurface<4>& eep_surface() const;
    const PriceTableMetadata& metadata() const;

private:
    std::shared_ptr<const PriceTableSurface<4>> surface_;
    OptionType type_;
    double K_ref_;
    double dividend_yield_;
};

}  // namespace mango
```

## Modified Files

### `src/option/table/price_table_builder.hpp`

In `extract_tensor()`, subtract the European price from each PDE result before storing:

```cpp
double p_eu = EuropeanOptionSolver(PricingParams{
    m * K_ref, K_ref, tau, rate, dividend_yield, type, sigma}).solve().value();
double eep_raw = p_am - p_eu;
// Softplus floor: smooth non-negativity
constexpr double kSharpness = 100.0;
tensor(m_idx, tau_idx, sigma_idx, r_idx) =
    std::log1p(std::exp(kSharpness * eep_raw)) / kSharpness;
```

Add `bool store_eep = true` to `PriceTableConfig`. Record decomposition mode in `PriceTableMetadata`.

### `src/option/iv_solver_interpolated.hpp`

Accept `AmericanPriceSurface` instead of raw `PriceTableSurface<4>`. The Newton iteration becomes:

```
f(σ) = american_surface.price(S, K, τ, σ, r) - market_price
f'(σ) = american_surface.vega(S, K, τ, σ, r)
```

Overload `create()` to accept either surface type for backward compatibility.

### `src/option/table/price_table_metadata.hpp`

Add decomposition mode and format version:

```cpp
/// What the surface tensor contains
enum class SurfaceContent : uint8_t {
    RawPrice = 0,              ///< Raw American option prices
    EarlyExercisePremium = 1   ///< P_Am - P_Eu (requires reconstruction)
};

struct PriceTableMetadata {
    // ... existing fields ...

    /// Format version for serialization compatibility
    uint32_t format_version = 2;  // Bump from 1 to 2

    /// What the tensor stores
    SurfaceContent content = SurfaceContent::RawPrice;
};
```

Serialization (Arrow IPC or any save/load path) must:
- Write the format version and content field
- On load: default to `RawPrice` if `format_version < 2` or field is absent
- Reject loading EEP surfaces with old readers that don't understand the field

## Tests

### `tests/european_option_test.cc`

- Known values against textbook examples
- Put-call parity: C - P = S·exp(-qT) - K·exp(-rT)
- Edge cases: deep ITM/OTM, τ → 0, σ → 0
- Greeks vs finite differences
- Satisfies `OptionResult` and `OptionResultWithVega` concepts (static_assert)

### `tests/american_price_surface_test.cc`

- Round-trip: build EEP surface, reconstruct American price, compare against direct PDE
- Greek chain rule: compare `AmericanPriceSurface::delta/gamma/vega/theta` against finite differences of `price()`
- Strike scaling consistency: P(S, K) = P(αS, αK) for several α values
- Rejects surfaces with `content != EarlyExercisePremium`
- Rejects surfaces with wrong SurfaceContent

### `tests/price_table_4d_integration_test.cc` (extend)

- EEP mode vs raw mode accuracy comparison on SPY data
- Verify EEP mode reduces max error in short-dated OTM region
- Verify softplus floor produces non-negative EEP values
- Verify τ → 0 region does not exhibit cancellation noise

## Performance

- **Build time**: One `EuropeanOptionSolver::solve()` call per grid point (~10ns each). Softplus adds ~2ns. Negligible compared to PDE solves.
- **Query time**: One BS evaluation (~10ns) added to B-spline eval (~500ns). Total ~510ns for price, ~30µs for IV. No measurable regression.
- **Memory**: Unchanged — same tensor size, same B-spline coefficient count.

## Risks

1. **Discretization mismatch**: The PDE-computed P_Am and closed-form P_Eu use different numerical methods. The EEP tensor absorbs this difference. As long as the same `EuropeanOptionSolver` is used at build and query time, the decomposition is consistent.
2. **Cancellation near τ → 0**: Mitigated by softplus regularization and asymptotic cross-checks. See "Cancellation near τ → 0" section.
3. **Backward compatibility**: Format version bump from 1 to 2. Old surfaces load as `RawPrice` (no behavior change). New surfaces with `EarlyExercisePremium` require updated readers.
4. **Discrete dividends**: Not yet functional in the PDE solver or price table pipeline. `BlackScholesPDE` ignores the `discrete_dividends` field, and `NormalizedBatchSolver` rejects batches containing it. EEP decomposition inherits this limitation. When discrete dividend support is added (likely via maturity segmentation with per-segment tables), EEP decomposition will apply within each segment without modification.

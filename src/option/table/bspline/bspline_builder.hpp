// SPDX-License-Identifier: MIT
#pragma once

#include "mango/math/bspline_nd_separable.hpp"
#include "mango/option/table/price_table_config.hpp"
#include "mango/option/table/price_table_axes.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/price_tensor.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/option_grid.hpp"
#include "mango/option/table/price_table_grid_estimator.hpp"
#include "mango/support/error_types.hpp"
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

namespace mango {

#ifndef NDEBUG
namespace testing {
template <size_t N> struct PriceTableBuilderAccess;
}  // namespace testing
#endif


/// Result from price table build with diagnostics
template <size_t N>
struct PriceTableResult {
    std::shared_ptr<const PriceTableSurfaceND<N>> surface = nullptr;  ///< Immutable surface
    size_t n_pde_solves = 0;                    ///< Number of PDE solves performed
    double precompute_time_seconds = 0.0;       ///< Wall-clock build time
    BSplineFittingStats<double, N> fitting_stats;  ///< B-spline fitting diagnostics
    // Failure and repair tracking
    size_t failed_pde_slices = 0;               ///< Count of (σ,r) slices where PDE failed
    size_t failed_spline_points = 0;            ///< Count of (σ,r,τ) points where spline failed
    size_t repaired_full_slices = 0;            ///< Full slices repaired via neighbor copy
    size_t repaired_partial_points = 0;         ///< Points repaired via τ-interpolation
    size_t total_slices = 0;                    ///< Total (σ,r) slices in grid
    size_t total_points = 0;                    ///< Total (σ,r,τ) points in grid
};

/// Result from tensor extraction with failure tracking
template <size_t N>
struct ExtractionResult {
    PriceTensorND<N> tensor;
    size_t total_slices;
    std::vector<size_t> failed_pde;
    std::vector<std::tuple<size_t, size_t, size_t>> failed_spline;
};

/// Statistics from failure repair
struct RepairStats {
    size_t repaired_full_slices;
    size_t repaired_partial_points;
};

/// Builder for N-dimensional price table surfaces
///
/// Orchestrates PDE solves across grid points, fits B-spline coefficients,
/// and constructs immutable PriceTableSurfaceND.
///
/// @tparam N Number of dimensions
template <size_t N>
class PriceTableBuilderND {
public:
    /// Construct builder with configuration
    /// Result type for factory methods: builder + axes pair
    using Setup = std::expected<std::pair<PriceTableBuilderND, PriceTableAxesND<N>>, PriceTableError>;

    explicit PriceTableBuilderND(PriceTableConfig config);

    /// Optional tensor transform applied between extraction and fitting.
    /// Used for EEP decomposition on the standard path.
    using TensorTransformFn = std::function<void(PriceTensorND<N>&, const PriceTableAxesND<N>&)>;

    /// Build price table surface
    ///
    /// @param axes Grid points for each dimension
    /// @param content Metadata tag for what the surface stores
    /// @param transform Optional transform applied to tensor after extraction (e.g., EEP decompose)
    /// @return PriceTableResult with surface and diagnostics, or error
    [[nodiscard]] std::expected<PriceTableResult<N>, PriceTableError>
    build(const PriceTableAxesND<N>& axes,
          SurfaceContent content = SurfaceContent::NormalizedPrice,
          TensorTransformFn transform = nullptr);

    /// When true, bypasses the τ>0 validation to allow τ=0 in the maturity grid
    void set_allow_tau_zero(bool allow) { allow_tau_zero_ = allow; }

    /// Factory from vectors (returns builder AND axes)
    ///
    /// Creates a PriceTableBuilderND and axes from explicit vectors.
    /// Sorts and deduplicates each input vector.
    /// Validates positivity for maturity, volatility, K_ref.
    /// Rates may be negative.
    ///
    /// @param log_moneyness Log-moneyness values (ln(S/K))
    /// @param maturity Time to expiration values in years (must be > 0)
    /// @param volatility Volatility values (must be > 0)
    /// @param rate Risk-free rate values (may be negative)
    /// @param K_ref Reference strike price (must be > 0)
    /// @param pde_grid PDE grid: PDEGridConfig{grid_spec, n_time} or GridAccuracyParams
    /// @param type Option type (PUT or CALL)
    /// @param dividend_yield Continuous dividend yield (default 0.0)
    /// @param max_failure_rate Maximum tolerable failure rate (default 0.0)
    /// @return Pair of (builder, axes) or error
    static Setup
    from_vectors(
        std::vector<double> log_moneyness,
        std::vector<double> maturity,
        std::vector<double> volatility,
        std::vector<double> rate,
        double K_ref,
        PDEGridSpec pde_grid = GridAccuracyParams{},
        OptionType type = OptionType::PUT,
        double dividend_yield = 0.0,
        double max_failure_rate = 0.0);

    /// Factory from strikes (auto-computes log-moneyness)
    ///
    /// Creates a PriceTableBuilderND and axes from spot and strike prices.
    /// Computes log-moneyness = ln(spot/strike), sorts ascending.
    /// Sorts and deduplicates all input vectors.
    ///
    /// @param spot Current underlying price (must be > 0)
    /// @param strikes Strike prices (must be > 0)
    /// @param maturities Time to expiration values in years (must be > 0)
    /// @param volatilities Volatility values (must be > 0)
    /// @param rates Risk-free rate values (may be negative)
    /// @param pde_grid PDE grid: PDEGridConfig{grid_spec, n_time} or GridAccuracyParams
    /// @param type Option type (PUT or CALL)
    /// @param dividend_yield Continuous dividend yield (default 0.0)
    /// @param max_failure_rate Maximum tolerable failure rate (default 0.0)
    /// @return Pair of (builder, axes) or error
    static Setup
    from_strikes(
        double spot,
        std::vector<double> strikes,
        std::vector<double> maturities,
        std::vector<double> volatilities,
        std::vector<double> rates,
        PDEGridSpec pde_grid = GridAccuracyParams{},
        OptionType type = OptionType::PUT,
        double dividend_yield = 0.0,
        double max_failure_rate = 0.0);

    /// Factory from option grid
    ///
    /// Creates a PriceTableBuilderND and axes from an OptionGrid.
    /// Extracts spot, strikes, maturities, vols, rates from grid.
    /// Uses grid.dividend_yield.
    ///
    /// @param chain Option grid data
    /// @param pde_grid PDE grid: PDEGridConfig{grid_spec, n_time} or GridAccuracyParams
    /// @param type Option type (PUT or CALL)
    /// @param max_failure_rate Maximum tolerable failure rate (default 0.0)
    /// @return Pair of (builder, axes) or error
    static Setup
    from_grid(
        const OptionGrid& chain,
        PDEGridSpec pde_grid = GridAccuracyParams{},
        OptionType type = OptionType::PUT,
        double max_failure_rate = 0.0);

    /// Factory from option grid with automatic grid estimation
    ///
    /// Creates a PriceTableBuilderND with optimal grids estimated from target accuracy.
    /// Uses curvature-based budget allocation to minimize PDE solves while achieving
    /// the specified IV error tolerance.
    ///
    /// @param chain Option grid (provides domain bounds from strikes, maturities, vols, rates)
    /// @param pde_grid PDE grid: PDEGridConfig{grid_spec, n_time} or GridAccuracyParams
    /// @param type Option type (PUT or CALL)
    /// @param accuracy Grid accuracy parameters (controls target error and point allocation)
    /// @return Pair of (builder, axes) or error
    static Setup
    from_grid_auto(
        const OptionGrid& chain,
        PDEGridSpec pde_grid = GridAccuracyParams{},
        OptionType type = OptionType::PUT,
        const PriceTableGridAccuracyParams<4>& accuracy = {});

    /// Top-level wrapper: estimate both price table grids and PDE grid from profiles
    ///
    /// Uses grid estimation for table axes (m, tau, sigma, r) and
    /// computes a PDE grid/time domain via estimate_batch_pde_grid().
    ///
    /// @param chain Option grid (provides domain bounds)
    /// @param grid_profile Accuracy profile for price table grid estimation
    /// @param pde_profile Accuracy profile for PDE grid/time domain estimation
    /// @param type Option type (PUT or CALL)
    /// @return Pair of (builder, axes) or error
    static Setup
    from_grid_auto_profile(
        const OptionGrid& chain,
        PriceTableGridProfile grid_profile = PriceTableGridProfile::High,
        GridAccuracyProfile pde_profile = GridAccuracyProfile::High,
        OptionType type = OptionType::PUT);


private:
    /// Internal result from B-spline coefficient fitting
    struct FitCoeffsResult {
        std::vector<double> coefficients;
        BSplineFittingStats<double, N> stats;
    };
    /// Generate batch of PricingParams from axes
    [[nodiscard]] std::vector<PricingParams> make_batch(
        const PriceTableAxesND<N>& axes) const;

    /// Estimate PDE grid from batch parameters using pde_accuracy config
    [[nodiscard]] std::pair<GridSpec<double>, TimeDomain> estimate_pde_grid(
        const std::vector<PricingParams>& batch,
        const PriceTableAxesND<N>& axes) const;

    /// Solve batch of options with snapshot registration
    [[nodiscard]] BatchAmericanOptionResult solve_batch(
        const std::vector<PricingParams>& batch,
        const PriceTableAxesND<N>& axes) const;

    /// Extract PriceTensorND from batch results using cubic spline interpolation
    [[nodiscard]] std::expected<ExtractionResult<N>, PriceTableError> extract_tensor(
        const BatchAmericanOptionResult& batch,
        const PriceTableAxesND<N>& axes) const;

    /// Fit B-spline coefficients from tensor
    [[nodiscard]] std::expected<FitCoeffsResult, PriceTableError> fit_coeffs(
        const PriceTensorND<N>& tensor,
        const PriceTableAxesND<N>& axes) const;

    /// Repair failed slices using neighbor interpolation
    [[nodiscard]] std::expected<RepairStats, PriceTableError> repair_failed_slices(
        PriceTensorND<N>& tensor,
        const std::vector<size_t>& failed_pde,
        const std::vector<std::tuple<size_t, size_t, size_t>>& failed_spline,
        const PriceTableAxesND<N>& axes) const;

    /// Find nearest valid neighbor in (σ,r) grid using Manhattan distance
    [[nodiscard]] std::optional<std::pair<size_t, size_t>> find_nearest_valid_neighbor(
        size_t σ_idx, size_t r_idx, size_t Nσ, size_t Nr,
        const std::vector<bool>& slice_valid) const;

    friend class AdaptiveGridBuilder;
    friend class SegmentedPriceTableBuilder;
#ifndef NDEBUG
    template <size_t M> friend struct testing::PriceTableBuilderAccess;
#endif

    PriceTableConfig config_;
    bool allow_tau_zero_ = false;
};

/// Convenience alias for the common 4D case.
using PriceTableBuilder = PriceTableBuilderND<kPriceTableDim>;

} // namespace mango

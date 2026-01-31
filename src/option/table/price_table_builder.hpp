// SPDX-License-Identifier: MIT
#pragma once

#include "src/math/bspline_nd_separable.hpp"
#include "src/option/table/price_table_config.hpp"
#include "src/option/table/price_table_axes.hpp"
#include "src/option/table/price_table_surface.hpp"
#include "src/option/table/price_tensor.hpp"
#include "src/option/american_option.hpp"
#include "src/option/american_option_batch.hpp"
#include "src/option/option_chain.hpp"
#include "src/option/table/price_table_grid_estimator.hpp"
#include "src/support/error_types.hpp"
#include <expected>
#include <memory>
#include <tuple>
#include <vector>

namespace mango {

/// Result from price table build with diagnostics
template <size_t N>
struct PriceTableResult {
    std::shared_ptr<const PriceTableSurface<N>> surface = nullptr;  ///< Immutable surface
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
    PriceTensor<N> tensor;
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
/// and constructs immutable PriceTableSurface.
///
/// @tparam N Number of dimensions
template <size_t N>
class PriceTableBuilder {
public:
    /// Construct builder with configuration
    /// Result type for factory methods: builder + axes pair
    using Setup = std::expected<std::pair<PriceTableBuilder, PriceTableAxes<N>>, PriceTableError>;

    explicit PriceTableBuilder(PriceTableConfig config);

    /// Build price table surface
    ///
    /// @param axes Grid points for each dimension
    /// @return PriceTableResult with surface and diagnostics, or error
    [[nodiscard]] std::expected<PriceTableResult<N>, PriceTableError>
    build(const PriceTableAxes<N>& axes);

    /// Factory from vectors (returns builder AND axes)
    ///
    /// Creates a PriceTableBuilder and axes from explicit vectors.
    /// Sorts and deduplicates each input vector.
    /// Validates positivity for moneyness, maturity, volatility, K_ref.
    /// Rates may be negative.
    ///
    /// @param moneyness Moneyness values (spot/strike ratios, must be > 0)
    /// @param maturity Time to expiration values in years (must be > 0)
    /// @param volatility Volatility values (must be > 0)
    /// @param rate Risk-free rate values (may be negative)
    /// @param K_ref Reference strike price (must be > 0)
    /// @param pde_grid PDE grid: ExplicitPDEGrid{grid_spec, n_time} or GridAccuracyParams
    /// @param type Option type (PUT or CALL)
    /// @param dividend_yield Continuous dividend yield (default 0.0)
    /// @param max_failure_rate Maximum tolerable failure rate (default 0.0)
    /// @return Pair of (builder, axes) or error
    static Setup
    from_vectors(
        std::vector<double> moneyness,
        std::vector<double> maturity,
        std::vector<double> volatility,
        std::vector<double> rate,
        double K_ref,
        PDEGridSpec pde_grid = ExplicitPDEGrid{},
        OptionType type = OptionType::PUT,
        double dividend_yield = 0.0,
        double max_failure_rate = 0.0,
        bool store_eep = true);

    /// Factory from strikes (auto-computes moneyness)
    ///
    /// Creates a PriceTableBuilder and axes from spot and strike prices.
    /// Computes moneyness = spot/strike, sorts ascending.
    /// Sorts and deduplicates all input vectors.
    ///
    /// @param spot Current underlying price (must be > 0)
    /// @param strikes Strike prices (must be > 0)
    /// @param maturities Time to expiration values in years (must be > 0)
    /// @param volatilities Volatility values (must be > 0)
    /// @param rates Risk-free rate values (may be negative)
    /// @param pde_grid PDE grid: ExplicitPDEGrid{grid_spec, n_time} or GridAccuracyParams
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
        PDEGridSpec pde_grid = ExplicitPDEGrid{},
        OptionType type = OptionType::PUT,
        double dividend_yield = 0.0,
        double max_failure_rate = 0.0,
        bool store_eep = true);

    /// Factory from option chain
    ///
    /// Creates a PriceTableBuilder and axes from an OptionChain.
    /// Extracts spot, strikes, maturities, vols, rates from chain.
    /// Uses chain.dividend_yield.
    ///
    /// @param chain Option chain data
    /// @param pde_grid PDE grid: ExplicitPDEGrid{grid_spec, n_time} or GridAccuracyParams
    /// @param type Option type (PUT or CALL)
    /// @param max_failure_rate Maximum tolerable failure rate (default 0.0)
    /// @return Pair of (builder, axes) or error
    static Setup
    from_chain(
        const OptionChain& chain,
        PDEGridSpec pde_grid = ExplicitPDEGrid{},
        OptionType type = OptionType::PUT,
        double max_failure_rate = 0.0);

    /// Factory from option chain with automatic grid estimation
    ///
    /// Creates a PriceTableBuilder with optimal grids estimated from target accuracy.
    /// Uses curvature-based budget allocation to minimize PDE solves while achieving
    /// the specified IV error tolerance.
    ///
    /// @param chain Option chain (provides domain bounds from strikes, maturities, vols, rates)
    /// @param pde_grid PDE grid: ExplicitPDEGrid{grid_spec, n_time} or GridAccuracyParams
    /// @param type Option type (PUT or CALL)
    /// @param accuracy Grid accuracy parameters (controls target error and point allocation)
    /// @return Pair of (builder, axes) or error
    static Setup
    from_chain_auto(
        const OptionChain& chain,
        PDEGridSpec pde_grid = ExplicitPDEGrid{},
        OptionType type = OptionType::PUT,
        const PriceTableGridAccuracyParams<4>& accuracy = {});

    /// Top-level wrapper: estimate both price table grids and PDE grid from profiles
    ///
    /// Uses grid estimation for table axes (m, tau, sigma, r) and
    /// computes a PDE grid/time domain via compute_global_grid_for_batch().
    ///
    /// @param chain Option chain (provides domain bounds)
    /// @param grid_profile Accuracy profile for price table grid estimation
    /// @param pde_profile Accuracy profile for PDE grid/time domain estimation
    /// @param type Option type (PUT or CALL)
    /// @return Pair of (builder, axes) or error
    static Setup
    from_chain_auto_profile(
        const OptionChain& chain,
        PriceTableGridProfile grid_profile = PriceTableGridProfile::High,
        GridAccuracyProfile pde_profile = GridAccuracyProfile::High,
        OptionType type = OptionType::PUT);

    /// For testing: expose make_batch method
    [[nodiscard]] std::vector<AmericanOptionParams> make_batch_for_testing(
        const PriceTableAxes<N>& axes) const {
        return make_batch(axes);
    }

    /// For testing: expose solve_batch method
    [[nodiscard]] BatchAmericanOptionResult solve_batch_for_testing(
        const std::vector<AmericanOptionParams>& batch,
        const PriceTableAxes<N>& axes) const {
        return solve_batch(batch, axes);
    }

    /// For testing: expose extract_tensor method
    [[nodiscard]] std::expected<ExtractionResult<N>, PriceTableError> extract_tensor_for_testing(
        const BatchAmericanOptionResult& batch,
        const PriceTableAxes<N>& axes) const {
        return extract_tensor(batch, axes);
    }

    /// For testing: expose fit_coeffs method
    [[nodiscard]] std::expected<std::vector<double>, PriceTableError> fit_coeffs_for_testing(
        const PriceTensor<N>& tensor,
        const PriceTableAxes<N>& axes) const {
        auto result = fit_coeffs(tensor, axes);
        if (!result.has_value()) {
            return std::unexpected(result.error());
        }
        return std::move(result.value().coefficients);
    }

    /// For testing: expose find_nearest_valid_neighbor method
    [[nodiscard]] std::optional<std::pair<size_t, size_t>> find_nearest_valid_neighbor_for_testing(
        size_t σ_idx, size_t r_idx, size_t Nσ, size_t Nr,
        const std::vector<bool>& slice_valid) const {
        return find_nearest_valid_neighbor(σ_idx, r_idx, Nσ, Nr, slice_valid);
    }

    /// For testing: expose repair_failed_slices method
    [[nodiscard]] std::expected<RepairStats, PriceTableError> repair_failed_slices_for_testing(
        PriceTensor<N>& tensor,
        const std::vector<size_t>& failed_pde,
        const std::vector<std::tuple<size_t, size_t, size_t>>& failed_spline,
        const PriceTableAxes<N>& axes) const {
        return repair_failed_slices(tensor, failed_pde, failed_spline, axes);
    }

    // =========================================================================
    // Internal API for AdaptiveGridBuilder
    // These methods expose internal functionality for incremental builds.
    // =========================================================================

    /// Internal API: generate batch of AmericanOptionParams from axes
    /// Used by AdaptiveGridBuilder for incremental builds
    [[nodiscard]] std::vector<AmericanOptionParams> make_batch_internal(
        const PriceTableAxes<N>& axes) const {
        return make_batch(axes);
    }

    /// Internal API: solve batch of options
    /// Used by AdaptiveGridBuilder for incremental builds
    [[nodiscard]] BatchAmericanOptionResult solve_batch_internal(
        const std::vector<AmericanOptionParams>& batch,
        const PriceTableAxes<N>& axes) const {
        return solve_batch(batch, axes);
    }

    /// Internal API: extract tensor from batch results
    /// Used by AdaptiveGridBuilder for incremental builds
    [[nodiscard]] std::expected<ExtractionResult<N>, PriceTableError> extract_tensor_internal(
        const BatchAmericanOptionResult& batch,
        const PriceTableAxes<N>& axes) const {
        return extract_tensor(batch, axes);
    }

    /// Internal API: fit B-spline coefficients from tensor
    /// Used by AdaptiveGridBuilder for incremental builds
    [[nodiscard]] std::expected<std::vector<double>, PriceTableError> fit_coeffs_internal(
        const PriceTensor<N>& tensor,
        const PriceTableAxes<N>& axes) const {
        auto result = fit_coeffs(tensor, axes);
        if (!result.has_value()) {
            return std::unexpected(result.error());
        }
        return std::move(result.value().coefficients);
    }

    /// Internal API: repair failed slices using neighbor interpolation
    /// Used by AdaptiveGridBuilder for incremental builds
    [[nodiscard]] std::expected<RepairStats, PriceTableError> repair_failed_slices_internal(
        PriceTensor<N>& tensor,
        const std::vector<size_t>& failed_pde,
        const std::vector<std::tuple<size_t, size_t, size_t>>& failed_spline,
        const PriceTableAxes<N>& axes) const {
        return repair_failed_slices(tensor, failed_pde, failed_spline, axes);
    }

private:
    /// Internal result from B-spline coefficient fitting
    struct FitCoeffsResult {
        std::vector<double> coefficients;
        BSplineFittingStats<double, N> stats;
    };
    /// Generate batch of AmericanOptionParams from axes
    [[nodiscard]] std::vector<AmericanOptionParams> make_batch(
        const PriceTableAxes<N>& axes) const;

    /// Estimate PDE grid from batch parameters using pde_accuracy config
    [[nodiscard]] std::pair<GridSpec<double>, TimeDomain> estimate_pde_grid(
        const std::vector<AmericanOptionParams>& batch,
        const PriceTableAxes<N>& axes) const;

    /// Solve batch of options with snapshot registration
    [[nodiscard]] BatchAmericanOptionResult solve_batch(
        const std::vector<AmericanOptionParams>& batch,
        const PriceTableAxes<N>& axes) const;

    /// Extract PriceTensor from batch results using cubic spline interpolation
    [[nodiscard]] std::expected<ExtractionResult<N>, PriceTableError> extract_tensor(
        const BatchAmericanOptionResult& batch,
        const PriceTableAxes<N>& axes) const;

    /// Fit B-spline coefficients from tensor
    [[nodiscard]] std::expected<FitCoeffsResult, PriceTableError> fit_coeffs(
        const PriceTensor<N>& tensor,
        const PriceTableAxes<N>& axes) const;

    /// Repair failed slices using neighbor interpolation
    [[nodiscard]] std::expected<RepairStats, PriceTableError> repair_failed_slices(
        PriceTensor<N>& tensor,
        const std::vector<size_t>& failed_pde,
        const std::vector<std::tuple<size_t, size_t, size_t>>& failed_spline,
        const PriceTableAxes<N>& axes) const;

    /// Find nearest valid neighbor in (σ,r) grid using Manhattan distance
    [[nodiscard]] std::optional<std::pair<size_t, size_t>> find_nearest_valid_neighbor(
        size_t σ_idx, size_t r_idx, size_t Nσ, size_t Nr,
        const std::vector<bool>& slice_valid) const;

    PriceTableConfig config_;
};

} // namespace mango

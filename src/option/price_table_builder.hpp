#pragma once

#include "src/option/price_table_config.hpp"
#include "src/option/price_table_axes.hpp"
#include "src/option/price_table_surface.hpp"
#include "src/option/price_tensor.hpp"
#include "src/option/american_option.hpp"
#include "src/option/american_option_batch.hpp"
#include "src/option/option_chain.hpp"
#include <expected>
#include <string>
#include <memory>
#include <vector>
#include <tuple>

namespace mango {

/// B-spline fitting diagnostics (extracted from BSplineNDSeparable)
struct BSplineFittingStats {
    double max_residual_axis0 = 0.0;
    double max_residual_axis1 = 0.0;
    double max_residual_axis2 = 0.0;
    double max_residual_axis3 = 0.0;
    double max_residual_overall = 0.0;

    double condition_axis0 = 0.0;
    double condition_axis1 = 0.0;
    double condition_axis2 = 0.0;
    double condition_axis3 = 0.0;
    double condition_max = 0.0;

    size_t failed_slices_axis0 = 0;
    size_t failed_slices_axis1 = 0;
    size_t failed_slices_axis2 = 0;
    size_t failed_slices_axis3 = 0;
    size_t failed_slices_total = 0;
};

/// Result from price table build with diagnostics
template <size_t N>
struct PriceTableResult {
    std::shared_ptr<const PriceTableSurface<N>> surface = nullptr;  ///< Immutable surface
    size_t n_pde_solves = 0;                    ///< Number of PDE solves performed
    double precompute_time_seconds = 0.0;       ///< Wall-clock build time
    BSplineFittingStats fitting_stats;          ///< B-spline fitting diagnostics
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
    explicit PriceTableBuilder(PriceTableConfig config);

    /// Build price table surface
    ///
    /// @param axes Grid points for each dimension
    /// @return PriceTableResult with surface and diagnostics, or error message
    [[nodiscard]] std::expected<PriceTableResult<N>, std::string>
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
    /// @param grid_spec PDE spatial grid specification
    /// @param n_time Number of time steps
    /// @param type Option type (PUT or CALL)
    /// @param dividend_yield Continuous dividend yield (default 0.0)
    /// @param max_failure_rate Maximum tolerable failure rate, 0.0 = strict, 0.1 = allow 10% (default 0.0)
    /// @return Pair of (builder, axes) or error message
    static std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, std::string>
    from_vectors(
        std::vector<double> moneyness,
        std::vector<double> maturity,
        std::vector<double> volatility,
        std::vector<double> rate,
        double K_ref,
        GridSpec<double> grid_spec,
        size_t n_time,
        OptionType type = OptionType::PUT,
        double dividend_yield = 0.0,
        double max_failure_rate = 0.0);

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
    /// @param grid_spec PDE spatial grid specification
    /// @param n_time Number of time steps
    /// @param type Option type (PUT or CALL)
    /// @param dividend_yield Continuous dividend yield (default 0.0)
    /// @param max_failure_rate Maximum tolerable failure rate, 0.0 = strict, 0.1 = allow 10% (default 0.0)
    /// @return Pair of (builder, axes) or error message
    static std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, std::string>
    from_strikes(
        double spot,
        std::vector<double> strikes,
        std::vector<double> maturities,
        std::vector<double> volatilities,
        std::vector<double> rates,
        GridSpec<double> grid_spec,
        size_t n_time,
        OptionType type = OptionType::PUT,
        double dividend_yield = 0.0,
        double max_failure_rate = 0.0);

    /// Factory from option chain
    ///
    /// Creates a PriceTableBuilder and axes from an OptionChain.
    /// Extracts spot, strikes, maturities, vols, rates from chain.
    /// Uses chain.dividend_yield.
    ///
    /// @param chain Option chain data
    /// @param grid_spec PDE spatial grid specification
    /// @param n_time Number of time steps
    /// @param type Option type (PUT or CALL)
    /// @param max_failure_rate Maximum tolerable failure rate, 0.0 = strict, 0.1 = allow 10% (default 0.0)
    /// @return Pair of (builder, axes) or error message
    static std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, std::string>
    from_chain(
        const OptionChain& chain,
        GridSpec<double> grid_spec,
        size_t n_time,
        OptionType type = OptionType::PUT,
        double max_failure_rate = 0.0);

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
    [[nodiscard]] std::expected<ExtractionResult<N>, std::string> extract_tensor_for_testing(
        const BatchAmericanOptionResult& batch,
        const PriceTableAxes<N>& axes) const {
        return extract_tensor(batch, axes);
    }

    /// For testing: expose fit_coeffs method
    [[nodiscard]] std::expected<std::vector<double>, std::string> fit_coeffs_for_testing(
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
    [[nodiscard]] std::expected<RepairStats, std::string> repair_failed_slices_for_testing(
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
        BSplineFittingStats stats;
    };
    /// Generate batch of AmericanOptionParams from axes
    [[nodiscard]] std::vector<AmericanOptionParams> make_batch(
        const PriceTableAxes<N>& axes) const;

    /// Solve batch of options with snapshot registration
    [[nodiscard]] BatchAmericanOptionResult solve_batch(
        const std::vector<AmericanOptionParams>& batch,
        const PriceTableAxes<N>& axes) const;

    /// Extract PriceTensor from batch results using cubic spline interpolation
    [[nodiscard]] std::expected<ExtractionResult<N>, std::string> extract_tensor(
        const BatchAmericanOptionResult& batch,
        const PriceTableAxes<N>& axes) const;

    /// Fit B-spline coefficients from tensor
    [[nodiscard]] std::expected<FitCoeffsResult, std::string> fit_coeffs(
        const PriceTensor<N>& tensor,
        const PriceTableAxes<N>& axes) const;

    /// Repair failed slices using neighbor interpolation
    [[nodiscard]] std::expected<RepairStats, std::string> repair_failed_slices(
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

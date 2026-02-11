// SPDX-License-Identifier: MIT
#pragma once

#include "mango/math/bspline_nd_separable.hpp"
#include "mango/math/safe_math.hpp"
#include "mango/option/grid_spec_types.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/american_option_batch.hpp"
#include "mango/option/option_grid.hpp"
#include "mango/support/aligned_allocator.hpp"
#include "mango/support/error_types.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <expected>
#include <experimental/mdspan>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace mango {

// ---------------------------------------------------------------------------
// PriceTensorND (merged from price_tensor.hpp)
// ---------------------------------------------------------------------------

/// N-dimensional tensor with aligned storage and mdspan view
///
/// Uses AlignedVector (64-byte aligned) for SIMD-friendly memory layout.
/// The tensor owns its storage via shared_ptr for safe sharing.
///
/// @tparam N Number of dimensions
template <size_t N>
struct PriceTensorND {
    std::shared_ptr<AlignedVector<double>> storage;  ///< Owns the memory
    std::experimental::mdspan<double, std::experimental::dextents<size_t, N>> view;  ///< Type-safe view

    /// Create tensor with given shape
    ///
    /// Allocated memory is uninitialized. Caller must initialize all elements
    /// before use. The tensor owns a shared reference to the storage, keeping
    /// memory alive until all references are destroyed.
    ///
    /// @param shape Number of elements per dimension
    /// @return PriceTensorND on success, or error message on failure
    ///         Error conditions:
    ///         - Shape overflow (product of dimensions exceeds SIZE_MAX)
    [[nodiscard]] static std::expected<PriceTensorND, std::string>
    create(std::array<size_t, N> shape) {
        // Calculate total elements with overflow check
        auto total_result = safe_product(shape);
        if (!total_result.has_value()) {
            return std::unexpected("Tensor shape overflow: product of dimensions exceeds SIZE_MAX");
        }
        size_t total = total_result.value();

        // Allocate aligned storage
        auto storage_ptr = std::make_shared<AlignedVector<double>>(total);

        // Create mdspan view
        PriceTensorND tensor;
        tensor.storage = storage_ptr;

        // Construct mdspan with dextents
        std::experimental::dextents<size_t, N> extents;
        if constexpr (N == 1) {
            extents = std::experimental::dextents<size_t, 1>(shape[0]);
        } else if constexpr (N == 2) {
            extents = std::experimental::dextents<size_t, 2>(shape[0], shape[1]);
        } else if constexpr (N == 3) {
            extents = std::experimental::dextents<size_t, 3>(shape[0], shape[1], shape[2]);
        } else if constexpr (N == 4) {
            extents = std::experimental::dextents<size_t, 4>(shape[0], shape[1], shape[2], shape[3]);
        } else if constexpr (N == 5) {
            extents = std::experimental::dextents<size_t, 5>(shape[0], shape[1], shape[2], shape[3], shape[4]);
        } else {
            static_assert(N <= 5, "PriceTensorND supports up to 5 dimensions");
        }

        tensor.view = std::experimental::mdspan<double, std::experimental::dextents<size_t, N>>(
            storage_ptr->data(), extents);

        return tensor;
    }
};

/// Convenience alias for the common 4D case.
using PriceTensor = PriceTensorND<kPriceTableDim>;

// ---------------------------------------------------------------------------
// PriceTableConfig (merged from price_table_config.hpp)
// ---------------------------------------------------------------------------

/// Configuration for price table pre-computation
struct PriceTableConfig {
    OptionType option_type = OptionType::PUT;  ///< Option type (call/put)
    double K_ref = 100.0;                      ///< Reference strike price for normalization
    PDEGridSpec pde_grid = GridAccuracyParams{};  ///< PDE grid: explicit or auto-estimated
    DividendSpec dividends;                    ///< Continuous yield + discrete schedule
    double max_failure_rate = 0.0;             ///< Maximum tolerable failure rate: 0.0 = strict, 0.1 = allow 10%
};

/// Validate PriceTableConfig fields
/// @param config Configuration to validate
/// @return Error message if invalid, nullopt if valid
inline std::optional<std::string> validate_config(const PriceTableConfig& config) {
    if (config.max_failure_rate < 0.0 || config.max_failure_rate > 1.0) {
        return "max_failure_rate must be in [0.0, 1.0], got " +
               std::to_string(config.max_failure_rate);
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// PriceTableGridEstimator (merged from price_table_grid_estimator.hpp)
// ---------------------------------------------------------------------------

enum class PriceTableGridProfile {
    Low,
    Medium,
    High,
    Ultra
};

/**
 * @brief Grid estimation accuracy parameters for N-dimensional price table
 *
 * Controls the tradeoff between accuracy and computation cost.
 * The estimator uses curvature-based weights to allocate grid points
 * where the price surface has highest variation.
 *
 * @tparam N Number of dimensions
 */
template <size_t N>
struct PriceTableGridAccuracyParams {
    /// Target IV error in absolute terms (default: 10 bps = 0.001)
    double target_iv_error = 0.001;

    /// Minimum points per dimension (B-spline requires >= 4)
    size_t min_points = 4;

    /// Maximum points per dimension (cost control)
    size_t max_points = 50;

    /// Curvature weights for budget allocation (dimension-specific)
    std::array<double, N> curvature_weights = {};

    /// Scale factor calibrated from benchmark data
    double scale_factor = 1e-6;
};

/**
 * @brief Specialization for 4D price table [m, tau, sigma, r]
 */
template <>
struct PriceTableGridAccuracyParams<4> {
    double target_iv_error = 0.001;
    size_t min_points = 4;
    size_t max_points = 50;
    std::array<double, 4> curvature_weights = {1.0, 1.0, 1.5, 0.6};  // [m, tau, sigma, r]
    double scale_factor = 2.0;  // Calibrated from benchmark: 12^4 * 0.0001 ~ 2
};

/**
 * @brief Result of N-dimensional grid estimation
 *
 * @tparam N Number of dimensions
 */
template <size_t N>
struct PriceTableGridEstimate {
    std::array<std::vector<double>, N> grids;  ///< Grid vectors for each dimension
    size_t estimated_pde_solves = 0;           ///< Estimated computational cost
};

/**
 * @brief Specialization for 4D price table with named accessors
 */
template <>
struct PriceTableGridEstimate<4> {
    std::array<std::vector<double>, 4> grids;  ///< Grid vectors [m, tau, sigma, r]
    size_t estimated_pde_solves = 0;           ///< n_vol * n_rate (PDE solves per slice)

    /// Named accessors for clarity
    std::vector<double>& moneyness_grid() { return grids[0]; }
    std::vector<double>& maturity_grid() { return grids[1]; }
    std::vector<double>& volatility_grid() { return grids[2]; }
    std::vector<double>& rate_grid() { return grids[3]; }

    const std::vector<double>& moneyness_grid() const { return grids[0]; }
    const std::vector<double>& maturity_grid() const { return grids[1]; }
    const std::vector<double>& volatility_grid() const { return grids[2]; }
    const std::vector<double>& rate_grid() const { return grids[3]; }
};

namespace detail {

/// Generate uniform grid
inline std::vector<double> uniform_grid(double min_val, double max_val, size_t n) {
    std::vector<double> grid(n);
    for (size_t i = 0; i < n; ++i) {
        grid[i] = min_val + (max_val - min_val) * static_cast<double>(i) / static_cast<double>(n - 1);
    }
    return grid;
}

/// Generate log-uniform grid (uniform in log-space)
inline std::vector<double> log_uniform_grid(double min_val, double max_val, size_t n) {
    std::vector<double> grid(n);
    double log_min = std::log(min_val);
    double log_max = std::log(max_val);
    for (size_t i = 0; i < n; ++i) {
        double log_val = log_min + (log_max - log_min) * static_cast<double>(i) / static_cast<double>(n - 1);
        grid[i] = std::exp(log_val);
    }
    return grid;
}

/// Generate sqrt-uniform grid (uniform in sqrt-space, concentrates near min)
inline std::vector<double> sqrt_uniform_grid(double min_val, double max_val, size_t n) {
    std::vector<double> grid(n);
    double sqrt_min = std::sqrt(min_val);
    double sqrt_max = std::sqrt(max_val);
    for (size_t i = 0; i < n; ++i) {
        double sqrt_val = sqrt_min + (sqrt_max - sqrt_min) * static_cast<double>(i) / static_cast<double>(n - 1);
        grid[i] = sqrt_val * sqrt_val;
    }
    return grid;
}

}  // namespace detail

/**
 * @brief Estimate optimal grid for 4D price table based on target accuracy
 */
inline PriceTableGridEstimate<4> estimate_grid_for_price_table(
    double m_min, double m_max,
    double tau_min, double tau_max,
    double sigma_min, double sigma_max,
    double r_min, double r_max,
    const PriceTableGridAccuracyParams<4>& params = {})
{
    double base_points = std::pow(params.scale_factor / params.target_iv_error, 0.25);

    auto clamp_points = [&](double weighted) -> size_t {
        size_t n = static_cast<size_t>(std::ceil(base_points * weighted));
        return std::clamp(n, params.min_points, params.max_points);
    };

    size_t n_m = clamp_points(params.curvature_weights[0]);
    size_t n_tau = clamp_points(params.curvature_weights[1]);
    size_t n_sigma = clamp_points(params.curvature_weights[2]);
    size_t n_rate = clamp_points(params.curvature_weights[3]);

    PriceTableGridEstimate<4> estimate;
    estimate.grids[0] = detail::uniform_grid(m_min, m_max, n_m);
    estimate.grids[1] = detail::sqrt_uniform_grid(tau_min, tau_max, n_tau);
    estimate.grids[2] = detail::uniform_grid(sigma_min, sigma_max, n_sigma);
    estimate.grids[3] = detail::uniform_grid(r_min, r_max, n_rate);
    estimate.estimated_pde_solves = n_sigma * n_rate;

    return estimate;
}

/**
 * @brief Estimate grid from domain bounds extracted from option chain
 */
inline PriceTableGridEstimate<4> estimate_grid_from_grid_bounds(
    const std::vector<double>& strikes,
    double spot,
    const std::vector<double>& maturities,
    const std::vector<double>& vols,
    const std::vector<double>& rates,
    const PriceTableGridAccuracyParams<4>& params = {})
{
    if (strikes.empty() || maturities.empty() || vols.empty() || rates.empty()) {
        return PriceTableGridEstimate<4>{};
    }
    if (spot <= 0.0) {
        return PriceTableGridEstimate<4>{};
    }

    double log_m_min = std::log(spot / *std::max_element(strikes.begin(), strikes.end()));
    double log_m_max = std::log(spot / *std::min_element(strikes.begin(), strikes.end()));

    if (log_m_min > log_m_max) std::swap(log_m_min, log_m_max);

    double pad = 0.01 * (log_m_max - log_m_min);
    if (pad < 0.01) pad = 0.01;
    log_m_min -= pad;
    log_m_max += pad;

    auto [tau_min_it, tau_max_it] = std::minmax_element(maturities.begin(), maturities.end());
    auto [sigma_min_it, sigma_max_it] = std::minmax_element(vols.begin(), vols.end());
    auto [r_min_it, r_max_it] = std::minmax_element(rates.begin(), rates.end());

    double tau_min = *tau_min_it * 0.9;
    double tau_max = *tau_max_it * 1.1;
    double sigma_min = std::max(0.01, *sigma_min_it * 0.9);
    double sigma_max = *sigma_max_it * 1.1;
    double r_min = *r_min_it - 0.005;
    double r_max = *r_max_it + 0.005;

    return estimate_grid_for_price_table(
        log_m_min, log_m_max,
        tau_min, tau_max,
        sigma_min, sigma_max,
        r_min, r_max,
        params);
}

inline PriceTableGridAccuracyParams<4> make_price_table_grid_accuracy(
    PriceTableGridProfile profile)
{
    PriceTableGridAccuracyParams<4> params;
    params.curvature_weights = {1.0, 1.0, 2.5, 0.6};
    params.min_points = 4;
    switch (profile) {
        case PriceTableGridProfile::Low:
            params.target_iv_error = 5e-4;
            params.max_points = 80;
            break;
        case PriceTableGridProfile::Medium:
            params.target_iv_error = 1e-4;
            params.max_points = 120;
            break;
        case PriceTableGridProfile::High:
            params.target_iv_error = 2e-5;
            params.max_points = 160;
            break;
        case PriceTableGridProfile::Ultra:
            params.target_iv_error = 7e-6;
            params.max_points = 200;
            break;
    }
    return params;
}

// ---------------------------------------------------------------------------
// Recursion helpers (merged from recursion_helpers.hpp)
// ---------------------------------------------------------------------------

/// Recursively iterate over all combinations of axis indices
template <size_t Axis, size_t N, typename Func>
void for_each_axis_index_impl(
    const PriceTableAxesND<N>& axes,
    std::array<size_t, N>& indices,
    Func&& func)
{
    if constexpr (Axis == N) {
        func(indices);
    } else {
        for (size_t i = 0; i < axes.grids[Axis].size(); ++i) {
            indices[Axis] = i;
            for_each_axis_index_impl<Axis + 1>(axes, indices, std::forward<Func>(func));
        }
    }
}

/// Public entry point for axis index iteration
template <size_t StartAxis, size_t N, typename Func>
void for_each_axis_index(const PriceTableAxesND<N>& axes, Func&& func) {
    std::array<size_t, N> indices{};
    for_each_axis_index_impl<StartAxis>(axes, indices, std::forward<Func>(func));
}

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
    /// @param transform Optional transform applied to tensor after extraction (e.g., EEP decompose)
    /// @return PriceTableResult with surface and diagnostics, or error
    [[nodiscard]] std::expected<PriceTableResult<N>, PriceTableError>
    build(const PriceTableAxesND<N>& axes,
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


    // -----------------------------------------------------------------
    // Pipeline steps (public for use by free-standing builder helpers)
    // -----------------------------------------------------------------

    /// Result from B-spline coefficient fitting
    struct FitCoeffsResult {
        std::vector<double> coefficients;
        BSplineFittingStats<double, N> stats;
    };

    /// Generate batch of PricingParams from axes
    [[nodiscard]] std::vector<PricingParams> make_batch(
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

private:
    /// Estimate PDE grid from batch parameters using pde_accuracy config
    [[nodiscard]] std::pair<GridSpec<double>, TimeDomain> estimate_pde_grid(
        const std::vector<PricingParams>& batch,
        const PriceTableAxesND<N>& axes) const;

    /// Solve batch of options with snapshot registration
    [[nodiscard]] BatchAmericanOptionResult solve_batch(
        const std::vector<PricingParams>& batch,
        const PriceTableAxesND<N>& axes) const;

    /// Find nearest valid neighbor in (σ,r) grid using Manhattan distance
    [[nodiscard]] std::optional<std::pair<size_t, size_t>> find_nearest_valid_neighbor(
        size_t σ_idx, size_t r_idx, size_t Nσ, size_t Nr,
        const std::vector<bool>& slice_valid) const;

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

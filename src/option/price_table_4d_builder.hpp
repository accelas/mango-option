/**
 * @file price_table_4d_builder.hpp
 * @brief Build 4D option price tables with B-spline interpolation
 *
 * Orchestrates multi-run PDE solves to build 4D price surfaces
 * (moneyness × maturity × volatility × rate) and fits B-spline
 * coefficients for fast interpolation.
 *
 * Usage:
 *   // Define 4D grids
 *   auto builder = PriceTable4DBuilder::create(
 *       {0.7, 0.8, ..., 1.3},   // 50 moneyness points
 *       {0.027, 0.1, ..., 2.0}, // 30 maturity points
 *       {0.10, 0.15, ..., 0.80},// 20 volatility points
 *       {0.0, 0.02, ..., 0.10}, // 10 rate points
 *       100.0                    // K_ref
 *   );
 *
 *   // Pre-compute prices (200 PDE solves, ~5 minutes on 16 cores)
 *   builder.precompute(OptionType::PUT, 101, 1000);
 *
 *   // Get fast interpolator (~500ns per query)
 *   auto evaluator = builder.get_evaluator();
 *   double price = evaluator.eval(1.05, 0.25, 0.20, 0.05);
 *
 * Performance:
 * - Pre-computation: ~72 options/sec × 200 = ~3 minutes single-threaded
 * - With OpenMP: ~848 options/sec × 200 = ~24 seconds (16 cores)
 * - Query time: ~500ns per price lookup
 * - IV calculation: <30µs (vs 143ms FDM, 4800× speedup)
 */

#pragma once

#include "src/option/american_option.hpp"
#include "src/option/bspline_price_table.hpp"
#include "src/math/bspline_nd_separable.hpp"
#include <expected>
#include "src/support/error_types.hpp"
#include "src/option/price_table_workspace.hpp"
#include <cassert>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <optional>
#include <utility>

namespace mango {

/// Statistics from B-spline fitting process
struct BSplineFittingStats {
    double max_residual_axis0 = 0.0;       ///< Max residual along axis 0
    double max_residual_axis1 = 0.0;       ///< Max residual along axis 1
    double max_residual_axis2 = 0.0;       ///< Max residual along axis 2
    double max_residual_axis3 = 0.0;       ///< Max residual along axis 3
    double max_residual_overall = 0.0;     ///< Max residual across all axes

    double condition_axis0 = 0.0;          ///< Condition number estimate (axis 0)
    double condition_axis1 = 0.0;          ///< Condition number estimate (axis 1)
    double condition_axis2 = 0.0;          ///< Condition number estimate (axis 2)
    double condition_axis3 = 0.0;          ///< Condition number estimate (axis 3)
    double condition_max = 0.0;            ///< Maximum condition number

    size_t failed_slices_axis0 = 0;        ///< Failed fits along axis 0
    size_t failed_slices_axis1 = 0;        ///< Failed fits along axis 1
    size_t failed_slices_axis2 = 0;        ///< Failed fits along axis 2
    size_t failed_slices_axis3 = 0;        ///< Failed fits along axis 3
    size_t failed_slices_total = 0;        ///< Total failed fits
};

/// Configuration for PDE solves performed by the builder
struct PriceTableConfig {
    OptionType option_type = OptionType::PUT;
    size_t n_space = 101;
    size_t n_time = 1000;
    double dividend_yield = 0.0;
    std::optional<std::pair<double, double>> x_bounds;
};

/// Market option chain data (from exchanges)
///
/// Represents raw option chain data as typically received from market data
/// feeds or exchanges. Can contain duplicate strikes/maturities (e.g., multiple
/// options with same parameters but different bid/ask spreads).
struct OptionChain {
    std::string ticker;                  ///< Underlying ticker symbol
    double spot = 0.0;                   ///< Current underlying price
    std::vector<double> strikes;         ///< Strike prices (may have duplicates)
    std::vector<double> maturities;      ///< Times to expiration in years (may have duplicates)
    std::vector<double> implied_vols;    ///< Market implied volatilities (for grid)
    std::vector<double> rates;           ///< Risk-free rates (may have duplicates)
    double dividend_yield = 0.0;         ///< Continuous dividend yield
};

/// Thin value object exposing a user-friendly interface to the price surface
class PriceTableSurface {
public:
    PriceTableSurface() = default;

    /// Construct from workspace (zero-copy ready, supports save/load)
    ///
    /// @param workspace Shared workspace containing all data
    explicit PriceTableSurface(std::shared_ptr<PriceTableWorkspace> workspace)
        : workspace_(std::move(workspace))
        , evaluator_(nullptr)
    {
        if (workspace_) {
            auto spline_result = BSpline4D::create(*workspace_);
            // Precondition: Workspace must be valid (already validated by PriceTable4DBuilder)
            // BSpline4D creation should not fail if workspace is valid
            assert(spline_result.has_value() && "BSpline4D creation failed (programming error)");
            evaluator_ = std::make_unique<BSpline4D>(std::move(spline_result.value()));
        }
    }

    bool valid() const {
        return workspace_ != nullptr;
    }

    double eval(double m, double tau, double sigma, double rate) const {
        // Precondition: Surface must be initialized before evaluation
        // This is a programming error, not a runtime condition
        assert(valid() && "PriceTableSurface not initialized (programming error)");
        return evaluator_->eval(m, tau, sigma, rate);
    }

    double K_ref() const {
        return workspace_->K_ref();
    }

    double dividend_yield() const {
        return workspace_->dividend_yield();
    }

    std::pair<double, double> moneyness_range() const {
        auto span = workspace_->moneyness();
        return {span.front(), span.back()};
    }

    std::pair<double, double> maturity_range() const {
        auto span = workspace_->maturity();
        return {span.front(), span.back()};
    }

    std::pair<double, double> volatility_range() const {
        auto span = workspace_->volatility();
        return {span.front(), span.back()};
    }

    std::pair<double, double> rate_range() const {
        auto span = workspace_->rate();
        return {span.front(), span.back()};
    }

    /// Access workspace
    std::shared_ptr<PriceTableWorkspace> workspace() const { return workspace_; }

private:
    std::shared_ptr<PriceTableWorkspace> workspace_;
    std::unique_ptr<BSpline4D> evaluator_;
};

/// Result of 4D price table building
struct PriceTable4DResult {
    PriceTableSurface surface;                ///< Friendly interface to evaluator
    std::shared_ptr<BSpline4D> evaluator;     ///< Fast B-spline evaluator
    std::vector<double> prices_4d;            ///< Raw 4D price array
    size_t n_pde_solves;                      ///< Number of PDE solves performed
    double precompute_time_seconds;           ///< Wall-clock time for pre-computation
    BSplineFittingStats fitting_stats;        ///< B-spline fitting diagnostics
};

/// 4D Price Table Builder
///
/// Builds interpolation-ready 4D option price surfaces by:
/// 1. Running PDE solver for each (σ, r) combination
/// 2. Evaluating prices at (m, τ) grid points
/// 3. Assembling into 4D array
/// 4. Fitting B-spline coefficients
///
/// Memory: O(Nm × Nτ × Nσ × Nr) for price storage
/// Time: O(Nσ × Nr × PDE_solve_time) for pre-computation
class PriceTable4DBuilder {
public:
    /// Create builder with 4D grids
    ///
    /// @param moneyness Moneyness grid m = S/K (sorted, ≥4 points)
    /// @param maturity Maturity grid τ in years (sorted, ≥4 points)
    /// @param volatility Volatility grid σ (sorted, ≥4 points)
    /// @param rate Risk-free rate grid r (sorted, ≥4 points)
    /// @param K_ref Reference strike price
    /// @return Builder on success, error message on validation failure
    static std::expected<PriceTable4DBuilder, std::string> create(
        std::vector<double> moneyness,
        std::vector<double> maturity,
        std::vector<double> volatility,
        std::vector<double> rate,
        double K_ref)
    {
        // Construct builder (no validation yet)
        PriceTable4DBuilder builder(
            std::move(moneyness),
            std::move(maturity),
            std::move(volatility),
            std::move(rate),
            K_ref,
            /* skip_validation = */ true
        );

        // Validate after construction
        auto validation = builder.validate_grids();
        if (!validation) {
            return std::unexpected(validation.error());
        }

        return builder;
    }


    /// Create builder from strike prices (auto-computes moneyness)
    ///
    /// This convenience method allows users to work with raw strike prices
    /// instead of pre-computing moneyness ratios. Automatically selects the
    /// ATM strike as the reference strike (K_ref).
    ///
    /// @param spot Current underlying price
    /// @param strikes Strike prices (should be sorted)
    /// @param maturities Time to expiration (years)
    /// @param volatilities Volatility grid
    /// @param rates Rate grid
    /// @return Builder ready for precomputation
    ///
    /// Usage example:
    /// ```cpp
    /// auto result = PriceTable4DBuilder::from_strikes(
    ///     450.0,  // spot price
    ///     {400, 425, 450, 475, 500},  // strikes
    ///     {0.1, 0.25, 0.5, 1.0},      // maturities
    ///     {0.15, 0.20, 0.25, 0.30},   // volatilities
    ///     {0.03, 0.04, 0.05}          // rates
    /// );
    /// if (!result) { /* handle error */ }
    /// ```
    static std::expected<PriceTable4DBuilder, std::string> from_strikes(
        double spot,
        std::vector<double> strikes,
        std::vector<double> maturities,
        std::vector<double> volatilities,
        std::vector<double> rates)
    {
        // Validate inputs
        if (strikes.empty()) {
            return std::unexpected("Strike array cannot be empty");
        }
        if (spot <= 0.0) {
            return std::unexpected("Spot price must be positive");
        }

        // Auto-compute moneyness: m = spot / strike
        // Note: strikes are sorted ascending, so moneyness will be descending
        std::vector<double> moneyness;
        moneyness.reserve(strikes.size());
        for (double K : strikes) {
            if (K <= 0.0) {
                return std::unexpected("All strikes must be positive");
            }
            moneyness.push_back(spot / K);
        }

        // Reverse to get ascending moneyness (required by PriceTable4DBuilder)
        std::reverse(moneyness.begin(), moneyness.end());

        // Use ATM strike as reference
        auto atm_it = std::lower_bound(strikes.begin(), strikes.end(), spot);
        double K_ref = (atm_it != strikes.end()) ? *atm_it : strikes[strikes.size()/2];

        return create(
            std::move(moneyness),
            std::move(maturities),
            std::move(volatilities),
            std::move(rates),
            K_ref
        );
    }

    /// Create builder from market option chain data
    ///
    /// Convenience method that extracts unique strikes, maturities, volatilities,
    /// and rates from raw market chain data (which may contain duplicates), sorts
    /// them, and builds a price table grid.
    ///
    /// @param chain Market option chain data from exchange
    /// @return Builder ready for precomputation
    ///
    /// Usage example:
    /// ```cpp
    /// OptionChain spy_chain{
    ///     .ticker = "SPY",
    ///     .spot = 450.0,
    ///     .strikes = {400, 425, 425, 450, 475, 500},  // duplicates OK
    ///     .maturities = {0.1, 0.25, 0.25, 0.5, 1.0},  // duplicates OK
    ///     .implied_vols = {0.15, 0.20, 0.25, 0.30},
    ///     .rates = {0.03, 0.04, 0.05},
    ///     .dividend_yield = 0.015
    /// };
    /// auto result = PriceTable4DBuilder::from_chain(spy_chain);
    /// if (!result) { /* handle error */ }
    /// ```
    static std::expected<PriceTable4DBuilder, std::string> from_chain(const OptionChain& chain)
    {
        // Helper to extract unique sorted values
        auto unique_sorted = [](std::vector<double> vec) {
            std::sort(vec.begin(), vec.end());
            auto last = std::unique(vec.begin(), vec.end());
            vec.erase(last, vec.end());
            return vec;
        };

        // Extract unique sorted values from potentially duplicated chain data
        auto strikes = unique_sorted(chain.strikes);
        auto maturities = unique_sorted(chain.maturities);
        auto vols = unique_sorted(chain.implied_vols);
        auto rates = unique_sorted(chain.rates);

        return from_strikes(
            chain.spot,
            std::move(strikes),
            std::move(maturities),
            std::move(vols),
            std::move(rates)
        );
    }

    /// Pre-compute all option prices on 4D grid (standard bounds)
    ///
    /// Runs PDE solver for each (σ, r) combination using standard
    /// log-moneyness bounds [-3.0, 3.0]. For custom bounds, use the
    /// overload that accepts x_min and x_max parameters.
    ///
    /// @param option_type Call or Put
    /// @param n_space Number of spatial grid points
    /// @param n_time Number of time steps
    /// @param dividend_yield Continuous dividend yield (default: 0)
    /// @return Result with fitted B-spline evaluator
    std::expected<PriceTable4DResult, std::string> precompute(
        OptionType option_type,
        size_t n_space,
        size_t n_time,
        double dividend_yield = 0.0);

    /// Pre-compute all option prices using a single configuration struct
    std::expected<PriceTable4DResult, std::string> precompute(
        const PriceTableConfig& config);

    /// Pre-compute all option prices on 4D grid (custom bounds)
    ///
    /// Runs PDE solver for each (σ, r) combination using custom
    /// log-moneyness bounds. Use this when your moneyness grid
    /// requires wider bounds than the standard [-3.0, 3.0] range.
    ///
    /// Example: For moneyness range [0.5, 2.0]:
    /// ```cpp
    /// double x_min = std::log(0.5);  // ≈ -0.693
    /// double x_max = std::log(2.0);  // ≈ 0.693
    /// builder.precompute(OptionType::PUT, x_min, x_max, 101, 1000);
    /// ```
    ///
    /// @param option_type Call or Put
    /// @param x_min Minimum log-moneyness (must be ≤ ln(moneyness.front()))
    /// @param x_max Maximum log-moneyness (must be ≥ ln(moneyness.back()))
    /// @param n_space Number of spatial grid points
    /// @param n_time Number of time steps
    /// @param dividend_yield Continuous dividend yield (default: 0)
    /// @return Result with fitted B-spline evaluator
    std::expected<PriceTable4DResult, std::string> precompute(
        OptionType option_type,
        double x_min,
        double x_max,
        size_t n_space,
        size_t n_time,
        double dividend_yield = 0.0);

    /// Get grid dimensions
    std::tuple<size_t, size_t, size_t, size_t> dimensions() const {
        return {moneyness_.size(), maturity_.size(), volatility_.size(), rate_.size()};
    }

private:
    PriceTable4DBuilder(
        std::vector<double> moneyness,
        std::vector<double> maturity,
        std::vector<double> volatility,
        std::vector<double> rate,
        double K_ref,
        bool skip_validation = false)
        : moneyness_(std::move(moneyness))
        , maturity_(std::move(maturity))
        , volatility_(std::move(volatility))
        , rate_(std::move(rate))
        , K_ref_(K_ref)
    {
        // Note: Validation is now handled by static factories
        // Constructor does not throw - callers must check std::expected
        (void)skip_validation;  // Parameter used to document intent
    }

    std::expected<void, std::string> validate_grids() const;

    std::vector<double> moneyness_;
    std::vector<double> maturity_;
    std::vector<double> volatility_;
    std::vector<double> rate_;
    double K_ref_;
};

}  // namespace mango

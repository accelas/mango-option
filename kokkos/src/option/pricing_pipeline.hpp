#pragma once

/// @file pricing_pipeline.hpp
/// @brief High-level pricing pipeline orchestrating all Kokkos components
///
/// Provides unified API for:
/// 1. Price table precomputation (PriceTableBuilder4D)
/// 2. Fast IV via interpolation (IVSolverInterpolated)
/// 3. Accurate IV via FDM (IVSolverFDM)
/// 4. Batch option pricing (BatchAmericanSolver)
///
/// Typical workflow:
/// ```cpp
/// PricingPipelineConfig config{...};
/// PricingPipeline<Kokkos::HostSpace> pipeline(config);
///
/// // Step 1: Build price table (once at startup)
/// pipeline.build_price_table();
///
/// // Step 2: Fast IV queries (microseconds)
/// auto iv_results = pipeline.solve_iv_interpolated(queries);
///
/// // Step 3: Ground truth IV for validation (milliseconds)
/// auto iv_fdm_results = pipeline.solve_iv_fdm(queries);
/// ```

#include <Kokkos_Core.hpp>
#include <expected>
#include <array>
#include <optional>
#include <memory>
#include "kokkos/src/option/price_table.hpp"
#include "kokkos/src/option/batch_solver.hpp"
#include "kokkos/src/option/iv_solver_interpolated.hpp"
#include "kokkos/src/option/iv_solver_fdm.hpp"
#include "kokkos/src/support/execution_space.hpp"

namespace mango::kokkos {

/// Configuration for pricing pipeline
struct PricingPipelineConfig {
    // Price table grid configurations
    struct GridConfig {
        double min;
        double max;
        size_t size;
    };

    GridConfig moneyness{0.7, 1.3, 21};    ///< Moneyness grid (S/K)
    GridConfig maturity{0.1, 2.0, 20};     ///< Maturity grid (years)
    GridConfig volatility{0.1, 0.5, 21};   ///< Volatility grid
    GridConfig rate{0.0, 0.1, 11};         ///< Rate grid

    // PDE solver configuration
    size_t n_space = 101;                   ///< Spatial grid points
    size_t n_time = 500;                    ///< Time steps

    // Option parameters
    double K_ref = 100.0;                   ///< Reference strike
    double dividend_yield = 0.0;            ///< Dividend yield
    bool is_put = true;                     ///< Option type

    // IV solver configurations
    IVSolverConfig iv_interp_config{};      ///< Interpolated IV config
    IVSolverFDMConfig iv_fdm_config{};      ///< FDM IV config
};

/// Error codes for pricing pipeline
enum class PipelineError {
    PriceTableNotBuilt,
    InvalidConfiguration,
    BuildFailed,
    SolveFailed
};

/// Unified pricing pipeline for American options
///
/// Orchestrates all components:
/// - Price table precomputation for fast lookups
/// - Interpolated IV solving for speed
/// - FDM IV solving for accuracy
/// - Batch option pricing
///
/// @tparam MemSpace Kokkos memory space (HostSpace, CudaSpace, etc.)
template <typename MemSpace>
class PricingPipeline {
public:
    using view_1d = Kokkos::View<double*, MemSpace>;

    /// Construct pipeline with configuration
    explicit PricingPipeline(const PricingPipelineConfig& config)
        : config_(config)
    {}

    /// Build the 4D price table (call once at startup)
    ///
    /// This is a one-time expensive operation (seconds) that enables
    /// fast IV solving via interpolation.
    ///
    /// @return Success or error
    [[nodiscard]] std::expected<void, PipelineError> build_price_table() {
        // Create grid Views
        auto m_grid = create_grid(config_.moneyness);
        auto tau_grid = create_grid(config_.maturity);
        auto sigma_grid = create_grid(config_.volatility);
        auto r_grid = create_grid(config_.rate);

        // Configure price table builder
        PriceTableConfig table_config{
            .n_space = config_.n_space,
            .n_time = config_.n_time,
            .K_ref = config_.K_ref,
            .q = config_.dividend_yield,
            .is_put = config_.is_put
        };

        // Build price table
        PriceTableBuilder4D<MemSpace> builder(
            m_grid, tau_grid, sigma_grid, r_grid, table_config);

        auto result = builder.build();
        if (!result.has_value()) {
            return std::unexpected(PipelineError::BuildFailed);
        }

        price_table_ = std::move(result.value());

        // Create interpolated IV solver
        auto solver_result = IVSolverInterpolated<MemSpace>::create(
            price_table_, config_.iv_interp_config);

        if (!solver_result.has_value()) {
            return std::unexpected(PipelineError::BuildFailed);
        }

        iv_solver_interpolated_ = std::move(solver_result.value());

        return {};
    }

    /// Price batch of options using BatchAmericanSolver
    ///
    /// All options share the same volatility, rate, maturity, and dividend yield.
    /// Each option has its own strike and spot price.
    ///
    /// @param strikes View of strike prices
    /// @param spots View of spot prices
    /// @return View of pricing results
    [[nodiscard]] std::expected<Kokkos::View<BatchOptionResult*, MemSpace>, PipelineError>
    price_options(view_1d strikes, view_1d spots) const {
        BatchPricingParams params{
            .maturity = (config_.maturity.min + config_.maturity.max) * 0.5,
            .volatility = (config_.volatility.min + config_.volatility.max) * 0.5,
            .rate = (config_.rate.min + config_.rate.max) * 0.5,
            .dividend_yield = config_.dividend_yield,
            .is_put = config_.is_put
        };

        BatchAmericanSolver<MemSpace> solver(
            params, strikes, spots, config_.n_space, config_.n_time);

        auto result = solver.solve();
        if (!result.has_value()) {
            return std::unexpected(PipelineError::SolveFailed);
        }

        return result.value();
    }

    /// Price batch of options with custom parameters
    ///
    /// @param params Batch pricing parameters
    /// @param strikes View of strike prices
    /// @param spots View of spot prices
    /// @return View of pricing results
    [[nodiscard]] std::expected<Kokkos::View<BatchOptionResult*, MemSpace>, PipelineError>
    price_options(const BatchPricingParams& params, view_1d strikes, view_1d spots) const {
        BatchAmericanSolver<MemSpace> solver(
            params, strikes, spots, config_.n_space, config_.n_time);

        auto result = solver.solve();
        if (!result.has_value()) {
            return std::unexpected(PipelineError::SolveFailed);
        }

        return result.value();
    }

    /// Solve IV using interpolated price table (fast, microseconds)
    ///
    /// Requires price table to be built via build_price_table() first.
    ///
    /// @param queries View of IV queries
    /// @return View of IV results
    [[nodiscard]] std::expected<Kokkos::View<IVResult*, MemSpace>, PipelineError>
    solve_iv_interpolated(const Kokkos::View<IVQuery*, MemSpace>& queries) const {
        if (!iv_solver_interpolated_.has_value()) {
            return std::unexpected(PipelineError::PriceTableNotBuilt);
        }

        auto result = iv_solver_interpolated_->solve_batch(queries);
        if (!result.has_value()) {
            return std::unexpected(PipelineError::SolveFailed);
        }

        return result.value();
    }

    /// Solve IV using FDM (accurate, milliseconds)
    ///
    /// Uses full PDE solver for each query. More accurate but slower
    /// than interpolated approach. Useful for validation.
    ///
    /// @param queries View of IV queries (host)
    /// @return View of FDM IV results
    [[nodiscard]] std::expected<Kokkos::View<IVResultFDM*, MemSpace>, PipelineError>
    solve_iv_fdm(const Kokkos::View<IVQuery*, MemSpace>& queries) const {
        IVSolverFDM<MemSpace> solver(config_.iv_fdm_config);
        auto results = solver.solve_batch(queries);
        return results;
    }

    /// Get underlying price table (if built)
    ///
    /// @return Optional reference to price table
    [[nodiscard]] std::optional<std::reference_wrapper<const PriceTable4D>>
    get_price_table() const {
        if (price_table_.shape[0] == 0) {
            return std::nullopt;
        }
        return std::cref(price_table_);
    }

    /// Check if price table has been built
    [[nodiscard]] bool is_price_table_built() const noexcept {
        return iv_solver_interpolated_.has_value();
    }

    /// Get pipeline configuration
    [[nodiscard]] const PricingPipelineConfig& config() const noexcept {
        return config_;
    }

private:
    /// Create uniform grid from configuration
    [[nodiscard]] view_1d create_grid(const PricingPipelineConfig::GridConfig& cfg) const {
        view_1d grid("grid", cfg.size);
        auto grid_h = Kokkos::create_mirror_view(grid);

        const double dx = (cfg.max - cfg.min) / static_cast<double>(cfg.size - 1);
        for (size_t i = 0; i < cfg.size; ++i) {
            grid_h(i) = cfg.min + static_cast<double>(i) * dx;
        }

        Kokkos::deep_copy(grid, grid_h);
        return grid;
    }

    PricingPipelineConfig config_;
    PriceTable4D price_table_;
    std::optional<IVSolverInterpolated<MemSpace>> iv_solver_interpolated_;
};

}  // namespace mango::kokkos

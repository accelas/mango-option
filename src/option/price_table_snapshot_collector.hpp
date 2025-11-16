#pragma once

#include "src/pde/core/snapshot.hpp"
#include "src/math/cubic_spline_solver.hpp"
#include "src/option/american_option.hpp"  // For OptionType enum
#include "src/support/memory/solver_memory_arena.hpp"
#include "common/ivcalc_trace.h"
#include <expected>
#include "src/support/error_types.hpp"
#include <span>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <optional>
#include <memory_resource>
#include <memory>

namespace mango {

// Memory module identifier for tracing
#define MODULE_PRICE_TABLE_COLLECTOR 9

/// Cubic spline interpolator for snapshot data
///
/// Modern C++ implementation using CubicSpline<double>.
/// Supports interpolation from pre-computed derivative arrays.
class SnapshotInterpolator {
public:
    /// Constructor with default memory resource
    SnapshotInterpolator() = default;

    /// Constructor with explicit memory resource
    explicit SnapshotInterpolator(std::pmr::memory_resource* resource)
        : x_(resource), y_(resource), cache_{DerivedSplineCache{}, DerivedSplineCache{}}
    {
    }

    // Rule of five: default move, delete copy
    SnapshotInterpolator(const SnapshotInterpolator&) = delete;
    SnapshotInterpolator& operator=(const SnapshotInterpolator&) = delete;
    SnapshotInterpolator(SnapshotInterpolator&&) = default;
    SnapshotInterpolator& operator=(SnapshotInterpolator&&) = default;
    ~SnapshotInterpolator() = default;

    /// Build spline from snapshot data
    ///
    /// @param x X-coordinates (must be strictly increasing)
    /// @param y Y-coordinates
    /// @return Optional error message (nullopt on success)
    ///
    /// @note On failure, the interpolator is reset to unbuilt state.
    ///       Previous state is NOT preserved across failed rebuilds.
    [[nodiscard]] std::optional<std::string_view> build(
        std::span<const double> x,
        std::span<const double> y)
    {
        // Reset to unbuilt state BEFORE attempting build
        // This prevents state corruption if build fails
        built_ = false;

        // CRITICAL: Invalidate derived spline cache when grid changes
        // The derived spline caches interval widths (h_) from the grid,
        // so it must be rebuilt from scratch when x_ changes
        invalidate_derived_spline();

        // Build cubic spline first (validates input)
        auto error = spline_.build(x, y);
        if (error.has_value()) {
            // Leave in unbuilt state on failure
            x_.clear();
            y_.clear();
            return error;
        }

        // Only update grid storage after successful build
        x_.assign(x.begin(), x.end());
        y_.assign(y.begin(), y.end());
        built_ = true;
        return std::nullopt;
    }

    /// Rebuild spline with new y-values on the same x-grid
    ///
    /// PERFORMANCE: Much faster than build() when grid unchanged.
    /// Reuses cached interval widths and grid structure.
    ///
    /// @param y New Y-coordinates (must match existing grid)
    /// @return Optional error message (nullopt on success)
    ///
    /// @pre build() must have been called successfully at least once
    [[nodiscard]] std::optional<std::string_view> rebuild_same_grid(
        std::span<const double> y)
    {
        if (!built_) {
            return "Must call build() before rebuild_same_grid()";
        }

        auto error = spline_.rebuild_same_grid(y);
        if (error.has_value()) {
            built_ = false;
            return error;
        }

        // Update stored y-values
        y_.assign(y.begin(), y.end());

        // Bump epoch to invalidate stale derivative caches
        ++data_epoch_;

        return std::nullopt;
    }

    /// Get current data epoch
    ///
    /// The epoch increments whenever rebuild_same_grid() succeeds.
    /// Callers can use this to track data freshness for derivative caches.
    ///
    /// @return Current epoch value
    [[nodiscard]] uint64_t current_epoch() const noexcept {
        return data_epoch_;
    }

    /// Evaluate interpolant
    ///
    /// @param x_eval Evaluation point
    /// @return Interpolated value (or 0 if not built)
    [[nodiscard]] double eval(double x_eval) const noexcept {
        if (!built_) return 0.0;
        return spline_.eval(x_eval);
    }

    /// Interpolate from pre-computed data array
    ///
    /// Uses the same grid as build() but evaluates with different data.
    /// Useful for evaluating derivatives without re-building the spline.
    ///
    /// Uses cubic spline interpolation for smoothness.
    ///
    /// PERFORMANCE: Uses epoch-based caching to avoid thrashing when
    /// alternating between different arrays (e.g., first and second derivatives).
    /// Maintains a 2-slot LRU cache. Cache entries are keyed by (pointer, epoch)
    /// for O(1) freshness checks without O(n) content comparison.
    ///
    /// @param x_eval Evaluation point
    /// @param data Pre-computed values at grid points (same grid as build())
    /// @param epoch Data epoch from current_epoch() after rebuild_same_grid()
    /// @return Interpolated value
    [[nodiscard]] double eval_from_data(double x_eval, std::span<const double> data, uint64_t epoch) const {
        if (x_.empty() || data.size() != x_.size()) {
            return 0.0;
        }

        // PERFORMANCE: Epoch-based cache using pointer + epoch comparison (O(1))
        // Handles both alternating arrays AND reused buffers with changed values
        const double* data_ptr = data.data();

        // Check cache[0]: pointer match + epoch match
        if (cache_[0].data_ptr == data_ptr && cache_[0].built && cache_[0].epoch == epoch) {
            // Cache hit: same pointer, same epoch
            return cache_[0].spline.eval(x_eval);
        }

        // Check cache[1]: pointer match + epoch match
        if (cache_[1].data_ptr == data_ptr && cache_[1].built && cache_[1].epoch == epoch) {
            // Cache hit in slot 1: promote to slot 0 (LRU)
            std::swap(cache_[0], cache_[1]);
            return cache_[0].spline.eval(x_eval);
        }

        // Cache miss (pointer or epoch mismatch): rebuild
        // Evict LRU (slot 1) and build in slot 0
        cache_[1] = std::move(cache_[0]);  // Demote slot 0 to slot 1

        auto error = cache_[0].spline.build(std::span{x_}, data);
        if (error.has_value()) {
            // Fallback to linear interpolation on error
            cache_[0].built = false;
            cache_[0].data_ptr = nullptr;
            cache_[0].epoch = 0;
            return eval_from_data_linear(x_eval, data);
        }

        cache_[0].built = true;
        cache_[0].data_ptr = data_ptr;
        cache_[0].epoch = epoch;
        return cache_[0].spline.eval(x_eval);
    }

    /// Check if spline has been built
    [[nodiscard]] bool is_built() const noexcept {
        return built_;
    }

    /// Get the underlying spline (for advanced use)
    [[nodiscard]] const CubicSpline<double>& get_spline() const noexcept {
        return spline_;
    }

private:
    /// Cache entry for derived splines
    struct DerivedSplineCache {
        CubicSpline<double> spline;
        const double* data_ptr = nullptr;     ///< Pointer for fast lookup
        uint64_t epoch = 0;                   ///< Data epoch for O(1) freshness check
        bool built = false;
    };

    /// Invalidate derived spline cache
    ///
    /// CRITICAL: Must be called whenever the grid (x_) changes.
    /// The derived spline caches interval widths from x_, so a grid
    /// change makes its cached state invalid.
    void invalidate_derived_spline() noexcept {
        cache_[0] = DerivedSplineCache{};
        cache_[1] = DerivedSplineCache{};
    }

    /// Linear interpolation fallback
    [[nodiscard]] double eval_from_data_linear(double x_eval, std::span<const double> data) const noexcept {
        // Find bracketing interval using binary search
        if (x_eval <= x_.front()) {
            return data.front();
        }
        if (x_eval >= x_.back()) {
            return data.back();
        }

        // Binary search for interval
        auto it = std::lower_bound(x_.begin(), x_.end(), x_eval);
        size_t i = std::distance(x_.begin(), it);

        // Adjust to get the interval [x[i], x[i+1]]
        if (i > 0 && (i == x_.size() || x_[i] > x_eval)) {
            --i;
        }
        i = std::min(i, x_.size() - 2);

        // Linear interpolation
        const double t = (x_eval - x_[i]) / (x_[i+1] - x_[i]);
        return (1.0 - t) * data[i] + t * data[i+1];
    }

    CubicSpline<double> spline_;
    std::pmr::vector<double> x_;  // Grid points (for eval_from_data)
    std::pmr::vector<double> y_;  // Values (for eval_from_data)
    bool built_ = false;

    // Epoch counter for tracking data freshness (incremented on rebuild_same_grid)
    uint64_t data_epoch_ = 0;

    // LRU cache for derived splines (size 2: handles first + second derivative)
    // Mutable for lazy initialization in const eval_from_data
    mutable DerivedSplineCache cache_[2];
};



struct PriceTableSnapshotCollectorConfig {
    std::span<const double> moneyness;
    std::span<const double> tau;
    double K_ref;
    OptionType option_type = OptionType::CALL;  // Used for obstacle computation
    const void* payoff_params = nullptr;
};

/// Collects snapshots into price table format
///
/// PERFORMANCE: Builds interpolators ONCE per snapshot (not O(n²))
/// CORRECTNESS: PDE provides ∂²V/∂S² directly - no transformation needed!
///
/// OPTIMIZATION: Caches spatial grid and reuses interpolators when grid unchanged.
/// For a (σ,r) slice with N maturities sharing the same spatial grid:
///   - 1 full build (first snapshot) + (N-1) fast rebuilds (remaining snapshots)
///   - Typical speedup: 2-3x for precomputation workloads
///
/// THREAD-SAFETY: NOT thread-safe. Each collector instance must be used
/// by a single thread only. For parallel precomputation, each thread should
/// create its own collector instance. The cached state (interpolators, grid)
/// is not protected by synchronization.
///
/// INVARIANTS:
///   - After first collect(): interpolators_built_ == true
///   - cached_grid_ contains the spatial grid from the most recent snapshot
///   - value_interp_ and lu_interp_ are valid and ready for evaluation
class PriceTableSnapshotCollector : public SnapshotCollector {
public:
    /// Constructor without memory arena (uses default memory resource)
    explicit PriceTableSnapshotCollector(const PriceTableSnapshotCollectorConfig& config)
        : PriceTableSnapshotCollector(config, nullptr)
    {
    }

    /// Constructor with memory arena for PMR allocations
    explicit PriceTableSnapshotCollector(const PriceTableSnapshotCollectorConfig& config,
                                       std::shared_ptr<memory::SolverMemoryArena> arena)
        : moneyness_(config.moneyness.begin(), config.moneyness.end())
        , tau_(config.tau.begin(), config.tau.end())
        , K_ref_(config.K_ref)
        , option_type_(config.option_type)
        , payoff_params_(config.payoff_params)
        , prices_(arena ? arena->resource() : std::pmr::get_default_resource())
        , deltas_(arena ? arena->resource() : std::pmr::get_default_resource())
        , gammas_(arena ? arena->resource() : std::pmr::get_default_resource())
        , thetas_(arena ? arena->resource() : std::pmr::get_default_resource())
        , log_moneyness_(arena ? arena->resource() : std::pmr::get_default_resource())
        , spot_values_(arena ? arena->resource() : std::pmr::get_default_resource())
        , inv_spot_(arena ? arena->resource() : std::pmr::get_default_resource())
        , inv_spot_sq_(arena ? arena->resource() : std::pmr::get_default_resource())
        , value_interp_(arena ? arena->resource() : std::pmr::get_default_resource())
        , lu_interp_(arena ? arena->resource() : std::pmr::get_default_resource())
        , cached_grid_(arena ? arena->resource() : std::pmr::get_default_resource())
        , memory_arena_(arena)
    {
        const size_t n = moneyness_.size() * tau_.size();
        prices_.resize(n, 0.0);
        deltas_.resize(n, 0.0);
        gammas_.resize(n, 0.0);
        thetas_.resize(n, 0.0);

        // PERFORMANCE: Precompute log-moneyness and scaling factors
        // These are constant across all snapshots, so cache them to avoid
        // repeated transcendentals and divisions in the hot path
        log_moneyness_.resize(moneyness_.size());
        spot_values_.resize(moneyness_.size());
        inv_spot_.resize(moneyness_.size());
        inv_spot_sq_.resize(moneyness_.size());

        for (size_t i = 0; i < moneyness_.size(); ++i) {
            const double m = moneyness_[i];
            log_moneyness_[i] = std::log(m);           // x = ln(m)
            spot_values_[i] = m * K_ref_;               // S = m * K_ref
            inv_spot_[i] = 1.0 / spot_values_[i];      // 1/S
            inv_spot_sq_[i] = inv_spot_[i] * inv_spot_[i];  // 1/S²
        }

        MANGO_TRACE_ALGO_START(MODULE_PRICE_TABLE_COLLECTOR,
                               static_cast<int>(moneyness_.size()),
                               static_cast<int>(tau_.size()),
                               static_cast<int>(n));
    }

    /// Collect snapshot data with exception-safe expected pattern
    /// @param snapshot Read-only snapshot data
    /// @return std::expected<void, std::string> - success or error message
    std::expected<void, std::string> collect_expected(const Snapshot& snapshot) {
        MANGO_TRACE_ALGO_START(MODULE_PRICE_TABLE_COLLECTOR,
                               static_cast<int>(snapshot.user_index),
                               static_cast<int>(moneyness_.size()),
                               static_cast<int>(snapshot.spatial_grid.size()));

        // FIXED: Use user_index to match tau directly (no float comparison!)
        // Snapshot user_index IS the tau index
        const size_t tau_idx = snapshot.user_index;

        // SAFETY: Validate tau_idx is in bounds before use
        if (tau_idx >= tau_.size()) {
            MANGO_TRACE_VALIDATION_ERROR(MODULE_PRICE_TABLE_COLLECTOR, 1,
                static_cast<int>(tau_idx), static_cast<int>(tau_.size()));
            return std::unexpected(
                std::string("tau index out of range: ") + std::to_string(tau_idx) +
                " >= " + std::to_string(tau_.size()) +
                ". Verify PDE time grid matches price table maturity grid.");
        }

        // PERFORMANCE: Cache grid and reuse interpolators
        // For a given (σ,r) pair, all snapshots share the same spatial grid
        const bool grid_changed = !grids_match(snapshot.spatial_grid);

        if (grid_changed || !interpolators_built_) {
            MANGO_TRACE_ALGO_PROGRESS(MODULE_PRICE_TABLE_COLLECTOR, 1, 10,
                                      "Building interpolators from scratch");
            // Grid changed or first snapshot: build interpolators from scratch
            auto V_error = value_interp_.build(snapshot.spatial_grid, snapshot.solution);
            if (V_error.has_value()) {
                MANGO_TRACE_CONVERGENCE_FAILED(MODULE_PRICE_TABLE_COLLECTOR, 1, 10, 0.0);
                return std::unexpected(std::string("Failed to build value interpolator: ") +
                                std::string(V_error.value()));
            }

            auto Lu_error = lu_interp_.build(snapshot.spatial_grid, snapshot.spatial_operator);
            if (Lu_error.has_value()) {
                MANGO_TRACE_CONVERGENCE_FAILED(MODULE_PRICE_TABLE_COLLECTOR, 2, 10, 0.0);
                return std::unexpected(std::string("Failed to build spatial operator interpolator: ") +
                                std::string(Lu_error.value()));
            }

            // Cache the grid
            cached_grid_.assign(snapshot.spatial_grid.begin(), snapshot.spatial_grid.end());
            interpolators_built_ = true;
        } else {
            MANGO_TRACE_ALGO_PROGRESS(MODULE_PRICE_TABLE_COLLECTOR, 2, 10,
                                      "Rebuilding interpolators with same grid");
            // Grid same as before: fast rebuild with new data
            auto V_error = value_interp_.rebuild_same_grid(snapshot.solution);
            if (V_error.has_value()) {
                MANGO_TRACE_CONVERGENCE_FAILED(MODULE_PRICE_TABLE_COLLECTOR, 3, 10, 0.0);
                return std::unexpected(std::string("Failed to rebuild value interpolator: ") +
                                std::string(V_error.value()));
            }

            auto Lu_error = lu_interp_.rebuild_same_grid(snapshot.spatial_operator);
            if (Lu_error.has_value()) {
                MANGO_TRACE_CONVERGENCE_FAILED(MODULE_PRICE_TABLE_COLLECTOR, 4, 10, 0.0);
                return std::unexpected(std::string("Failed to rebuild spatial operator interpolator: ") +
                                std::string(Lu_error.value()));
            }
        }

        // PERFORMANCE: Capture epoch after rebuild for O(1) cache freshness checks
        // The epoch increments on every rebuild_same_grid(), allowing derivative
        // cache to detect buffer reuse without O(n) content comparison
        const uint64_t value_epoch = value_interp_.current_epoch();

        // Fill price table for all moneyness points
        for (size_t m_idx = 0; m_idx < moneyness_.size(); ++m_idx) {
            MANGO_TRACE_ALGO_PROGRESS(MODULE_PRICE_TABLE_COLLECTOR, static_cast<int>(m_idx + 1),
                                      static_cast<int>(moneyness_.size()), "Processing moneyness point");
            // PERFORMANCE: Use precomputed values instead of recomputing
            const double x = log_moneyness_[m_idx];      // Cached ln(m)
            const double S = spot_values_[m_idx];         // Cached m * K_ref
            const double inv_S = inv_spot_[m_idx];        // Cached 1/S
            const double inv_S2 = inv_spot_sq_[m_idx];    // Cached 1/S²

            const size_t table_idx = m_idx * tau_.size() + tau_idx;

            // Interpolate NORMALIZED price at log-moneyness x
            const double V_norm = value_interp_.eval(x);

            // Convert to DOLLAR price: V_dollar = K_ref * V_norm
            prices_[table_idx] = K_ref_ * V_norm;

            // Interpolate normalized delta from PDE data: dV_norm/dx
            // Pass epoch for O(1) cache freshness check (avoids O(n) std::equal)
            const double dVnorm_dx = value_interp_.eval_from_data(x, snapshot.first_derivative, value_epoch);

            // Transform to dollar delta using chain rule:
            // PERFORMANCE: Use FMA for better precision and potential FMA instruction
            const double delta_scale = K_ref_ * inv_S;
            deltas_[table_idx] = delta_scale * dVnorm_dx;

            // Interpolate normalized second derivative: d²V_norm/dx²
            // Pass epoch for O(1) cache freshness check
            const double d2Vnorm_dx2 = value_interp_.eval_from_data(x, snapshot.second_derivative, value_epoch);

            // Transform to dollar gamma using chain rule:
            // gamma = (K_ref/S²) * [d²V_norm/dx² - dV_norm/dx]
            // PERFORMANCE: Use FMA to reduce rounding and enable fused instructions
            const double gamma_scale = K_ref_ * inv_S2;
            gammas_[table_idx] = std::fma(gamma_scale, d2Vnorm_dx2, -gamma_scale * dVnorm_dx);

            // Theta computation
            // American exercise: theta = -L(V) in continuation region, NaN at boundary
            const double obstacle = compute_american_obstacle(S, snapshot.time);
            const double BOUNDARY_TOLERANCE = 1e-6;

            if (std::abs(prices_[table_idx] - obstacle) < BOUNDARY_TOLERANCE) {
                // At exercise boundary
                thetas_[table_idx] = std::numeric_limits<double>::quiet_NaN();
            } else {
                // In continuation region
                const double Lu_norm = lu_interp_.eval(x);
                thetas_[table_idx] = -(K_ref_ * Lu_norm);
            }
        }

        MANGO_TRACE_ALGO_COMPLETE(MODULE_PRICE_TABLE_COLLECTOR, 0, 0);
        return {};  // Success
    }

    /// Legacy collect method - delegates to expected pattern but throws on error
    void collect(const Snapshot& snapshot) override {
        auto result = collect_expected(snapshot);
        if (!result.has_value()) {
            throw std::runtime_error(result.error());
        }
    }

    std::span<const double> prices() const { return prices_; }
    std::span<const double> deltas() const { return deltas_; }
    std::span<const double> gammas() const { return gammas_; }
    std::span<const double> thetas() const { return thetas_; }

    /// Span accessors for workspace borrowing (zero-copy)
    std::span<double> prices_span() { return prices_; }
    std::span<double> deltas_span() { return deltas_; }
    std::span<double> gammas_span() { return gammas_; }
    std::span<double> thetas_span() { return thetas_; }

private:
    /// Get memory resource from arena or default
    std::pmr::memory_resource* get_memory_resource() const {
        // Note: This is safe to call during construction because memory_arena_
        // is initialized before the pmr::vectors in the constructor
        if (memory_arena_) {
            return memory_arena_->resource();
        }
        return std::pmr::get_default_resource();
    }

    /// Check if spatial grid matches cached grid
    [[nodiscard]] bool grids_match(std::span<const double> grid) const noexcept {
        if (cached_grid_.size() != grid.size()) {
            return false;
        }
        // Grid should be identical (not just approximately equal)
        // because it comes from the same solver instance
        return std::equal(cached_grid_.begin(), cached_grid_.end(), grid.begin());
    }

    std::pmr::vector<double> moneyness_;
    std::pmr::vector<double> tau_;
    double K_ref_;
    OptionType option_type_;
    const void* payoff_params_;

    std::pmr::vector<double> prices_;
    std::pmr::vector<double> deltas_;
    std::pmr::vector<double> gammas_;
    std::pmr::vector<double> thetas_;

    // PERFORMANCE: Precomputed values to avoid repeated transcendentals
    std::pmr::vector<double> log_moneyness_;  ///< Cached ln(m) for each moneyness point
    std::pmr::vector<double> spot_values_;    ///< Cached S = m * K_ref
    std::pmr::vector<double> inv_spot_;       ///< Cached 1/S
    std::pmr::vector<double> inv_spot_sq_;    ///< Cached 1/S²

    SnapshotInterpolator value_interp_;
    SnapshotInterpolator lu_interp_;

    // PERFORMANCE: Cached spatial grid for fast rebuild detection
    std::pmr::vector<double> cached_grid_;
    bool interpolators_built_ = false;

    // Memory arena for PMR allocations
    std::shared_ptr<memory::SolverMemoryArena> memory_arena_;

    double compute_american_obstacle(double S, double /*tau*/) const {
        // American option intrinsic value (exercise boundary)
        if (option_type_ == OptionType::CALL) {
            return std::max(S - K_ref_, 0.0);  // Call: max(S - K, 0)
        } else {
            return std::max(K_ref_ - S, 0.0);  // Put: max(K - S, 0)
        }
    }
};

}  // namespace mango

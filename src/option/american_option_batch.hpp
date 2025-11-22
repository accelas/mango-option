/**
 * @file american_option_batch.hpp
 * @brief Batch American option pricing solver for parallel processing
 *
 * Includes both regular batch solver and normalized chain solver (scale-invariant).
 */

#ifndef MANGO_AMERICAN_OPTION_BATCH_HPP
#define MANGO_AMERICAN_OPTION_BATCH_HPP

#include "src/option/american_option.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include "src/pde/core/grid.hpp"
#include "src/math/cubic_spline_solver.hpp"
#include "src/support/error_types.hpp"
#include "src/support/parallel.hpp"
#include <vector>
#include <expected>
#include <span>
#include <functional>
#include <memory>
#include <memory_resource>
#include <tuple>
#include <optional>

namespace mango {

// Forward declaration for PDE parameter grouping
struct PDEParameterGroup;

/**
 * Batch solver result containing individual results and aggregate statistics.
 */
struct BatchAmericanOptionResult {
    std::vector<std::expected<AmericanOptionResult, SolverError>> results;
    size_t failed_count;  ///< Number of failed solves

    /// Check if all solves succeeded
    bool all_succeeded() const { return failed_count == 0; }
};

/// Batch American Option Solver
///
/// Solves multiple American options in parallel using OpenMP.
/// This is significantly faster than solving options sequentially
/// for embarrassingly parallel workloads.
///
/// **Basic usage (per-option grids):**
/// ```cpp
/// std::vector<AmericanOptionParams> batch;
/// batch.emplace_back(spot, strike, maturity, rate, dividend_yield, type, sigma);
///
/// BatchAmericanOptionSolver solver;
/// auto results = solver.solve_batch(batch);
/// ```
///
/// **Adjusting grid accuracy:**
/// ```cpp
/// BatchAmericanOptionSolver solver;
/// GridAccuracyParams accuracy;
/// accuracy.tol = 1e-6;  // High accuracy mode
/// solver.set_grid_accuracy(accuracy);
/// auto results = solver.solve_batch(batch);
/// ```
///
/// **Price table usage (shared grid):**
/// ```cpp
/// // use_shared_grid=true: all options share one global grid
/// BatchAmericanOptionSolver solver;
/// auto results = solver.solve_batch(batch, true);
///
/// // Results contain full surfaces for interpolation
/// for (const auto& result_expected : results.results) {
///     if (result_expected.has_value()) {
///         const auto& result = result_expected.value();
///         auto spatial_solution = result.at_time(step_idx);
///     }
/// }
/// ```
///
/// Performance:
/// - Single-threaded: ~72 options/sec (101x1000 grid, tol=1e-3)
/// - Parallel (32 cores): ~848 options/sec (11.8x speedup)
class BatchAmericanOptionSolver {
public:
    /// Setup callback: called before each solve() to configure solver
    /// @param index Index of current option in params vector
    /// @param solver Reference to solver for pre-solve configuration
    using SetupCallback = std::function<void(size_t index, AmericanOptionSolver& solver)>;

    /// Set grid accuracy parameters
    /// @param accuracy Grid accuracy parameters controlling size/resolution tradeoff
    void set_grid_accuracy(const GridAccuracyParams& accuracy) {
        grid_accuracy_ = accuracy;
    }

    /// Get current grid accuracy parameters
    const GridAccuracyParams& grid_accuracy() const {
        return grid_accuracy_;
    }

    /// Disable normalized chain optimization (for benchmarking/debugging)
    void set_use_normalized(bool enable) {
        use_normalized_ = enable;
    }

    bool use_normalized() const {
        return use_normalized_;
    }

    /// Solve a batch of American options
    ///
    /// @param params Vector of option parameters
    /// @param use_shared_grid If true, all options share one global grid (required for price tables).
    ///                        If false (default), each option gets its own optimal grid.
    ///                        Shared grid enables at_time() access by populating surface_2d.
    /// @param setup Optional callback invoked after solver creation, before solve()
    /// @return Batch result with individual results and failure count
    BatchAmericanOptionResult solve_batch(
        std::span<const AmericanOptionParams> params,
        bool use_shared_grid = false,
        SetupCallback setup = nullptr)
    {
        if (params.empty()) {
            return BatchAmericanOptionResult{.results = {}, .failed_count = 0};
        }

        // Pre-allocate results vector with sentinel errors (parallel access requires pre-sized vector)
        // Since AmericanOptionResult is not copyable, we construct each element in-place
        std::vector<std::expected<AmericanOptionResult, SolverError>> results;
        results.reserve(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            results.emplace_back(std::unexpected(SolverError{
                .code = SolverErrorCode::InvalidConfiguration,
                .message = "Not yet computed",
                .iterations = 0
            }));
        }
        size_t failed_count = 0;

        // Precompute shared grid if needed
        std::optional<std::tuple<GridSpec<double>, size_t>> shared_grid;
        if (use_shared_grid) {
            shared_grid = compute_global_grid_for_batch(params, grid_accuracy_);
        }

        // Precompute workspace size outside parallel region
        size_t workspace_size_elements = 0;
        size_t shared_n_space = 0;

        if (use_shared_grid) {
            auto [grid_spec, n_time] = shared_grid.value();
            shared_n_space = grid_spec.n_points();
            workspace_size_elements = PDEWorkspace::required_size(shared_n_space);
        } else {
            // For per-option grids, estimate max workspace size across all options
            for (const auto& p : params) {
                auto [grid_spec, n_time] = estimate_grid_for_option(p, grid_accuracy_);
                size_t n = grid_spec.n_points();
                workspace_size_elements = std::max(workspace_size_elements, PDEWorkspace::required_size(n));
            }
        }

        // Convert to bytes for monotonic_buffer_resource
        const size_t workspace_size_bytes = workspace_size_elements * sizeof(double);

        MANGO_PRAGMA_PARALLEL
        {
            // Per-thread monotonic buffer sized for workspace reuse
            // Using monotonic_buffer_resource instead of unsynchronized_pool for:
            // - Predictable allocation (no fragmentation)
            // - Faster allocation (bump pointer)
            // - Zero overhead release() (just resets offset)
            std::pmr::monotonic_buffer_resource thread_pool(workspace_size_bytes);

            // Per-thread shared grid (only for shared grid strategy)
            std::shared_ptr<Grid<double>> thread_grid;
            std::pmr::vector<double> thread_buffer(&thread_pool);

            if (use_shared_grid) {
                auto [grid_spec, n_time] = shared_grid.value();
                TimeDomain time_domain = TimeDomain::from_n_steps(0.0, 1.0, n_time);  // Temp time domain for batch

                // Create Grid with solution storage
                auto grid_result = Grid<double>::create(grid_spec, time_domain);
                if (grid_result.has_value()) {
                    thread_grid = grid_result.value();

                    // Allocate buffer for shared workspace (in elements, not bytes)
                    thread_buffer.resize(workspace_size_elements);
                }
                // If creation failed, thread_grid remains null and we'll fail in loop
            }

            // Use static scheduling to avoid false sharing on results vector
            // Each thread gets a contiguous block of iterations
            MANGO_PRAGMA_FOR_STATIC
            for (size_t i = 0; i < params.size(); ++i) {
                // Get or create grid and workspace for this iteration
                std::shared_ptr<Grid<double>> grid;
                std::pmr::vector<double> buffer(&thread_pool);
                PDEWorkspace* workspace_ptr = nullptr;
                std::optional<PDEWorkspace> workspace_storage;

                if (use_shared_grid) {
                    // Shared grid: reuse thread grid and buffer
                    grid = thread_grid;
                    if (grid && !thread_buffer.empty()) {
                        auto workspace_result = PDEWorkspace::from_buffer(thread_buffer, shared_n_space);
                        if (workspace_result.has_value()) {
                            workspace_storage = workspace_result.value();
                            workspace_ptr = &workspace_storage.value();
                        }
                    }
                } else {
                    // Per-option grid: create workspace for this option
                    auto [grid_spec, n_time] = estimate_grid_for_option(params[i], grid_accuracy_);
                    TimeDomain time_domain = TimeDomain::from_n_steps(0.0, params[i].maturity, n_time);

                    // Create Grid with solution storage
                    auto grid_result = Grid<double>::create(grid_spec, time_domain);
                    if (grid_result.has_value()) {
                        grid = grid_result.value();

                        // Allocate buffer and create workspace
                        size_t n = grid_spec.n_points();
                        buffer.resize(PDEWorkspace::required_size(n));
                        auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
                        if (workspace_result.has_value()) {
                            workspace_storage = workspace_result.value();
                            workspace_ptr = &workspace_storage.value();
                        }
                    }
                }

                // Fallback to heap if PMR pool allocation failed
                if (!grid || !workspace_ptr) {
                    // Try allocating from default resource (heap) as fallback
                    std::pmr::vector<double> heap_buffer(std::pmr::get_default_resource());

                    if (!use_shared_grid) {
                        auto [grid_spec, n_time] = estimate_grid_for_option(params[i], grid_accuracy_);
                        TimeDomain time_domain = TimeDomain::from_n_steps(0.0, params[i].maturity, n_time);

                        auto grid_result = Grid<double>::create(grid_spec, time_domain);
                        if (grid_result.has_value()) {
                            grid = grid_result.value();

                            size_t n = grid_spec.n_points();
                            heap_buffer.resize(PDEWorkspace::required_size(n));
                            auto workspace_result = PDEWorkspace::from_buffer(heap_buffer, n);
                            if (workspace_result.has_value()) {
                                workspace_storage = workspace_result.value();
                                workspace_ptr = &workspace_storage.value();
                            }
                        }
                    }

                    // If still failed after heap fallback, report error
                    if (!grid || !workspace_ptr) {
                        results[i] = std::unexpected(SolverError{
                            .code = SolverErrorCode::InvalidConfiguration,
                            .message = "Failed to create workspace (pool and heap fallback failed)",
                            .iterations = 0
                        });
                        MANGO_PRAGMA_ATOMIC
                        ++failed_count;
                        continue;
                    }
                }

                // Create solver using PDEWorkspace API
                AmericanOptionSolver solver(params[i], *workspace_ptr);

                // Invoke setup callback if provided
                if (setup) {
                    setup(i, solver);
                }

                // Solve (use placement new to avoid copy/move assignment issues)
                auto solve_result = solver.solve();
                results[i].~expected();  // Destroy sentinel value
                new (&results[i]) std::expected<AmericanOptionResult, SolverError>(std::move(solve_result));

                if (!results[i].has_value()) {
                    MANGO_PRAGMA_ATOMIC
                    ++failed_count;
                }

                // Release monotonic buffer for next iteration (per-option grid only)
                // For shared grid, thread_buffer stays alive across all iterations
                // For per-option grid, release() resets the allocation offset,
                // allowing efficient reuse across loop iterations
                if (!use_shared_grid) {
                    thread_pool.release();
                }
            }
        }

        return BatchAmericanOptionResult{
            .results = std::move(results),
            .failed_count = failed_count
        };
    }

    /// Solve a batch of American options (vector overload)
    BatchAmericanOptionResult solve_batch(
        const std::vector<AmericanOptionParams>& params,
        bool use_shared_grid = false,
        SetupCallback setup = nullptr)
    {
        return solve_batch(std::span{params}, use_shared_grid, setup);
    }

private:
    GridAccuracyParams grid_accuracy_;  ///< Grid accuracy parameters for automatic estimation

    // Normalized chain solver eligibility constants
    static constexpr double MAX_WIDTH = 5.8;       ///< Convergence limit (log-units)
    static constexpr double MAX_DX = 0.05;         ///< Von Neumann stability
    static constexpr double MIN_MARGIN_ABS = 0.35; ///< 6-cell ghost zone minimum

    bool use_normalized_ = true;  ///< Enable normalized chain optimization

    /// Check if batch qualifies for normalized solving
    bool is_normalized_eligible(
        std::span<const AmericanOptionParams> params,
        bool use_shared_grid) const;

    /// Trace why normalized path wasn't used
    void trace_ineligibility_reason(
        std::span<const AmericanOptionParams> params,
        bool use_shared_grid) const;

    /// Group options by PDE parameters for normalized solving
    std::vector<PDEParameterGroup> group_by_pde_parameters(
        std::span<const AmericanOptionParams> params) const;
};

/// Solve a single American option with automatic grid determination
///
/// Convenience API that automatically determines optimal grid parameters
/// based on option characteristics, eliminating need for manual grid specification.
///
/// Note: Allocates temporary workspace buffer (discarded after solve).
/// For reusable workspaces, caller should manage buffer and use PDEWorkspace directly.
///
/// @param params Option parameters
/// @return Expected containing result on success, error on failure
inline std::expected<AmericanOptionResult, SolverError> solve_american_option_auto(
    const AmericanOptionParams& params)
{
    // Estimate grid for this option
    auto [grid_spec, n_time] = estimate_grid_for_option(params);

    // Allocate workspace buffer (local, temporary)
    size_t n = grid_spec.n_points();
    std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), std::pmr::get_default_resource());

    // Create workspace spans from buffer
    auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
    if (!workspace_result.has_value()) {
        return std::unexpected(SolverError{
            .code = SolverErrorCode::InvalidConfiguration,
            .message = "Failed to create PDEWorkspace: " + workspace_result.error(),
            .iterations = 0
        });
    }

    // Create and solve using PDEWorkspace API
    // Buffer stays alive during solve(), result contains Grid with solution
    AmericanOptionSolver solver(params, workspace_result.value());
    return solver.solve();
}

// ============================================================================
// Normalized Chain Solver (Scale-Invariant)
// ============================================================================

/**
 * Request for normalized PDE solve.
 *
 * Solves dimensionless PDE: ∂u/∂t = 0.5σ²(∂²u/∂x² - ∂u/∂x) + (r-q)∂u/∂x - ru
 * where u = V/K, x = ln(S/K)
 *
 * Output: u(x,τ) on specified grid
 * Caller converts to prices: V = K·u(ln(S/K), τ)
 */
struct NormalizedSolveRequest {
    // PDE coefficients
    double sigma;              ///< Volatility
    double rate;               ///< Risk-free rate
    double dividend;           ///< Continuous dividend yield
    OptionType option_type;    ///< Call or Put

    // Grid configuration
    double x_min;              ///< Minimum log-moneyness
    double x_max;              ///< Maximum log-moneyness
    size_t n_space;            ///< Spatial grid points
    size_t n_time;             ///< Time steps
    double T_max;              ///< Maximum maturity (years)

    // Snapshot collection
    std::span<const double> tau_snapshots;  ///< Maturities to collect

    /// Validate request parameters
    std::expected<void, std::string> validate() const;
};

/**
 * View of normalized solution surface u(x,τ).
 *
 * Provides interpolation interface for querying u at arbitrary (x,τ).
 * Uses separable cubic spline interpolation for C² smoothness and accurate Greeks.
 * Caller scales results: V = K·u
 */
class NormalizedSurfaceView {
public:
    NormalizedSurfaceView(
        std::span<const double> x_grid,
        std::span<const double> tau_grid,
        std::span<const double> values)
        : x_grid_(x_grid)
        , tau_grid_(tau_grid)
        , values_(values)
    {}

    /// Interpolate u(x,τ) using separable cubic spline interpolation
    ///
    /// Provides C² continuity for smooth derivatives (required for Greeks).
    /// Uses natural boundary conditions (f''=0 at endpoints).
    /// Automatically builds cache on first call.
    double interpolate(double x, double tau) const;

    /// Build cached interpolation structure (called once after surface is populated)
    ///
    /// Constructs 2D cubic spline for the entire surface.
    /// Must be called before interpolate() is used.
    /// @return Error message on failure, nullopt on success
    std::optional<std::string> build_cache();

    /// Access raw data (for testing)
    std::span<const double> x_grid() const { return x_grid_; }
    std::span<const double> tau_grid() const { return tau_grid_; }
    std::span<const double> values() const { return values_; }

private:
    std::span<const double> x_grid_;
    std::span<const double> tau_grid_;
    std::span<const double> values_;
    mutable CubicSpline2D<double> spline2d_;  ///< 2D cubic spline interpolator
};

/**
 * Reusable workspace for normalized solves.
 *
 * Allocates all buffers needed for PDE solve + interpolation surface.
 * Thread-safe: each thread creates its own workspace instance.
 */
class NormalizedWorkspace {
public:
    /// Create workspace for given request parameters
    /// Caller must provide PMR buffer for PDEWorkspace (use PDEWorkspace::required_size)
    static std::expected<NormalizedWorkspace, std::string> create(
        const NormalizedSolveRequest& request,
        std::span<double> pde_buffer,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource());

    /// Get view of solution surface (after solve completes)
    NormalizedSurfaceView surface_view();

    // No copying (expensive)
    NormalizedWorkspace(const NormalizedWorkspace&) = delete;
    NormalizedWorkspace& operator=(const NormalizedWorkspace&) = delete;

    // Moving OK
    NormalizedWorkspace(NormalizedWorkspace&&) = default;
    NormalizedWorkspace& operator=(NormalizedWorkspace&&) = default;

private:
    NormalizedWorkspace() = default;

    friend class NormalizedChainSolver;

    PDEWorkspace pde_workspace_;                // Non-owning spans (caller provides buffer)
    std::shared_ptr<Grid<double>> grid_;
    std::vector<double> x_grid_;
    std::vector<double> tau_grid_;
    std::vector<double> values_;  // u(x,τ) [row-major: Nx × Ntau]

public:
    // Accessor for workspace (after create() succeeds)
    PDEWorkspace& workspace() { return pde_workspace_; }
    const PDEWorkspace& workspace() const { return pde_workspace_; }
};

/**
 * Eligibility limits for normalized solver.
 *
 * Thresholds derived from numerical stability and convergence analysis:
 * - Margin: ≥6 ghost cells to avoid boundary reflection (<0.5bp error)
 * - Width: ≤5.8 log-units for convergence (empirical from sweeps)
 * - Grid spacing: ≤0.05 for Von Neumann stability at σ=200%, τ=2y
 */
struct EligibilityLimits {
    static constexpr double MAX_WIDTH = 5.8;      ///< Convergence limit (log-units)
    static constexpr double MAX_DX = 0.05;        ///< Truncation error O(dx²)
    static constexpr double MIN_MARGIN_ABS = 0.35; ///< 6-cell ghost zone minimum

    /// Minimum margin (6 ghost cells or 0.35, whichever is larger)
    static double min_margin(double dx) {
        return std::max(MIN_MARGIN_ABS, 6.0 * dx);
    }

    /// Maximum ratio K_max/K_min given dx
    static double max_ratio(double dx) {
        double margin = min_margin(dx);
        return std::exp(MAX_WIDTH - 2.0 * margin);
    }
};

/**
 * Normalized chain solver.
 *
 * Solves American option PDE in dimensionless coordinates exploiting
 * scale invariance: V(S,K,τ) = K·u(ln(S/K), τ)
 *
 * One PDE solve yields prices for all (S,K) combinations via interpolation.
 */
class NormalizedChainSolver {
public:
    /**
     * Solve normalized PDE.
     *
     * Solves: ∂u/∂t = 0.5σ²(∂²u/∂x² - ∂u/∂x) + (r-q)∂u/∂x - ru
     * Terminal condition: u(x,0) = max(eˣ-1, 0) (call) or max(1-eˣ, 0) (put)
     * Boundary conditions: American exercise constraint u ≥ intrinsic
     *
     * @param request PDE configuration and snapshot times
     * @param workspace Pre-allocated buffers (reusable across solves)
     * @param surface_view Output view (references workspace.values_)
     * @return Success or solver error
     */
    static std::expected<void, SolverError> solve(
        const NormalizedSolveRequest& request,
        NormalizedWorkspace& workspace,
        NormalizedSurfaceView& surface_view);

    /**
     * Check eligibility for normalized solving.
     *
     * Criteria:
     * 1. Grid spacing dx ≤ MAX_DX (0.05)
     * 2. Domain width ≤ MAX_WIDTH (5.8)
     * 3. Margins ≥ min_margin(dx) on both boundaries
     *
     * @param request Request to validate
     * @param moneyness_grid Moneyness values m = K/S from price table
     * @return Success or reason for ineligibility
     */
    static std::expected<void, std::string> check_eligibility(
        const NormalizedSolveRequest& request,
        std::span<const double> moneyness_grid);
};

}  // namespace mango

#endif  // MANGO_AMERICAN_OPTION_BATCH_HPP

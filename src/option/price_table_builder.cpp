#include "src/option/price_table_builder.hpp"
#include "src/option/recursion_helpers.hpp"
#include "src/math/cubic_spline_solver.hpp"
#include "src/math/bspline_nd_separable.hpp"
#include "src/support/memory/aligned_arena.hpp"
#include "src/support/ivcalc_trace.h"
#include "src/pde/core/time_domain.hpp"
#include "src/support/parallel.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <mutex>
#include <tuple>
#include <map>
#include <unordered_set>

namespace mango {

template <size_t N>
PriceTableBuilder<N>::PriceTableBuilder(PriceTableConfig config)
    : config_(std::move(config)) {}

template <size_t N>
std::expected<PriceTableResult<N>, std::string>
PriceTableBuilder<N>::build(const PriceTableAxes<N>& axes) {
    static_assert(N == 4, "PriceTableBuilder only supports N=4");

    // Validate config
    if (auto err = validate_config(config_); err.has_value()) {
        return std::unexpected("Invalid config: " + err.value());
    }

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Step 1: Validate axes
    auto axes_valid = axes.validate();
    if (!axes_valid.has_value()) {
        auto err = axes_valid.error();
        return std::unexpected(
            "Invalid axes (error code " + std::to_string(static_cast<int>(err.code)) +
            ", value=" + std::to_string(err.value) + ")");
    }

    // Check minimum 4 points per axis (B-spline requirement)
    for (size_t i = 0; i < N; ++i) {
        if (axes.grids[i].size() < 4) {
            return std::unexpected("Axis " + std::to_string(i) +
                                   " has only " + std::to_string(axes.grids[i].size()) +
                                   " points (need >=4 for cubic B-splines)");
        }
    }

    // Check positive moneyness (needed for log)
    if (axes.grids[0].front() <= 0.0) {
        return std::unexpected("Moneyness must be positive (needed for log)");
    }

    // Check positive maturity (strict > 0)
    if (axes.grids[1].front() <= 0.0) {
        return std::unexpected("Maturity must be positive (tau > 0 required for PDE time domain)");
    }

    // Check positive volatility
    if (axes.grids[2].front() <= 0.0) {
        return std::unexpected("Volatility must be positive");
    }

    // Check K_ref > 0
    if (config_.K_ref <= 0.0) {
        return std::unexpected("Reference strike K_ref must be positive");
    }

    // Check PDE domain coverage
    const double x_min_requested = std::log(axes.grids[0].front());
    const double x_max_requested = std::log(axes.grids[0].back());
    const double x_min = config_.grid_estimator.x_min();
    const double x_max = config_.grid_estimator.x_max();

    if (x_min_requested < x_min || x_max_requested > x_max) {
        return std::unexpected(
            "Requested moneyness range [" + std::to_string(axes.grids[0].front()) + ", " +
            std::to_string(axes.grids[0].back()) + "] in spot ratios "
            "maps to log-moneyness [" + std::to_string(x_min_requested) + ", " +
            std::to_string(x_max_requested) + "], "
            "which exceeds PDE grid bounds [" + std::to_string(x_min) + ", " +
            std::to_string(x_max) + "]. "
            "Narrow the moneyness grid or expand the PDE domain."
        );
    }

    // Step 2: Generate batch (Nσ × Nr entries)
    auto batch_params = make_batch(axes);
    if (batch_params.empty()) {
        return std::unexpected("make_batch returned empty batch");
    }

    // Step 3: Solve batch with snapshot registration
    auto batch_result = solve_batch(batch_params, axes);
    if (batch_result.failed_count > 0) {
        return std::unexpected(
            "solve_batch had " + std::to_string(batch_result.failed_count) +
            " failures out of " + std::to_string(batch_result.results.size()));
    }

    // Count PDE solves (successful results)
    size_t n_pde_solves = batch_result.results.size() - batch_result.failed_count;

    // Step 4: Extract tensor via interpolation
    auto extraction = extract_tensor(batch_result, axes);
    if (!extraction.has_value()) {
        return std::unexpected("extract_tensor failed: " + extraction.error());
    }

    // Step 5: Fit B-spline coefficients
    auto coeffs_result = fit_coeffs(extraction->tensor, axes);
    if (!coeffs_result.has_value()) {
        return std::unexpected("fit_coeffs failed: " + coeffs_result.error());
    }

    auto& fit_result = coeffs_result.value();
    auto coefficients = std::move(fit_result.coefficients);
    BSplineFittingStats fitting_stats = fit_result.stats;

    // Step 6: Create metadata
    PriceTableMetadata metadata{
        .K_ref = config_.K_ref,
        .dividend_yield = config_.dividend_yield,
        .discrete_dividends = config_.discrete_dividends
    };

    // Step 7: Build immutable surface
    auto surface_result = PriceTableSurface<N>::build(axes, std::move(coefficients), metadata);
    if (!surface_result.has_value()) {
        return std::unexpected("Surface build failed: " + surface_result.error());
    }

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    // Return full result with diagnostics
    return PriceTableResult<N>{
        .surface = std::move(surface_result.value()),
        .n_pde_solves = n_pde_solves,
        .precompute_time_seconds = elapsed,
        .fitting_stats = fitting_stats
    };
}

template <size_t N>
std::vector<AmericanOptionParams>
PriceTableBuilder<N>::make_batch(const PriceTableAxes<N>& axes) const {
    static_assert(N == 4, "PriceTableBuilder only supports N=4");

    std::vector<AmericanOptionParams> batch;

    // Iterate only over high-cost axes: axes[2] (σ) and axes[3] (r)
    // This creates Nσ × Nr batch entries, NOT Nm × Nt × Nσ × Nr
    // Each solve produces a surface over (m, τ) that gets reused
    const size_t Nσ = axes.grids[2].size();
    const size_t Nr = axes.grids[3].size();
    batch.reserve(Nσ * Nr);

    // Normalized parameters: Spot = Strike = K_ref
    // Moneyness and maturity are handled via grid interpolation in extract_tensor
    const double K_ref = config_.K_ref;

    for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
        for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
            double sigma = axes.grids[2][σ_idx];
            double r = axes.grids[3][r_idx];

            // Normalized solve: Spot = Strike = K_ref
            // Surface will be interpolated across m and τ in extract_tensor
            AmericanOptionParams params(
                K_ref,                          // spot
                K_ref,                          // strike
                axes.grids[1].back(),           // maturity (max for this σ,r)
                r,                              // rate
                config_.dividend_yield,         // dividend_yield
                config_.option_type,            // type
                sigma,                          // volatility
                config_.discrete_dividends      // discrete_dividends
            );

            batch.push_back(params);
        }
    }

    return batch;
}

template <size_t N>
BatchAmericanOptionResult
PriceTableBuilder<N>::solve_batch(
    const std::vector<AmericanOptionParams>& batch,
    const PriceTableAxes<N>& axes) const
{
    static_assert(N == 4, "PriceTableBuilder only supports N=4");

    BatchAmericanOptionSolver solver;

    // Register maturity grid as snapshot times
    // This enables extract_tensor to access surfaces at each maturity point
    solver.set_snapshot_times(axes.grids[1]);  // axes.grids[1] = maturity axis

    // Solver stability constraints (from BatchAmericanOptionSolver)
    constexpr double MAX_WIDTH = 5.8;   // Convergence limit (log-units)
    constexpr double MAX_DX = 0.05;     // Von Neumann stability

    // Check if user's grid_spec meets solver constraints
    // Note: Domain coverage (PDE grid covers moneyness range) is already
    // validated in build() at lines 69-85, so we don't re-check here.
    const double grid_width = config_.grid_estimator.x_max() - config_.grid_estimator.x_min();

    // Compute actual max spacing for non-uniform grids
    // Sinh grids concentrate points at center, so max spacing is in wings
    // Using average dx would underestimate and potentially violate Von Neumann
    double max_dx;
    if (config_.grid_estimator.type() == GridSpec<double>::Type::Uniform) {
        // Uniform grid: all spacings equal
        max_dx = grid_width / static_cast<double>(config_.grid_estimator.n_points() - 1);
    } else {
        // Non-uniform grid: generate and find actual max spacing
        auto grid_buffer = config_.grid_estimator.generate();
        max_dx = 0.0;
        for (size_t i = 1; i < grid_buffer.size(); ++i) {
            double spacing = grid_buffer[i] - grid_buffer[i-1];
            max_dx = std::max(max_dx, spacing);
        }
    }

    // Compute minimum required width based on option parameters
    // For accuracy, grid should cover ~3σ√τ on each side of log-moneyness
    double max_sigma_sqrt_tau = 0.0;
    for (const auto& p : batch) {
        double sigma_sqrt_tau = p.volatility * std::sqrt(p.maturity);
        max_sigma_sqrt_tau = std::max(max_sigma_sqrt_tau, sigma_sqrt_tau);
    }
    const double min_required_width = 6.0 * max_sigma_sqrt_tau;  // 3σ√τ each side

    const bool grid_meets_constraints =
        (grid_width <= MAX_WIDTH) &&
        (max_dx <= MAX_DX) &&
        (grid_width >= min_required_width);

    if (grid_meets_constraints) {
        // Grid meets constraints: use custom_grid directly
        // This honors user's exact spatial resolution request
        const double max_maturity = axes.grids[1].back();
        TimeDomain time_domain = TimeDomain::from_n_steps(0.0, max_maturity, config_.n_time);
        auto custom_grid = std::make_pair(config_.grid_estimator, time_domain);
        return solver.solve_batch(batch, true, nullptr, custom_grid);
    } else {
        // Grid violates constraints: use auto-estimation with configured bounds
        // This ensures solver stability while covering requested domain
        GridAccuracyParams accuracy;
        const size_t n_points = config_.grid_estimator.n_points();
        const size_t clamped = std::clamp(n_points, size_t(100), size_t(1200));
        accuracy.min_spatial_points = clamped;
        accuracy.max_spatial_points = clamped;
        accuracy.max_time_steps = config_.n_time;

        // Extract alpha parameter for sinh-spaced grids
        if (config_.grid_estimator.type() == GridSpec<double>::Type::SinhSpaced) {
            accuracy.alpha = config_.grid_estimator.concentration();
        }

        // Compute n_sigma to cover user's requested domain bounds
        // Domain is centered at x=0 (ATM), with half-width = n_sigma * max_sigma_sqrt_tau
        // User's domain: [x_min, x_max] from grid_estimator
        // Required: n_sigma >= max(|x_min|, |x_max|) / max_sigma_sqrt_tau
        const double x_min = config_.grid_estimator.x_min();
        const double x_max = config_.grid_estimator.x_max();
        const double max_abs_x = std::max(std::abs(x_min), std::abs(x_max));

        // Safety margin (10%) for boundary effects in PDE solver
        constexpr double DOMAIN_MARGIN_FACTOR = 1.1;

        // Compute required n_sigma, guarding against near-zero sigma*sqrt(tau)
        // (which could happen with very short maturities or near-zero volatility)
        if (max_sigma_sqrt_tau < 1e-10) {
            // Fallback to default n_sigma when volatility × sqrt(maturity) ≈ 0
            accuracy.n_sigma = 5.0;
        } else {
            double required_n_sigma = (max_abs_x / max_sigma_sqrt_tau) * DOMAIN_MARGIN_FACTOR;
            // Use at least the default (5.0) but expand if needed for user's domain
            accuracy.n_sigma = std::max(5.0, required_n_sigma);
        }

        solver.set_grid_accuracy(accuracy);
        return solver.solve_batch(batch, true);  // use_shared_grid = true, auto-estimation
    }
}

template <size_t N>
std::expected<ExtractionResult<N>, std::string>
PriceTableBuilder<N>::extract_tensor(
    const BatchAmericanOptionResult& batch,
    const PriceTableAxes<N>& axes) const
{
    static_assert(N == 4, "PriceTableBuilder only supports N=4");

    const size_t Nm = axes.grids[0].size();  // moneyness
    const size_t Nt = axes.grids[1].size();  // maturity
    const size_t Nσ = axes.grids[2].size();  // volatility
    const size_t Nr = axes.grids[3].size();  // rate

    // Verify batch size matches (σ, r) grid
    const size_t expected_batch_size = Nσ * Nr;
    if (batch.results.size() != expected_batch_size) {
        MANGO_TRACE_RUNTIME_ERROR(MODULE_PRICE_TABLE,
            batch.results.size(), expected_batch_size);
        return std::unexpected(
            "Batch size mismatch: expected " + std::to_string(expected_batch_size) +
            " results (Nσ × Nr), got " + std::to_string(batch.results.size()));
    }

    // Create tensor (use axes.total_points() for consistency with validation)
    const size_t total_points = axes.total_points();
    const size_t tensor_bytes = total_points * sizeof(double);
    const size_t arena_bytes = tensor_bytes + 64;  // 64-byte alignment padding

    auto arena = memory::AlignedArena::create(arena_bytes);
    if (!arena.has_value()) {
        return std::unexpected("Failed to create arena: " + arena.error());
    }

    std::array<size_t, N> shape = {Nm, Nt, Nσ, Nr};
    auto tensor_result = PriceTensor<N>::create(shape, arena.value());
    if (!tensor_result.has_value()) {
        return std::unexpected("Failed to create tensor: " + tensor_result.error());
    }

    auto tensor = tensor_result.value();

    // Precompute log-moneyness for interpolation
    std::vector<double> log_moneyness(Nm);
    for (size_t i = 0; i < Nm; ++i) {
        log_moneyness[i] = std::log(axes.grids[0][i]);
    }

    // Failure tracking
    std::vector<size_t> failed_pde;
    std::vector<std::tuple<size_t, size_t, size_t>> failed_spline;
    std::mutex failed_mutex;

    // Extract prices from each (σ, r) surface (parallelized)
    MANGO_PRAGMA_PARALLEL
    {
        MANGO_PRAGMA_FOR_COLLAPSE2
        for (size_t σ_idx = 0; σ_idx < Nσ; ++σ_idx) {
            for (size_t r_idx = 0; r_idx < Nr; ++r_idx) {
                size_t batch_idx = σ_idx * Nr + r_idx;
                const auto& result_expected = batch.results[batch_idx];

                if (!result_expected.has_value()) {
                    // Track PDE failure
                    {
                        std::lock_guard<std::mutex> lock(failed_mutex);
                        failed_pde.push_back(batch_idx);
                    }
                    // Fill with NaN for failed solves
                    for (size_t i = 0; i < Nm; ++i) {
                        for (size_t j = 0; j < Nt; ++j) {
                            tensor.view[i, j, σ_idx, r_idx] = std::numeric_limits<double>::quiet_NaN();
                        }
                    }
                    continue;
                }

                const auto& result = result_expected.value();
                auto grid = result.grid();
                auto x_grid = grid->x();  // Spatial grid (log-moneyness)

                // For each maturity snapshot
                for (size_t j = 0; j < Nt; ++j) {
                    // Get spatial solution at this maturity
                    std::span<const double> spatial_solution = result.at_time(j);

                    // Interpolate across moneyness using cubic spline
                    // This resamples the PDE solution onto our moneyness grid
                    CubicSpline<double> spline;
                    auto build_error = spline.build(x_grid, spatial_solution);

                    if (build_error.has_value()) {
                        // Track spline failure
                        {
                            std::lock_guard<std::mutex> lock(failed_mutex);
                            failed_spline.emplace_back(σ_idx, r_idx, j);
                        }
                        // Spline fitting failed, fill with NaN
                        for (size_t i = 0; i < Nm; ++i) {
                            tensor.view[i, j, σ_idx, r_idx] = std::numeric_limits<double>::quiet_NaN();
                        }
                        continue;
                    }

                    // Evaluate spline at each moneyness point and scale by K_ref
                    // PDE solves are normalized (Spot=Strike=K_ref), so V_normalized
                    // needs to be scaled back to actual prices: V_actual = K_ref * V_norm
                    const double K_ref = config_.K_ref;
                    for (size_t i = 0; i < Nm; ++i) {
                        double normalized_price = spline.eval(log_moneyness[i]);
                        tensor.view[i, j, σ_idx, r_idx] = K_ref * normalized_price;
                    }
                }
            }
        }
    }

    return ExtractionResult<N>{
        .tensor = std::move(tensor),
        .total_slices = Nσ * Nr,
        .failed_pde = std::move(failed_pde),
        .failed_spline = std::move(failed_spline)
    };
}

template <size_t N>
std::expected<typename PriceTableBuilder<N>::FitCoeffsResult, std::string>
PriceTableBuilder<N>::fit_coeffs(
    const PriceTensor<N>& tensor,
    const PriceTableAxes<N>& axes) const
{
    static_assert(N == 4, "PriceTableBuilder only supports N=4");

    // Extract grids for BSplineNDSeparable
    std::array<std::vector<double>, N> grids;
    for (size_t i = 0; i < N; ++i) {
        grids[i] = axes.grids[i];
    }

    // Create fitter
    auto fitter_result = BSplineNDSeparable<double, N>::create(std::move(grids));
    if (!fitter_result.has_value()) {
        return std::unexpected("Failed to create fitter: " + fitter_result.error());
    }

    // Extract values from tensor (convert mdspan to vector)
    size_t total_points = axes.total_points();
    std::vector<double> values;
    values.reserve(total_points);

    // Extract in row-major order using for_each_axis_index
    if constexpr (N == 4) {
        for_each_axis_index<0>(axes, [&](const std::array<size_t, N>& indices) {
            values.push_back(tensor.view[indices[0], indices[1], indices[2], indices[3]]);
        });
    }

    // Fit B-spline coefficients
    auto fit_result = fitter_result->fit(values);
    if (!fit_result.has_value()) {
        return std::unexpected("B-spline fitting failed: " + fit_result.error());
    }

    const auto& result = fit_result.value();

    // Map BSplineNDSeparableResult to BSplineFittingStats
    BSplineFittingStats stats;
    stats.max_residual_axis0 = result.max_residual_per_axis[0];
    stats.max_residual_axis1 = result.max_residual_per_axis[1];
    stats.max_residual_axis2 = result.max_residual_per_axis[2];
    stats.max_residual_axis3 = result.max_residual_per_axis[3];
    stats.max_residual_overall = *std::max_element(
        result.max_residual_per_axis.begin(),
        result.max_residual_per_axis.end()
    );

    stats.condition_axis0 = result.condition_per_axis[0];
    stats.condition_axis1 = result.condition_per_axis[1];
    stats.condition_axis2 = result.condition_per_axis[2];
    stats.condition_axis3 = result.condition_per_axis[3];
    stats.condition_max = *std::max_element(
        result.condition_per_axis.begin(),
        result.condition_per_axis.end()
    );

    stats.failed_slices_axis0 = result.failed_slices[0];
    stats.failed_slices_axis1 = result.failed_slices[1];
    stats.failed_slices_axis2 = result.failed_slices[2];
    stats.failed_slices_axis3 = result.failed_slices[3];
    stats.failed_slices_total = std::accumulate(
        result.failed_slices.begin(),
        result.failed_slices.end(),
        size_t(0)
    );

    return FitCoeffsResult{
        .coefficients = std::move(result.coefficients),
        .stats = stats
    };
}

// Helper: sort and dedupe a vector
namespace {
std::vector<double> sort_and_dedupe(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}
}  // namespace

// Factory method implementations (explicit specialization for N=4)
template <>
std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, std::string>
PriceTableBuilder<4>::from_vectors(
    std::vector<double> moneyness,
    std::vector<double> maturity,
    std::vector<double> volatility,
    std::vector<double> rate,
    double K_ref,
    GridSpec<double> grid_spec,
    size_t n_time,
    OptionType type,
    double dividend_yield,
    double max_failure_rate)
{
    // Sort and dedupe
    moneyness = sort_and_dedupe(std::move(moneyness));
    maturity = sort_and_dedupe(std::move(maturity));
    volatility = sort_and_dedupe(std::move(volatility));
    rate = sort_and_dedupe(std::move(rate));

    // Validate positivity
    if (!moneyness.empty() && moneyness.front() <= 0.0) {
        return std::unexpected("Moneyness must be positive");
    }
    if (!maturity.empty() && maturity.front() <= 0.0) {
        return std::unexpected("Maturity must be positive");
    }
    if (!volatility.empty() && volatility.front() <= 0.0) {
        return std::unexpected("Volatility must be positive");
    }
    if (K_ref <= 0.0) {
        return std::unexpected("K_ref must be positive");
    }

    // Build axes
    PriceTableAxes<4> axes;
    axes.grids[0] = std::move(moneyness);
    axes.grids[1] = std::move(maturity);
    axes.grids[2] = std::move(volatility);
    axes.grids[3] = std::move(rate);
    axes.names = {"moneyness", "maturity", "volatility", "rate"};

    // Build config
    PriceTableConfig config;
    config.option_type = type;
    config.K_ref = K_ref;
    config.grid_estimator = grid_spec;
    config.n_time = n_time;
    config.dividend_yield = dividend_yield;
    config.max_failure_rate = max_failure_rate;

    // Validate config
    if (auto err = validate_config(config); err.has_value()) {
        return std::unexpected(err.value());
    }

    return std::make_pair(PriceTableBuilder<4>(config), std::move(axes));
}

template <>
std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, std::string>
PriceTableBuilder<4>::from_strikes(
    double spot,
    std::vector<double> strikes,
    std::vector<double> maturities,
    std::vector<double> volatilities,
    std::vector<double> rates,
    GridSpec<double> grid_spec,
    size_t n_time,
    OptionType type,
    double dividend_yield,
    double max_failure_rate)
{
    if (spot <= 0.0) {
        return std::unexpected("Spot must be positive");
    }

    // Sort and dedupe
    strikes = sort_and_dedupe(std::move(strikes));
    maturities = sort_and_dedupe(std::move(maturities));
    volatilities = sort_and_dedupe(std::move(volatilities));
    rates = sort_and_dedupe(std::move(rates));

    // Validate strikes positive
    if (!strikes.empty() && strikes.front() <= 0.0) {
        return std::unexpected("Strikes must be positive");
    }

    // Compute moneyness = spot/strike
    std::vector<double> moneyness;
    moneyness.reserve(strikes.size());
    for (double K : strikes) {
        moneyness.push_back(spot / K);
    }
    // Note: if strikes are ascending, moneyness is descending
    // Sort to make ascending
    std::sort(moneyness.begin(), moneyness.end());

    return from_vectors(
        std::move(moneyness),
        std::move(maturities),
        std::move(volatilities),
        std::move(rates),
        spot,  // K_ref = spot
        grid_spec,
        n_time,
        type,
        dividend_yield,
        max_failure_rate
    );
}

template <>
std::expected<std::pair<PriceTableBuilder<4>, PriceTableAxes<4>>, std::string>
PriceTableBuilder<4>::from_chain(
    const OptionChain& chain,
    GridSpec<double> grid_spec,
    size_t n_time,
    OptionType type,
    double max_failure_rate)
{
    return from_strikes(
        chain.spot,
        chain.strikes,
        chain.maturities,
        chain.implied_vols,
        chain.rates,
        grid_spec,
        n_time,
        type,
        chain.dividend_yield,
        max_failure_rate
    );
}

template <size_t N>
std::optional<std::pair<size_t, size_t>>
PriceTableBuilder<N>::find_nearest_valid_neighbor(
    size_t σ_idx, size_t r_idx, size_t Nσ, size_t Nr,
    const std::vector<bool>& slice_valid) const
{
    const size_t max_dist = (Nσ - 1) + (Nr - 1);

    for (size_t dist = 1; dist <= max_dist; ++dist) {
        for (int dσ = -static_cast<int>(dist); dσ <= static_cast<int>(dist); ++dσ) {
            int dr = static_cast<int>(dist) - std::abs(dσ);
            for (int sign : {-1, 1}) {
                int nσ = static_cast<int>(σ_idx) + dσ;
                int nr = static_cast<int>(r_idx) + sign * dr;
                if (nσ >= 0 && nσ < static_cast<int>(Nσ) &&
                    nr >= 0 && nr < static_cast<int>(Nr)) {
                    if (slice_valid[nσ * Nr + nr]) {
                        return std::make_pair(static_cast<size_t>(nσ),
                                              static_cast<size_t>(nr));
                    }
                }
            }
        }
    }
    return std::nullopt;
}

// Explicit instantiation (only N=4 supported)
template class PriceTableBuilder<4>;

} // namespace mango

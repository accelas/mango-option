// SPDX-License-Identifier: MIT
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/recursion_helpers.hpp"
#include "src/math/cubic_spline_solver.hpp"
#include "src/math/bspline_nd_separable.hpp"
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
#include <ranges>

namespace mango {

template <size_t N>
PriceTableBuilder<N>::PriceTableBuilder(PriceTableConfig config)
    : config_(std::move(config)) {}

template <size_t N>
std::expected<PriceTableResult<N>, PriceTableError>
PriceTableBuilder<N>::build(const PriceTableAxes<N>& axes) {
    static_assert(N == 4, "PriceTableBuilder only supports N=4");

    // Validate config
    if (auto err = validate_config(config_); err.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Step 1: Validate axes
    auto axes_valid = axes.validate();
    if (!axes_valid.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    // Check minimum 4 points per axis (B-spline requirement)
    for (size_t i = 0; i < N; ++i) {
        if (axes.grids[i].size() < 4) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::InsufficientGridPoints, i, axes.grids[i].size()});
        }
    }

    // Check positive moneyness (needed for log)
    if (axes.grids[0].front() <= 0.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::NonPositiveValue, 0});
    }

    // Check positive maturity (strict > 0)
    if (axes.grids[1].front() <= 0.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::NonPositiveValue, 1});
    }

    // Check positive volatility
    if (axes.grids[2].front() <= 0.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::NonPositiveValue, 2});
    }

    // Check K_ref > 0
    if (config_.K_ref <= 0.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::NonPositiveValue, 4});
    }

    // Check PDE domain coverage (only for explicit grids; auto-estimated grids
    // are computed from batch parameters and always cover the needed domain)
    if (auto* explicit_grid = std::get_if<ExplicitPDEGrid>(&config_.pde_grid)) {
        const double x_min_requested = std::log(axes.grids[0].front());
        const double x_max_requested = std::log(axes.grids[0].back());
        const double x_min = explicit_grid->grid_spec.x_min();
        const double x_max = explicit_grid->grid_spec.x_max();

        if (x_min_requested < x_min || x_max_requested > x_max) {
            return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
        }
    }

    // Step 2: Generate batch (Nσ × Nr entries)
    auto batch_params = make_batch(axes);
    if (batch_params.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::EmptyBatch});
    }

    // Step 3: Solve batch with snapshot registration
    auto batch_result = solve_batch(batch_params, axes);

    // Check failure rate against threshold
    const double failure_rate = static_cast<double>(batch_result.failed_count) /
                                static_cast<double>(batch_result.results.size());
    if (failure_rate > config_.max_failure_rate) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::ExtractionFailed, 0, batch_result.failed_count});
    }

    // Count PDE solves (successful results)
    size_t n_pde_solves = batch_result.results.size() - batch_result.failed_count;

    // Step 4: Extract tensor via interpolation
    auto extraction = extract_tensor(batch_result, axes);
    if (!extraction.has_value()) {
        return std::unexpected(extraction.error());
    }

    // Step 4b: Repair failed slices
    auto repair_result = repair_failed_slices(
        extraction->tensor, extraction->failed_pde, extraction->failed_spline, axes);
    if (!repair_result.has_value()) {
        return std::unexpected(repair_result.error());
    }
    auto repair_stats = repair_result.value();

    // Step 5: Fit B-spline coefficients
    auto coeffs_result = fit_coeffs(extraction->tensor, axes);
    if (!coeffs_result.has_value()) {
        return std::unexpected(coeffs_result.error());
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
        return std::unexpected(PriceTableError{PriceTableErrorCode::SurfaceBuildFailed});
    }

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    // Return full result with diagnostics
    const size_t Nt = axes.grids[1].size();
    return PriceTableResult<N>{
        .surface = std::move(surface_result.value()),
        .n_pde_solves = n_pde_solves,
        .precompute_time_seconds = elapsed,
        .fitting_stats = fitting_stats,
        .failed_pde_slices = extraction->failed_pde.size(),
        .failed_spline_points = extraction->failed_spline.size(),
        .repaired_full_slices = repair_stats.repaired_full_slices,
        .repaired_partial_points = repair_stats.repaired_partial_points,
        .total_slices = extraction->total_slices,
        .total_points = extraction->total_slices * Nt
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

    // C++23 cartesian_product: iterate over (σ, r) combinations
    for (auto [sigma, r] : std::views::cartesian_product(axes.grids[2], axes.grids[3])) {
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

    return batch;
}

/// Ensure n_sigma is large enough so the PDE domain covers [log(m_min), log(m_max)].
/// The batch solves use spot=strike=K_ref (x0=0), so the grid half-width is
/// n_sigma * max(σ√T).  If the moneyness axis extends beyond that, we bump n_sigma.
template <size_t N>
static void ensure_moneyness_coverage(
    GridAccuracyParams& accuracy,
    const std::vector<AmericanOptionParams>& batch,
    const PriceTableAxes<N>& axes)
{
    const double log_m_min = std::log(axes.grids[0].front());
    const double log_m_max = std::log(axes.grids[0].back());
    const double required_half_width = std::max(std::abs(log_m_min), std::abs(log_m_max));

    // Compute max σ√T across the batch (floor to avoid division by zero)
    double max_sigma_sqrt_T = 0.0;
    for (const auto& p : batch) {
        max_sigma_sqrt_T = std::max(max_sigma_sqrt_T,
                                     p.volatility * std::sqrt(p.maturity));
    }
    max_sigma_sqrt_T = std::max(max_sigma_sqrt_T, 1e-10);

    constexpr double MARGIN = 1.1;  // 10% margin for boundary effects
    double required_n_sigma = (required_half_width / max_sigma_sqrt_T) * MARGIN;
    accuracy.n_sigma = std::max(accuracy.n_sigma, required_n_sigma);
}

template <size_t N>
std::pair<GridSpec<double>, TimeDomain>
PriceTableBuilder<N>::estimate_pde_grid(
    const std::vector<AmericanOptionParams>& batch,
    const PriceTableAxes<N>& axes) const
{
    auto accuracy = std::get<GridAccuracyParams>(config_.pde_grid);
    ensure_moneyness_coverage<N>(accuracy, batch, axes);

    auto [grid_spec, time_domain] = compute_global_grid_for_batch(
        std::span<const AmericanOptionParams>(batch), accuracy);

    // Extend time domain to cover max maturity from axes
    const double max_maturity = axes.grids[1].back();
    time_domain = TimeDomain::from_n_steps(0.0, max_maturity, time_domain.n_steps());

    return {grid_spec, time_domain};
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

    return std::visit([&](auto&& grid) -> BatchAmericanOptionResult {
        using T = std::decay_t<decltype(grid)>;

        if constexpr (std::is_same_v<T, GridAccuracyParams>) {
            // Auto-estimate PDE grid from batch parameters
            auto custom_grid = estimate_pde_grid(batch, axes);
            return solver.solve_batch(batch, true, nullptr, custom_grid);
        } else {
            // Explicit grid: check solver stability constraints
            const auto& grid_spec = grid.grid_spec;
            const auto n_time = grid.n_time;

            constexpr double MAX_WIDTH = 5.8;   // Convergence limit (log-units)
            constexpr double MAX_DX = 0.05;     // Von Neumann stability

            const double grid_width = grid_spec.x_max() - grid_spec.x_min();

            // Compute actual max spacing for non-uniform grids
            double max_dx;
            if (grid_spec.type() == GridSpec<double>::Type::Uniform) {
                max_dx = grid_width / static_cast<double>(grid_spec.n_points() - 1);
            } else {
                auto grid_buffer = grid_spec.generate();
                auto spacings = grid_buffer.span() | std::views::pairwise
                                                   | std::views::transform([](auto pair) {
                                                         auto [a, b] = pair;
                                                         return b - a;
                                                     });
                max_dx = std::ranges::max(spacings);
            }

            // Compute minimum required width: ~3σ√τ on each side
            auto sigma_sqrt_tau = [](const AmericanOptionParams& p) {
                return p.volatility * std::sqrt(p.maturity);
            };
            const double max_sigma_sqrt_tau = std::ranges::max(
                batch | std::views::transform(sigma_sqrt_tau));
            const double min_required_width = 6.0 * max_sigma_sqrt_tau;

            const bool grid_meets_constraints =
                (grid_width <= MAX_WIDTH) &&
                (max_dx <= MAX_DX) &&
                (grid_width >= min_required_width);

            if (grid_meets_constraints) {
                const double max_maturity = axes.grids[1].back();
                TimeDomain time_domain = TimeDomain::from_n_steps(0.0, max_maturity, n_time);
                auto custom_grid = std::make_pair(grid_spec, time_domain);
                return solver.solve_batch(batch, true, nullptr, custom_grid);
            } else {
                // Grid violates constraints: fall back to auto-estimation
                GridAccuracyParams accuracy;
                const size_t n_points = grid_spec.n_points();
                const size_t clamped = std::clamp(n_points, size_t(100), size_t(1200));
                accuracy.min_spatial_points = clamped;
                accuracy.max_spatial_points = clamped;
                accuracy.max_time_steps = n_time;

                if (grid_spec.type() == GridSpec<double>::Type::SinhSpaced) {
                    accuracy.alpha = grid_spec.concentration();
                }

                const double x_min = grid_spec.x_min();
                const double x_max = grid_spec.x_max();
                const double max_abs_x = std::max(std::abs(x_min), std::abs(x_max));
                constexpr double DOMAIN_MARGIN_FACTOR = 1.1;

                const double safe_sigma_sqrt_tau = std::max(max_sigma_sqrt_tau, 1e-10);
                double required_n_sigma = (max_abs_x / safe_sigma_sqrt_tau) * DOMAIN_MARGIN_FACTOR;
                accuracy.n_sigma = std::max(5.0, required_n_sigma);

                // Also ensure coverage of the moneyness axis
                ensure_moneyness_coverage<N>(accuracy, batch, axes);

                solver.set_grid_accuracy(accuracy);
                return solver.solve_batch(batch, true);
            }
        }
    }, config_.pde_grid);
}

template <size_t N>
std::expected<ExtractionResult<N>, PriceTableError>
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
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::ExtractionFailed, 0, batch.results.size()});
    }

    // Create tensor
    std::array<size_t, N> shape = {Nm, Nt, Nσ, Nr};
    auto tensor_result = PriceTensor<N>::create(shape);
    if (!tensor_result.has_value()) {
        return std::unexpected(PriceTableError{
            PriceTableErrorCode::TensorCreationFailed, 0, axes.total_points()});
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
std::expected<typename PriceTableBuilder<N>::FitCoeffsResult, PriceTableError>
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
        return std::unexpected(PriceTableError{PriceTableErrorCode::FittingFailed});
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
        return std::unexpected(PriceTableError{PriceTableErrorCode::FittingFailed});
    }

    const auto& result = fit_result.value();

    // Convert BSplineNDSeparableResult to BSplineFittingStats
    auto stats = result.to_stats();

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
PriceTableBuilder<4>::Setup
PriceTableBuilder<4>::from_vectors(
    std::vector<double> moneyness,
    std::vector<double> maturity,
    std::vector<double> volatility,
    std::vector<double> rate,
    double K_ref,
    PDEGridSpec pde_grid,
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
        return std::unexpected(PriceTableError{PriceTableErrorCode::NonPositiveValue, 0});
    }
    if (!maturity.empty() && maturity.front() <= 0.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::NonPositiveValue, 1});
    }
    if (!volatility.empty() && volatility.front() <= 0.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::NonPositiveValue, 2});
    }
    if (K_ref <= 0.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::NonPositiveValue, 4});
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
    config.pde_grid = std::move(pde_grid);
    config.dividend_yield = dividend_yield;
    config.max_failure_rate = max_failure_rate;

    // Validate config
    if (auto err = validate_config(config); err.has_value()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    return std::make_pair(PriceTableBuilder<4>(config), std::move(axes));
}

template <>
PriceTableBuilder<4>::Setup
PriceTableBuilder<4>::from_strikes(
    double spot,
    std::vector<double> strikes,
    std::vector<double> maturities,
    std::vector<double> volatilities,
    std::vector<double> rates,
    PDEGridSpec pde_grid,
    OptionType type,
    double dividend_yield,
    double max_failure_rate)
{
    if (spot <= 0.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::NonPositiveValue, 4});
    }

    // Sort and dedupe
    strikes = sort_and_dedupe(std::move(strikes));
    maturities = sort_and_dedupe(std::move(maturities));
    volatilities = sort_and_dedupe(std::move(volatilities));
    rates = sort_and_dedupe(std::move(rates));

    // Validate strikes positive
    if (!strikes.empty() && strikes.front() <= 0.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::NonPositiveValue, 0});
    }

    // Compute moneyness = spot/strike
    // C++23 ranges::to materializes the transform view into a vector
    auto moneyness = strikes
        | std::views::transform([spot](double K) { return spot / K; })
        | std::ranges::to<std::vector>();
    // Note: if strikes are ascending, moneyness is descending
    // Sort to make ascending
    std::ranges::sort(moneyness);

    return from_vectors(
        std::move(moneyness),
        std::move(maturities),
        std::move(volatilities),
        std::move(rates),
        spot,  // K_ref = spot
        std::move(pde_grid),
        type,
        dividend_yield,
        max_failure_rate
    );
}

template <>
PriceTableBuilder<4>::Setup
PriceTableBuilder<4>::from_chain(
    const OptionChain& chain,
    PDEGridSpec pde_grid,
    OptionType type,
    double max_failure_rate)
{
    return from_strikes(
        chain.spot,
        chain.strikes,
        chain.maturities,
        chain.implied_vols,
        chain.rates,
        std::move(pde_grid),
        type,
        chain.dividend_yield,
        max_failure_rate
    );
}

template <>
PriceTableBuilder<4>::Setup
PriceTableBuilder<4>::from_chain_auto(
    const OptionChain& chain,
    PDEGridSpec pde_grid,
    OptionType type,
    const PriceTableGridAccuracyParams<4>& accuracy)
{
    // Validate chain inputs before calling estimator
    if (chain.spot <= 0.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::NonPositiveValue, 4});
    }
    if (chain.strikes.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 0, 0});
    }
    if (chain.maturities.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 1, 0});
    }
    if (chain.implied_vols.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 2, 0});
    }
    if (chain.rates.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 3, 0});
    }

    // Estimate optimal grids based on target accuracy
    auto estimate = estimate_grid_from_chain_bounds(
        chain.strikes,
        chain.spot,
        chain.maturities,
        chain.implied_vols,
        chain.rates,
        accuracy
    );

    // Check for empty grids (indicates estimation failure)
    if (estimate.grids[0].empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    // Use estimated grids with from_vectors
    return from_vectors(
        std::move(estimate.grids[0]),
        std::move(estimate.grids[1]),
        std::move(estimate.grids[2]),
        std::move(estimate.grids[3]),
        chain.spot,  // K_ref = spot
        std::move(pde_grid),
        type,
        chain.dividend_yield,
        0.0  // max_failure_rate = 0 (strict)
    );
}

template <>
PriceTableBuilder<4>::Setup
PriceTableBuilder<4>::from_chain_auto_profile(
    const OptionChain& chain,
    PriceTableGridProfile grid_profile,
    GridAccuracyProfile pde_profile,
    OptionType type)
{
    // Validate chain inputs before calling estimator
    if (chain.spot <= 0.0) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::NonPositiveValue, 4});
    }
    if (chain.strikes.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 0, 0});
    }
    if (chain.maturities.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 1, 0});
    }
    if (chain.implied_vols.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 2, 0});
    }
    if (chain.rates.empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InsufficientGridPoints, 3, 0});
    }

    // Estimate optimal grids based on target accuracy profile
    auto grid_params = grid_accuracy_profile(grid_profile);
    auto estimate = estimate_grid_from_chain_bounds(
        chain.strikes,
        chain.spot,
        chain.maturities,
        chain.implied_vols,
        chain.rates,
        grid_params
    );

    // Check for empty grids (indicates estimation failure)
    if (estimate.grids[0].empty()) {
        return std::unexpected(PriceTableError{PriceTableErrorCode::InvalidConfig});
    }

    // Auto-estimate PDE grid from batch parameters
    return from_vectors(
        std::move(estimate.grids[0]),
        std::move(estimate.grids[1]),
        std::move(estimate.grids[2]),
        std::move(estimate.grids[3]),
        chain.spot,  // K_ref = spot
        grid_accuracy_profile(pde_profile),
        type,
        chain.dividend_yield,
        0.0  // max_failure_rate = 0 (strict)
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

template <size_t N>
std::expected<RepairStats, PriceTableError>
PriceTableBuilder<N>::repair_failed_slices(
    PriceTensor<N>& tensor,
    const std::vector<size_t>& failed_pde,
    const std::vector<std::tuple<size_t, size_t, size_t>>& failed_spline,
    const PriceTableAxes<N>& axes) const
{
    const size_t Nm = axes.grids[0].size();
    const size_t Nt = axes.grids[1].size();
    const size_t Nσ = axes.grids[2].size();
    const size_t Nr = axes.grids[3].size();

    // Group spline failures by (σ,r) to detect full-slice vs partial failures
    std::map<std::pair<size_t, size_t>, std::vector<size_t>> spline_failures_by_slice;
    for (auto [σ_idx, r_idx, τ_idx] : failed_spline) {
        spline_failures_by_slice[{σ_idx, r_idx}].push_back(τ_idx);
    }

    // Collect slices that need full neighbor copy (PDE failures + all-maturity spline failures)
    std::unordered_set<size_t> full_slice_set(failed_pde.begin(), failed_pde.end());
    size_t partial_spline_points = 0;

    for (auto& [slice_key, τ_failures] : spline_failures_by_slice) {
        if (τ_failures.size() == Nt) {
            auto [σ_idx, r_idx] = slice_key;
            size_t flat_idx = σ_idx * Nr + r_idx;
            full_slice_set.insert(flat_idx);
        } else {
            partial_spline_points += τ_failures.size();
        }
    }

    std::vector<size_t> full_slice_failures(full_slice_set.begin(), full_slice_set.end());

    // Track which (σ,r) slices are valid donors
    std::vector<bool> slice_valid(Nσ * Nr, true);
    for (size_t flat_idx : full_slice_failures) {
        slice_valid[flat_idx] = false;
    }

    // ========== PHASE 1: Repair partial spline failures via τ-interpolation ==========
    for (auto& [slice_key, τ_failures] : spline_failures_by_slice) {
        auto [σ_idx, r_idx] = slice_key;

        // Skip full-slice failures (handled in Phase 2)
        if (τ_failures.size() == Nt) continue;

        // Partial failures: interpolate along τ axis
        for (size_t τ_idx : τ_failures) {
            std::optional<size_t> τ_before, τ_after;
            for (size_t j = τ_idx; j-- > 0; ) {
                if (!std::isnan(tensor.view[0, j, σ_idx, r_idx])) {
                    τ_before = j; break;
                }
            }
            for (size_t j = τ_idx + 1; j < Nt; ++j) {
                if (!std::isnan(tensor.view[0, j, σ_idx, r_idx])) {
                    τ_after = j; break;
                }
            }

            // At least one must exist (not all_maturities_failed)
            for (size_t i = 0; i < Nm; ++i) {
                if (τ_before && τ_after) {
                    double t = static_cast<double>(τ_idx - *τ_before) /
                               static_cast<double>(*τ_after - *τ_before);
                    tensor.view[i, τ_idx, σ_idx, r_idx] =
                        (1.0 - t) * tensor.view[i, *τ_before, σ_idx, r_idx] +
                        t * tensor.view[i, *τ_after, σ_idx, r_idx];
                } else if (τ_before) {
                    tensor.view[i, τ_idx, σ_idx, r_idx] = tensor.view[i, *τ_before, σ_idx, r_idx];
                } else {
                    tensor.view[i, τ_idx, σ_idx, r_idx] = tensor.view[i, *τ_after, σ_idx, r_idx];
                }
            }
        }
    }

    // ========== PHASE 2: Repair full-slice failures via neighbor copy ==========
    size_t repaired_full_count = 0;
    for (size_t flat_idx : full_slice_failures) {
        size_t σ_idx = flat_idx / Nr;
        size_t r_idx = flat_idx % Nr;

        auto neighbor = find_nearest_valid_neighbor(σ_idx, r_idx, Nσ, Nr, slice_valid);
        if (!neighbor.has_value()) {
            return std::unexpected(PriceTableError{
                PriceTableErrorCode::RepairFailed, σ_idx * Nr + r_idx, full_slice_failures.size()});
        }
        auto [nσ, nr] = neighbor.value();

        // Copy entire (m,τ) surface from neighbor
        for (size_t i = 0; i < Nm; ++i) {
            for (size_t j = 0; j < Nt; ++j) {
                tensor.view[i, j, σ_idx, r_idx] = tensor.view[i, j, nσ, nr];
            }
        }

        // Mark as valid so this slice can be a donor for subsequent repairs
        slice_valid[flat_idx] = true;
        ++repaired_full_count;
    }

    return RepairStats{
        .repaired_full_slices = repaired_full_count,
        .repaired_partial_points = partial_spline_points
    };
}

// Explicit instantiation (only N=4 supported)
template class PriceTableBuilder<4>;

} // namespace mango

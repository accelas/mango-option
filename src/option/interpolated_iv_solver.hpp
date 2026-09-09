// SPDX-License-Identifier: MIT
/**
 * @file interpolated_iv_solver.hpp
 * @brief Implied volatility solver using pre-computed price interpolation
 *
 * Provides:
 * - InterpolatedIVSolver<Surface>: Brent IV solver on certified price tables
 * - AnyInterpIVSolver: type-erased wrapper for convenient use
 * - make_interpolated_iv_solver(): factory that builds the price surface and solver
 *
 * Two construction paths:
 * 1. Direct: build your own PriceTable, then InterpolatedIVSolver::create()
 * 2. Factory: fill IVSolverFactoryConfig, call make_interpolated_iv_solver()
 *
 * Grid density is controlled via IVGrid.  When `adaptive` is set, the grid
 * values serve as domain bounds for automatic refinement; otherwise they are
 * exact interpolation knots.
 */

#pragma once

#include "mango/option/dividend_utils.hpp"
#include "mango/option/option_spec.hpp"
#include "mango/option/iv_result.hpp"
#include "mango/option/table/price_table.hpp"
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/option/table/bspline/bspline_3d_surface.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_3d_surface.hpp"
#include "mango/option/table/chebyshev/chebyshev_adaptive.hpp"
#include "mango/option/table/chebyshev/chebyshev_surface.hpp"
#include "mango/support/error_types.hpp"
#include "mango/support/parallel.hpp"
#include "mango/math/root_finding.hpp"
#include <expected>
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>
#include <optional>

namespace mango {

/// Configuration for interpolation-based IV solver
struct InterpolatedIVSolverConfig {
    size_t max_iter = 50;          ///< Maximum Brent iterations
    double tolerance = 1e-6;       ///< Absolute root-finding tolerance
    double sigma_min = 0.01;       ///< Minimum volatility (1%)
    double sigma_max = 3.0;        ///< Maximum volatility (300%)

    /// Minimum vega to attempt IV solve.  Evaluates surface vega at the
    /// quartile points of the *actual* solve bracket (25%, 50%, 75% of
    /// [sigma_min, sigma_max]) and takes the **signed** maximum.  A strongly
    /// negative vega is a broken surface, not a healthy one, so the check
    /// never takes an absolute value.  If the signed maximum is below the
    /// threshold, the option has no usable sensitivity to volatility and IV
    /// is effectively undefined: returns VegaTooSmall immediately (~600 ns)
    /// instead of running a doomed Brent search.  A non-finite probe vega
    /// returns NumericalInstability. Set to 0 to disable this pre-check.
    /// Every returned root still requires finite positive vega and sufficient
    /// sensitivity to resolve a representable price change at the solver's
    /// volatility resolution. A positive threshold also applies at that root.
    double vega_threshold = 1e-4;

};

/// Interpolation-based IV Solver
///
/// Uses a pre-computed price surface for ultra-fast IV calculation.
/// Solves: Find s such that Price(m, tau, s, r) = Market_Price
///
/// Thread-safe: Fully thread-safe for both single and batch queries (immutable surface)
///
/// Rate handling: The price surface uses a scalar rate axis (designed for SOFR/flat rates).
/// When a YieldCurve is provided, it is collapsed to a zero rate: -ln(D(T))/T.
/// This provides a reasonable approximation but does not capture term structure dynamics.
/// For full yield curve support, use IVSolver instead.
/// When rate approximation is used, IVSuccess::used_rate_approximation is set to true.
///
/// Construction requires a payload-bound proof that the represented physical
/// price is nondecreasing in sigma over its admitted domain. Every returned
/// root must also have finite positive sensitivity sufficient to resolve a
/// representable price change at the configured volatility resolution.
/// Legitimate flat prices remain priceable; their IV is refused as unidentifiable.
///
/// @tparam Surface A PriceTable<Inner> instantiation
template <typename Surface>
class InterpolatedIVSolver {
public:
    /// Create a solver from a certified immutable library price table.
    ///
    /// Only the six supported PriceTable representations and their library
    /// shared-handle adapters are admitted. Callback claims are not evidence.
    /// `build_dividends`, when supplied, is a consistency check against the
    /// table's stored model; it cannot change that model. Continuous tables
    /// have a known empty cash-dividend schedule. Segmented tables retain the
    /// exact fixed-expiry anchor and canonical schedule used by their build.
    /// @return IV solver or a typed representation/model/configuration error.
    static std::expected<InterpolatedIVSolver, ValidationError> create(
        Surface surface,
        const InterpolatedIVSolverConfig& config = {},
        std::optional<std::vector<Dividend>> build_dividends = std::nullopt);

    /// Solve for implied volatility (single query)
    ///
    /// Uses Brent's method with the certified price surface.
    ///
    /// @param query Option specification and market price
    /// @return Success with IV and diagnostics, or error with details
    std::expected<IVSuccess, IVError> solve(const IVQuery& query) const noexcept;

    /// Solve for implied volatility (batch with OpenMP)
    ///
    /// Trivially parallel since the surface is immutable and thread-safe.
    ///
    /// @param queries Input queries (as vector for convenience)
    /// @return BatchIVResult with individual results and failure count
    BatchIVResult solve_batch(const std::vector<IVQuery>& queries) const noexcept;

private:
    /// Private constructor (use create() factory method)
    InterpolatedIVSolver(
        Surface surface,
        std::pair<double, double> m_range,
        std::pair<double, double> tau_range,
        std::pair<double, double> sigma_range,
        std::pair<double, double> r_range,
        OptionType option_type,
        double dividend_yield,
        std::optional<std::vector<Dividend>> build_dividends,
        double reference_maturity,
        const InterpolatedIVSolverConfig& config)
        : surface_(std::move(surface))
        , m_range_(m_range)
        , tau_range_(tau_range)
        , sigma_range_(sigma_range)
        , r_range_(r_range)
        , config_(config)
        , option_type_(option_type)
        , dividend_yield_(dividend_yield)
        , build_dividends_(std::move(build_dividends))
        , reference_maturity_(reference_maturity)
    {}

    Surface surface_;
    std::pair<double, double> m_range_, tau_range_, sigma_range_, r_range_;
    InterpolatedIVSolverConfig config_;
    OptionType option_type_;
    double dividend_yield_;
    /// Canonical schedule retained by the immutable table model. Public
    /// construction always resolves this, including known-empty continuous models.
    std::optional<std::vector<Dividend>> build_dividends_;
    double reference_maturity_;

    /// Evaluate option price using surface interpolation with strike scaling
    double eval_price(double moneyness, double maturity, double vol, double rate, double strike) const;

    /// Check if query parameters are within surface bounds
    bool is_in_bounds(const IVQuery& query, double vol) const {
        if constexpr (requires { surface_.contains_maturity(query.maturity); }) {
            if (!surface_.contains_maturity(query.maturity)) return false;
        }
        if constexpr (requires { surface_.contains_strike(query.strike); }) {
            if (!surface_.contains_strike(query.strike)) return false;
        }
        const bool moneyness_inside = [&] {
            if constexpr (requires { surface_.contains_moneyness(query.spot, query.strike); }) {
                return surface_.contains_moneyness(query.spot, query.strike);
            } else {
                const double x = std::log(query.spot / query.strike);
                return x >= m_range_.first && x <= m_range_.second;
            }
        }();

        // Extract zero rate for bounds check - must match what solve uses
        // Using get_zero_rate() ensures consistency: -ln(D(T))/T for curves
        double rate_value = get_zero_rate(query.rate, query.maturity);

        return moneyness_inside &&
               query.maturity >= tau_range_.first && query.maturity <= tau_range_.second &&
               vol >= sigma_range_.first && vol <= sigma_range_.second &&
               rate_value >= r_range_.first && rate_value <= r_range_.second;
    }

    /// Validate query parameters
    std::optional<ValidationError> validate_query(const IVQuery& query) const;

    /// Determine adaptive volatility bounds based on intrinsic value
    std::pair<double, double> adaptive_bounds(const IVQuery& query) const;
};

// =====================================================================
// Factory: config types, type-erased solver, and factory function
// =====================================================================

/// B-spline interpolation backend
struct BSplineBackend {
    std::vector<double> maturity_grid;  ///< Tau knots (continuous) / chain maturities (adaptive)
};

/// Chebyshev tensor interpolation backend
struct ChebyshevBackend {
    double maturity = 2.0;                             ///< Domain upper bound for tau
    /// CGL nodes per axis for manual continuous builds only. Adaptive builds
    /// choose their CC levels using AdaptiveGridParams.
    std::array<size_t, 4> num_pts = {16, 12, 12, 8};
};

/// Dimensionless 3D interpolation backend
///
/// Collapses (sigma, r) into kappa = 2r/sigma^2, reducing to 3D (x, tau', ln kappa).
/// Fewer PDE solves but sigma/r coupling limits accuracy.
///
/// Constraints: dividend_yield must be 0, no discrete dividends, rate > 0.
///
/// IVGrid values define domain bounds only (not exact knots).  The B-spline
/// path derives its own linspace grid; the Chebyshev path uses CGL nodes.
struct DimensionlessBackend {
    enum class Interpolant { BSpline, Chebyshev };

    double maturity = 2.0;                    ///< Domain upper bound for physical tau
    Interpolant interpolant = Interpolant::BSpline;
    std::array<size_t, 3> chebyshev_pts = {16, 16, 12};  ///< CGL nodes (x, tau', ln_kappa)
};

/// Discrete dividend configuration (optional, orthogonal to backend choice)
struct DiscreteDividendConfig {
    double maturity = 1.0;                  ///< Surface maturity
    std::vector<Dividend> discrete_dividends;
    MultiKRefConfig kref_config;            ///< defaults to auto
    std::optional<StrikeBounds> strike_bounds = std::nullopt; ///< requested absolute K interval
};

/// Configuration for the IV solver factory
///
/// Backend choice (B-spline vs Chebyshev) and dividend type (continuous vs
/// discrete) are orthogonal.  All four combinations are supported:
///   - BSpline + continuous:  standard B-spline surface
///   - BSpline + discrete:   segmented multi-K_ref B-spline surface
///   - Chebyshev + continuous: Chebyshev tensor surface
///   - Chebyshev + discrete:  segmented Chebyshev surface (requires adaptive)
struct IVSolverFactoryConfig {
    OptionType option_type = OptionType::PUT;
    double spot = 100.0;
    double dividend_yield = 0.0;
    IVGrid grid;                                    ///< Grid points (exact or domain bounds)
    std::optional<AdaptiveGridParams> adaptive;     ///< If set, refine grid adaptively
    InterpolatedIVSolverConfig solver_config;       ///< Newton config
    std::variant<BSplineBackend, ChebyshevBackend, DimensionlessBackend> backend;
    std::optional<DiscreteDividendConfig> discrete_dividends;
};

/// Type-erased IV solver wrapping any PriceTable backend
///
/// Impl is defined in the .cpp — only the factory can construct instances.
class AnyInterpIVSolver {
public:
    /// Solve for implied volatility (single query)
    std::expected<IVSuccess, IVError> solve(const IVQuery& query) const;

    /// Solve for implied volatility (batch with OpenMP)
    BatchIVResult solve_batch(const std::vector<IVQuery>& queries) const;

    /// Diagnostics from adaptive grid refinement (spec D7), propagated from
    /// the `AnyPriceTable` this solver was built from.  `nullopt` when the
    /// table was built manually or loaded from Parquet.
    [[nodiscard]] std::optional<BuildDiagnostics> build_diagnostics() const;

    /// Same immutable final assessment owned by the source table, when present.
    [[nodiscard]] std::shared_ptr<const AccuracyReport> accuracy_report() const;


    // Pimpl: move-only, defined in .cpp
    struct Impl;
    explicit AnyInterpIVSolver(std::unique_ptr<Impl> impl);
    AnyInterpIVSolver(AnyInterpIVSolver&&) noexcept;
    AnyInterpIVSolver& operator=(AnyInterpIVSolver&&) noexcept;
    ~AnyInterpIVSolver();

private:
    friend class AnyPriceTable;
    void attach_accuracy_report(std::shared_ptr<const AccuracyReport> report);
    std::unique_ptr<Impl> impl_;
};

namespace detail {

/// Cheap immutable surface view used when building solvers from AnyPriceTable.
template <typename Table>
class SharedPriceTableSurface {
public:
    static constexpr bool requires_fixed_expiry = Table::requires_fixed_expiry;
    using inner_type = typename Table::inner_type;

    explicit SharedPriceTableSurface(std::shared_ptr<const Table> table)
        : table_(*table) {}

    [[nodiscard]] double price(double spot, double strike,
                               double tau, double sigma, double rate) const {
        return table_.price(spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] double vega(double spot, double strike,
                              double tau, double sigma, double rate) const {
        return table_.vega(spot, strike, tau, sigma, rate);
    }

    [[nodiscard]] std::expected<double, GreekError>
    delta(const PricingParams& params) const { return table_.delta(params); }

    [[nodiscard]] std::expected<double, GreekError>
    gamma(const PricingParams& params) const { return table_.gamma(params); }

    [[nodiscard]] std::expected<double, GreekError>
    theta(const PricingParams& params) const { return table_.theta(params); }

    [[nodiscard]] std::expected<double, GreekError>
    rho(const PricingParams& params) const { return table_.rho(params); }

    [[nodiscard]] double m_min() const noexcept { return table_.m_min(); }
    [[nodiscard]] double m_max() const noexcept { return table_.m_max(); }
    [[nodiscard]] double tau_min() const noexcept { return table_.tau_min(); }
    [[nodiscard]] double tau_max() const noexcept { return table_.tau_max(); }
    [[nodiscard]] const std::optional<FixedExpiryMetadata>& fixed_expiry() const noexcept {
        return table_.fixed_expiry();
    }
    [[nodiscard]] bool contains_moneyness(double spot, double strike) const noexcept {
        return table_.contains_moneyness(spot, strike);
    }
    [[nodiscard]] bool contains_strike(double strike) const noexcept {
        return table_.contains_strike(strike);
    }
    [[nodiscard]] bool contains_maturity(double tau) const noexcept {
        return table_.contains_maturity(tau);
    }
    [[nodiscard]] double sigma_min() const noexcept { return table_.sigma_min(); }
    [[nodiscard]] double sigma_max() const noexcept { return table_.sigma_max(); }
    [[nodiscard]] double rate_min() const noexcept { return table_.rate_min(); }
    [[nodiscard]] double rate_max() const noexcept { return table_.rate_max(); }
    [[nodiscard]] OptionType option_type() const noexcept { return table_.option_type(); }
    [[nodiscard]] double dividend_yield() const noexcept { return table_.dividend_yield(); }
    [[nodiscard]] PriceProofStatus proof_status() const noexcept { return table_.proof_status(); }

private:
    // Snapshot the immutable payload handle, not a caller-reassignable wrapper.
    Table table_;
};

// Closed library representations, not a callback's self-reported status.
template <class Table>
inline constexpr bool supported_price_table =
    std::same_as<Table, BSplinePriceTable> || std::same_as<Table, BSpline3DPriceTable> ||
    std::same_as<Table, BSplineMultiKRefSurface> || std::same_as<Table, ChebyshevSurface> ||
    std::same_as<Table, Chebyshev3DPriceTable> || std::same_as<Table, ChebyshevMultiKRefSurface>;
template <class Surface>
struct SupportedIVSurface : std::bool_constant<supported_price_table<Surface>> {};
template <class Table>
struct SupportedIVSurface<SharedPriceTableSurface<Table>>
    : std::bool_constant<supported_price_table<Table>> {};

}  // namespace detail

template <typename Surface>
using SharedPriceTableSolver =
    InterpolatedIVSolver<detail::SharedPriceTableSurface<Surface>>;

[[nodiscard]] AnyInterpIVSolver make_any_interpolated_solver(
    InterpolatedIVSolver<BSplinePriceTable> solver,
    std::optional<BuildDiagnostics> diagnostics = std::nullopt);
[[nodiscard]] AnyInterpIVSolver make_any_interpolated_solver(
    InterpolatedIVSolver<BSplineMultiKRefSurface> solver,
    std::optional<BuildDiagnostics> diagnostics = std::nullopt);
[[nodiscard]] AnyInterpIVSolver make_any_interpolated_solver(
    InterpolatedIVSolver<ChebyshevSurface> solver,
    std::optional<BuildDiagnostics> diagnostics = std::nullopt);
[[nodiscard]] AnyInterpIVSolver make_any_interpolated_solver(
    InterpolatedIVSolver<ChebyshevMultiKRefSurface> solver,
    std::optional<BuildDiagnostics> diagnostics = std::nullopt);
[[nodiscard]] AnyInterpIVSolver make_any_interpolated_solver(
    InterpolatedIVSolver<BSpline3DPriceTable> solver,
    std::optional<BuildDiagnostics> diagnostics = std::nullopt);
[[nodiscard]] AnyInterpIVSolver make_any_interpolated_solver(
    InterpolatedIVSolver<Chebyshev3DPriceTable> solver,
    std::optional<BuildDiagnostics> diagnostics = std::nullopt);
[[nodiscard]] AnyInterpIVSolver make_any_interpolated_solver(
    SharedPriceTableSolver<BSplinePriceTable> solver,
    std::optional<BuildDiagnostics> diagnostics = std::nullopt);
[[nodiscard]] AnyInterpIVSolver make_any_interpolated_solver(
    SharedPriceTableSolver<BSplineMultiKRefSurface> solver,
    std::optional<BuildDiagnostics> diagnostics = std::nullopt);
[[nodiscard]] AnyInterpIVSolver make_any_interpolated_solver(
    SharedPriceTableSolver<ChebyshevSurface> solver,
    std::optional<BuildDiagnostics> diagnostics = std::nullopt);
[[nodiscard]] AnyInterpIVSolver make_any_interpolated_solver(
    SharedPriceTableSolver<ChebyshevMultiKRefSurface> solver,
    std::optional<BuildDiagnostics> diagnostics = std::nullopt);
[[nodiscard]] AnyInterpIVSolver make_any_interpolated_solver(
    SharedPriceTableSolver<BSpline3DPriceTable> solver,
    std::optional<BuildDiagnostics> diagnostics = std::nullopt);
[[nodiscard]] AnyInterpIVSolver make_any_interpolated_solver(
    SharedPriceTableSolver<Chebyshev3DPriceTable> solver,
    std::optional<BuildDiagnostics> diagnostics = std::nullopt);

/// Factory function: build price surface and IV solver from config
///
/// Dispatches on backend × dividend type:
///   - BSpline + continuous → standard B-spline surface
///   - BSpline + discrete  → segmented multi-K_ref B-spline surface
///   - Chebyshev + continuous → Chebyshev tensor surface
///   - Chebyshev + discrete → segmented Chebyshev surface (requires adaptive)
///
/// When adaptive is set, uses adaptive refinement to automatically refine
/// grid density until the target IV error is met.
///
/// @param config Solver configuration
/// @return Type-erased AnyInterpIVSolver or ValidationError
std::expected<AnyInterpIVSolver, ValidationError> make_interpolated_iv_solver(const IVSolverFactoryConfig& config);

// =====================================================================
// Template implementation (must be in header for template instantiation)
// =====================================================================

template <typename Surface>
std::expected<InterpolatedIVSolver<Surface>, ValidationError>
InterpolatedIVSolver<Surface>::create(
    Surface surface,
    const InterpolatedIVSolverConfig& config,
    std::optional<std::vector<Dividend>> build_dividends)
{
    if constexpr (!detail::SupportedIVSurface<Surface>::value) {
        return std::unexpected(ValidationError{ValidationErrorCode::UnsupportedRepresentation});
    } else {
        if (!std::isfinite(config.tolerance) || config.tolerance <= 0.0) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::InvalidBounds, config.tolerance});
        }
        if (!std::isfinite(config.vega_threshold) || config.vega_threshold < 0.0) {
            return std::unexpected(ValidationError{
                ValidationErrorCode::InvalidBounds, config.vega_threshold});
        }
        if (surface.proof_status() != PriceProofStatus::Certified) {
            return std::unexpected(ValidationError{ValidationErrorCode::CertificationIndeterminate});
        }
        // Read the bounds attached to this exact immutable proof payload.
        auto m_range = std::make_pair(surface.m_min(), surface.m_max());
        auto tau_range = std::make_pair(surface.tau_min(), surface.tau_max());
        auto sigma_range = std::make_pair(surface.sigma_min(), surface.sigma_max());
        auto r_range = std::make_pair(surface.rate_min(), surface.rate_max());

        // Validate bounds
        if (m_range.first > m_range.second ||
            tau_range.first > tau_range.second ||
            sigma_range.first >= sigma_range.second ||
            r_range.first > r_range.second) {
            return std::unexpected(ValidationError(ValidationErrorCode::InvalidGridSize, 0.0));
        }

        auto option_type = surface.option_type();
        auto dividend_yield = surface.dividend_yield();

        double reference_maturity = tau_range.second;
        if constexpr (requires { surface.fixed_expiry(); }) {
            const auto& model = surface.fixed_expiry();
            if (model) {
                if (!model->valid(tau_range.second)) {
                    return std::unexpected(ValidationError{ValidationErrorCode::InvalidBounds});
                }
                reference_maturity = model->reference_maturity;
                if (build_dividends) {
                    const auto supplied = filter_and_merge_dividends(*build_dividends, reference_maturity);
                    const bool same = supplied.size() == model->discrete_dividends.size()
                        && std::equal(supplied.begin(), supplied.end(), model->discrete_dividends.begin(),
                            [](const Dividend& a, const Dividend& b) {
                                return a.calendar_time == b.calendar_time && a.amount == b.amount;
                            });
                    if (!same) return std::unexpected(
                        ValidationError{ValidationErrorCode::DiscreteDividendMismatch});
                }
                build_dividends = model->discrete_dividends;
            } else {
                if constexpr (Surface::requires_fixed_expiry) {
                    return std::unexpected(ValidationError{ValidationErrorCode::InvalidBounds});
                }
                // A continuous payload has known absence of discrete events.
                // Caller provenance cannot change the retained numerical model.
                if (build_dividends && !filter_and_merge_dividends(
                        *build_dividends, reference_maturity).empty()) {
                    return std::unexpected(ValidationError{ValidationErrorCode::DiscreteDividendMismatch});
                }
                build_dividends = std::vector<Dividend>{};
            }
        }
        // The immutable table model controls both schedule and expiry anchor.
        if (build_dividends) {
            build_dividends = filter_and_merge_dividends(*build_dividends, reference_maturity);
        }

        return InterpolatedIVSolver(
            std::move(surface),
            m_range,
            tau_range,
            sigma_range,
            r_range,
            option_type,
            dividend_yield,
            std::move(build_dividends),
            reference_maturity,
            config);
    }
}

template <typename Surface>
double InterpolatedIVSolver<Surface>::eval_price(
    double moneyness, double maturity, double vol, double rate, double strike) const
{
    double spot = moneyness * strike;
    return surface_.price(spot, strike, maturity, vol, rate);
}



template <typename Surface>
std::optional<ValidationError>
InterpolatedIVSolver<Surface>::validate_query(const IVQuery& query) const
{
    if (query.option_type != option_type_) {
        return ValidationError{ValidationErrorCode::OptionTypeMismatch,
            static_cast<double>(query.option_type), 0};
    }

    if (std::abs(query.dividend_yield - dividend_yield_) > 1e-10) {
        return ValidationError{ValidationErrorCode::DividendYieldMismatch,
            query.dividend_yield, 0};
    }

    // Discrete dividend schedule check (#440 item 1). An empty query
    // schedule is always valid: for segmented surfaces the build-time
    // schedule is authoritative. A non-empty schedule must match the
    // build schedule rolled to the query valuation, when it is known.
    //
    // Roll the schedule from its numerical model anchor, which may exceed
    // the largest published query maturity. At an exact dividend instant
    // that event has elapsed (post-dividend calendar side). A query's own
    // out-of-life entries stay visible so they cannot validate accidentally.
    if (!query.discrete_dividends.empty() && build_dividends_.has_value()) {
        constexpr double kTimeTol = 1e-6;    // years (~30 seconds)
        constexpr double kAmountTol = 1e-6;  // dollars
        auto expected = rolled_dividends(
            *build_dividends_, reference_maturity_, query.maturity);
        std::vector<Dividend> actual = filter_and_merge_dividends(
            query.discrete_dividends, std::numeric_limits<double>::infinity());
        if (expected.size() != actual.size()) {
            return ValidationError{
                ValidationErrorCode::DiscreteDividendMismatch,
                static_cast<double>(actual.size()), expected.size()};
        }
        for (size_t i = 0; i < actual.size(); ++i) {
            bool time_mismatch = std::abs(actual[i].calendar_time -
                                           expected[i].calendar_time) > kTimeTol;
            bool amount_mismatch = std::abs(actual[i].amount -
                                             expected[i].amount) > kAmountTol;
            if (time_mismatch || amount_mismatch) {
                double reported = time_mismatch ? actual[i].calendar_time
                                                 : actual[i].amount;
                return ValidationError{
                    ValidationErrorCode::DiscreteDividendMismatch,
                    reported, i};
            }
        }
    }

    // Use common validation for option spec, market price, and arbitrage checks
    auto validation = validate_iv_query(query);
    if (!validation.has_value()) {
        return validation.error();
    }

    return std::nullopt;
}

template <typename Surface>
std::pair<double, double>
InterpolatedIVSolver<Surface>::adaptive_bounds(const IVQuery& query) const
{
    double intrinsic = intrinsic_value(query.spot, query.strike, query.option_type);

    // Analyze time value to set adaptive bounds
    const double time_value = query.market_price - intrinsic;
    const double time_value_pct = time_value / query.market_price;

    double sigma_upper;
    if (time_value_pct > 0.5) {
        sigma_upper = 3.0;  // 300%
    } else if (time_value_pct > 0.2) {
        sigma_upper = 2.0;  // 200%
    } else {
        sigma_upper = 1.5;  // 150%
    }

    double sigma_min = std::max(config_.sigma_min, sigma_range_.first);
    double sigma_max = std::min({sigma_upper, config_.sigma_max, sigma_range_.second});

    if (sigma_min >= sigma_max) {
        sigma_min = sigma_range_.first;
        sigma_max = sigma_range_.second;
    }

    return {sigma_min, sigma_max};
}

template <typename Surface>
std::expected<IVSuccess, IVError>
InterpolatedIVSolver<Surface>::solve(const IVQuery& query) const noexcept
{
    // Validate input using centralized validation
    auto error = validate_query(query);
    if (error.has_value()) {
        // Convert ValidationError to IVError using shared mapping
        return std::unexpected(validation_error_to_iv_error(*error));
    }

    const double moneyness = query.spot / query.strike;

    // Get adaptive bounds
    auto [sigma_min, sigma_max] = adaptive_bounds(query);

    // Check if query is within surface bounds
    if (!is_in_bounds(query, sigma_min) || !is_in_bounds(query, sigma_max)) {
        return std::unexpected(IVError{
            .code = IVErrorCode::InvalidGridConfig,
            .iterations = 0,
            .final_error = 0.0,
            .last_vol = std::nullopt
        });
    }

    // Extract zero rate for surface lookup
    // For yield curves, use zero rate = -ln(D(T))/T which matches how surfaces are built
    // Using instantaneous forward rate curve.rate(T) would be incorrect as it only
    // reflects the rate at maturity, not the integrated discount factor
    //
    // Note: When a YieldCurve is provided, we collapse it to a single zero rate.
    // This loses term structure dynamics. For full curve support, use IVSolver.
    const bool rate_is_curve = is_yield_curve(query.rate);
    double rate_value = get_zero_rate(query.rate, query.maturity);

    // Vega pre-check: reject queries where the option has no usable
    // sensitivity to volatility.  Probes are the quartile points of the
    // actual bracket (fixed probe vols could fall outside it entirely) and
    // the maximum is signed: a uniformly negative vega is a broken surface,
    // not a healthy one.  ~600 ns, saves a doomed Brent search.
    if (config_.vega_threshold > 0.0) {
        const double vega_span = sigma_max - sigma_min;
        const double probe_vols[3] = {sigma_min + 0.25 * vega_span,
                                      sigma_min + 0.50 * vega_span,
                                      sigma_min + 0.75 * vega_span};
        double max_vega = -std::numeric_limits<double>::infinity();
        for (double sv : probe_vols) {
            const double v = surface_.vega(query.spot, query.strike,
                                           query.maturity, sv, rate_value);
            if (!std::isfinite(v)) {
                return std::unexpected(IVError{
                    .code = IVErrorCode::NumericalInstability,
                    .iterations = 0,
                    .final_error = std::numeric_limits<double>::quiet_NaN(),
                    .last_vol = sv
                });
            }
            max_vega = std::max(max_vega, v);
        }
        if (max_vega < config_.vega_threshold) {
            return std::unexpected(IVError{
                .code = IVErrorCode::VegaTooSmall,
                .iterations = 0,
                .final_error = max_vega,
                .last_vol = std::nullopt
            });
        }
    }

    // Define objective function: f(s) = Price(s) - Market_Price
    auto objective = [&](double sigma) -> double {
        return eval_price(moneyness, query.maturity, sigma, rate_value, query.strike) - query.market_price;
    };

    // Every success, including an endpoint, must be sensitive at the
    // returned root. A healthy bracket probe cannot establish this property.
    auto admit_root = [&](IVSuccess success) -> std::expected<IVSuccess, IVError> {
        const double vega = surface_.vega(query.spot, query.strike,
            query.maturity, success.implied_vol, rate_value);
        if (!std::isfinite(vega)) {
            return std::unexpected(IVError{
                .code = IVErrorCode::NumericalInstability,
                .iterations = success.iterations,
                .final_error = vega,
                .last_vol = success.implied_vol
            });
        }
        // Brent currently uses tolerance for both price residual and sigma
        // distance. At that sigma resolution, the local price response must
        // reach at least one representable quote increment. This numerical
        // admission is scale-dependent; it is not an IV accuracy estimate.
        const double sigma_ulp = std::nextafter(success.implied_vol,
            std::numeric_limits<double>::infinity()) - success.implied_vol;
        const double sigma_resolution = std::min(sigma_max - sigma_min,
            std::max(config_.tolerance, sigma_ulp));
        const double price_ulp_down = query.market_price - std::nextafter(
            query.market_price, 0.0);
        const double price_ulp_up = std::nextafter(query.market_price,
            std::numeric_limits<double>::infinity()) - query.market_price;
        const double price_ulp = std::isfinite(price_ulp_up)
            ? std::max(price_ulp_down, price_ulp_up) : price_ulp_down;
        const double resolution_floor = price_ulp / sigma_resolution;
        if (!(vega > 0.0) || vega < config_.vega_threshold ||
            vega < resolution_floor) {
            return std::unexpected(IVError{
                .code = IVErrorCode::VegaTooSmall,
                .iterations = success.iterations,
                .final_error = vega,
                .last_vol = success.implied_vol
            });
        }
        success.vega = vega;
        success.used_rate_approximation = rate_is_curve;
        return success;
    };

    // Brent's method
    RootFindingConfig brent_config{
        .max_iter = config_.max_iter,
        .brent_tol_abs = config_.tolerance
    };

    auto result = find_root(objective, sigma_min, sigma_max, brent_config);

    // Check convergence - transform RootFindingError to IVError
    if (!result.has_value()) {
        const auto& root_error = result.error();
        IVErrorCode error_code;
        switch (root_error.code) {
            case RootFindingErrorCode::MaxIterationsExceeded:
                error_code = IVErrorCode::MaxIterationsExceeded;
                break;
            case RootFindingErrorCode::InvalidBracket:
                error_code = IVErrorCode::BracketingFailed;
                break;
            case RootFindingErrorCode::NumericalInstability:
                error_code = IVErrorCode::NumericalInstability;
                break;
            case RootFindingErrorCode::NoProgress:
                error_code = IVErrorCode::NumericalInstability;
                break;
            default:
                error_code = IVErrorCode::NumericalInstability;
                break;
        }

        return std::unexpected(IVError{
            .code = error_code,
            .iterations = root_error.iterations,
            .final_error = root_error.final_error,
            .last_vol = root_error.last_value
        });
    }

    return admit_root(IVSuccess{
        .implied_vol = result->root,
        .iterations = result->iterations,
        .final_error = result->final_error,
        .vega = std::nullopt,
        .used_rate_approximation = rate_is_curve
    });
}

template <typename Surface>
BatchIVResult
InterpolatedIVSolver<Surface>::solve_batch(const std::vector<IVQuery>& queries) const noexcept
{
    std::vector<std::expected<IVSuccess, IVError>> results(queries.size());
    size_t failed_count = 0;

    // Trivially parallel: surface is immutable and thread-safe
    MANGO_PRAGMA_PARALLEL_FOR
    for (size_t i = 0; i < queries.size(); ++i) {
        results[i] = solve(queries[i]);
        if (!results[i].has_value()) {
            MANGO_PRAGMA_ATOMIC
            ++failed_count;
        }
    }

    return BatchIVResult{
        .results = std::move(results),
        .failed_count = failed_count
    };
}

}  // namespace mango

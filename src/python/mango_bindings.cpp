// SPDX-License-Identifier: MIT
/**
 * @file mango_bindings.cpp
 * @brief Python bindings for mango-option library using pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <sstream>
#include "src/option/option_spec.hpp"
#include "src/option/iv_solver_fdm.hpp"
#include "src/option/iv_solver_interpolated.hpp"
#include "src/option/american_option.hpp"
#include "src/option/option_chain.hpp"
#include "src/option/table/price_table_builder.hpp"
#include "src/option/table/price_table_workspace.hpp"
#include "src/option/table/price_table_surface.hpp"
#include "src/option/table/american_price_surface.hpp"
#include "src/option/table/price_table_metadata.hpp"
#include "src/option/table/price_table_axes.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include "src/math/yield_curve.hpp"
#include "src/option/american_option_batch.hpp"

namespace py = pybind11;

// Helper to convert Python object to RateSpec
mango::RateSpec python_to_rate_spec(const py::object& obj) {
    if (py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj)) {
        return obj.cast<double>();
    } else if (py::isinstance<mango::YieldCurve>(obj)) {
        return obj.cast<mango::YieldCurve>();
    } else {
        throw py::type_error("rate must be a float or YieldCurve");
    }
}

// Helper to convert RateSpec to Python object
py::object rate_spec_to_python(const mango::RateSpec& spec) {
    return std::visit([](const auto& arg) -> py::object {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, double>) {
            return py::cast(arg);
        } else {
            return py::cast(arg);
        }
    }, spec);
}

// Helper to format rate for __repr__
std::string rate_spec_to_string(const mango::RateSpec& spec) {
    return std::visit([](const auto& arg) -> std::string {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, double>) {
            return std::to_string(arg);
        } else {
            return "<YieldCurve>";
        }
    }, spec);
}

PYBIND11_MODULE(mango_option, m) {
    m.doc() = "Python bindings for mango-option American option pricing and IV solver";

    // OptionType enum
    py::enum_<mango::OptionType>(m, "OptionType")
        .value("CALL", mango::OptionType::CALL)
        .value("PUT", mango::OptionType::PUT);

    // Price table grid profile enums
    py::enum_<mango::PriceTableGridProfile>(m, "PriceTableGridProfile")
        .value("LOW", mango::PriceTableGridProfile::Low)
        .value("MEDIUM", mango::PriceTableGridProfile::Medium)
        .value("HIGH", mango::PriceTableGridProfile::High)
        .value("ULTRA", mango::PriceTableGridProfile::Ultra);

    py::enum_<mango::GridAccuracyProfile>(m, "GridAccuracyProfile")
        .value("LOW", mango::GridAccuracyProfile::Low)
        .value("MEDIUM", mango::GridAccuracyProfile::Medium)
        .value("HIGH", mango::GridAccuracyProfile::High)
        .value("ULTRA", mango::GridAccuracyProfile::Ultra);

    // GridAccuracyParams structure
    py::class_<mango::GridAccuracyParams>(m, "GridAccuracyParams")
        .def(py::init<>())
        .def_readwrite("n_sigma", &mango::GridAccuracyParams::n_sigma)
        .def_readwrite("alpha", &mango::GridAccuracyParams::alpha)
        .def_readwrite("tol", &mango::GridAccuracyParams::tol)
        .def_readwrite("c_t", &mango::GridAccuracyParams::c_t)
        .def_readwrite("min_spatial_points", &mango::GridAccuracyParams::min_spatial_points)
        .def_readwrite("max_spatial_points", &mango::GridAccuracyParams::max_spatial_points)
        .def_readwrite("max_time_steps", &mango::GridAccuracyParams::max_time_steps);

    // OptionChain data container
    py::class_<mango::OptionChain>(m, "OptionChain")
        .def(py::init<>())
        .def_readwrite("ticker", &mango::OptionChain::ticker)
        .def_readwrite("spot", &mango::OptionChain::spot)
        .def_readwrite("strikes", &mango::OptionChain::strikes)
        .def_readwrite("maturities", &mango::OptionChain::maturities)
        .def_readwrite("implied_vols", &mango::OptionChain::implied_vols)
        .def_readwrite("rates", &mango::OptionChain::rates)
        .def_readwrite("dividend_yield", &mango::OptionChain::dividend_yield)
        .def("__repr__", [](const mango::OptionChain& chain) {
            return "<OptionChain spot=" + std::to_string(chain.spot) +
                   " strikes=" + std::to_string(chain.strikes.size()) +
                   " maturities=" + std::to_string(chain.maturities.size()) +
                   " vols=" + std::to_string(chain.implied_vols.size()) +
                   " rates=" + std::to_string(chain.rates.size()) + ">";
        });

    // TenorPoint structure (for YieldCurve construction)
    py::class_<mango::TenorPoint>(m, "TenorPoint")
        .def(py::init<>())
        .def(py::init([](double tenor, double log_discount) {
            return mango::TenorPoint{tenor, log_discount};
        }), py::arg("tenor"), py::arg("log_discount"))
        .def_readwrite("tenor", &mango::TenorPoint::tenor)
        .def_readwrite("log_discount", &mango::TenorPoint::log_discount);

    // YieldCurve class
    py::class_<mango::YieldCurve>(m, "YieldCurve")
        .def(py::init<>())
        .def_static("flat", &mango::YieldCurve::flat, py::arg("rate"),
            "Create a flat yield curve with constant rate")
        .def_static("from_discounts", [](const std::vector<double>& tenors,
                                          const std::vector<double>& discounts) {
            auto result = mango::YieldCurve::from_discounts(tenors, discounts);
            if (!result.has_value()) {
                throw py::value_error(result.error());
            }
            return result.value();
        }, py::arg("tenors"), py::arg("discounts"),
            "Create yield curve from tenor and discount factor vectors")
        .def("rate", &mango::YieldCurve::rate, py::arg("t"),
            "Get instantaneous forward rate at time t")
        .def("discount", &mango::YieldCurve::discount, py::arg("t"),
            "Get discount factor D(t) at time t")
        .def("zero_rate", &mango::YieldCurve::zero_rate, py::arg("t"),
            "Get zero rate at time t: -ln(D(t))/t")
        .def("__repr__", [](const mango::YieldCurve&) {
            return "<YieldCurve>";
        });

    // IVQuery structure (replaces IVParams)
    py::class_<mango::IVQuery>(m, "IVQuery")
        .def(py::init<>())
        .def(py::init<double, double, double, double, double, mango::OptionType, double>(),
             py::arg("spot"), py::arg("strike"), py::arg("maturity"),
             py::arg("rate"), py::arg("dividend_yield"), py::arg("type"),
             py::arg("market_price"))
        .def_readwrite("spot", &mango::IVQuery::spot)
        .def_readwrite("strike", &mango::IVQuery::strike)
        .def_readwrite("maturity", &mango::IVQuery::maturity)
        .def_property("rate",
            [](const mango::IVQuery& q) { return rate_spec_to_python(q.rate); },
            [](mango::IVQuery& q, const py::object& obj) { q.rate = python_to_rate_spec(obj); },
            "Risk-free rate (float or YieldCurve)")
        .def_readwrite("dividend_yield", &mango::IVQuery::dividend_yield)
        .def_readwrite("type", &mango::IVQuery::type)
        .def_readwrite("market_price", &mango::IVQuery::market_price)
        .def("__repr__", [](const mango::IVQuery& q) {
            return "<IVQuery spot=" + std::to_string(q.spot) +
                   " strike=" + std::to_string(q.strike) +
                   " maturity=" + std::to_string(q.maturity) +
                   " rate=" + rate_spec_to_string(q.rate) +
                   " dividend_yield=" + std::to_string(q.dividend_yield) +
                   " type=" + (q.type == mango::OptionType::CALL ? "CALL" : "PUT") +
                   " market_price=" + std::to_string(q.market_price) + ">";
        });

    // RootFindingConfig structure
    py::class_<mango::RootFindingConfig>(m, "RootFindingConfig")
        .def(py::init<>())
        .def_readwrite("max_iter", &mango::RootFindingConfig::max_iter)
        .def_readwrite("tolerance", &mango::RootFindingConfig::tolerance)
        .def_readwrite("jacobian_fd_epsilon", &mango::RootFindingConfig::jacobian_fd_epsilon)
        .def_readwrite("brent_tol_abs", &mango::RootFindingConfig::brent_tol_abs);

    // IVSolverFDMConfig structure
    py::class_<mango::IVSolverFDMConfig>(m, "IVSolverFDMConfig")
        .def(py::init<>())
        .def_readwrite("root_config", &mango::IVSolverFDMConfig::root_config)
        .def_readwrite("use_manual_grid", &mango::IVSolverFDMConfig::use_manual_grid)
        .def_readwrite("grid_n_space", &mango::IVSolverFDMConfig::grid_n_space)
        .def_readwrite("grid_n_time", &mango::IVSolverFDMConfig::grid_n_time)
        .def_readwrite("grid_x_min", &mango::IVSolverFDMConfig::grid_x_min)
        .def_readwrite("grid_x_max", &mango::IVSolverFDMConfig::grid_x_max)
        .def_readwrite("grid_alpha", &mango::IVSolverFDMConfig::grid_alpha)
        .def_readwrite("grid_accuracy", &mango::IVSolverFDMConfig::grid_accuracy);

    // IVSuccess structure (std::expected success type)
    py::class_<mango::IVSuccess>(m, "IVSuccess")
        .def(py::init<>())
        .def_readwrite("implied_vol", &mango::IVSuccess::implied_vol)
        .def_readwrite("iterations", &mango::IVSuccess::iterations)
        .def_readwrite("final_error", &mango::IVSuccess::final_error)
        .def_readwrite("vega", &mango::IVSuccess::vega)
        .def("__repr__", [](const mango::IVSuccess& r) {
            std::string repr = "<IVSuccess iv=" + std::to_string(r.implied_vol) +
                       " iters=" + std::to_string(r.iterations) +
                       " error=" + std::to_string(r.final_error);
            if (r.vega.has_value()) {
                repr += " vega=" + std::to_string(*r.vega);
            }
            return repr + ">";
        });

    // IVError structure (std::expected error type)
    py::class_<mango::IVError>(m, "IVError")
        .def(py::init<>())
        .def_readwrite("code", &mango::IVError::code)
        .def_readwrite("iterations", &mango::IVError::iterations)
        .def_readwrite("final_error", &mango::IVError::final_error)
        .def_readwrite("last_vol", &mango::IVError::last_vol)
        .def_property_readonly("message", [](const mango::IVError& e) {
            // Helper function to convert error code to human-readable string
            switch (e.code) {
                case mango::IVErrorCode::NegativeSpot: return "Negative spot price";
                case mango::IVErrorCode::NegativeStrike: return "Negative strike price";
                case mango::IVErrorCode::NegativeMaturity: return "Negative maturity";
                case mango::IVErrorCode::NegativeMarketPrice: return "Negative market price";
                case mango::IVErrorCode::ArbitrageViolation: return "Arbitrage violation";
                case mango::IVErrorCode::MaxIterationsExceeded: return "Maximum iterations exceeded";
                case mango::IVErrorCode::BracketingFailed: return "Bracketing failed";
                case mango::IVErrorCode::NumericalInstability: return "Numerical instability";
                case mango::IVErrorCode::InvalidGridConfig: return "Invalid grid configuration";
                case mango::IVErrorCode::PDESolveFailed: return "PDE solve failed";
                default: return "Unknown error";
            }
        })
        .def("__repr__", [](const mango::IVError& e) {
            std::string repr = "<IVError code=" + std::to_string(static_cast<int>(e.code));
            if (e.last_vol.has_value()) {
                repr += " last_vol=" + std::to_string(*e.last_vol);
            }
            return repr + ">";
        });

    // IVErrorCode enum
    py::enum_<mango::IVErrorCode>(m, "IVErrorCode")
        .value("NegativeSpot", mango::IVErrorCode::NegativeSpot)
        .value("NegativeStrike", mango::IVErrorCode::NegativeStrike)
        .value("NegativeMaturity", mango::IVErrorCode::NegativeMaturity)
        .value("NegativeMarketPrice", mango::IVErrorCode::NegativeMarketPrice)
        .value("ArbitrageViolation", mango::IVErrorCode::ArbitrageViolation)
        .value("MaxIterationsExceeded", mango::IVErrorCode::MaxIterationsExceeded)
        .value("BracketingFailed", mango::IVErrorCode::BracketingFailed)
        .value("NumericalInstability", mango::IVErrorCode::NumericalInstability)
        .value("InvalidGridConfig", mango::IVErrorCode::InvalidGridConfig)
        .value("PDESolveFailed", mango::IVErrorCode::PDESolveFailed)
        .export_values();

    // IVSolver class (now using FDM solver with std::expected)
    py::class_<mango::IVSolverFDM>(m, "IVSolverFDM")
        .def(py::init<const mango::IVSolverFDMConfig&>(),
             py::arg("config"))
        .def("solve_impl", [](const mango::IVSolverFDM& solver, const mango::IVQuery& query) {
            auto result = solver.solve_impl(query);
            if (result.has_value()) {
                return py::make_tuple(true, result.value(), mango::IVError{});
            } else {
                return py::make_tuple(false, mango::IVSuccess{}, result.error());
            }
        },
        py::arg("query"),
        "Solve for implied volatility. Returns (success: bool, result: IVSuccess, error: IVError)");

    // Note: Batch solver removed - users should use IVSolverInterpolated for batch queries

    // AmericanOptionParams structure
    py::class_<mango::AmericanOptionParams>(m, "AmericanOptionParams")
        .def(py::init<>())
        .def_readwrite("strike", &mango::AmericanOptionParams::strike)
        .def_readwrite("spot", &mango::AmericanOptionParams::spot)
        .def_readwrite("maturity", &mango::AmericanOptionParams::maturity)
        .def_readwrite("volatility", &mango::AmericanOptionParams::volatility)
        .def_property("rate",
            [](const mango::AmericanOptionParams& p) { return rate_spec_to_python(p.rate); },
            [](mango::AmericanOptionParams& p, const py::object& obj) { p.rate = python_to_rate_spec(obj); },
            "Risk-free rate (float or YieldCurve)")
        .def_readwrite("dividend_yield", &mango::AmericanOptionParams::dividend_yield)
        .def_readwrite("type", &mango::AmericanOptionParams::type)
        .def_readwrite("discrete_dividends", &mango::AmericanOptionParams::discrete_dividends);

    // AmericanOptionResult structure
    py::class_<mango::AmericanOptionResult>(m, "AmericanOptionResult")
        .def("value_at", &mango::AmericanOptionResult::value_at, py::arg("spot"),
             "Interpolate to get option value at specific spot price")
        .def("delta", &mango::AmericanOptionResult::delta,
             "Compute delta (∂V/∂S) at spot price")
        .def("gamma", &mango::AmericanOptionResult::gamma,
             "Compute gamma (∂²V/∂S²) at spot price")
        .def("theta", &mango::AmericanOptionResult::theta,
             "Compute theta (∂V/∂t) at spot price");

    m.def(
        "american_option_price",
        [](const mango::AmericanOptionParams& params,
           std::optional<mango::GridAccuracyProfile> accuracy_profile) {
            // Validate parameters before grid estimation to avoid
            // division-by-zero or extreme allocations
            auto validation = mango::validate_pricing_params(params);
            if (!validation.has_value()) {
                auto err = validation.error();
                std::string msg = "Invalid pricing parameters: ";
                switch (err.code) {
                    case mango::ValidationErrorCode::InvalidSpotPrice:
                        msg += "spot price must be positive"; break;
                    case mango::ValidationErrorCode::InvalidStrike:
                        msg += "strike must be positive"; break;
                    case mango::ValidationErrorCode::InvalidMaturity:
                        msg += "maturity must be positive"; break;
                    case mango::ValidationErrorCode::InvalidVolatility:
                        msg += "volatility must be positive"; break;
                    case mango::ValidationErrorCode::InvalidRate:
                        msg += "invalid rate"; break;
                    case mango::ValidationErrorCode::InvalidDividend:
                        msg += "invalid dividend yield"; break;
                    default:
                        msg += "validation error code " +
                            std::to_string(static_cast<int>(err.code));
                        break;
                }
                msg += " (value=" + std::to_string(err.value) + ")";
                throw py::value_error(msg);
            }

            mango::GridAccuracyParams accuracy;
            if (accuracy_profile.has_value()) {
                accuracy = mango::grid_accuracy_profile(accuracy_profile.value());
            }

            // Estimate grid automatically (sinh-spaced, clustered near strike)
            auto [grid_spec, time_domain] = mango::estimate_grid_for_option(params, accuracy);

            // Allocate workspace buffer
            size_t n = grid_spec.n_points();
            std::vector<double> buffer(mango::PDEWorkspace::required_size(n));

            auto workspace_result = mango::PDEWorkspace::from_buffer(buffer, n);
            if (!workspace_result) {
                throw py::value_error(
                    "Failed to create workspace: " + workspace_result.error());
            }

            mango::AmericanOptionSolver solver(
                params, workspace_result.value(), std::nullopt,
                std::make_pair(grid_spec, time_domain));
            auto solve_result = solver.solve();
            if (!solve_result) {
                auto error = solve_result.error();
                throw py::value_error(
                    "American option solve failed (error code " +
                    std::to_string(static_cast<int>(error.code)) + ")");
            }

            return std::move(solve_result.value());
        },
        py::arg("params"),
        py::arg("accuracy") = py::none(),
        R"pbdoc(
            Price an American option using the PDE solver with automatic grid estimation.

            Uses sinh-spaced grids with clustering near the strike for optimal accuracy.
            Supports yield curves (via params.rate) and discrete dividends
            (via params.discrete_dividends).

            Args:
                params: AmericanOptionParams with contract and market parameters.
                accuracy: Optional GridAccuracyProfile (LOW/MEDIUM/HIGH/ULTRA).
                          If not specified, uses default parameters.

            Returns:
                AmericanOptionResult with value and Greeks.
        )pbdoc");

    // =========================================================================
    // Batch American Option Solver (with normalized chain optimization)
    // =========================================================================

    // SolverErrorCode enum
    py::enum_<mango::SolverErrorCode>(m, "SolverErrorCode")
        .value("ConvergenceFailure", mango::SolverErrorCode::ConvergenceFailure)
        .value("LinearSolveFailure", mango::SolverErrorCode::LinearSolveFailure)
        .value("InvalidConfiguration", mango::SolverErrorCode::InvalidConfiguration)
        .value("Unknown", mango::SolverErrorCode::Unknown)
        .export_values();

    // SolverError structure
    py::class_<mango::SolverError>(m, "SolverError")
        .def(py::init<>())
        .def_readwrite("code", &mango::SolverError::code)
        .def_readwrite("iterations", &mango::SolverError::iterations)
        .def_readwrite("residual", &mango::SolverError::residual)
        .def("__repr__", [](const mango::SolverError& e) {
            return "<SolverError code=" + std::to_string(static_cast<int>(e.code)) +
                   " iterations=" + std::to_string(e.iterations) +
                   " residual=" + std::to_string(e.residual) + ">";
        });

    py::class_<mango::BatchAmericanOptionSolver>(m, "BatchAmericanOptionSolver")
        .def(py::init<>())
        .def("set_grid_accuracy",
            [](mango::BatchAmericanOptionSolver& self, mango::GridAccuracyProfile profile) {
                self.set_grid_accuracy(mango::grid_accuracy_profile(profile));
                return &self;
            },
            py::arg("profile"),
            py::return_value_policy::reference_internal,
            R"pbdoc(
                Set grid accuracy using a profile.

                Args:
                    profile: GridAccuracyProfile (LOW/MEDIUM/HIGH/ULTRA)

                Returns:
                    Self for method chaining
            )pbdoc")
        .def("set_grid_accuracy_params",
            [](mango::BatchAmericanOptionSolver& self, const mango::GridAccuracyParams& params) {
                self.set_grid_accuracy(params);
                return &self;
            },
            py::arg("params"),
            py::return_value_policy::reference_internal,
            R"pbdoc(
                Set grid accuracy using explicit parameters.

                Args:
                    params: GridAccuracyParams with fine-grained control

                Returns:
                    Self for method chaining
            )pbdoc")
        .def("set_use_normalized",
            [](mango::BatchAmericanOptionSolver& self, bool enable) {
                self.set_use_normalized(enable);
                return &self;
            },
            py::arg("enable") = true,
            py::return_value_policy::reference_internal,
            "Enable/disable normalized chain optimization")
        .def("solve_batch",
            [](mango::BatchAmericanOptionSolver& self,
               const std::vector<mango::AmericanOptionParams>& params,
               bool use_shared_grid) {
                auto batch_result = self.solve_batch(params, use_shared_grid);

                py::list results;
                for (auto& r : batch_result.results) {
                    if (r.has_value()) {
                        results.append(py::make_tuple(true, std::move(r.value()),
                            mango::SolverError{}));
                    } else {
                        results.append(py::make_tuple(false, py::none(), r.error()));
                    }
                }
                return py::make_tuple(results, batch_result.failed_count);
            },
            py::arg("params"),
            py::arg("use_shared_grid") = false,
            R"pbdoc(
                Solve a batch of American options in parallel.

                Automatically routes to the normalized chain solver when eligible
                (same maturity, same type, no discrete dividends, use_shared_grid=True).
                The normalized path solves one PDE and reuses it for all strikes.

                Args:
                    params: List of AmericanOptionParams
                    use_shared_grid: If True, all options share one global grid
                                     (required for normalized chain optimization)

                Returns:
                    Tuple of (results, failed_count) where results is a list of
                    (success: bool, result: AmericanOptionResult|None, error: SolverError) tuples
            )pbdoc");

    // =========================================================================
    // Price Table File Storage (Arrow IPC format)
    // =========================================================================

    // PriceTableWorkspace LoadError enum
    py::enum_<mango::PriceTableWorkspace::LoadError>(m, "PriceTableLoadError")
        .value("NOT_ARROW_FILE", mango::PriceTableWorkspace::LoadError::NOT_ARROW_FILE)
        .value("UNSUPPORTED_VERSION", mango::PriceTableWorkspace::LoadError::UNSUPPORTED_VERSION)
        .value("INSUFFICIENT_GRID_POINTS", mango::PriceTableWorkspace::LoadError::INSUFFICIENT_GRID_POINTS)
        .value("SIZE_MISMATCH", mango::PriceTableWorkspace::LoadError::SIZE_MISMATCH)
        .value("COEFFICIENT_SIZE_MISMATCH", mango::PriceTableWorkspace::LoadError::COEFFICIENT_SIZE_MISMATCH)
        .value("GRID_NOT_SORTED", mango::PriceTableWorkspace::LoadError::GRID_NOT_SORTED)
        .value("MMAP_FAILED", mango::PriceTableWorkspace::LoadError::MMAP_FAILED)
        .value("INVALID_ALIGNMENT", mango::PriceTableWorkspace::LoadError::INVALID_ALIGNMENT)
        .value("FILE_NOT_FOUND", mango::PriceTableWorkspace::LoadError::FILE_NOT_FOUND)
        .value("SCHEMA_MISMATCH", mango::PriceTableWorkspace::LoadError::SCHEMA_MISMATCH)
        .value("ARROW_READ_ERROR", mango::PriceTableWorkspace::LoadError::ARROW_READ_ERROR)
        .value("CORRUPTED_COEFFICIENTS", mango::PriceTableWorkspace::LoadError::CORRUPTED_COEFFICIENTS)
        .value("CORRUPTED_GRIDS", mango::PriceTableWorkspace::LoadError::CORRUPTED_GRIDS)
        .value("CORRUPTED_KNOTS", mango::PriceTableWorkspace::LoadError::CORRUPTED_KNOTS)
        .export_values();

    // PriceTableWorkspace class
    py::class_<mango::PriceTableWorkspace>(m, "PriceTableWorkspace")
        .def_static("create",
            [](py::array_t<double> log_m_grid,
               py::array_t<double> tau_grid,
               py::array_t<double> sigma_grid,
               py::array_t<double> r_grid,
               py::array_t<double> coefficients,
               double K_ref,
               double dividend_yield,
               double m_min,
               double m_max) {
                auto result = mango::PriceTableWorkspace::create(
                    std::span<const double>(log_m_grid.data(), log_m_grid.size()),
                    std::span<const double>(tau_grid.data(), tau_grid.size()),
                    std::span<const double>(sigma_grid.data(), sigma_grid.size()),
                    std::span<const double>(r_grid.data(), r_grid.size()),
                    std::span<const double>(coefficients.data(), coefficients.size()),
                    K_ref, dividend_yield, m_min, m_max);
                if (!result.has_value()) {
                    throw py::value_error(result.error());
                }
                return std::move(result.value());
            },
            py::arg("log_m_grid"), py::arg("tau_grid"),
            py::arg("sigma_grid"), py::arg("r_grid"),
            py::arg("coefficients"), py::arg("K_ref"),
            py::arg("dividend_yield"), py::arg("m_min"), py::arg("m_max"),
            R"pbdoc(
                Create a PriceTableWorkspace from grid data and coefficients.

                Args:
                    log_m_grid: Log-moneyness grid (ln(S/K), sorted ascending, >= 4 points)
                    tau_grid: Maturity grid (years, sorted ascending, >= 4 points)
                    sigma_grid: Volatility grid (sorted ascending, >= 4 points)
                    r_grid: Rate grid (sorted ascending, >= 4 points)
                    coefficients: B-spline coefficients (size = n_m * n_tau * n_sigma * n_r)
                    K_ref: Reference strike price
                    dividend_yield: Continuous dividend yield
                    m_min: Minimum moneyness (S/K) for user-facing bounds
                    m_max: Maximum moneyness (S/K) for user-facing bounds

                Returns:
                    PriceTableWorkspace instance

                Raises:
                    ValueError: If validation fails (grid size, sorting, etc.)
            )pbdoc")
        .def_static("load",
            [](const std::string& filepath) {
                auto result = mango::PriceTableWorkspace::load(filepath);
                if (!result.has_value()) {
                    std::string msg = "Failed to load price table: ";
                    switch (result.error()) {
                        case mango::PriceTableWorkspace::LoadError::NOT_ARROW_FILE:
                            msg += "Not an Arrow file"; break;
                        case mango::PriceTableWorkspace::LoadError::UNSUPPORTED_VERSION:
                            msg += "Unsupported version"; break;
                        case mango::PriceTableWorkspace::LoadError::INSUFFICIENT_GRID_POINTS:
                            msg += "Insufficient grid points"; break;
                        case mango::PriceTableWorkspace::LoadError::SIZE_MISMATCH:
                            msg += "Size mismatch"; break;
                        case mango::PriceTableWorkspace::LoadError::COEFFICIENT_SIZE_MISMATCH:
                            msg += "Coefficient size mismatch"; break;
                        case mango::PriceTableWorkspace::LoadError::GRID_NOT_SORTED:
                            msg += "Grid not sorted"; break;
                        case mango::PriceTableWorkspace::LoadError::MMAP_FAILED:
                            msg += "Memory mapping failed"; break;
                        case mango::PriceTableWorkspace::LoadError::INVALID_ALIGNMENT:
                            msg += "Invalid alignment"; break;
                        case mango::PriceTableWorkspace::LoadError::FILE_NOT_FOUND:
                            msg += "File not found"; break;
                        case mango::PriceTableWorkspace::LoadError::SCHEMA_MISMATCH:
                            msg += "Schema mismatch"; break;
                        case mango::PriceTableWorkspace::LoadError::ARROW_READ_ERROR:
                            msg += "Arrow read error"; break;
                        case mango::PriceTableWorkspace::LoadError::CORRUPTED_COEFFICIENTS:
                            msg += "Corrupted coefficients"; break;
                        case mango::PriceTableWorkspace::LoadError::CORRUPTED_GRIDS:
                            msg += "Corrupted grids"; break;
                        case mango::PriceTableWorkspace::LoadError::CORRUPTED_KNOTS:
                            msg += "Corrupted knots"; break;
                    }
                    throw py::value_error(msg);
                }
                return std::move(result.value());
            },
            py::arg("filepath"),
            R"pbdoc(
                Load a PriceTableWorkspace from an Arrow IPC file.

                Args:
                    filepath: Path to the Arrow IPC file

                Returns:
                    PriceTableWorkspace instance

                Raises:
                    ValueError: If loading fails (file not found, corrupted, etc.)
            )pbdoc")
        .def("save",
            [](const mango::PriceTableWorkspace& self, const std::string& filepath,
               const std::string& ticker, uint8_t option_type) {
                auto result = self.save(filepath, ticker, option_type);
                if (!result.has_value()) {
                    throw py::value_error(result.error());
                }
            },
            py::arg("filepath"), py::arg("ticker"), py::arg("option_type"),
            R"pbdoc(
                Save the workspace to an Arrow IPC file.

                Args:
                    filepath: Output file path
                    ticker: Underlying symbol (e.g., "SPY")
                    option_type: 0=PUT, 1=CALL

                Raises:
                    ValueError: If saving fails
            )pbdoc")
        .def_property_readonly("log_moneyness",
            [](const mango::PriceTableWorkspace& self) {
                auto span = self.log_moneyness();
                return py::array_t<double>(span.size(), span.data());
            }, "Log-moneyness grid (ln(S/K))")
        .def_property_readonly("maturity",
            [](const mango::PriceTableWorkspace& self) {
                auto span = self.maturity();
                return py::array_t<double>(span.size(), span.data());
            }, "Maturity grid (years)")
        .def_property_readonly("volatility",
            [](const mango::PriceTableWorkspace& self) {
                auto span = self.volatility();
                return py::array_t<double>(span.size(), span.data());
            }, "Volatility grid")
        .def_property_readonly("rate",
            [](const mango::PriceTableWorkspace& self) {
                auto span = self.rate();
                return py::array_t<double>(span.size(), span.data());
            }, "Rate grid")
        .def_property_readonly("coefficients",
            [](const mango::PriceTableWorkspace& self) {
                auto span = self.coefficients();
                return py::array_t<double>(span.size(), span.data());
            }, "B-spline coefficients")
        .def_property_readonly("K_ref", &mango::PriceTableWorkspace::K_ref,
            "Reference strike price")
        .def_property_readonly("dividend_yield", &mango::PriceTableWorkspace::dividend_yield,
            "Continuous dividend yield")
        .def_property_readonly("m_min", &mango::PriceTableWorkspace::m_min,
            "Minimum moneyness (S/K)")
        .def_property_readonly("m_max", &mango::PriceTableWorkspace::m_max,
            "Maximum moneyness (S/K)")
        .def_property_readonly("dimensions",
            [](const mango::PriceTableWorkspace& self) {
                auto [nm, nt, nv, nr] = self.dimensions();
                return py::make_tuple(nm, nt, nv, nr);
            }, "Grid dimensions (n_m, n_tau, n_sigma, n_r)")
        .def("__repr__", [](const mango::PriceTableWorkspace& self) {
            auto [nm, nt, nv, nr] = self.dimensions();
            return "<PriceTableWorkspace dims=(" + std::to_string(nm) + "," +
                   std::to_string(nt) + "," + std::to_string(nv) + "," +
                   std::to_string(nr) + ") K_ref=" + std::to_string(self.K_ref()) + ">";
        });

    // =========================================================================
    // PriceTableSurface (4D B-spline interpolation)
    // =========================================================================

    // SurfaceContent enum
    py::enum_<mango::SurfaceContent>(m, "SurfaceContent")
        .value("RawPrice", mango::SurfaceContent::RawPrice)
        .value("EarlyExercisePremium", mango::SurfaceContent::EarlyExercisePremium);

    // PriceTableMetadata
    py::class_<mango::PriceTableMetadata>(m, "PriceTableMetadata")
        .def(py::init<>())
        .def_readwrite("K_ref", &mango::PriceTableMetadata::K_ref)
        .def_readwrite("dividend_yield", &mango::PriceTableMetadata::dividend_yield)
        .def_readwrite("m_min", &mango::PriceTableMetadata::m_min)
        .def_readwrite("m_max", &mango::PriceTableMetadata::m_max)
        .def_readwrite("content", &mango::PriceTableMetadata::content)
        .def_readwrite("discrete_dividends", &mango::PriceTableMetadata::discrete_dividends);

    // PriceTableAxes<4>
    py::class_<mango::PriceTableAxes<4>>(m, "PriceTableAxes4D")
        .def(py::init<>())
        .def_property("grids",
            [](const mango::PriceTableAxes<4>& self) {
                py::list result;
                for (const auto& grid : self.grids) {
                    result.append(py::array_t<double>(grid.size(), grid.data()));
                }
                return result;
            },
            [](mango::PriceTableAxes<4>& self, const py::list& grids) {
                if (grids.size() != 4) {
                    throw py::value_error("Must provide exactly 4 grids");
                }
                for (size_t i = 0; i < 4; ++i) {
                    auto arr = grids[i].cast<py::array_t<double>>();
                    self.grids[i] = std::vector<double>(arr.data(), arr.data() + arr.size());
                }
            })
        .def_property("names",
            [](const mango::PriceTableAxes<4>& self) {
                py::list result;
                for (const auto& name : self.names) {
                    result.append(name);
                }
                return result;
            },
            [](mango::PriceTableAxes<4>& self, const py::list& names) {
                if (names.size() != 4) {
                    throw py::value_error("Must provide exactly 4 names");
                }
                for (size_t i = 0; i < 4; ++i) {
                    self.names[i] = names[i].cast<std::string>();
                }
            })
        .def("total_points", [](const mango::PriceTableAxes<4>& self) {
            return self.total_points();
        })
        .def("shape", [](const mango::PriceTableAxes<4>& self) {
            auto s = self.shape();
            return py::make_tuple(s[0], s[1], s[2], s[3]);
        });

    // PriceTableSurface<4>
    // Note: We use shared_ptr<mango::PriceTableSurface<4>> as holder, even though
    // the C++ API returns shared_ptr<const ...>. pybind11 will handle the const conversion.
    py::class_<mango::PriceTableSurface<4>, std::shared_ptr<mango::PriceTableSurface<4>>>(
        m, "PriceTableSurface4D")
        .def_static("build",
            [](mango::PriceTableAxes<4> axes, py::array_t<double> coeffs,
               mango::PriceTableMetadata metadata) {
                std::vector<double> coeffs_vec(coeffs.data(), coeffs.data() + coeffs.size());
                auto result = mango::PriceTableSurface<4>::build(
                    std::move(axes), std::move(coeffs_vec), std::move(metadata));
                if (!result.has_value()) {
                    throw py::value_error("Failed to build surface: error code " +
                        std::to_string(static_cast<int>(result.error().code)));
                }
                return result.value();
            },
            py::arg("axes"), py::arg("coefficients"), py::arg("metadata"),
            R"pbdoc(
                Build a 4D price table surface from axes and coefficients.

                Args:
                    axes: PriceTableAxes4D with grid points for each dimension
                    coefficients: B-spline coefficients (flattened, row-major)
                    metadata: PriceTableMetadata with K_ref, dividend info

                Returns:
                    PriceTableSurface4D instance

                Raises:
                    ValueError: If building fails
            )pbdoc")
        .def("value",
            [](const mango::PriceTableSurface<4>& self, double m, double tau, double sigma, double r) {
                return self.value({m, tau, sigma, r});
            },
            py::arg("moneyness"), py::arg("maturity"), py::arg("volatility"), py::arg("rate"),
            R"pbdoc(
                Evaluate price at query point.

                Args:
                    moneyness: S/K ratio
                    maturity: Time to maturity (years)
                    volatility: Implied volatility
                    rate: Risk-free rate

                Returns:
                    Interpolated option price
            )pbdoc")
        .def("partial",
            [](const mango::PriceTableSurface<4>& self, size_t axis,
               double m, double tau, double sigma, double r) {
                return self.partial(axis, {m, tau, sigma, r});
            },
            py::arg("axis"), py::arg("moneyness"), py::arg("maturity"),
            py::arg("volatility"), py::arg("rate"),
            R"pbdoc(
                Compute partial derivative along specified axis.

                Args:
                    axis: 0=moneyness, 1=maturity, 2=volatility, 3=rate
                    moneyness: S/K ratio
                    maturity: Time to maturity (years)
                    volatility: Implied volatility
                    rate: Risk-free rate

                Returns:
                    Partial derivative estimate
            )pbdoc")
        .def_property_readonly("axes", &mango::PriceTableSurface<4>::axes)
        .def_property_readonly("metadata", &mango::PriceTableSurface<4>::metadata);

    // =========================================================================
    // AmericanPriceSurface (EEP + European reconstruction)
    // =========================================================================

    py::class_<mango::AmericanPriceSurface>(m, "AmericanPriceSurface")
        .def_static("create",
            [](std::shared_ptr<const mango::PriceTableSurface<4>> eep_surface,
               mango::OptionType type) {
                auto result = mango::AmericanPriceSurface::create(
                    std::move(eep_surface), type);
                if (!result.has_value()) {
                    throw py::value_error(
                        "Failed to create AmericanPriceSurface: validation error code " +
                        std::to_string(static_cast<int>(result.error().code)));
                }
                return std::move(result.value());
            },
            py::arg("eep_surface"), py::arg("option_type"),
            R"pbdoc(
                Create an AmericanPriceSurface from an EEP B-spline surface.

                Reconstructs full American prices as:
                  P = (K/K_ref) * EEP(m, tau, sigma, r) + P_European(S, K, tau, sigma, r, q)

                Args:
                    eep_surface: Pre-computed PriceTableSurface4D with EEP data
                    option_type: OptionType.PUT or OptionType.CALL

                Returns:
                    AmericanPriceSurface instance

                Raises:
                    ValueError: If validation fails
            )pbdoc")
        .def("price", &mango::AmericanPriceSurface::price,
            py::arg("spot"), py::arg("strike"), py::arg("tau"),
            py::arg("sigma"), py::arg("rate"),
            "Compute American option price")
        .def("delta", &mango::AmericanPriceSurface::delta,
            py::arg("spot"), py::arg("strike"), py::arg("tau"),
            py::arg("sigma"), py::arg("rate"),
            "Compute delta (dP/dS)")
        .def("gamma", &mango::AmericanPriceSurface::gamma,
            py::arg("spot"), py::arg("strike"), py::arg("tau"),
            py::arg("sigma"), py::arg("rate"),
            "Compute gamma (d²P/dS²)")
        .def("vega", &mango::AmericanPriceSurface::vega,
            py::arg("spot"), py::arg("strike"), py::arg("tau"),
            py::arg("sigma"), py::arg("rate"),
            "Compute vega (dP/dsigma)")
        .def("theta", &mango::AmericanPriceSurface::theta,
            py::arg("spot"), py::arg("strike"), py::arg("tau"),
            py::arg("sigma"), py::arg("rate"),
            "Compute theta (dV/dt, calendar time, negative for decay)")
        .def_property_readonly("metadata", &mango::AmericanPriceSurface::metadata)
        .def("__repr__", [](const mango::AmericanPriceSurface& self) {
            return "<AmericanPriceSurface K_ref=" +
                   std::to_string(self.metadata().K_ref) + ">";
        });

    // =========================================================================
    // Price table builder convenience wrapper (auto-grid profiles)
    // =========================================================================
    m.def("build_price_table_surface_from_chain_auto_profile",
        [](double spot,
           const std::vector<double>& strikes,
           const std::vector<double>& maturities,
           const std::vector<double>& implied_vols,
           const std::vector<double>& rates,
           double dividend_yield,
           mango::OptionType type,
           mango::PriceTableGridProfile grid_profile,
           mango::GridAccuracyProfile pde_profile) {
            mango::OptionChain chain;
            chain.spot = spot;
            chain.strikes = strikes;
            chain.maturities = maturities;
            chain.implied_vols = implied_vols;
            chain.rates = rates;
            chain.dividend_yield = dividend_yield;

            auto builder_axes = mango::PriceTableBuilder<4>::from_chain_auto_profile(
                chain, grid_profile, pde_profile, type);
            if (!builder_axes.has_value()) {
                std::ostringstream oss;
                oss << "from_chain_auto_profile failed: " << builder_axes.error();
                throw py::value_error(oss.str());
            }

            auto [builder, axes] = std::move(builder_axes.value());
            auto build_result = builder.build(axes);
            if (!build_result.has_value()) {
                std::ostringstream oss;
                oss << "price table build failed: " << build_result.error();
                throw py::value_error(oss.str());
            }

            return build_result.value().surface;
        },
        py::arg("spot"),
        py::arg("strikes"),
        py::arg("maturities"),
        py::arg("implied_vols"),
        py::arg("rates"),
        py::arg("dividend_yield") = 0.0,
        py::arg("option_type") = mango::OptionType::PUT,
        py::arg("grid_profile") = mango::PriceTableGridProfile::High,
        py::arg("pde_profile") = mango::GridAccuracyProfile::High,
        R"pbdoc(
            Build a 4D price table surface from an option chain using auto-grid profiles.

            Args:
                spot: Underlying spot price
                strikes: Strike prices
                maturities: Times to expiration (years)
                implied_vols: Implied volatility samples for grid bounds
                rates: Risk-free rates
                dividend_yield: Continuous dividend yield (default 0.0)
                option_type: OptionType.PUT or OptionType.CALL
                grid_profile: PriceTableGridProfile (LOW/MEDIUM/HIGH/ULTRA)
                pde_profile: GridAccuracyProfile for PDE grid/time steps

            Returns:
                PriceTableSurface4D instance
        )pbdoc");

    m.def("build_price_table_surface_from_chain",
        [](const mango::OptionChain& chain,
           mango::OptionType type,
           mango::PriceTableGridProfile grid_profile,
           mango::GridAccuracyProfile pde_profile) {
            auto builder_axes = mango::PriceTableBuilder<4>::from_chain_auto_profile(
                chain, grid_profile, pde_profile, type);
            if (!builder_axes.has_value()) {
                std::ostringstream oss;
                oss << "from_chain_auto_profile failed: " << builder_axes.error();
                throw py::value_error(oss.str());
            }

            auto [builder, axes] = std::move(builder_axes.value());
            auto build_result = builder.build(axes);
            if (!build_result.has_value()) {
                std::ostringstream oss;
                oss << "price table build failed: " << build_result.error();
                throw py::value_error(oss.str());
            }

            return build_result.value().surface;
        },
        py::arg("chain"),
        py::arg("option_type") = mango::OptionType::PUT,
        py::arg("grid_profile") = mango::PriceTableGridProfile::High,
        py::arg("pde_profile") = mango::GridAccuracyProfile::High,
        R"pbdoc(
            Build a 4D price table surface from an OptionChain using auto-grid profiles.

            Args:
                chain: OptionChain with spot, strikes, maturities, implied_vols, rates
                option_type: OptionType.PUT or OptionType.CALL
                grid_profile: PriceTableGridProfile (LOW/MEDIUM/HIGH/ULTRA)
                pde_profile: GridAccuracyProfile for PDE grid/time steps

            Returns:
                PriceTableSurface4D instance
        )pbdoc");

    // =========================================================================
    // IVSolverInterpolated (fast IV solving using B-spline interpolation)
    // =========================================================================

    // IVSolverInterpolatedConfig
    py::class_<mango::IVSolverInterpolatedConfig>(m, "IVSolverInterpolatedConfig")
        .def(py::init<>())
        .def_readwrite("max_iterations", &mango::IVSolverInterpolatedConfig::max_iterations)
        .def_readwrite("tolerance", &mango::IVSolverInterpolatedConfig::tolerance)
        .def_readwrite("sigma_min", &mango::IVSolverInterpolatedConfig::sigma_min)
        .def_readwrite("sigma_max", &mango::IVSolverInterpolatedConfig::sigma_max);

    // IVSolverInterpolated
    py::class_<mango::IVSolverInterpolatedStandard>(m, "IVSolverInterpolated")
        .def_static("create",
            [](mango::AmericanPriceSurface american_surface,
               const mango::IVSolverInterpolatedConfig& config) {
                auto result = mango::IVSolverInterpolatedStandard::create(
                    std::move(american_surface), config);
                if (!result.has_value()) {
                    throw py::value_error("Failed to create solver: validation error");
                }
                return std::move(result.value());
            },
            py::arg("american_surface"),
            py::arg("config") = mango::IVSolverInterpolatedConfig{},
            R"pbdoc(
                Create an interpolation-based IV solver from an AmericanPriceSurface.

                Args:
                    american_surface: Pre-built AmericanPriceSurface
                    config: Optional solver configuration

                Returns:
                    IVSolverInterpolated instance

                Raises:
                    ValueError: If validation fails
            )pbdoc")
        .def("solve_impl",
            [](const mango::IVSolverInterpolatedStandard& solver, const mango::IVQuery& query) {
                auto result = solver.solve_impl(query);
                if (result.has_value()) {
                    return py::make_tuple(true, result.value(), mango::IVError{});
                } else {
                    return py::make_tuple(false, mango::IVSuccess{}, result.error());
                }
            },
            py::arg("query"),
            R"pbdoc(
                Solve for implied volatility (single query).

                Uses Newton-Raphson with B-spline interpolation (~3.5µs vs ~19ms FDM).

                Note: When a YieldCurve is provided, it is collapsed to zero rate: -ln(D(T))/T.
                This provides a reasonable approximation but does not capture term structure
                dynamics. For full yield curve support, use IVSolverFDM instead.

                Args:
                    query: IVQuery with option parameters and market price

                Returns:
                    Tuple of (success: bool, result: IVSuccess, error: IVError)
            )pbdoc")
        .def("solve_batch",
            [](const mango::IVSolverInterpolatedStandard& solver, const std::vector<mango::IVQuery>& queries) {
                auto batch_result = solver.solve_batch_impl(queries);
                py::list results;
                for (const auto& r : batch_result.results) {
                    if (r.has_value()) {
                        results.append(py::make_tuple(true, r.value(), mango::IVError{}));
                    } else {
                        results.append(py::make_tuple(false, mango::IVSuccess{}, r.error()));
                    }
                }
                return py::make_tuple(results, batch_result.failed_count);
            },
            py::arg("queries"),
            R"pbdoc(
                Solve for implied volatility (batch with OpenMP parallelization).

                Args:
                    queries: List of IVQuery objects

                Returns:
                    Tuple of (results: list of (success, IVSuccess, IVError), failed_count: int)
            )pbdoc");
}

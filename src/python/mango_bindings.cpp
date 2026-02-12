// SPDX-License-Identifier: MIT
/**
 * @file mango_bindings.cpp
 * @brief Python bindings for mango-option library using pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "mango/option/option_spec.hpp"
#include "mango/option/iv_solver.hpp"
#include "mango/option/interpolated_iv_solver.hpp"
#include "mango/option/american_option.hpp"
#include "mango/option/grid_spec_types.hpp"
#include "mango/option/option_grid.hpp"
#include "mango/option/table/bspline/bspline_builder.hpp"
#include "mango/option/table/bspline/bspline_surface.hpp"
#include "mango/option/table/adaptive_grid_types.hpp"
#include "mango/pde/core/pde_workspace.hpp"
#include "mango/math/yield_curve.hpp"
#include "mango/option/american_option_batch.hpp"

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

    // OptionGrid data container
    py::class_<mango::OptionGrid>(m, "OptionGrid")
        .def(py::init<>())
        .def_readwrite("ticker", &mango::OptionGrid::ticker)
        .def_readwrite("spot", &mango::OptionGrid::spot)
        .def_readwrite("strikes", &mango::OptionGrid::strikes)
        .def_readwrite("maturities", &mango::OptionGrid::maturities)
        .def_readwrite("implied_vols", &mango::OptionGrid::implied_vols)
        .def_readwrite("rates", &mango::OptionGrid::rates)
        .def_readwrite("dividend_yield", &mango::OptionGrid::dividend_yield)
        .def("__repr__", [](const mango::OptionGrid& chain) {
            return "<OptionGrid spot=" + std::to_string(chain.spot) +
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
        .def(py::init([](double spot, double strike, double maturity,
                         double rate, double dividend_yield, mango::OptionType type,
                         double market_price) {
                 return mango::IVQuery(
                     mango::OptionSpec{.spot = spot, .strike = strike, .maturity = maturity,
                         .rate = rate, .dividend_yield = dividend_yield, .option_type = type},
                     market_price);
             }),
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
        .def_readwrite("option_type", &mango::IVQuery::option_type)
        .def_readwrite("market_price", &mango::IVQuery::market_price)
        .def("__repr__", [](const mango::IVQuery& q) {
            return "<IVQuery spot=" + std::to_string(q.spot) +
                   " strike=" + std::to_string(q.strike) +
                   " maturity=" + std::to_string(q.maturity) +
                   " rate=" + rate_spec_to_string(q.rate) +
                   " dividend_yield=" + std::to_string(q.dividend_yield) +
                   " type=" + (q.option_type == mango::OptionType::CALL ? "CALL" : "PUT") +
                   " market_price=" + std::to_string(q.market_price) + ">";
        });

    // RootFindingConfig structure
    py::class_<mango::RootFindingConfig>(m, "RootFindingConfig")
        .def(py::init<>())
        .def_readwrite("max_iter", &mango::RootFindingConfig::max_iter)
        .def_readwrite("tolerance", &mango::RootFindingConfig::tolerance)
        .def_readwrite("jacobian_fd_epsilon", &mango::RootFindingConfig::jacobian_fd_epsilon)
        .def_readwrite("brent_tol_abs", &mango::RootFindingConfig::brent_tol_abs);

    // IVSolverConfig structure
    py::class_<mango::IVSolverConfig>(m, "IVSolverConfig")
        .def(py::init<>())
        .def_readwrite("root_config", &mango::IVSolverConfig::root_config)
        .def_readwrite("batch_parallel_threshold", &mango::IVSolverConfig::batch_parallel_threshold);
    // Note: PDEGridSpec variant binding deferred — Python users use default auto-estimation

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
        .value("OptionTypeMismatch", mango::IVErrorCode::OptionTypeMismatch)
        .value("DividendYieldMismatch", mango::IVErrorCode::DividendYieldMismatch)
        .value("PDESolveFailed", mango::IVErrorCode::PDESolveFailed)
        .export_values();

    // IVSolver class (FDM-based IV solver with std::expected)
    py::class_<mango::IVSolver>(m, "IVSolver")
        .def(py::init<const mango::IVSolverConfig&>(),
             py::arg("config"))
        .def("solve", [](const mango::IVSolver& solver, const mango::IVQuery& query) {
            auto result = solver.solve(query);
            if (result.has_value()) {
                return py::make_tuple(true, result.value(), mango::IVError{});
            } else {
                return py::make_tuple(false, mango::IVSuccess{}, result.error());
            }
        },
        py::arg("query"),
        "Solve for implied volatility. Returns (success: bool, result: IVSuccess, error: IVError)");

    // Note: Batch solver removed - users should use InterpolatedIVSolver for batch queries

    // Dividend structure (must be registered before PricingParams)
    py::class_<mango::Dividend>(m, "Dividend")
        .def(py::init<>())
        .def(py::init<double, double>(), py::arg("calendar_time"), py::arg("amount"))
        .def_readwrite("calendar_time", &mango::Dividend::calendar_time)
        .def_readwrite("amount", &mango::Dividend::amount)
        .def("__repr__", [](const mango::Dividend& d) {
            return "Dividend(t=" + std::to_string(d.calendar_time) +
                   ", amt=" + std::to_string(d.amount) + ")";
        });

    // DividendSpec structure
    py::class_<mango::DividendSpec>(m, "DividendSpec")
        .def(py::init<>())
        .def_readwrite("dividend_yield", &mango::DividendSpec::dividend_yield)
        .def_readwrite("discrete_dividends", &mango::DividendSpec::discrete_dividends);

    // PricingParams structure
    py::class_<mango::PricingParams>(m, "PricingParams")
        .def(py::init<>())
        .def_readwrite("strike", &mango::PricingParams::strike)
        .def_readwrite("spot", &mango::PricingParams::spot)
        .def_readwrite("maturity", &mango::PricingParams::maturity)
        .def_readwrite("volatility", &mango::PricingParams::volatility)
        .def_property("rate",
            [](const mango::PricingParams& p) { return rate_spec_to_python(p.rate); },
            [](mango::PricingParams& p, const py::object& obj) { p.rate = python_to_rate_spec(obj); },
            "Risk-free rate (float or YieldCurve)")
        .def_readwrite("dividend_yield", &mango::PricingParams::dividend_yield)
        .def_readwrite("option_type", &mango::PricingParams::option_type)
        .def_readwrite("discrete_dividends", &mango::PricingParams::discrete_dividends);

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
        [](const mango::PricingParams& params,
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
                accuracy = mango::make_grid_accuracy(accuracy_profile.value());
            }

            // Estimate grid automatically (sinh-spaced, clustered near strike)
            auto [grid_spec, time_domain] = mango::estimate_pde_grid(params, accuracy);

            // Allocate workspace buffer
            size_t n = grid_spec.n_points();
            std::vector<double> buffer(mango::PDEWorkspace::required_size(n));

            auto workspace_result = mango::PDEWorkspace::from_buffer(buffer, n);
            if (!workspace_result) {
                throw py::value_error(
                    "Failed to create workspace: " + workspace_result.error());
            }

            auto solver_result = mango::AmericanOptionSolver::create(
                params, workspace_result.value(),
                mango::PDEGridConfig{grid_spec, time_domain.n_steps(), {}});
            if (!solver_result) {
                throw py::value_error(
                    "Failed to create solver (validation error code " +
                    std::to_string(static_cast<int>(solver_result.error().code)) + ")");
            }
            auto& solver = solver_result.value();
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
                params: PricingParams with contract and market parameters.
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
                self.set_grid_accuracy(mango::make_grid_accuracy(profile));
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
               const std::vector<mango::PricingParams>& params,
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
                    params: List of PricingParams
                    use_shared_grid: If True, all options share one global grid
                                     (required for normalized chain optimization)

                Returns:
                    Tuple of (results, failed_count) where results is a list of
                    (success: bool, result: AmericanOptionResult|None, error: SolverError) tuples
            )pbdoc");

    // =========================================================================
    // PriceTableAxes (4D grid metadata)
    // =========================================================================

    // PriceTableAxes
    py::class_<mango::PriceTableAxes>(m, "PriceTableAxes")
        .def(py::init<>())
        .def_property("grids",
            [](const mango::PriceTableAxes& self) {
                py::list result;
                for (const auto& grid : self.grids) {
                    result.append(py::array_t<double>(grid.size(), grid.data()));
                }
                return result;
            },
            [](mango::PriceTableAxes& self, const py::list& grids) {
                if (grids.size() != 4) {
                    throw py::value_error("Must provide exactly 4 grids");
                }
                for (size_t i = 0; i < 4; ++i) {
                    auto arr = grids[i].cast<py::array_t<double>>();
                    self.grids[i] = std::vector<double>(arr.data(), arr.data() + arr.size());
                }
            })
        .def_property("names",
            [](const mango::PriceTableAxes& self) {
                py::list result;
                for (const auto& name : self.names) {
                    result.append(name);
                }
                return result;
            },
            [](mango::PriceTableAxes& self, const py::list& names) {
                if (names.size() != 4) {
                    throw py::value_error("Must provide exactly 4 names");
                }
                for (size_t i = 0; i < 4; ++i) {
                    self.names[i] = names[i].cast<std::string>();
                }
            })
        .def("total_points", [](const mango::PriceTableAxes& self) {
            return self.total_points();
        })
        .def("shape", [](const mango::PriceTableAxes& self) {
            auto s = self.shape();
            return py::make_tuple(s[0], s[1], s[2], s[3]);
        });

    // PriceTableSurface and builder convenience wrappers removed.
    // Use make_interpolated_iv_solver() for interpolation-based IV solving.

    // =========================================================================
    // InterpolatedIVSolver (fast IV solving using B-spline interpolation)
    // =========================================================================

    // InterpolatedIVSolverConfig
    py::class_<mango::InterpolatedIVSolverConfig>(m, "InterpolatedIVSolverConfig")
        .def(py::init<>())
        .def_readwrite("max_iter", &mango::InterpolatedIVSolverConfig::max_iter)
        .def_readwrite("tolerance", &mango::InterpolatedIVSolverConfig::tolerance)
        .def_readwrite("sigma_min", &mango::InterpolatedIVSolverConfig::sigma_min)
        .def_readwrite("sigma_max", &mango::InterpolatedIVSolverConfig::sigma_max);

    // IVGrid config
    py::class_<mango::IVGrid>(m, "IVGrid")
        .def(py::init<>())
        .def_readwrite("moneyness", &mango::IVGrid::moneyness)
        .def_readwrite("vol", &mango::IVGrid::vol)
        .def_readwrite("rate", &mango::IVGrid::rate);

    // AdaptiveGridParams config
    py::class_<mango::AdaptiveGridParams>(m, "AdaptiveGridParams")
        .def(py::init<>())
        .def_readwrite("target_iv_error", &mango::AdaptiveGridParams::target_iv_error)
        .def_readwrite("max_iter", &mango::AdaptiveGridParams::max_iter)
        .def_readwrite("max_points_per_dim", &mango::AdaptiveGridParams::max_points_per_dim)
        .def_readwrite("min_moneyness_points", &mango::AdaptiveGridParams::min_moneyness_points);

    // MultiKRefConfig
    py::class_<mango::MultiKRefConfig>(m, "MultiKRefConfig")
        .def(py::init<>())
        .def_readwrite("K_refs", &mango::MultiKRefConfig::K_refs)
        .def_readwrite("K_ref_count", &mango::MultiKRefConfig::K_ref_count)
        .def_readwrite("K_ref_span", &mango::MultiKRefConfig::K_ref_span);

    // BSplineBackend
    py::class_<mango::BSplineBackend>(m, "BSplineBackend")
        .def(py::init<>())
        .def_readwrite("maturity_grid", &mango::BSplineBackend::maturity_grid);

    // ChebyshevBackend
    py::class_<mango::ChebyshevBackend>(m, "ChebyshevBackend")
        .def(py::init<>())
        .def_readwrite("maturity", &mango::ChebyshevBackend::maturity)
        .def_readwrite("tucker_epsilon", &mango::ChebyshevBackend::tucker_epsilon);

    // DiscreteDividendConfig
    py::class_<mango::DiscreteDividendConfig>(m, "DiscreteDividendConfig")
        .def(py::init<>())
        .def_readwrite("maturity", &mango::DiscreteDividendConfig::maturity)
        .def_readwrite("discrete_dividends", &mango::DiscreteDividendConfig::discrete_dividends)
        .def_readwrite("kref_config", &mango::DiscreteDividendConfig::kref_config);

    // IVSolverFactoryConfig
    py::class_<mango::IVSolverFactoryConfig>(m, "IVSolverFactoryConfig")
        .def(py::init<>())
        .def_readwrite("option_type", &mango::IVSolverFactoryConfig::option_type)
        .def_readwrite("spot", &mango::IVSolverFactoryConfig::spot)
        .def_readwrite("dividend_yield", &mango::IVSolverFactoryConfig::dividend_yield)
        .def_readwrite("grid", &mango::IVSolverFactoryConfig::grid)
        .def_property("adaptive",
            [](const mango::IVSolverFactoryConfig& c) -> py::object {
                if (c.adaptive.has_value()) return py::cast(*c.adaptive);
                return py::none();
            },
            [](mango::IVSolverFactoryConfig& c, const py::object& obj) {
                if (obj.is_none()) c.adaptive = std::nullopt;
                else c.adaptive = obj.cast<mango::AdaptiveGridParams>();
            })
        .def_readwrite("solver_config", &mango::IVSolverFactoryConfig::solver_config)
        .def_property("backend",
            [](const mango::IVSolverFactoryConfig& c) -> py::object {
                return std::visit([](const auto& b) -> py::object {
                    return py::cast(b);
                }, c.backend);
            },
            [](mango::IVSolverFactoryConfig& c, const py::object& obj) {
                if (py::isinstance<mango::BSplineBackend>(obj))
                    c.backend = obj.cast<mango::BSplineBackend>();
                else if (py::isinstance<mango::ChebyshevBackend>(obj))
                    c.backend = obj.cast<mango::ChebyshevBackend>();
                else
                    throw py::type_error("backend must be BSplineBackend or ChebyshevBackend");
            })
        .def_property("discrete_dividends",
            [](const mango::IVSolverFactoryConfig& c) -> py::object {
                if (c.discrete_dividends.has_value()) return py::cast(*c.discrete_dividends);
                return py::none();
            },
            [](mango::IVSolverFactoryConfig& c, const py::object& obj) {
                if (obj.is_none()) c.discrete_dividends = std::nullopt;
                else c.discrete_dividends = obj.cast<mango::DiscreteDividendConfig>();
            });

    // AnyInterpIVSolver (exposed as InterpolatedIVSolver for Python)
    py::class_<mango::AnyInterpIVSolver>(m, "InterpolatedIVSolver")
        .def("solve",
            [](const mango::AnyInterpIVSolver& solver, const mango::IVQuery& query) {
                auto result = solver.solve(query);
                if (result.has_value()) {
                    return py::make_tuple(true, result.value(), mango::IVError{});
                } else {
                    return py::make_tuple(false, mango::IVSuccess{}, result.error());
                }
            },
            py::arg("query"),
            R"pbdoc(
                Solve for implied volatility (single query).

                Uses Newton-Raphson with B-spline interpolation (~3.5us per query).

                Args:
                    query: IVQuery with option parameters and market price

                Returns:
                    Tuple of (success: bool, result: IVSuccess, error: IVError)
            )pbdoc")
        .def("solve_batch",
            [](const mango::AnyInterpIVSolver& solver, const std::vector<mango::IVQuery>& queries) {
                auto batch_result = solver.solve_batch(queries);
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

    // Factory function
    m.def("make_interpolated_iv_solver",
        [](const mango::IVSolverFactoryConfig& config) {
            auto result = mango::make_interpolated_iv_solver(config);
            if (!result.has_value()) {
                throw py::value_error(
                    "Failed to create IV solver: validation error code " +
                    std::to_string(static_cast<int>(result.error().code)));
            }
            return std::move(*result);
        },
        py::arg("config"),
        R"pbdoc(
            Create an interpolation-based IV solver from configuration.

            Builds a price surface and wraps it in a solver. Supports both
            standard (continuous dividend) and segmented (discrete dividend) paths.

            Args:
                config: IVSolverFactoryConfig with grid, path, and solver parameters

            Returns:
                InterpolatedIVSolver instance

            Raises:
                ValueError: If validation or surface building fails
        )pbdoc");
}

/**
 * @file mango_bindings.cpp
 * @brief Python bindings for mango-iv library using pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "src/option/option_spec.hpp"
#include "src/option/iv_solver_fdm.hpp"
#include "src/option/american_option.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include "src/math/yield_curve.hpp"

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

PYBIND11_MODULE(mango_iv, m) {
    m.doc() = "Python bindings for mango-iv American option pricing and IV solver";

    // OptionType enum
    py::enum_<mango::OptionType>(m, "OptionType")
        .value("CALL", mango::OptionType::CALL)
        .value("PUT", mango::OptionType::PUT);

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
        .def_readwrite("grid_alpha", &mango::IVSolverFDMConfig::grid_alpha);

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
           double x_min,
           double x_max,
           size_t n_space,
           [[maybe_unused]] size_t n_time) {
            auto grid_spec_result = mango::GridSpec<double>::uniform(x_min, x_max, n_space);
            if (!grid_spec_result.has_value()) {
                auto err = grid_spec_result.error();
                throw py::value_error(
                    "Failed to create grid (error code " + std::to_string(static_cast<int>(err.code)) + ")");
            }

            // Allocate workspace buffer (local, temporary)
            size_t n = grid_spec_result.value().n_points();
            std::vector<double> buffer(mango::PDEWorkspace::required_size(n));

            auto workspace_result = mango::PDEWorkspace::from_buffer(buffer, n);
            if (!workspace_result) {
                throw py::value_error(
                    "Failed to create workspace: " + workspace_result.error());
            }

            mango::AmericanOptionSolver solver(params, workspace_result.value());
            auto solve_result = solver.solve();
            if (!solve_result) {
                auto error = solve_result.error();
                throw py::value_error(
                    "American option solve failed (error code " + std::to_string(static_cast<int>(error.code)) + ")");
            }

            return std::move(solve_result.value());
        },
        py::arg("params"),
        py::arg("x_min") = -3.0,
        py::arg("x_max") = 3.0,
        py::arg("n_space") = 201,
        py::arg("n_time") = 2000,
        R"pbdoc(
            Price an American option using the PDE solver.

            Args:
                params: AmericanOptionParams with contract information.
                x_min/x_max: Log-moneyness domain bounds.
                n_space: Number of spatial grid points.
                n_time: Number of time steps.

            Returns:
                AmericanOptionResult with value and Greeks.
        )pbdoc");
}

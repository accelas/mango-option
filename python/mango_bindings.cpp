/**
 * @file mango_bindings.cpp
 * @brief Python bindings for mango-iv library using pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "src/iv_solver.hpp"
#include "src/american_option.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mango_iv, m) {
    m.doc() = "Python bindings for mango-iv American option pricing and IV solver";

    // IVParams structure
    py::class_<mango::IVParams>(m, "IVParams")
        .def(py::init<>())
        .def_readwrite("spot_price", &mango::IVParams::spot_price)
        .def_readwrite("strike", &mango::IVParams::strike)
        .def_readwrite("time_to_maturity", &mango::IVParams::time_to_maturity)
        .def_readwrite("risk_free_rate", &mango::IVParams::risk_free_rate)
        .def_readwrite("market_price", &mango::IVParams::market_price)
        .def_readwrite("is_call", &mango::IVParams::is_call)
        .def("__repr__", [](const mango::IVParams& p) {
            return "<IVParams spot=" + std::to_string(p.spot_price) +
                   " strike=" + std::to_string(p.strike) +
                   " maturity=" + std::to_string(p.time_to_maturity) +
                   " rate=" + std::to_string(p.risk_free_rate) +
                   " price=" + std::to_string(p.market_price) +
                   " is_call=" + std::to_string(p.is_call) + ">";
        });

    // RootFindingConfig structure
    py::class_<mango::RootFindingConfig>(m, "RootFindingConfig")
        .def(py::init<>())
        .def_readwrite("max_iter", &mango::RootFindingConfig::max_iter)
        .def_readwrite("tolerance", &mango::RootFindingConfig::tolerance)
        .def_readwrite("jacobian_fd_epsilon", &mango::RootFindingConfig::jacobian_fd_epsilon)
        .def_readwrite("brent_tol_abs", &mango::RootFindingConfig::brent_tol_abs);

    // IVConfig structure
    py::class_<mango::IVConfig>(m, "IVConfig")
        .def(py::init<>())
        .def_readwrite("root_config", &mango::IVConfig::root_config)
        .def_readwrite("grid_n_space", &mango::IVConfig::grid_n_space)
        .def_readwrite("grid_n_time", &mango::IVConfig::grid_n_time)
        .def_readwrite("grid_s_max", &mango::IVConfig::grid_s_max);

    // IVResult structure
    py::class_<mango::IVResult>(m, "IVResult")
        .def(py::init<>())
        .def_readwrite("converged", &mango::IVResult::converged)
        .def_readwrite("iterations", &mango::IVResult::iterations)
        .def_readwrite("implied_vol", &mango::IVResult::implied_vol)
        .def_readwrite("final_error", &mango::IVResult::final_error)
        .def_readwrite("failure_reason", &mango::IVResult::failure_reason)
        .def_readwrite("vega", &mango::IVResult::vega)
        .def("__repr__", [](const mango::IVResult& r) {
            std::string repr = "<IVResult converged=" + std::to_string(r.converged);
            if (r.converged) {
                repr += " iv=" + std::to_string(r.implied_vol) +
                       " iters=" + std::to_string(r.iterations) +
                       " error=" + std::to_string(r.final_error);
            } else if (r.failure_reason.has_value()) {
                repr += " reason='" + *r.failure_reason + "'";
            }
            return repr + ">";
        });

    // IVSolver class
    py::class_<mango::IVSolver>(m, "IVSolver")
        .def(py::init<const mango::IVParams&, const mango::IVConfig&>(),
             py::arg("params"), py::arg("config"))
        .def("solve", &mango::IVSolver::solve,
             "Solve for implied volatility");

    // Batch IV solver convenience function
    m.def("solve_implied_vol_batch",
          py::overload_cast<const std::vector<mango::IVParams>&, const mango::IVConfig&>(
              &mango::solve_implied_vol_batch),
          py::arg("params"), py::arg("config"),
          "Solve implied volatility for a batch of options in parallel");

    // OptionType enum
    py::enum_<mango::OptionType>(m, "OptionType")
        .value("CALL", mango::OptionType::CALL)
        .value("PUT", mango::OptionType::PUT)
        .export_values();

    // AmericanOptionParams structure
    py::class_<mango::AmericanOptionParams>(m, "AmericanOptionParams")
        .def(py::init<>())
        .def_readwrite("strike", &mango::AmericanOptionParams::strike)
        .def_readwrite("spot", &mango::AmericanOptionParams::spot)
        .def_readwrite("maturity", &mango::AmericanOptionParams::maturity)
        .def_readwrite("volatility", &mango::AmericanOptionParams::volatility)
        .def_readwrite("rate", &mango::AmericanOptionParams::rate)
        .def_readwrite("continuous_dividend_yield", &mango::AmericanOptionParams::continuous_dividend_yield)
        .def_readwrite("option_type", &mango::AmericanOptionParams::option_type)
        .def_readwrite("discrete_dividends", &mango::AmericanOptionParams::discrete_dividends);

    // AmericanOptionGrid structure
    py::class_<mango::AmericanOptionGrid>(m, "AmericanOptionGrid")
        .def(py::init<>())
        .def_readwrite("n_space", &mango::AmericanOptionGrid::n_space)
        .def_readwrite("n_time", &mango::AmericanOptionGrid::n_time)
        .def_readwrite("x_min", &mango::AmericanOptionGrid::x_min)
        .def_readwrite("x_max", &mango::AmericanOptionGrid::x_max);

    // AmericanOptionResult structure
    py::class_<mango::AmericanOptionResult>(m, "AmericanOptionResult")
        .def(py::init<>())
        .def_readwrite("value", &mango::AmericanOptionResult::value)
        .def_readwrite("delta", &mango::AmericanOptionResult::delta)
        .def_readwrite("gamma", &mango::AmericanOptionResult::gamma)
        .def_readwrite("theta", &mango::AmericanOptionResult::theta)
        .def_readwrite("converged", &mango::AmericanOptionResult::converged);
}

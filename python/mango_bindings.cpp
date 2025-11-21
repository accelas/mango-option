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
#include "src/support/memory/solver_memory_arena.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mango_iv, m) {
    m.doc() = "Python bindings for mango-iv American option pricing and IV solver";

    // OptionType enum
    py::enum_<mango::OptionType>(m, "OptionType")
        .value("CALL", mango::OptionType::CALL)
        .value("PUT", mango::OptionType::PUT);

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
        .def_readwrite("rate", &mango::IVQuery::rate)
        .def_readwrite("dividend_yield", &mango::IVQuery::dividend_yield)
        .def_readwrite("type", &mango::IVQuery::type)
        .def_readwrite("market_price", &mango::IVQuery::market_price)
        .def("__repr__", [](const mango::IVQuery& q) {
            return "<IVQuery spot=" + std::to_string(q.spot) +
                   " strike=" + std::to_string(q.strike) +
                   " maturity=" + std::to_string(q.maturity) +
                   " rate=" + std::to_string(q.rate) +
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
        .def_readwrite("grid_n_space", &mango::IVSolverFDMConfig::grid_n_space)
        .def_readwrite("grid_n_time", &mango::IVSolverFDMConfig::grid_n_time)
        .def_readwrite("grid_s_max", &mango::IVSolverFDMConfig::grid_s_max);

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

    // IVSolver class (now using FDM solver)
    py::class_<mango::IVSolverFDM>(m, "IVSolverFDM")
        .def(py::init<const mango::IVSolverFDMConfig&>(),
             py::arg("config"))
        .def("solve", &mango::IVSolverFDM::solve_impl,
             py::arg("query"),
             "Solve for implied volatility");

    // Note: Batch solver removed - users should use IVSolverInterpolated for batch queries

    // AmericanOptionParams structure
    py::class_<mango::AmericanOptionParams>(m, "AmericanOptionParams")
        .def(py::init<>())
        .def_readwrite("strike", &mango::AmericanOptionParams::strike)
        .def_readwrite("spot", &mango::AmericanOptionParams::spot)
        .def_readwrite("maturity", &mango::AmericanOptionParams::maturity)
        .def_readwrite("volatility", &mango::AmericanOptionParams::volatility)
        .def_readwrite("rate", &mango::AmericanOptionParams::rate)
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

    // SolverMemoryArenaStats structure
    py::class_<mango::memory::SolverMemoryArenaStats>(m, "SolverMemoryArenaStats")
        .def(py::init<>())
        .def_readwrite("total_size", &mango::memory::SolverMemoryArenaStats::total_size)
        .def_readwrite("used_size", &mango::memory::SolverMemoryArenaStats::used_size)
        .def_readwrite("active_workspace_count", &mango::memory::SolverMemoryArenaStats::active_workspace_count)
        .def("__repr__", [](const mango::memory::SolverMemoryArenaStats& s) {
            return "<SolverMemoryArenaStats total_size=" + std::to_string(s.total_size) +
                   " used_size=" + std::to_string(s.used_size) +
                   " active_workspace_count=" + std::to_string(s.active_workspace_count) + ">";
        });

    // SolverMemoryArena class
    py::class_<mango::memory::SolverMemoryArena, std::shared_ptr<mango::memory::SolverMemoryArena>>(m, "SolverMemoryArena")
        .def("try_reset", [](mango::memory::SolverMemoryArena& self) {
            auto result = self.try_reset();
            if (!result.has_value()) {
                throw std::runtime_error(result.error());
            }
        })
        .def("increment_active", &mango::memory::SolverMemoryArena::increment_active)
        .def("decrement_active", &mango::memory::SolverMemoryArena::decrement_active)
        .def("get_stats", &mango::memory::SolverMemoryArena::get_stats)
        .def("__repr__", [](const mango::memory::SolverMemoryArena& arena) {
            auto stats = arena.get_stats();
            return "<SolverMemoryArena total_size=" + std::to_string(stats.total_size) +
                   " used_size=" + std::to_string(stats.used_size) +
                   " active_workspaces=" + std::to_string(stats.active_workspace_count) + ">";
        });

    // ActiveWorkspaceToken RAII wrapper
    py::class_<mango::memory::SolverMemoryArena::ActiveWorkspaceToken>(m, "ActiveWorkspaceToken")
        .def(py::init<>())
        .def(py::init<std::shared_ptr<mango::memory::SolverMemoryArena>>(), py::arg("arena"))
        .def("reset", &mango::memory::SolverMemoryArena::ActiveWorkspaceToken::reset)
        .def("is_active", &mango::memory::SolverMemoryArena::ActiveWorkspaceToken::is_active)
        .def_property_readonly(
            "resource",
            [](const mango::memory::SolverMemoryArena::ActiveWorkspaceToken& self) {
                auto* resource = self.resource();
                if (!resource) {
                    throw std::runtime_error("ActiveWorkspaceToken has no active arena");
                }
                return resource;
            },
            py::return_value_policy::reference)
        .def("shared", &mango::memory::SolverMemoryArena::ActiveWorkspaceToken::shared)
        .def("__enter__", [](mango::memory::SolverMemoryArena::ActiveWorkspaceToken& self) -> mango::memory::SolverMemoryArena::ActiveWorkspaceToken& {
            if (!self.is_active()) {
                throw std::runtime_error("ActiveWorkspaceToken must be constructed with an arena before use");
            }
            return self;
        }, py::return_value_policy::reference)
        .def("__exit__", [](mango::memory::SolverMemoryArena::ActiveWorkspaceToken& self, py::handle, py::handle, py::handle) {
            self.reset();
            return false;
        });

    m.def(
        "american_option_price",
        [](const mango::AmericanOptionParams& params,
           double x_min,
           double x_max,
           size_t n_space,
           size_t n_time) {
            auto grid_spec_result = mango::GridSpec<double>::uniform(x_min, x_max, n_space);
            if (!grid_spec_result.has_value()) {
                throw py::value_error(
                    "Failed to create grid: " + grid_spec_result.error());
            }

            // Allocate workspace buffer (local, temporary)
            size_t n = grid_spec_result.value().n_points();
            std::pmr::vector<double> buffer(mango::PDEWorkspace::required_size(n), std::pmr::get_default_resource());

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
                    "American option solve failed: " + error.message);
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

    // SolverMemoryArena factory function
    m.def(
        "create_arena",
        [](size_t arena_size) {
            auto result = mango::memory::SolverMemoryArena::create(arena_size);
            if (!result) {
                throw py::value_error("Failed to create SolverMemoryArena: " + result.error());
            }
            return result.value();
        },
        py::arg("arena_size"),
        R"pbdoc(
            Create a solver memory arena for PMR-based memory management.

            Args:
                arena_size: Size of the memory arena in bytes.

            Returns:
                SolverMemoryArena instance with shared_ptr ownership.
        )pbdoc");
}

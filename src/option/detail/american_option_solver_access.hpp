// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/american_option.hpp"

#include <utility>

namespace mango::detail {

/// Package-private access to AmericanOptionSolver setup hooks.
///
/// Callers must first construct the solver through AmericanOptionSolver::create;
/// this shim only installs a continuation surface after parameter validation.
/// Its Bazel visibility is restricted to the option implementation package(s)
/// that build such validated chained surfaces.
class AmericanOptionSolverAccess {
public:
    static void set_initial_condition(
        AmericanOptionSolver& solver,
        AmericanOptionSolver::InitialCondition initial_condition)
    {
        solver.set_initial_condition(std::move(initial_condition));
    }
};

}  // namespace mango::detail

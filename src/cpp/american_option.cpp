/**
 * @file american_option.cpp
 * @brief American option pricing solver implementation
 */

#include "src/cpp/american_option.hpp"
#include <algorithm>

namespace mango {

AmericanOptionSolver::AmericanOptionSolver(
    const AmericanOptionParams& params,
    const AmericanOptionGrid& grid,
    const TRBDF2Config& trbdf2_config,
    const RootFindingConfig& root_config)
    : params_(params)
    , grid_(grid)
{
    // Validate parameters
    params_.validate();
    grid_.validate();
}

void AmericanOptionSolver::register_dividend(double time, double amount) {
    if (time < 0.0 || time > params_.maturity) {
        throw std::invalid_argument("Dividend time must be in [0, maturity]");
    }
    if (amount < 0.0) {
        throw std::invalid_argument("Dividend amount must be non-negative");
    }
    dividends_.push_back({time, amount});

    // Sort by time (early dividends first)
    std::sort(dividends_.begin(), dividends_.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
}

AmericanOptionResult AmericanOptionSolver::solve() {
    // TODO: Implement in Task 8
    throw std::runtime_error("AmericanOptionSolver::solve() not yet implemented");
}

std::vector<double> AmericanOptionSolver::get_solution() const {
    if (!solved_) {
        throw std::runtime_error("Solver has not been run yet");
    }
    return solution_;
}

double AmericanOptionSolver::compute_delta() const {
    // TODO: Implement in Task 9
    return 0.0;
}

double AmericanOptionSolver::compute_gamma() const {
    // TODO: Implement in Task 9
    return 0.0;
}

double AmericanOptionSolver::compute_theta() const {
    // TODO: Implement in Task 9
    return 0.0;
}

}  // namespace mango

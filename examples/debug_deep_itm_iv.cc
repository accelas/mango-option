#include "src/option/iv_solver_fdm.hpp"
#include "src/option/american_option.hpp"
#include "src/pde/core/pde_workspace.hpp"
#include <iostream>
#include <iomanip>
#include <memory_resource>

int main() {
    using namespace mango;

    // Deep ITM put from failing IV test
    IVQuery query;
    query.spot = 50.0;        // Deep in the money (S/K = 0.5)
    query.strike = 100.0;
    query.maturity = 1.0;
    query.rate = 0.05;
    query.dividend_yield = 0.0;
    query.type = OptionType::PUT;
    query.market_price = 51.0;  // Intrinsic value is 50

    IVSolverFDMConfig config{
        .root_config = RootFindingConfig{
            .max_iter = 100,
            .tolerance = 1e-6,
            .brent_tol_abs = 1e-6
        },
        .use_manual_grid = true,
        .grid_n_space = 101,
        .grid_n_time = 1000,
        .grid_x_min = -3.0,
        .grid_x_max = 3.0,
        .grid_alpha = 2.0
    };

    std::cout << "=== Deep ITM Put IV Test ===\n";
    std::cout << "S=" << query.spot << " K=" << query.strike << " T=" << query.maturity << "\n";
    std::cout << "Market price: " << query.market_price << "\n";
    std::cout << "Intrinsic: " << (query.strike - query.spot) << "\n";
    std::cout << "Time value: " << (query.market_price - (query.strike - query.spot)) << "\n\n";

    // First, test if we can price the option directly with various volatilities
    std::cout << "Testing direct pricing with different volatilities:\n";

    for (double vol : {0.01, 0.05, 0.10, 0.15, 0.20, 0.30}) {
        AmericanOptionParams params(
            query.spot,
            query.strike,
            query.maturity,
            query.rate,
            query.dividend_yield,
            query.type,
            vol
        );

        double moneyness = query.spot / query.strike;
        double x_min = std::log(std::min(0.5, moneyness * 0.5));
        double x_max = std::log(std::max(2.0, 200.0 / query.strike));

        auto grid_spec = GridSpec<double>::uniform(x_min, x_max, config.grid_n_space);
        if (!grid_spec.has_value()) {
            std::cerr << "Grid creation failed\n";
            return 1;
        }

        // Allocate buffer for workspace
        std::pmr::synchronized_pool_resource pool;
        size_t n = grid_spec.value().n_points();
        std::pmr::vector<double> buffer(PDEWorkspace::required_size(n), &pool);

        auto workspace_result = PDEWorkspace::from_buffer(buffer, n);
        if (!workspace_result.has_value()) {
            std::cerr << "Workspace creation failed: " << workspace_result.error() << "\n";
            return 1;
        }

        AmericanOptionSolver solver(params, workspace_result.value());
        auto result = solver.solve();

        if (result.has_value()) {
            double price = result.value().value_at(query.spot);
            std::cout << "  σ=" << vol << ": Price=" << price
                      << " TimeValue=" << (price - (query.strike - query.spot)) << "\n";
        } else {
            std::cout << "  σ=" << vol << ": Pricing failed\n";
        }
    }
    std::cout << "\n";

    // Now try IV solver
    std::cout << "Testing IV solver:\n";
    IVSolverFDM iv_solver(config);
    IVResult iv_result = iv_solver.solve(query);

    std::cout << "  Converged: " << (iv_result.converged ? "Yes" : "No") << "\n";
    std::cout << "  Iterations: " << iv_result.iterations << "\n";
    std::cout << "  Implied vol: " << iv_result.implied_vol << "\n";
    std::cout << "  Final error: " << iv_result.final_error << "\n";

    if (iv_result.failure_reason.has_value()) {
        std::cout << "  Failure reason: " << *iv_result.failure_reason << "\n";
    }

    return iv_result.converged ? 0 : 1;
}

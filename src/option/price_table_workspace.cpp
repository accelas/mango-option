#include "src/option/price_table_workspace.hpp"
#include <algorithm>
#include <numeric>
#include <cstring>

namespace mango {

expected<void, std::string> PriceTableWorkspace::validate_inputs(
    const std::vector<double>& m_grid,
    const std::vector<double>& tau_grid,
    const std::vector<double>& sigma_grid,
    const std::vector<double>& r_grid,
    const std::vector<double>& coefficients)
{
    // Validate grid sizes
    if (m_grid.size() < 4) {
        return unexpected("Moneyness grid must have >= 4 points");
    }
    if (tau_grid.size() < 4) {
        return unexpected("Maturity grid must have >= 4 points");
    }
    if (sigma_grid.size() < 4) {
        return unexpected("Volatility grid must have >= 4 points");
    }
    if (r_grid.size() < 4) {
        return unexpected("Rate grid must have >= 4 points");
    }

    // Validate coefficient size
    size_t expected_size = m_grid.size() * tau_grid.size() *
                          sigma_grid.size() * r_grid.size();
    if (coefficients.size() != expected_size) {
        return unexpected("Coefficient size mismatch: expected " +
                         std::to_string(expected_size) + ", got " +
                         std::to_string(coefficients.size()));
    }

    // Validate monotonicity
    auto is_sorted = [](const std::vector<double>& v) {
        return std::is_sorted(v.begin(), v.end());
    };

    if (!is_sorted(m_grid)) {
        return unexpected("Moneyness grid must be sorted ascending");
    }
    if (!is_sorted(tau_grid)) {
        return unexpected("Maturity grid must be sorted ascending");
    }
    if (!is_sorted(sigma_grid)) {
        return unexpected("Volatility grid must be sorted ascending");
    }
    if (!is_sorted(r_grid)) {
        return unexpected("Rate grid must be sorted ascending");
    }

    return {};
}

expected<PriceTableWorkspace, std::string> PriceTableWorkspace::allocate_and_initialize(
    const std::vector<double>& m_grid,
    const std::vector<double>& tau_grid,
    const std::vector<double>& sigma_grid,
    const std::vector<double>& r_grid,
    const std::vector<double>& coefficients,
    double K_ref,
    double dividend_yield)
{
    PriceTableWorkspace ws;

    // Compute knot vectors (clamped cubic B-spline)
    auto knots_m = clamped_knots_cubic(m_grid);
    auto knots_tau = clamped_knots_cubic(tau_grid);
    auto knots_sigma = clamped_knots_cubic(sigma_grid);
    auto knots_r = clamped_knots_cubic(r_grid);

    // Calculate total arena size
    size_t total_size = m_grid.size() + tau_grid.size() +
                       sigma_grid.size() + r_grid.size() +
                       knots_m.size() + knots_tau.size() +
                       knots_sigma.size() + knots_r.size() +
                       coefficients.size();

    // Allocate with 64-byte alignment for AVX-512
    // Use over-allocation to ensure alignment
    ws.arena_.resize(total_size + 8);  // +8 for alignment padding

    // Find 64-byte aligned start within arena
    auto arena_ptr = reinterpret_cast<std::uintptr_t>(ws.arena_.data());
    auto aligned_offset = (64 - (arena_ptr % 64)) % 64;
    double* aligned_start = ws.arena_.data() + aligned_offset / sizeof(double);

    // Copy data into arena
    double* ptr = aligned_start;

    std::memcpy(ptr, m_grid.data(), m_grid.size() * sizeof(double));
    ws.moneyness_ = std::span<const double>(ptr, m_grid.size());
    ptr += m_grid.size();

    std::memcpy(ptr, tau_grid.data(), tau_grid.size() * sizeof(double));
    ws.maturity_ = std::span<const double>(ptr, tau_grid.size());
    ptr += tau_grid.size();

    std::memcpy(ptr, sigma_grid.data(), sigma_grid.size() * sizeof(double));
    ws.volatility_ = std::span<const double>(ptr, sigma_grid.size());
    ptr += sigma_grid.size();

    std::memcpy(ptr, r_grid.data(), r_grid.size() * sizeof(double));
    ws.rate_ = std::span<const double>(ptr, r_grid.size());
    ptr += r_grid.size();

    std::memcpy(ptr, knots_m.data(), knots_m.size() * sizeof(double));
    ws.knots_m_ = std::span<const double>(ptr, knots_m.size());
    ptr += knots_m.size();

    std::memcpy(ptr, knots_tau.data(), knots_tau.size() * sizeof(double));
    ws.knots_tau_ = std::span<const double>(ptr, knots_tau.size());
    ptr += knots_tau.size();

    std::memcpy(ptr, knots_sigma.data(), knots_sigma.size() * sizeof(double));
    ws.knots_sigma_ = std::span<const double>(ptr, knots_sigma.size());
    ptr += knots_sigma.size();

    std::memcpy(ptr, knots_r.data(), knots_r.size() * sizeof(double));
    ws.knots_r_ = std::span<const double>(ptr, knots_r.size());
    ptr += knots_r.size();

    std::memcpy(ptr, coefficients.data(), coefficients.size() * sizeof(double));
    ws.coefficients_ = std::span<const double>(ptr, coefficients.size());

    ws.K_ref_ = K_ref;
    ws.dividend_yield_ = dividend_yield;

    return ws;
}

expected<PriceTableWorkspace, std::string> PriceTableWorkspace::create(
    const std::vector<double>& m_grid,
    const std::vector<double>& tau_grid,
    const std::vector<double>& sigma_grid,
    const std::vector<double>& r_grid,
    const std::vector<double>& coefficients,
    double K_ref,
    double dividend_yield)
{
    // Validate inputs first
    auto validation = validate_inputs(m_grid, tau_grid, sigma_grid, r_grid, coefficients);
    if (!validation) {
        return unexpected(validation.error());
    }

    // Allocate and initialize workspace
    return allocate_and_initialize(m_grid, tau_grid, sigma_grid, r_grid,
                                   coefficients, K_ref, dividend_yield);
}

}  // namespace mango

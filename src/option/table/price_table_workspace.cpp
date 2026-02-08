// SPDX-License-Identifier: MIT
#include "mango/option/table/price_table_workspace.hpp"
#include "mango/math/bspline_basis.hpp"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <vector>
#include <span>
#include <string>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace mango {

std::expected<void, std::string> PriceTableWorkspace::validate_inputs(
    std::span<const double> m_grid,
    std::span<const double> tau_grid,
    std::span<const double> sigma_grid,
    std::span<const double> r_grid,
    std::span<const double> coefficients)
{
    // Validate grid sizes
    if (m_grid.size() < 4) {
        return std::unexpected("Log-moneyness grid must have >= 4 points");
    }
    if (tau_grid.size() < 4) {
        return std::unexpected("Maturity grid must have >= 4 points");
    }
    if (sigma_grid.size() < 4) {
        return std::unexpected("Volatility grid must have >= 4 points");
    }
    if (r_grid.size() < 4) {
        return std::unexpected("Rate grid must have >= 4 points");
    }

    // Validate coefficient size
    size_t expected_size = m_grid.size() * tau_grid.size() *
                          sigma_grid.size() * r_grid.size();
    if (coefficients.size() != expected_size) {
        return std::unexpected("Coefficient size mismatch: expected " +
                         std::to_string(expected_size) + ", got " +
                         std::to_string(coefficients.size()));
    }

    // Validate monotonicity
    auto is_sorted = [](std::span<const double> v) {
        return std::is_sorted(v.begin(), v.end());
    };

    if (!is_sorted(m_grid)) {
        return std::unexpected("Log-moneyness grid must be sorted ascending");
    }
    if (!is_sorted(tau_grid)) {
        return std::unexpected("Maturity grid must be sorted ascending");
    }
    if (!is_sorted(sigma_grid)) {
        return std::unexpected("Volatility grid must be sorted ascending");
    }
    if (!is_sorted(r_grid)) {
        return std::unexpected("Rate grid must be sorted ascending");
    }

    return {};
}

std::expected<PriceTableWorkspace, std::string> PriceTableWorkspace::allocate_and_initialize(
    std::span<const double> m_grid,
    std::span<const double> tau_grid,
    std::span<const double> sigma_grid,
    std::span<const double> r_grid,
    std::span<const double> coefficients,
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
    ws.log_moneyness_ = std::span<const double>(ptr, m_grid.size());
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

// Helper for zero-copy loading from raw buffers (friend of PriceTableWorkspace)
std::expected<PriceTableWorkspace, std::string> allocate_and_initialize_from_buffers(
    const double* m_data, size_t n_m,
    const double* tau_data, size_t n_tau,
    const double* sigma_data, size_t n_sigma,
    const double* r_data, size_t n_r,
    const double* coeff_data, size_t n_coeffs,
    double K_ref,
    double dividend_yield)
{
    PriceTableWorkspace ws;

    // Compute knot vectors (clamped cubic B-spline)
    // We need to create temporary vectors for clamped_knots_cubic
    std::vector<double> m_grid(m_data, m_data + n_m);
    std::vector<double> tau_grid(tau_data, tau_data + n_tau);
    std::vector<double> sigma_grid(sigma_data, sigma_data + n_sigma);
    std::vector<double> r_grid(r_data, r_data + n_r);

    auto knots_m = clamped_knots_cubic(m_grid);
    auto knots_tau = clamped_knots_cubic(tau_grid);
    auto knots_sigma = clamped_knots_cubic(sigma_grid);
    auto knots_r = clamped_knots_cubic(r_grid);

    // Calculate total arena size
    size_t total_size = n_m + n_tau + n_sigma + n_r +
                       knots_m.size() + knots_tau.size() +
                       knots_sigma.size() + knots_r.size() +
                       n_coeffs;

    // Allocate with 64-byte alignment for AVX-512
    ws.arena_.resize(total_size + 8);  // +8 for alignment padding

    // Find 64-byte aligned start within arena
    auto arena_ptr = reinterpret_cast<std::uintptr_t>(ws.arena_.data());
    auto aligned_offset = (64 - (arena_ptr % 64)) % 64;
    double* aligned_start = ws.arena_.data() + aligned_offset / sizeof(double);

    // Copy data into arena (single copy from Arrow buffers)
    double* ptr = aligned_start;

    std::memcpy(ptr, m_data, n_m * sizeof(double));
    ws.log_moneyness_ = std::span<const double>(ptr, n_m);
    ptr += n_m;

    std::memcpy(ptr, tau_data, n_tau * sizeof(double));
    ws.maturity_ = std::span<const double>(ptr, n_tau);
    ptr += n_tau;

    std::memcpy(ptr, sigma_data, n_sigma * sizeof(double));
    ws.volatility_ = std::span<const double>(ptr, n_sigma);
    ptr += n_sigma;

    std::memcpy(ptr, r_data, n_r * sizeof(double));
    ws.rate_ = std::span<const double>(ptr, n_r);
    ptr += n_r;

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

    std::memcpy(ptr, coeff_data, n_coeffs * sizeof(double));
    ws.coefficients_ = std::span<const double>(ptr, n_coeffs);

    ws.K_ref_ = K_ref;
    ws.dividend_yield_ = dividend_yield;

    return ws;
}

std::expected<PriceTableWorkspace, std::string> PriceTableWorkspace::create(
    std::span<const double> log_m_grid,
    std::span<const double> tau_grid,
    std::span<const double> sigma_grid,
    std::span<const double> r_grid,
    std::span<const double> coefficients,
    double K_ref,
    double dividend_yield,
    double m_min,
    double m_max,
    SurfaceContent surface_content)
{
    // Validate inputs first
    auto validation = validate_inputs(log_m_grid, tau_grid, sigma_grid, r_grid, coefficients);
    if (!validation) {
        return std::unexpected(validation.error());
    }

    // Allocate and initialize workspace
    auto result = allocate_and_initialize(log_m_grid, tau_grid, sigma_grid, r_grid,
                                          coefficients, K_ref, dividend_yield);
    if (result) {
        result->m_min_ = m_min;
        result->m_max_ = m_max;
        result->surface_content_ = surface_content;
    }
    return result;
}

std::expected<void, std::string> PriceTableWorkspace::save(
    const std::string& /*filepath*/,
    const std::string& /*ticker*/,
    uint8_t /*option_type*/) const
{
    return std::unexpected(std::string("Persistence temporarily disabled (issue #373)"));
}

std::expected<PriceTableWorkspace, PriceTableWorkspace::LoadError>
PriceTableWorkspace::load(const std::string& /*filepath*/)
{
    return std::unexpected(LoadError::UNSUPPORTED_VERSION);
}

}  // namespace mango

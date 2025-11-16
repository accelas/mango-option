#include "src/option/price_table_workspace_pmr.hpp"
#include "src/support/crc64.hpp"
#include "src/bspline/bspline_utils.hpp"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <chrono>
#include <fstream>
#include <optional>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

namespace mango {

std::expected<void, std::string> PriceTableWorkspacePMR::validate_inputs(
    std::span<const double> m_grid,
    std::span<const double> tau_grid,
    std::span<const double> sigma_grid,
    std::span<const double> r_grid,
    std::span<const double> coefficients)
{
    // Validate grid sizes
    if (m_grid.size() < 4) {
        return std::unexpected("Moneyness grid must have >= 4 points");
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
        return std::unexpected("Moneyness grid must be sorted ascending");
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

std::expected<PriceTableWorkspacePMR, std::string> PriceTableWorkspacePMR::allocate_and_initialize(
    std::span<const double> m_grid,
    std::span<const double> tau_grid,
    std::span<const double> sigma_grid,
    std::span<const double> r_grid,
    std::span<const double> coefficients,
    double K_ref,
    double dividend_yield)
{
    PriceTableWorkspacePMR ws;

    // Store logical sizes
    ws.n_m_ = m_grid.size();
    ws.n_tau_ = tau_grid.size();
    ws.n_sigma_ = sigma_grid.size();
    ws.n_r_ = r_grid.size();
    ws.n_coeffs_ = coefficients.size();
    ws.K_ref_ = K_ref;
    ws.dividend_yield_ = dividend_yield;

    // Create PMR vectors with proper padding
    ws.moneyness_ = ws.create_pmr_vector_from_span(m_grid);
    ws.maturity_ = ws.create_pmr_vector_from_span(tau_grid);
    ws.volatility_ = ws.create_pmr_vector_from_span(sigma_grid);
    ws.rate_ = ws.create_pmr_vector_from_span(r_grid);

    // Compute knot vectors (clamped cubic B-spline)
    auto knots_m = clamped_knots_cubic(m_grid);
    auto knots_tau = clamped_knots_cubic(tau_grid);
    auto knots_sigma = clamped_knots_cubic(sigma_grid);
    auto knots_r = clamped_knots_cubic(r_grid);

    ws.knots_m_ = ws.create_pmr_vector_from_span(std::span<const double>(knots_m));
    ws.knots_tau_ = ws.create_pmr_vector_from_span(std::span<const double>(knots_tau));
    ws.knots_sigma_ = ws.create_pmr_vector_from_span(std::span<const double>(knots_sigma));
    ws.knots_r_ = ws.create_pmr_vector_from_span(std::span<const double>(knots_r));

    // Coefficients with padding
    ws.coefficients_ = ws.create_pmr_vector_from_span(coefficients);

    return ws;
}

std::expected<PriceTableWorkspacePMR, std::string> PriceTableWorkspacePMR::create(
    std::span<const double> m_grid,
    std::span<const double> tau_grid,
    std::span<const double> sigma_grid,
    std::span<const double> r_grid,
    std::span<const double> coefficients,
    double K_ref,
    double dividend_yield)
{
    // Validate inputs
    auto validation_result = validate_inputs(m_grid, tau_grid, sigma_grid, r_grid, coefficients);
    if (!validation_result) {
        return std::unexpected(validation_result.error());
    }

    // Allocate and initialize
    return allocate_and_initialize(m_grid, tau_grid, sigma_grid, r_grid, coefficients, K_ref, dividend_yield);
}

std::expected<void, std::string> PriceTableWorkspacePMR::save(const std::string& filepath,
                                                        const std::string& ticker,
                                                        uint8_t option_type) const
{
    // Implementation would be similar to original, but using PMR data
    // For now, return not implemented
    return std::unexpected("Save not yet implemented for PMR version");
}

std::expected<PriceTableWorkspacePMR, PriceTableWorkspacePMR::LoadError> PriceTableWorkspacePMR::load(const std::string& filepath)
{
    // Implementation would be similar to original, but creating PMR workspace
    // For now, return not implemented
    return std::unexpected(LoadError::FILE_NOT_FOUND);
}

} // namespace mango
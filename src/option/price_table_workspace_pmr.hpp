#pragma once

#include <expected>
#include "src/support/error_types.hpp"
#include "src/bspline/bspline_utils.hpp"
#include "src/option/option_workspace_base.hpp"
#include <span>
#include <string>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace mango {

/**
 * PMR-aware PriceTableWorkspace using unified memory resource
 *
 * Replaces std::vector with pmr::vector for better memory management
 * and zero-copy operations within the unified memory arena.
 */
class PriceTableWorkspacePMR : public OptionWorkspaceBase {
public:
    /// Factory method with validation (PMR-aware)
    static std::expected<PriceTableWorkspacePMR, std::string> create(
        std::span<const double> m_grid,
        std::span<const double> tau_grid,
        std::span<const double> sigma_grid,
        std::span<const double> r_grid,
        std::span<const double> coefficients,
        double K_ref,
        double dividend_yield);

    /// Factory method from std::vector (for backward compatibility)
    static std::expected<PriceTableWorkspacePMR, std::string> create(
        const std::vector<double>& m_grid,
        const std::vector<double>& tau_grid,
        const std::vector<double>& sigma_grid,
        const std::vector<double>& r_grid,
        const std::vector<double>& coefficients,
        double K_ref,
        double dividend_yield) {
        return create(std::span<const double>(m_grid),
                     std::span<const double>(tau_grid),
                     std::span<const double>(sigma_grid),
                     std::span<const double>(r_grid),
                     std::span<const double>(coefficients),
                     K_ref, dividend_yield);
    }

    /// Grid accessors (zero-copy spans into arena)
    std::span<const double> moneyness() const { return get_logical_span(moneyness_, n_m_); }
    std::span<const double> maturity() const { return get_logical_span(maturity_, n_tau_); }
    std::span<const double> volatility() const { return get_logical_span(volatility_, n_sigma_); }
    std::span<const double> rate() const { return get_logical_span(rate_, n_r_); }

    /// Knot vector accessors (precomputed clamped cubic knots)
    std::span<const double> knots_moneyness() const { return get_logical_span(knots_m_, n_m_ + 4); }
    std::span<const double> knots_maturity() const { return get_logical_span(knots_tau_, n_tau_ + 4); }
    std::span<const double> knots_volatility() const { return get_logical_span(knots_sigma_, n_sigma_ + 4); }
    std::span<const double> knots_rate() const { return get_logical_span(knots_r_, n_r_ + 4); }

    /// Coefficient accessor (4D tensor in row-major layout)
    std::span<const double> coefficients() const { return get_logical_span(coefficients_, n_coeffs_); }

    /// Metadata accessors
    double K_ref() const { return K_ref_; }
    double dividend_yield() const { return dividend_yield_; }

    /// Grid dimensions
    std::tuple<size_t, size_t, size_t, size_t> dimensions() const {
        return {n_m_, n_tau_, n_sigma_, n_r_};
    }

    /// Get total memory usage
    size_t memory_usage() const {
        return moneyness_.size() + maturity_.size() + volatility_.size() + rate_.size() +
               knots_m_.size() + knots_tau_.size() + knots_sigma_.size() + knots_r_.size() +
               coefficients_.size();
    }

    /// Save workspace to Apache Arrow IPC file (same interface)
    std::expected<void, std::string> save(const std::string& filepath,
                                     const std::string& ticker,
                                     uint8_t option_type) const;

    /// Load error codes (same as original)
    enum class LoadError {
        NOT_ARROW_FILE = 1,
        UNSUPPORTED_VERSION,
        INSUFFICIENT_GRID_POINTS,
        SIZE_MISMATCH,
        COEFFICIENT_SIZE_MISMATCH,
        GRID_NOT_SORTED,
        MMAP_FAILED,
        INVALID_ALIGNMENT,
        FILE_NOT_FOUND,
        SCHEMA_MISMATCH,
        ARROW_READ_ERROR,
        CORRUPTED_COEFFICIENTS,
        CORRUPTED_GRIDS,
        CORRUPTED_KNOTS,
    };

    /// Load workspace from Apache Arrow IPC file with zero-copy mmap
    static std::expected<PriceTableWorkspacePMR, LoadError> load(const std::string& filepath);

    /// Move-only semantics (no copies of large arena)
    PriceTableWorkspacePMR(const PriceTableWorkspacePMR&) = delete;
    PriceTableWorkspacePMR& operator=(const PriceTableWorkspacePMR&) = delete;
    PriceTableWorkspacePMR(PriceTableWorkspacePMR&&) noexcept = default;
    PriceTableWorkspacePMR& operator=(PriceTableWorkspacePMR&&) noexcept = default;

private:
    PriceTableWorkspacePMR() = default;

    /// Allocate aligned arena and set up spans using PMR
    static std::expected<PriceTableWorkspacePMR, std::string> allocate_and_initialize(
        std::span<const double> m_grid,
        std::span<const double> tau_grid,
        std::span<const double> sigma_grid,
        std::span<const double> r_grid,
        std::span<const double> coefficients,
        double K_ref,
        double dividend_yield);

    /// Validate inputs (same as original but with spans)
    static std::expected<void, std::string> validate_inputs(
        std::span<const double> m_grid,
        std::span<const double> tau_grid,
        std::span<const double> sigma_grid,
        std::span<const double> r_grid,
        std::span<const double> coefficients);

    // PMR vectors for all data (single memory resource)
    pmr_vector moneyness_;
    pmr_vector maturity_;
    pmr_vector volatility_;
    pmr_vector rate_;

    pmr_vector knots_m_;
    pmr_vector knots_tau_;
    pmr_vector knots_sigma_;
    pmr_vector knots_r_;

    pmr_vector coefficients_;

    // Logical sizes (excluding padding)
    size_t n_m_ = 0;
    size_t n_tau_ = 0;
    size_t n_sigma_ = 0;
    size_t n_r_ = 0;
    size_t n_coeffs_ = 0;

    // Scalar metadata
    double K_ref_ = 0.0;
    double dividend_yield_ = 0.0;
};

}  // namespace mango
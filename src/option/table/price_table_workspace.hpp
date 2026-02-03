// SPDX-License-Identifier: MIT
#pragma once

#include <expected>
#include "mango/support/error_types.hpp"
#include "mango/math/bspline_basis.hpp"
#include <experimental/mdspan>
#include <vector>
#include <span>
#include <string>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace mango {

/// Workspace holding all data for PriceTableSurface in single contiguous allocation
///
/// Enables zero-copy mmap loading from Arrow IPC files. All numeric data is
/// 64-byte aligned for AVX-512 SIMD operations.
///
/// Memory layout (all contiguous):
///   [log_moneyness grid][maturity grid][volatility grid][rate grid]
///   [knots_m][knots_tau][knots_sigma][knots_r]
///   [coefficients]
///   [metadata: K_ref, dividend_yield, m_min, m_max]
///
/// Note: Axis 0 stores log-moneyness (ln(S/K)) for better B-spline interpolation.
/// The original moneyness bounds are stored in m_min, m_max for user-facing APIs.
///
/// Example:
///   auto ws = PriceTableWorkspace::create(log_m, tau, sigma, r, coeffs, K_ref, q, m_min, m_max);
///   BSpline4D spline(ws.value());
class PriceTableWorkspace {
public:
    /// Factory method with validation
    ///
    /// @param log_m_grid Log-moneyness grid (ln(S/K), sorted ascending, >= 4 points)
    /// @param tau_grid Maturity grid (years, sorted ascending, >= 4 points)
    /// @param sigma_grid Volatility grid (sorted ascending, >= 4 points)
    /// @param r_grid Rate grid (sorted ascending, >= 4 points)
    /// @param coefficients B-spline coefficients (size = n_m * n_tau * n_sigma * n_r)
    /// @param K_ref Reference strike price
    /// @param dividend_yield Continuous dividend yield
    /// @param m_min Minimum moneyness (S/K) for user-facing bounds
    /// @param m_max Maximum moneyness (S/K) for user-facing bounds
    /// @return Expected workspace or error message
    static std::expected<PriceTableWorkspace, std::string> create(
        std::span<const double> log_m_grid,
        std::span<const double> tau_grid,
        std::span<const double> sigma_grid,
        std::span<const double> r_grid,
        std::span<const double> coefficients,
        double K_ref,
        double dividend_yield,
        double m_min,
        double m_max,
        uint8_t surface_content = 0);

    /// Grid accessors (zero-copy spans into arena)
    /// Note: log_moneyness() returns ln(S/K), use m_min()/m_max() for original bounds
    std::span<const double> log_moneyness() const { return log_moneyness_; }
    std::span<const double> maturity() const { return maturity_; }
    std::span<const double> volatility() const { return volatility_; }
    std::span<const double> rate() const { return rate_; }

    /// Original moneyness bounds (for user-facing APIs)
    double m_min() const { return m_min_; }
    double m_max() const { return m_max_; }

    /// Knot vector accessors (precomputed clamped cubic knots)
    std::span<const double> knots_log_moneyness() const { return knots_m_; }
    std::span<const double> knots_maturity() const { return knots_tau_; }
    std::span<const double> knots_volatility() const { return knots_sigma_; }
    std::span<const double> knots_rate() const { return knots_r_; }

    /// Coefficient accessor (4D tensor in row-major layout)
    std::span<const double> coefficients() const { return coefficients_; }

    /// Type-safe 4D coefficient accessor via mdspan
    ///
    /// Provides multi-dimensional indexing: coeffs_view[m_idx, tau_idx, sigma_idx, r_idx]
    /// Layout: [moneyness, maturity, volatility, rate] in row-major order
    ///
    /// @return mdspan view of 4D coefficient tensor
    [[nodiscard]] auto coefficients_view() const {
        using std::experimental::mdspan;
        using std::experimental::dextents;
        const auto [Nm, Nt, Nv, Nr] = dimensions();
        return mdspan<const double, dextents<size_t, 4>>(coefficients_.data(), Nm, Nt, Nv, Nr);
    }

    /// Metadata accessors
    double K_ref() const { return K_ref_; }
    double dividend_yield() const { return dividend_yield_; }
    uint8_t surface_content() const { return surface_content_; }

    /// Grid dimensions
    std::tuple<size_t, size_t, size_t, size_t> dimensions() const {
        return {log_moneyness_.size(), maturity_.size(),
                volatility_.size(), rate_.size()};
    }

    /// Save workspace to Apache Arrow IPC file
    ///
    /// @param filepath Output file path
    /// @param ticker Underlying symbol (e.g., "SPY")
    /// @param option_type 0=PUT, 1=CALL
    /// @return Expected void or error message
    std::expected<void, std::string> save(const std::string& filepath,
                                     const std::string& ticker,
                                     uint8_t option_type) const;

    /// Load error codes
    enum class LoadError {
        NOT_ARROW_FILE,              // Missing "ARROW1" magic
        UNSUPPORTED_VERSION,         // format_version != 2
        INSUFFICIENT_GRID_POINTS,    // n < 4 for any axis
        SIZE_MISMATCH,               // Array length doesn't match metadata
        COEFFICIENT_SIZE_MISMATCH,   // coeffs.size() != n_m×n_tau×n_sigma×n_r
        GRID_NOT_SORTED,             // Monotonicity violation
        MMAP_FAILED,                 // OS mmap error
        INVALID_ALIGNMENT,           // Buffer not 64-byte aligned
        FILE_NOT_FOUND,              // File doesn't exist
        SCHEMA_MISMATCH,             // Missing required fields
        ARROW_READ_ERROR,            // Arrow library error
        CORRUPTED_COEFFICIENTS,      // CRC64 checksum mismatch for coefficients
        CORRUPTED_GRIDS,             // CRC64 checksum mismatch for grids
        CORRUPTED_KNOTS,             // Knot values don't match recomputed knots
    };

    /// Load workspace from Apache Arrow IPC file with zero-copy mmap
    ///
    /// @param filepath Input file path
    /// @return Expected workspace or error code
    static std::expected<PriceTableWorkspace, LoadError> load(const std::string& filepath);

    /// Move-only semantics (no copies of large arena)
    PriceTableWorkspace(const PriceTableWorkspace&) = delete;
    PriceTableWorkspace& operator=(const PriceTableWorkspace&) = delete;
    PriceTableWorkspace(PriceTableWorkspace&&) noexcept = default;
    PriceTableWorkspace& operator=(PriceTableWorkspace&&) noexcept = default;

private:
    PriceTableWorkspace() = default;

    /// Allocate aligned arena and set up spans
    static std::expected<PriceTableWorkspace, std::string> allocate_and_initialize(
        std::span<const double> m_grid,
        std::span<const double> tau_grid,
        std::span<const double> sigma_grid,
        std::span<const double> r_grid,
        std::span<const double> coefficients,
        double K_ref,
        double dividend_yield);

    /// Friend function for zero-copy loading from raw buffers
    friend std::expected<PriceTableWorkspace, std::string> allocate_and_initialize_from_buffers(
        const double* m_data, size_t n_m,
        const double* tau_data, size_t n_tau,
        const double* sigma_data, size_t n_sigma,
        const double* r_data, size_t n_r,
        const double* coeff_data, size_t n_coeffs,
        double K_ref,
        double dividend_yield);

    /// Validate grids before allocation
    static std::expected<void, std::string> validate_inputs(
        std::span<const double> m_grid,
        std::span<const double> tau_grid,
        std::span<const double> sigma_grid,
        std::span<const double> r_grid,
        std::span<const double> coefficients);

    // Single contiguous allocation (64-byte aligned)
    std::vector<double> arena_;

    // Views into arena (no ownership)
    std::span<const double> log_moneyness_;
    std::span<const double> maturity_;
    std::span<const double> volatility_;
    std::span<const double> rate_;

    std::span<const double> knots_m_;
    std::span<const double> knots_tau_;
    std::span<const double> knots_sigma_;
    std::span<const double> knots_r_;

    std::span<const double> coefficients_;

    // Scalar metadata
    double K_ref_ = 0.0;
    double dividend_yield_ = 0.0;
    double m_min_ = 0.0;  // Original moneyness bounds
    double m_max_ = 0.0;
    uint8_t surface_content_ = 0;  // 0 = RawPrice (default)
};

}  // namespace mango

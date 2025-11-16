#pragma once

#include "src/bspline/bspline_fitter_4d.hpp"
#include "src/option/option_workspace_base.hpp"
#include <span>

namespace mango {

/**
 * PMR-aware workspace for B-spline 4D fitting to reduce allocations
 *
 * Replaces std::vector with pmr::vector for better memory management
 * within the unified memory arena.
 */
struct BSplineFitter4DWorkspacePMR : public OptionWorkspaceBase {
    pmr_vector slice_buffer;     ///< Reusable buffer for slice extraction
    pmr_vector coeffs_buffer;    ///< Reusable buffer for fitted coefficients

    /// Create workspace sized for maximum axis dimension
    explicit BSplineFitter4DWorkspacePMR(size_t max_n, OptionWorkspaceBase* parent = nullptr)
        : OptionWorkspaceBase(parent ? parent->resource_.bytes_allocated() : 1024 * 1024)
        , slice_buffer(create_pmr_vector(max_n))
        , coeffs_buffer(create_pmr_vector(max_n))
    {
        // If we have a parent workspace, use its memory resource
        if (parent) {
            // Re-create buffers using parent's resource
            slice_buffer = parent->create_pmr_vector(max_n);
            coeffs_buffer = parent->create_pmr_vector(max_n);
        }
    }

    /// Get slice buffer as span (subspan for smaller axes)
    std::span<double> get_slice_buffer(size_t n) {
        assert(n <= slice_buffer.size());
        return std::span{slice_buffer.data(), n};
    }

    /// Get coefficients buffer as span
    std::span<double> get_coeffs_buffer(size_t n) {
        assert(n <= coeffs_buffer.size());
        return std::span{coeffs_buffer.data(), n};
    }

    /// Get const slice buffer as span
    std::span<const double> get_slice_buffer(size_t n) const {
        assert(n <= slice_buffer.size());
        return std::span{slice_buffer.data(), n};
    }

    /// Get const coefficients buffer as span
    std::span<const double> get_coeffs_buffer(size_t n) const {
        assert(n <= coeffs_buffer.size());
        return std::span{coeffs_buffer.data(), n};
    }
};

/**
 * PMR-aware BandedMatrixStorage for 4-diagonal matrices
 */
class BandedMatrixStoragePMR {
public:
    /// Construct banded storage for n×n matrix with bandwidth 4
    explicit BandedMatrixStoragePMR(size_t n, OptionWorkspaceBase* workspace)
        : n_(n)
        , band_values_(workspace->create_pmr_vector(4 * n, false))  // Don't pad matrix data
        , col_start_(workspace->create_pmr_vector(n, false))
    {
        // Initialize to zero
        std::fill(band_values_.begin(), band_values_.end(), 0.0);
        std::fill(col_start_.begin(), col_start_.end(), 0);
    }

    /// Get reference to band entry A[row, col]
    double& operator()(size_t row, size_t col) {
        assert(row < n_);
        assert(col >= col_start_[row] && col < col_start_[row] + 4);
        size_t k = col - col_start_[row];
        return band_values_[row * 4 + k];
    }

    /// Get const reference to band entry
    double operator()(size_t row, size_t col) const {
        assert(row < n_);
        assert(col >= col_start_[row] && col < col_start_[row] + 4);
        size_t k = col - col_start_[row];
        return band_values_[row * 4 + k];
    }

    /// Get starting column index for row
    size_t col_start(size_t row) const {
        assert(row < n_);
        return col_start_[row];
    }

    /// Set starting column index for row
    void set_col_start(size_t row, size_t col) {
        assert(row < n_);
        col_start_[row] = col;
    }

    /// Get matrix size
    size_t size() const { return n_; }

    /// Get raw band values for LAPACK (if needed)
    std::span<const double> band_values() const { return band_values_; }
    std::span<double> band_values() { return band_values_; }

    /// Get raw column start indices
    std::span<const size_t> col_start_data() const { return col_start_; }
    std::span<size_t> col_start_data() { return col_start_; }

private:
    size_t n_;
    OptionWorkspaceBase::pmr_vector band_values_;
    OptionWorkspaceBase::pmr_size_vector col_start_;
};

/**
 * PMR-aware BSplineCollocation1D
 */
class BSplineCollocation1DPMR {
public:
    BSplineCollocation1DPMR(std::span<const double> grid,
                           std::span<const double> knots,
                           OptionWorkspaceBase* workspace)
        : grid_(workspace->create_pmr_vector_from_span(grid))
        , knots_(workspace->create_pmr_vector_from_span(knots))
        , band_values_(workspace->create_pmr_vector(4 * grid.size(), false))
        , band_col_start_(workspace->create_pmr_vector(grid.size(), false))
        , n_(grid.size())
        , workspace_(workspace)
    {
        // Initialize band storage
        std::fill(band_values_.begin(), band_values_.end(), 0.0);
        std::fill(band_col_start_.begin(), band_col_start_.end(), 0);
    }

    /// Get grid points
    std::span<const double> grid() const {
        return std::span<const double>(grid_.data(), n_);
    }

    /// Get knot vector
    std::span<const double> knots() const {
        return std::span<const double>(knots_.data(), knots_.size());
    }

    /// Get banded matrix storage
    BandedMatrixStoragePMR create_matrix_storage() {
        return BandedMatrixStoragePMR(n_, workspace_);
    }

    /// Access to workspace for derived classes
    OptionWorkspaceBase* workspace() { return workspace_; }
    const OptionWorkspaceBase* workspace() const { return workspace_; }

private:
    OptionWorkspaceBase::pmr_vector grid_;
    OptionWorkspaceBase::pmr_vector knots_;
    OptionWorkspaceBase::pmr_vector band_values_;
    OptionWorkspaceBase::pmr_size_vector band_col_start_;
    size_t n_;
    OptionWorkspaceBase* workspace_;
};

} // namespace mango
#pragma once

#include <span>
#include <cstddef>

namespace mango {

/// View into tridiagonal Jacobian storage
///
/// Provides safe access to lower, diagonal, and upper bands of a
/// tridiagonal matrix stored in three separate arrays.
///
/// Memory layout:
/// - lower[i] represents J[i+1,i] (i = 0..n-2)
/// - diag[i] represents J[i,i] (i = 0..n-1)
/// - upper[i] represents J[i,i+1] (i = 0..n-1)
class JacobianView {
public:
    JacobianView(std::span<double> lower,
                 std::span<double> diag,
                 std::span<double> upper)
        : lower_(lower)
        , diag_(diag)
        , upper_(upper)
    {
        // Validate dimensions
        // lower has n-1 elements (indices 0..n-2)
        // diag has n elements (indices 0..n-1)
        // upper has n elements (indices 0..n-1)
        // Note: upper[n-1] is unused but present for alignment
    }

    /// Access lower diagonal: J[i+1,i]
    std::span<double> lower() { return lower_; }
    std::span<const double> lower() const { return lower_; }

    /// Access main diagonal: J[i,i]
    std::span<double> diag() { return diag_; }
    std::span<const double> diag() const { return diag_; }

    /// Access upper diagonal: J[i,i+1]
    std::span<double> upper() { return upper_; }
    std::span<const double> upper() const { return upper_; }

    /// Grid size
    size_t size() const { return diag_.size(); }

private:
    std::span<double> lower_;
    std::span<double> diag_;
    std::span<double> upper_;
};

} // namespace mango

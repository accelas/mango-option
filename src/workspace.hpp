#pragma once

#include <vector>
#include <span>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace mango {

/// Workspace storage for PDE solver arrays
///
/// **CPU-only implementation** - SYCL GPU specialization deferred to v2.1.
///
/// Manages all solver state in a single contiguous buffer for cache efficiency.
/// Arrays: u_current, u_next, u_stage, rhs, Lu, psi (6n doubles total).
///
/// **64-byte alignment** - Backing storage is allocated via std::aligned_alloc
/// so AVX/AVX-512 kernels can safely use aligned loads/stores.
///
/// **Pre-computed dx array** - Grid spacing computed once during construction
/// to avoid redundant S[i+1] - S[i] calculations in stencil operations.
///
/// Future GPU version (v2.1) will use SYCL unified shared memory (USM)
/// with explicit device allocation and host-device synchronization.
class WorkspaceStorage {
public:
    /// Construct workspace for n grid points
    ///
    /// @param n Number of grid points
    /// @param grid Grid coordinates for pre-computing dx
    ///
    /// Allocates 6n doubles (u_current, u_next, u_stage, rhs, Lu, psi)
    /// plus (n-1) doubles for pre-computed dx array.
    /// Memory is 64-byte aligned for AVX-512 SIMD operations.
    explicit WorkspaceStorage(size_t n, std::span<const double> grid)
        : n_(n)
        , buffer_(allocate_aligned(6 * n))
        , dx_(n - 1)
    {
        // Pre-compute grid spacing once during initialization
        for (size_t i = 0; i < n - 1; ++i) {
            dx_[i] = grid[i + 1] - grid[i];
        }

        assign_spans();
    }

    ~WorkspaceStorage() {
        std::free(buffer_);
    }

    WorkspaceStorage(const WorkspaceStorage&) = delete;
    WorkspaceStorage& operator=(const WorkspaceStorage&) = delete;

    WorkspaceStorage(WorkspaceStorage&& other) noexcept
        : n_(other.n_)
        , buffer_(other.buffer_)
        , dx_(std::move(other.dx_))
    {
        other.buffer_ = nullptr;
        assign_spans();
    }

    WorkspaceStorage& operator=(WorkspaceStorage&& other) noexcept {
        if (this == &other) return *this;
        std::free(buffer_);
        n_ = other.n_;
        buffer_ = other.buffer_;
        dx_ = std::move(other.dx_);
        other.buffer_ = nullptr;
        assign_spans();
        return *this;
    }

    // Access to arrays
    std::span<double> u_current() { return u_current_; }
    std::span<const double> u_current() const { return u_current_; }

    std::span<double> u_next() { return u_next_; }
    std::span<const double> u_next() const { return u_next_; }

    std::span<double> u_stage() { return u_stage_; }
    std::span<const double> u_stage() const { return u_stage_; }

    std::span<double> rhs() { return rhs_; }
    std::span<const double> rhs() const { return rhs_; }

    std::span<double> lu() { return lu_; }
    std::span<const double> lu() const { return lu_; }

    std::span<double> psi_buffer() { return psi_; }
    std::span<const double> psi_buffer() const { return psi_; }

    // Access to pre-computed dx
    std::span<const double> dx() const { return dx_; }

private:
    static double* allocate_aligned(size_t count) {
        const std::size_t alignment = 64;
        std::size_t bytes = count * sizeof(double);
        std::size_t padded = ((bytes + alignment - 1) / alignment) * alignment;
        void* ptr = std::aligned_alloc(alignment, padded);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<double*>(ptr);
    }

    void assign_spans() {
        if (!buffer_) {
            u_current_ = u_next_ = u_stage_ = rhs_ = lu_ = psi_ = {};
            return;
        }
        size_t offset = 0;
        u_current_ = std::span{buffer_ + offset, n_}; offset += n_;
        u_next_    = std::span{buffer_ + offset, n_}; offset += n_;
        u_stage_   = std::span{buffer_ + offset, n_}; offset += n_;
        rhs_       = std::span{buffer_ + offset, n_}; offset += n_;
        lu_        = std::span{buffer_ + offset, n_}; offset += n_;
        psi_       = std::span{buffer_ + offset, n_};
    }

    size_t n_{0};
    double* buffer_{nullptr};
    std::vector<double> dx_;

    // Spans into buffer_
    std::span<double> u_current_;
    std::span<double> u_next_;
    std::span<double> u_stage_;
    std::span<double> rhs_;
    std::span<double> lu_;
    std::span<double> psi_;
};

} // namespace mango

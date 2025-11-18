#pragma once

#include "src/pde/core/grid.hpp"
#include <memory_resource>
#include <span>
#include <expected>
#include <string>
#include <memory>
#include <algorithm>

namespace mango {

/**
 * PDEWorkspace: Unified memory workspace for PDE solver
 *
 * Uses PMR vectors for all storage. All accessors return SIMD-padded spans.
 * Caller extracts logical size with .subspan(0, logical_size()) when needed.
 */
class PDEWorkspace {
public:
    static constexpr size_t SIMD_WIDTH = 8;

    static constexpr size_t pad_to_simd(size_t n) {
        return ((n + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
    }

    static std::expected<std::shared_ptr<PDEWorkspace>, std::string>
    create(const GridSpec<double>& grid_spec,
           std::pmr::memory_resource* resource) {
        if (!resource) {
            return std::unexpected("Memory resource cannot be null");
        }

        auto grid_buffer = grid_spec.generate();
        size_t n = grid_buffer.size();

        if (n == 0) {
            return std::unexpected("Grid size must be positive");
        }

        return std::shared_ptr<PDEWorkspace>(
            new PDEWorkspace(n, grid_buffer.span(), resource));
    }

    // Accessors - all return SIMD-padded spans
    std::span<double> u_current() { return {u_current_.data(), padded_n_}; }
    std::span<const double> u_current() const { return {u_current_.data(), padded_n_}; }

    std::span<double> u_next() { return {u_next_.data(), padded_n_}; }
    std::span<const double> u_next() const { return {u_next_.data(), padded_n_}; }

    std::span<double> u_stage() { return {u_stage_.data(), padded_n_}; }
    std::span<const double> u_stage() const { return {u_stage_.data(), padded_n_}; }

    std::span<double> rhs() { return {rhs_.data(), padded_n_}; }
    std::span<const double> rhs() const { return {rhs_.data(), padded_n_}; }

    std::span<double> lu() { return {lu_.data(), padded_n_}; }
    std::span<const double> lu() const { return {lu_.data(), padded_n_}; }

    std::span<double> psi() { return {psi_.data(), padded_n_}; }
    std::span<const double> psi() const { return {psi_.data(), padded_n_}; }

    std::span<const double> grid() const { return {grid_.data(), padded_n_}; }

    std::span<const double> dx() const { return {dx_.data(), pad_to_simd(n_ - 1)}; }

    size_t logical_size() const { return n_; }
    size_t padded_size() const { return padded_n_; }

private:
    PDEWorkspace(size_t n, std::span<const double> grid_data,
                 std::pmr::memory_resource* mr)
        : n_(n)
        , padded_n_(pad_to_simd(n))
        , resource_(mr)
        , grid_(padded_n_, 0.0, mr)
        , u_current_(padded_n_, 0.0, mr)
        , u_next_(padded_n_, 0.0, mr)
        , u_stage_(padded_n_, 0.0, mr)
        , rhs_(padded_n_, 0.0, mr)
        , lu_(padded_n_, 0.0, mr)
        , psi_(padded_n_, 0.0, mr)
        , dx_(pad_to_simd(n - 1), 0.0, mr)
    {
        // Copy grid data
        std::copy(grid_data.begin(), grid_data.end(), grid_.begin());

        // Precompute dx
        for (size_t i = 0; i < n_ - 1; ++i) {
            dx_[i] = grid_[i + 1] - grid_[i];
        }
    }

    size_t n_;
    size_t padded_n_;
    std::pmr::memory_resource* resource_;

    std::pmr::vector<double> grid_;
    std::pmr::vector<double> u_current_;
    std::pmr::vector<double> u_next_;
    std::pmr::vector<double> u_stage_;
    std::pmr::vector<double> rhs_;
    std::pmr::vector<double> lu_;
    std::pmr::vector<double> psi_;
    std::pmr::vector<double> dx_;
};

}  // namespace mango

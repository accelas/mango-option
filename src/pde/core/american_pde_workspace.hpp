// src/pde/core/american_pde_workspace.hpp
#pragma once

#include "src/pde/core/pde_workspace.hpp"
#include "src/support/lifetime.hpp"
#include <span>
#include <expected>
#include <string>
#include <cstddef>

namespace mango {

/// Workspace for American PDE solver - slices byte buffer into PDEWorkspace
///
/// This is a thin wrapper that:
/// 1. Takes a byte buffer from ThreadWorkspaceBuffer
/// 2. Starts lifetime of double array using start_array_lifetime
/// 3. Creates PDEWorkspace over the typed buffer
///
/// Design:
/// - Accepts std::span<std::byte> from ThreadWorkspaceBuffer
/// - Uses start_array_lifetime<double> for proper object lifetime management
/// - Delegates all operations to inner PDEWorkspace
/// - 64-byte alignment matches ThreadWorkspaceBuffer::ALIGNMENT
///
/// Example:
///   ThreadWorkspaceBuffer buffer(AmericanPDEWorkspace::required_bytes(n));
///   auto ws = AmericanPDEWorkspace::from_bytes(buffer.bytes(), n).value();
///   // Use ws.dx(), ws.u_stage(), etc.
///
struct AmericanPDEWorkspace {
    static constexpr size_t ALIGNMENT = 64;

    /// Calculate required buffer size in BYTES
    ///
    /// This accounts for:
    /// - PDEWorkspace double array requirements
    /// - 64-byte alignment padding
    ///
    /// @param n Grid size
    /// @return Required buffer size in bytes
    static size_t required_bytes(size_t n) {
        size_t doubles = PDEWorkspace::required_size(n);
        // Add padding for 64-byte alignment
        return align_up(doubles * sizeof(double), ALIGNMENT);
    }

    /// Create from byte buffer
    ///
    /// Starts lifetime of double array and constructs PDEWorkspace.
    ///
    /// @param buffer Byte buffer (must be at least required_bytes(n))
    /// @param n Grid size (must be at least 2)
    /// @return AmericanPDEWorkspace on success, error message on failure
    static std::expected<AmericanPDEWorkspace, std::string>
    from_bytes(std::span<std::byte> buffer, size_t n) {
        if (n < 2) {
            return std::unexpected("Grid size must be at least 2");
        }

        size_t required = required_bytes(n);
        if (buffer.size() < required) {
            return std::unexpected("Buffer too small for AmericanPDEWorkspace");
        }

        // Start lifetime of double array
        size_t double_count = PDEWorkspace::required_size(n);
        double* typed = start_array_lifetime<double>(buffer.data(), double_count);

        // Create PDEWorkspace from typed buffer
        auto ws_result = PDEWorkspace::from_buffer(
            std::span<double>(typed, double_count), n);

        if (!ws_result.has_value()) {
            return std::unexpected(ws_result.error());
        }

        return AmericanPDEWorkspace{std::move(ws_result.value())};
    }

    /// Get inner PDEWorkspace reference (used by AmericanOptionSolver)
    PDEWorkspace& workspace() { return inner_; }
    const PDEWorkspace& workspace() const { return inner_; }

    size_t size() const { return inner_.size(); }

private:
    explicit AmericanPDEWorkspace(PDEWorkspace ws) : inner_(std::move(ws)) {}
    PDEWorkspace inner_;
};

} // namespace mango

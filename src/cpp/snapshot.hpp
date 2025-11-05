#pragma once

#include <span>
#include <cstddef>

namespace mango {

/// Snapshot of PDE solution at a specific time
///
/// Contains V(x,t) and derivatives for a single time point.
/// Passed to SnapshotCollector callbacks during PDE solve.
///
/// Thread-Safety: Read-only after construction (safe for collectors)
struct Snapshot {
    // Time and indexing
    double time;                              ///< Solution time
    size_t user_index;                        ///< User-provided index for matching

    // Spatial domain (for interpolation to different grids)
    std::span<const double> spatial_grid;     ///< PDE grid x-coordinates
    std::span<const double> dx;               ///< Grid spacing (size = n-1)

    // Solution data (all size = n)
    std::span<const double> solution;         ///< V(x,t)
    std::span<const double> spatial_operator; ///< L(V) from PDE
    std::span<const double> first_derivative; ///< ∂V/∂x
    std::span<const double> second_derivative;///< ∂²V/∂x²

    // Problem context (optional, for collector use)
    const void* problem_params = nullptr;     ///< User-defined context
};

/// Collector callback interface
///
/// Called by PDESolver when snapshot times are reached.
/// Implementations must be thread-safe if used with parallel precompute.
class SnapshotCollector {
public:
    virtual ~SnapshotCollector() = default;

    /// Collect snapshot data
    ///
    /// @param snapshot Read-only snapshot data
    virtual void collect(const Snapshot& snapshot) = 0;
};

}  // namespace mango

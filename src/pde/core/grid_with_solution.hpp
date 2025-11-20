#ifndef MANGO_PDE_CORE_GRID_WITH_SOLUTION_HPP
#define MANGO_PDE_CORE_GRID_WITH_SOLUTION_HPP

#include <memory>
#include <span>
#include <vector>
#include <expected>
#include <string>
#include "src/pde/core/grid.hpp"
#include "src/pde/core/time_domain.hpp"

namespace mango {

/// Grid with persistent solution storage and metadata
/// Outlives PDESolver, passed via shared_ptr for lifetime management
template<typename T = double>
class GridWithSolution {
public:
    /// Create grid with solution storage
    /// @param grid_spec Grid specification (uniform, sinh, etc)
    /// @param time_domain Time domain information
    /// @return Grid instance or error message
    static std::expected<std::shared_ptr<GridWithSolution<T>>, std::string>
    create(const GridSpec<T>& grid_spec, const TimeDomain& time_domain) {
        // Generate grid buffer
        auto grid_buffer = grid_spec.generate();
        auto grid_view = grid_buffer.view();

        // Create GridSpacing
        auto spacing = GridSpacing<T>(grid_view);

        size_t n = grid_view.size();

        // Allocate solution storage (2 × n for current + previous)
        std::vector<T> solution(2 * n);

        // Create instance (private constructor, so use new)
        auto grid = std::shared_ptr<GridWithSolution<T>>(
            new GridWithSolution<T>(
                std::move(grid_buffer),
                std::move(spacing),
                time_domain,
                std::move(solution)
            )
        );

        return grid;
    }

    // Accessors

    /// Spatial grid points (read-only)
    std::span<const T> x() const {
        return grid_buffer_.span();
    }

    /// Grid spacing object (reference, safe since Grid outlives solver)
    const GridSpacing<T>& spacing() const {
        return spacing_;
    }

    /// Time domain information
    const TimeDomain& time() const {
        return time_;
    }

    /// Current solution (last time step)
    std::span<T> solution() {
        return std::span{solution_.data(), n_space()};
    }

    std::span<const T> solution() const {
        return std::span{solution_.data(), n_space()};
    }

    /// Previous solution (second-to-last time step)
    std::span<T> solution_prev() {
        return std::span{solution_.data() + n_space(), n_space()};
    }

    std::span<const T> solution_prev() const {
        return std::span{solution_.data() + n_space(), n_space()};
    }

    /// Number of spatial points
    size_t n_space() const {
        return grid_buffer_.size();
    }

    /// Time step size
    double dt() const {
        return time_.dt();
    }

private:
    // Private constructor (use factory method)
    GridWithSolution(GridBuffer<T>&& grid_buffer,
                     GridSpacing<T>&& spacing,
                     const TimeDomain& time,
                     std::vector<T>&& solution)
        : grid_buffer_(std::move(grid_buffer))
        , spacing_(std::move(spacing))
        , time_(time)
        , solution_(std::move(solution))
    {}

    GridBuffer<T> grid_buffer_;     // Spatial grid points
    GridSpacing<T> spacing_;        // Grid spacing (uniform or non-uniform)
    TimeDomain time_;               // Time domain metadata
    std::vector<T> solution_;       // [u_current | u_prev] (2 × n_space)
};

}  // namespace mango

#endif  // MANGO_PDE_CORE_GRID_WITH_SOLUTION_HPP

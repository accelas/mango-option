// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <cstddef>
#include <cmath>
#include <algorithm>

namespace mango {

/// Time domain configuration for PDE solver
///
/// Defines the time interval [t_start, t_end] and time step dt.
/// Computes the number of time steps needed to reach t_end.
class TimeDomain {
public:
    /// Construct time domain from time step size
    ///
    /// @param t_start Initial time
    /// @param t_end Final time
    /// @param dt Time step size
    TimeDomain(double t_start, double t_end, double dt)
        : t_start_(t_start)
        , t_end_(t_end)
        , dt_(dt)
        , n_steps_(static_cast<size_t>(std::ceil((t_end - t_start) / dt)))
    {}

    /// Construct time domain from number of steps (avoids floating-point rounding)
    ///
    /// Preferred when n_steps is known exactly to avoid ceil() rounding issues.
    ///
    /// @param t_start Initial time
    /// @param t_end Final time
    /// @param n_steps Number of time steps
    static TimeDomain from_n_steps(double t_start, double t_end, size_t n_steps) {
        TimeDomain td;
        td.t_start_ = t_start;
        td.t_end_ = t_end;
        td.n_steps_ = n_steps;
        td.dt_ = (t_end - t_start) / static_cast<double>(n_steps);
        return td;
    }

    /// Construct a time domain that includes mandatory time points
    ///
    /// Subdivides each interval between mandatory points so that no step
    /// exceeds the requested dt.  When mandatory is empty, falls back to
    /// a uniform grid identical to from_n_steps().
    ///
    /// @param t_start Initial time
    /// @param t_end Final time
    /// @param dt Maximum time step size
    /// @param mandatory Points that must appear in the grid
    static TimeDomain with_mandatory_points(
        double t_start, double t_end, double dt,
        std::vector<double> mandatory)
    {
        // Filter mandatory points to strict interior of [t_start, t_end]
        std::vector<double> breaks;
        for (double t : mandatory) {
            if (t > t_start + 1e-14 && t < t_end - 1e-14) {
                breaks.push_back(t);
            }
        }
        std::sort(breaks.begin(), breaks.end());

        // Remove duplicates
        breaks.erase(std::unique(breaks.begin(), breaks.end(),
            [](double a, double b) { return std::abs(a - b) < 1e-14; }),
            breaks.end());

        if (breaks.empty()) {
            size_t n = static_cast<size_t>(std::ceil((t_end - t_start) / dt));
            return from_n_steps(t_start, t_end, n);
        }

        // Build segment boundaries
        std::vector<double> boundaries;
        boundaries.push_back(t_start);
        for (double b : breaks) boundaries.push_back(b);
        boundaries.push_back(t_end);

        // Subdivide each segment so no step exceeds dt
        std::vector<double> points;
        points.push_back(t_start);

        for (size_t seg = 0; seg + 1 < boundaries.size(); ++seg) {
            double seg_start = boundaries[seg];
            double seg_end = boundaries[seg + 1];
            double seg_len = seg_end - seg_start;
            size_t n_sub = std::max(size_t{1},
                static_cast<size_t>(std::ceil(seg_len / dt)));
            double sub_dt = seg_len / static_cast<double>(n_sub);
            for (size_t j = 1; j <= n_sub; ++j) {
                points.push_back(seg_start + j * sub_dt);
            }
        }

        TimeDomain td;
        td.t_start_ = t_start;
        td.t_end_ = t_end;
        td.n_steps_ = points.size() - 1;
        td.dt_ = (t_end - t_start) / static_cast<double>(td.n_steps_);
        td.time_points_ = std::move(points);
        return td;
    }

private:
    // Private default constructor for from_n_steps factory
    TimeDomain() = default;

public:

    double t_start() const { return t_start_; }
    double t_end() const { return t_end_; }
    double dt() const { return dt_; }
    size_t n_steps() const { return n_steps_; }

    /// Whether this time domain has explicit (non-uniform) time points
    bool has_time_points() const { return !time_points_.empty(); }

    /// Read-only reference to stored time points (empty for uniform grids)
    const std::vector<double>& time_points_ref() const { return time_points_; }

    /// Time step size at a given step index
    ///
    /// For non-uniform grids returns the actual step size; for uniform
    /// grids returns the constant dt.
    double dt_at(size_t step) const {
        if (!time_points_.empty() && step < time_points_.size() - 1) {
            return time_points_[step + 1] - time_points_[step];
        }
        return dt_;
    }

    /// Generate vector of time points from t_start to t_end
    std::vector<double> time_points() const {
        if (!time_points_.empty()) {
            return time_points_;
        }

        std::vector<double> times;
        times.reserve(n_steps_ + 1);

        for (size_t i = 0; i <= n_steps_; ++i) {
            times.push_back(t_start_ + i * dt_);
        }

        return times;
    }

private:
    double t_start_;
    double t_end_;
    double dt_;
    size_t n_steps_;
    std::vector<double> time_points_;  ///< Explicit points for non-uniform grids (empty = uniform)
};

}  // namespace mango

// SPDX-License-Identifier: MIT
#pragma once

#include <vector>
#include <cstddef>
#include <cmath>

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

private:
    // Private default constructor for from_n_steps factory
    TimeDomain() = default;

public:

    double t_start() const { return t_start_; }
    double t_end() const { return t_end_; }
    double dt() const { return dt_; }
    size_t n_steps() const { return n_steps_; }

    /// Generate vector of time points from t_start to t_end
    std::vector<double> time_points() const {
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
};

}  // namespace mango

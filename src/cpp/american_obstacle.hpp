#ifndef MANGO_AMERICAN_OBSTACLE_HPP
#define MANGO_AMERICAN_OBSTACLE_HPP

#include <span>
#include <cmath>
#include <algorithm>

namespace mango {

/**
 * American put option obstacle in log-moneyness coordinates.
 *
 * Intrinsic value: ψ(x) = max(1 - exp(x), 0)
 * where x = ln(S/K).
 *
 * This represents the payoff max(K - S, 0) in log-moneyness.
 */
class AmericanPutObstacle {
public:
    /**
     * Evaluate obstacle at all grid points.
     *
     * @param t Current time (unused - intrinsic value time-independent)
     * @param x Grid points (log-moneyness values)
     * @param psi Output: obstacle values ψ(x)
     */
    void operator()(double t, std::span<const double> x,
                    std::span<double> psi) const {
        [[maybe_unused]] auto unused = t;  // Intrinsic value is time-independent

        #pragma omp simd
        for (size_t i = 0; i < x.size(); ++i) {
            // ψ(x) = max(1 - exp(x), 0)
            psi[i] = std::max(1.0 - std::exp(x[i]), 0.0);
        }
    }
};

/**
 * American call option obstacle in log-moneyness coordinates.
 *
 * Intrinsic value: ψ(x) = max(exp(x) - 1, 0)
 * where x = ln(S/K).
 *
 * This represents the payoff max(S - K, 0) in log-moneyness.
 */
class AmericanCallObstacle {
public:
    /**
     * Evaluate obstacle at all grid points.
     *
     * @param t Current time (unused - intrinsic value time-independent)
     * @param x Grid points (log-moneyness values)
     * @param psi Output: obstacle values ψ(x)
     */
    void operator()(double t, std::span<const double> x,
                    std::span<double> psi) const {
        [[maybe_unused]] auto unused = t;  // Intrinsic value is time-independent

        #pragma omp simd
        for (size_t i = 0; i < x.size(); ++i) {
            // ψ(x) = max(exp(x) - 1, 0)
            psi[i] = std::max(std::exp(x[i]) - 1.0, 0.0);
        }
    }
};

}  // namespace mango

#endif  // MANGO_AMERICAN_OBSTACLE_HPP

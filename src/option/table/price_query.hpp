// SPDX-License-Identifier: MIT
#pragma once

#include <vector>

namespace mango {

/// Configuration for multi-K_ref surface construction.
/// Used by both manual and adaptive grid builders.
struct MultiKRefConfig {
    std::vector<double> K_refs;   ///< explicit list; if empty, use auto selection
    int K_ref_count = 11;         ///< used when K_refs is empty
    double K_ref_span = 0.3;      ///< +/-span around spot for auto mode (log-spaced)
};

struct PriceQuery {
    double spot;
    double strike;
    double tau;
    double sigma;
    double rate;
};

}  // namespace mango

// SPDX-License-Identifier: MIT
#pragma once

#include "mango/option/option_spec.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace mango {

/// Filter dividends: keep only those strictly inside (0, T) with positive
/// amount.  Sort by calendar time.  Merge duplicates at the same date
/// (within 1e-12 tolerance) by summing amounts.
inline std::vector<Dividend> filter_and_merge_dividends(
    const std::vector<Dividend>& divs, double T)
{
    std::vector<Dividend> filtered;
    for (const auto& div : divs) {
        if (div.calendar_time > 0.0 && div.calendar_time < T && div.amount > 0.0) {
            filtered.push_back(div);
        }
    }
    std::sort(filtered.begin(), filtered.end(),
              [](const Dividend& a, const Dividend& b) {
                  return a.calendar_time < b.calendar_time;
              });

    // Merge same-date dividends
    std::vector<Dividend> merged;
    for (const auto& div : filtered) {
        if (!merged.empty() &&
            std::abs(merged.back().calendar_time - div.calendar_time) < 1e-12) {
            merged.back().amount += div.amount;
        } else {
            merged.push_back(div);
        }
    }
    return merged;
}

}  // namespace mango

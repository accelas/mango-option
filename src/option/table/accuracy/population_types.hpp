// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>

namespace mango {

/// Final accuracy work ceilings, independent of adaptive training samples.
struct AccuracyEvaluationLimits {
    size_t max_population_rows=512;
    size_t max_reference_requests=8192;
    size_t max_qualification_rounds=2;
    [[nodiscard]] bool valid() const noexcept {
        return max_population_rows>0 && max_reference_requests>0 && max_qualification_rounds>0 && max_qualification_rounds<=3;
    }
};

enum class AccuracyStratum : uint32_t {
    GlobalCorner=1u<<0, RegimeInterior=1u<<1, EventSide=1u<<2,
    EventAdmission=1u<<3, ExpiryAdmission=1u<<4, ZeroRate=1u<<5,
    MoneynessStrip=1u<<6,
};

[[nodiscard]] constexpr bool has_stratum(uint32_t mask, AccuracyStratum stratum) noexcept {
    return (mask & static_cast<uint32_t>(stratum))!=0;
}

/// Empirical profile scope; this is not a uniform whole-domain error bound.
struct AccuracyPopulationCounts {
    uint32_t profile_version=1;
    size_t declared_rows=0;
    size_t pricing_rows=0;
    size_t admission_rows=0;
    size_t refusal_rows=0;
    size_t temporal_regimes=0;
    size_t declared_moneyness_node_limit=0;
    size_t moneyness_strip_rows=0;
    bool off_node_moneyness_available=false;
};

} // namespace mango

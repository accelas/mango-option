// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <optional>

namespace mango {

/// Actual provider work. An attempted solve may complete or fail.
struct PdeWork {
    size_t attempted = 0;
    size_t completed = 0;
    size_t failed = 0;

    PdeWork& operator+=(const PdeWork& other) {
        attempted += other.attempted;
        completed += other.completed;
        failed += other.failed;
        return *this;
    }
};

/// Generic callback requests are distinct from reported numerical work.
/// An empty ledger is known zero; any unreported callback makes its total unknown.
struct OperationWork {
    size_t requests = 0;
    size_t failed_requests = 0;
    std::optional<PdeWork> pde = PdeWork{};

    void record(bool success, const std::optional<PdeWork>& reported) {
        ++requests;
        if (!success) ++failed_requests;
        if (pde && reported) *pde += *reported;
        else pde.reset();
    }
    OperationWork& operator+=(const OperationWork& other) {
        requests += other.requests;
        failed_requests += other.failed_requests;
        if (pde && other.pde) *pde += *other.pde;
        else pde.reset();
        return *this;
    }
};

/// Owned ledger survives failed refinement as well as successful delivery.
struct RefinementWork {
    OperationWork references;
    OperationWork tables;
    OperationWork selection;  ///< Reference-set evaluator calls, not per-row preparations

    [[nodiscard]] std::optional<PdeWork> total_pde() const {
        if (!references.pde || !tables.pde || !selection.pde) return std::nullopt;
        auto total = *references.pde;
        total += *tables.pde;
        total += *selection.pde;
        return total;
    }
    [[nodiscard]] std::optional<size_t> total_pde_attempts() const {
        const auto total = total_pde();
        if (!total) return std::nullopt;
        return total->attempted;
    }
    RefinementWork& operator+=(const RefinementWork& other) {
        references += other.references;
        tables += other.tables;
        selection += other.selection;
        return *this;
    }
};

} // namespace mango

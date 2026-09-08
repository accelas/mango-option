// SPDX-License-Identifier: MIT
#pragma once
#include "mango/math/proof/interval.hpp"
#include <cstddef>
#include <expected>
#include <span>
#include <utility>
#include <vector>

namespace mango::detail::proof {
enum class InputError { Shape, Nonfinite, Axis, Knots, Cell };

/// A tensor Bernstein polynomial on [0,1]^rank. Coefficients enclose the
/// actual polynomial coefficients; all dimensions use row-major ordering.
/// This preparation kernel accepts rank <= 4, degree <= 64, and at most
/// 4096 coefficients. These are explicit private arithmetic capacity limits.
class BernsteinTensor {
  public:
    static std::expected<BernsteinTensor, InputError> create(std::vector<std::size_t> degrees,
                                                             std::vector<Interval> coefficients);
    [[nodiscard]] Interval bounds() const;
    [[nodiscard]] std::expected<std::pair<BernsteinTensor, BernsteinTensor>, InputError>
    split(std::size_t axis) const;
    const std::vector<std::size_t> &degrees() const { return degrees_; }
    const std::vector<Interval> &coefficients() const { return coefficients_; }

  private:
    BernsteinTensor(std::vector<std::size_t> degrees, std::vector<Interval> coefficients)
        : degrees_(std::move(degrees)), coefficients_(std::move(coefficients)) {}
    std::vector<std::size_t> degrees_;
    std::vector<Interval> coefficients_;
};
enum class ProofStatus { Certified, NegativeWitness, Indeterminate };
enum class StopReason { None, NodeBudget, DepthBudget, Arithmetic };
struct ProofBudget {
    std::size_t max_nodes = 4096;
    // Unit-box witness labels are exact dyadics; the implementation caps
    // subdivision depth at 52 even when a larger limit is supplied.
    std::size_t max_depth = 24;
};
struct ProofResult {
    ProofStatus status = ProofStatus::Indeterminate;
    StopReason reason = StopReason::None;
    std::size_t nodes = 0;
    std::size_t deepest = 0;
    // Only populated for a rigorous negative witness, in original unit-box
    // coordinates. This is not a sample-based indication of a violation.
    std::vector<std::pair<double, double>> witness_box;
    Interval witness_bound{0};
};
ProofResult prove_nonnegative(const BernsteinTensor &polynomial, ProofBudget budget = {});
} // namespace mango::detail::proof

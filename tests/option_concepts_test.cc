// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include "src/option/option_concepts.hpp"
#include "src/option/american_option_result.hpp"

namespace mango {
namespace {

static_assert(OptionResult<AmericanOptionResult>,
    "AmericanOptionResult must satisfy OptionResult concept");

static_assert(!OptionResultWithVega<AmericanOptionResult>,
    "AmericanOptionResult should not satisfy OptionResultWithVega");

TEST(OptionConceptsTest, StaticAssertionsCompile) {
    SUCCEED();
}

}  // namespace
}  // namespace mango

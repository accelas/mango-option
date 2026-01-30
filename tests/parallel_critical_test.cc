// SPDX-License-Identifier: MIT
#include "src/support/parallel.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <vector>

TEST(ParallelCriticalTest, AtomicCounterCorrectness) {
    std::atomic<int> counter{0};
    const int iterations = 1000;

    MANGO_PRAGMA_PARALLEL
    {
        MANGO_PRAGMA_FOR_STATIC
        for (int i = 0; i < iterations; ++i) {
            MANGO_PRAGMA_CRITICAL
            {
                ++counter;
            }
        }
    }

    EXPECT_EQ(counter.load(), iterations);
}

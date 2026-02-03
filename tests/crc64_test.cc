// SPDX-License-Identifier: MIT
#include "mango/support/crc64.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(CRC64, ComputesConsistentChecksum) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};

    uint64_t checksum1 = mango::CRC64::compute(data.data(), data.size());
    uint64_t checksum2 = mango::CRC64::compute(data.data(), data.size());

    EXPECT_EQ(checksum1, checksum2) << "Same data should produce same checksum";
    EXPECT_NE(checksum1, 0) << "Checksum should not be zero for non-zero data";
}

TEST(CRC64, DetectsSingleBitChange) {
    std::vector<double> data1 = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> data2 = {1.0, 2.0, 3.0, 4.0, 5.0};

    uint64_t checksum1 = mango::CRC64::compute(data1.data(), data1.size());

    // Flip one bit in one value
    data2[2] = 3.0000000001;

    uint64_t checksum2 = mango::CRC64::compute(data2.data(), data2.size());

    EXPECT_NE(checksum1, checksum2) << "Different data should produce different checksums";
}

TEST(CRC64, DifferentDataProducesDifferentChecksum) {
    std::vector<double> data1 = {1.0, 2.0, 3.0};
    std::vector<double> data2 = {1.0, 2.0, 4.0};

    uint64_t checksum1 = mango::CRC64::compute(data1.data(), data1.size());
    uint64_t checksum2 = mango::CRC64::compute(data2.data(), data2.size());

    EXPECT_NE(checksum1, checksum2);
}

TEST(CRC64, HandlesEmptyData) {
    std::vector<double> data;

    uint64_t checksum = mango::CRC64::compute(data.data(), 0);

    // CRC64-ECMA of empty data should be specific value
    // Initial XOR with final XOR of all 1s
    EXPECT_EQ(checksum, 0ULL);
}

TEST(CRC64, HandlesLargeData) {
    std::vector<double> data(10000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<double>(i);
    }

    uint64_t checksum = mango::CRC64::compute(data.data(), data.size());

    EXPECT_NE(checksum, 0) << "Large data should produce non-zero checksum";

    // Verify consistency
    uint64_t checksum2 = mango::CRC64::compute(data.data(), data.size());
    EXPECT_EQ(checksum, checksum2);
}

TEST(CRC64, OrderMatters) {
    std::vector<double> data1 = {1.0, 2.0, 3.0};
    std::vector<double> data2 = {3.0, 2.0, 1.0};

    uint64_t checksum1 = mango::CRC64::compute(data1.data(), data1.size());
    uint64_t checksum2 = mango::CRC64::compute(data2.data(), data2.size());

    EXPECT_NE(checksum1, checksum2) << "Order should matter";
}

TEST(CRC64, ECMA182Standard) {
    // Test ECMA-182 standard compliance
    // For empty input, CRC should be 0 (init=0, final_xor=0)
    std::vector<double> empty_data;
    uint64_t empty_checksum = mango::CRC64::compute(empty_data.data(), 0);
    EXPECT_EQ(empty_checksum, 0ULL) << "CRC64-ECMA-182 of empty data should be 0";

    // Test with known byte pattern: ASCII string "123456789"
    // CRC64-ECMA-182 of "123456789" is 0x62EC59E3F1A4F00A
    // Note: Currently we only test with double* API, not raw bytes

    // CRC64/WE (init=0xFFFF..., final=0xFFFF...) produces different results
    // than CRC64-ECMA-182 (init=0x0, final=0x0)

    // Verify we're NOT using CRC64/WE (which would give all 0xFFs for empty)
    EXPECT_NE(empty_checksum, 0xFFFFFFFFFFFFFFFFULL)
        << "Should not use CRC64/WE variant (init/final 0xFFFF...)";
}

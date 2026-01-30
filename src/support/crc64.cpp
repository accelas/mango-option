// SPDX-License-Identifier: MIT
#include "src/support/crc64.hpp"
#include <cstring>
#include <mutex>

namespace mango {

// Static member initialization
uint64_t CRC64::table_[256];
std::once_flag CRC64::init_flag_;

void CRC64::init_table() {
    for (size_t i = 0; i < 256; ++i) {
        uint64_t crc = i;
        for (size_t j = 0; j < 8; ++j) {
            if (crc & 1) {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc = crc >> 1;
            }
        }
        table_[i] = crc;
    }
}

uint64_t CRC64::compute(const double* data, size_t count) {
    // Thread-safe lazy initialization
    std::call_once(init_flag_, init_table);

    // Convert double array to byte array for CRC computation
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    size_t byte_count = count * sizeof(double);

    // CRC64-ECMA-182: initial value is 0x0, final XOR is 0x0
    uint64_t crc = 0x0ULL;

    for (size_t i = 0; i < byte_count; ++i) {
        uint8_t index = static_cast<uint8_t>(crc ^ bytes[i]);
        crc = (crc >> 8) ^ table_[index];
    }

    // No final XOR for ECMA-182 standard
    return crc;
}

}  // namespace mango

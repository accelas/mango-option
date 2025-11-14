#include "src/option/crc64.hpp"
#include <cstring>

namespace mango {

// Static member initialization
uint64_t CRC64::table_[256];
bool CRC64::table_initialized_ = false;

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
    table_initialized_ = true;
}

uint64_t CRC64::compute(const double* data, size_t count) {
    if (!table_initialized_) {
        init_table();
    }

    // Convert double array to byte array for CRC computation
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    size_t byte_count = count * sizeof(double);

    // CRC64-ECMA: initial value is 0xFFFFFFFFFFFFFFFF, final XOR is 0xFFFFFFFFFFFFFFFF
    uint64_t crc = 0xFFFFFFFFFFFFFFFFULL;

    for (size_t i = 0; i < byte_count; ++i) {
        uint8_t index = static_cast<uint8_t>(crc ^ bytes[i]);
        crc = (crc >> 8) ^ table_[index];
    }

    // Final XOR
    return crc ^ 0xFFFFFFFFFFFFFFFFULL;
}

}  // namespace mango

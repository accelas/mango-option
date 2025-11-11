#include "src/cpu/feature_detection.hpp"
#include <gtest/gtest.h>
#include <iostream>

TEST(CPUFeatureDetectionTest, DetectFeatures) {
    auto features = mango::cpu::detect_cpu_features();

    // x86-64 baseline guarantees SSE2
    EXPECT_TRUE(features.has_sse2);

    // Print detected features for diagnostic
    std::cout << "SSE2: " << features.has_sse2 << "\n";
    std::cout << "AVX2: " << features.has_avx2 << "\n";
    std::cout << "AVX512F: " << features.has_avx512f << "\n";
    std::cout << "FMA: " << features.has_fma << "\n";
}

TEST(CPUFeatureDetectionTest, SelectISATarget) {
    auto target = mango::cpu::select_isa_target();
    auto name = mango::cpu::isa_target_name(target);

    std::cout << "Selected ISA: " << name << "\n";

    // Should return one of the known targets
    EXPECT_TRUE(target == mango::cpu::ISATarget::DEFAULT ||
                target == mango::cpu::ISATarget::AVX2 ||
                target == mango::cpu::ISATarget::AVX512F);
}

TEST(CPUFeatureDetectionTest, ISATargetNames) {
    EXPECT_EQ(mango::cpu::isa_target_name(mango::cpu::ISATarget::DEFAULT), "SSE2");
    EXPECT_EQ(mango::cpu::isa_target_name(mango::cpu::ISATarget::AVX2), "AVX2+FMA");
    EXPECT_EQ(mango::cpu::isa_target_name(mango::cpu::ISATarget::AVX512F), "AVX512F");
}

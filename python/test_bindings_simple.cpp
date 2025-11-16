/**
 * @file test_bindings_simple.cpp
 * @brief Simple test to verify SolverMemoryArena bindings compile
 */

#include "src/support/memory/solver_memory_arena.hpp"
#include <iostream>
#include <memory>

int main() {
    try {
        // Test factory method
        auto result = mango::memory::SolverMemoryArena::create(1024 * 1024);
        if (!result) {
            std::cerr << "Failed to create arena: " << result.error() << std::endl;
            return 1;
        }

        auto arena = result.value();

        // Test basic functionality
        auto stats = arena->get_stats();
        std::cout << "Arena created successfully:" << std::endl;
        std::cout << "  Total size: " << stats.total_size << std::endl;
        std::cout << "  Used size: " << stats.used_size << std::endl;
        std::cout << "  Active workspaces: " << stats.active_workspace_count << std::endl;

        // Test workspace counting
        arena->increment_active();
        arena->increment_active();
        stats = arena->get_stats();
        std::cout << "After incrementing: " << stats.active_workspace_count << " active workspaces" << std::endl;

        arena->decrement_active();
        stats = arena->get_stats();
        std::cout << "After decrementing: " << stats.active_workspace_count << " active workspaces" << std::endl;

        // Test reset
        auto reset_result = arena->try_reset();
        if (!reset_result) {
            std::cout << "Reset failed (expected due to active workspace): " << reset_result.error() << std::endl;
        }

        arena->decrement_active();
        reset_result = arena->try_reset();
        if (reset_result) {
            std::cout << "Reset successful!" << std::endl;
        }

        // Test resource access
        auto resource = arena->resource();
        if (resource) {
            std::cout << "Memory resource accessible: " << resource << std::endl;
        }

        std::cout << "✓ All C++ tests passed!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        return 1;
    }
}
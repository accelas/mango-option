// src/support/lifetime.hpp
#pragma once

#include <cstddef>
#include <cstring>  // std::memmove
#include <type_traits>
#include <memory>  // std::launder

namespace mango {

/// Start lifetime of T[n] array at given memory location
///
/// Emulates std::start_lifetime_as_array (C++23) using std::memmove.
/// std::memmove implicitly creates objects in the destination region
/// per [intro.object]/13 and [cstring.syn]/2.
/// The memmove(p, p, n) idiom forces implicit object creation without moving memory.
///
/// See: https://stackoverflow.com/questions/79164176/emulate-stdstart-lifetime-as-array-in-c20
///
/// IMPORTANT: T must be trivially destructible because no destructor is ever
/// called when the workspace goes out of scope.
///
/// @tparam T Element type (must be trivially destructible and trivially copyable)
/// @param p Pointer to suitably aligned memory
/// @param n Number of elements
/// @return Pointer to first element of the array
template<typename T>
T* start_array_lifetime(void* p, size_t n) {
    static_assert(std::is_trivially_destructible_v<T>,
        "start_array_lifetime requires trivially destructible types because "
        "no destructor is called when the workspace goes out of scope");
    static_assert(std::is_trivially_copyable_v<T>,
        "start_array_lifetime requires trivially copyable types for implicit lifetime creation");

    if (n == 0) return static_cast<T*>(p);

    // Emulate std::start_lifetime_as_array (C++23) using std::memmove.
    // std::memmove implicitly creates objects in the destination region
    // per [intro.object]/13 and [cstring.syn]/2.
    // The memmove(p, p, n) idiom forces implicit object creation without moving memory.
    // See: https://stackoverflow.com/questions/79164176/emulate-stdstart-lifetime-as-array-in-c20
    return std::launder(static_cast<T*>(std::memmove(p, p, sizeof(T) * n)));
}

/// Align offset up to specified alignment
constexpr size_t align_up(size_t offset, size_t alignment) noexcept {
    return (offset + alignment - 1) & ~(alignment - 1);
}

}  // namespace mango

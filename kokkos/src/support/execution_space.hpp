#pragma once

/// @file execution_space.hpp
/// @brief Execution space and memory space aliases for Kokkos backends
///
/// Provides type aliases for different Kokkos execution targets (CPU, SYCL, CUDA, HIP)
/// and common View templates for 1D-4D arrays.

#include <Kokkos_Core.hpp>

namespace mango {

/// Execution target for runtime selection of compute backend
enum class ExecutionTarget {
    CPU,   ///< OpenMP or Serial CPU execution
    SYCL,  ///< Intel SYCL (oneAPI) GPU execution
    CUDA,  ///< NVIDIA CUDA GPU execution
    HIP    ///< AMD HIP GPU execution
};

// Memory space aliases based on enabled backends
#if defined(KOKKOS_ENABLE_SYCL)
using SYCLSpace = Kokkos::Experimental::SYCL;
using SYCLMemSpace = Kokkos::Experimental::SYCLSharedUSMSpace;
#endif

#if defined(KOKKOS_ENABLE_CUDA)
using CUDASpace = Kokkos::Cuda;
using CUDAMemSpace = Kokkos::CudaSpace;
#endif

#if defined(KOKKOS_ENABLE_HIP)
using HIPSpace = Kokkos::HIP;
using HIPMemSpace = Kokkos::HIPSpace;
#endif

using HostSpace = Kokkos::DefaultHostExecutionSpace;
using HostMemSpace = Kokkos::HostSpace;

// Default execution/memory space based on enabled backends (priority: SYCL > CUDA > HIP > Host)
#if defined(KOKKOS_ENABLE_SYCL)
using DefaultExecSpace = Kokkos::Experimental::SYCL;
using DefaultMemSpace = Kokkos::Experimental::SYCLSharedUSMSpace;
#elif defined(KOKKOS_ENABLE_CUDA)
using DefaultExecSpace = Kokkos::Cuda;
using DefaultMemSpace = Kokkos::CudaSpace;
#elif defined(KOKKOS_ENABLE_HIP)
using DefaultExecSpace = Kokkos::HIP;
using DefaultMemSpace = Kokkos::HIPSpace;
#else
using DefaultExecSpace = Kokkos::DefaultHostExecutionSpace;
using DefaultMemSpace = Kokkos::HostSpace;
#endif

// Common View type aliases
template <typename T, typename MemSpace>
using View1D = Kokkos::View<T*, MemSpace>;

template <typename T, typename MemSpace>
using View2D = Kokkos::View<T**, MemSpace>;

template <typename T, typename MemSpace>
using View3D = Kokkos::View<T***, MemSpace>;

template <typename T, typename MemSpace>
using View4D = Kokkos::View<T****, MemSpace>;

}  // namespace mango

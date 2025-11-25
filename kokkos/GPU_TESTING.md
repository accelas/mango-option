# GPU Testing Guide for Kokkos Backend

This guide covers building and testing the Kokkos implementation on AMD GPUs using the HIP backend.

## Prerequisites

### AMD ROCm Installation

ROCm 6.0+ is required for HIP support:

```bash
# Check ROCm installation
rocminfo | head -20

# Verify HIP
hipcc --version
```

### Building Kokkos with HIP Backend

The default Bazel build uses OpenMP. For GPU testing, build Kokkos manually with HIP:

```bash
# Clone Kokkos
cd /tmp
wget https://github.com/kokkos/kokkos/archive/refs/tags/4.3.00.tar.gz
tar xzf 4.3.00.tar.gz
cd kokkos-4.3.00

# Configure for AMD GPU (adjust architecture as needed)
mkdir build && cd build
cmake .. \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DCMAKE_INSTALL_PREFIX=/usr/local/kokkos-hip \
    -DKokkos_ENABLE_HIP=ON \
    -DKokkos_ARCH_AMD_GFX1030=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_ROCTHRUST=OFF \
    -DBUILD_SHARED_LIBS=OFF

# Build and install
make -j$(nproc)
sudo make install
```

### Supported AMD GPU Architectures

| Architecture | GPUs |
|-------------|------|
| GFX906 | Radeon VII, MI50 |
| GFX908 | MI100 |
| GFX90A | MI200 series |
| GFX1030 | RX 6800/6900, integrated RDNA2 |
| GFX1100 | RX 7900 series |

For integrated GPUs (e.g., Ryzen APU), use the closest supported architecture.

## Compiling Tests with HIP

### TartanLlama/expected Dependency

The code uses `std::expected` (C++23). For C++17 compatibility with HIP, use the TartanLlama polyfill:

```bash
cd /tmp
git clone --depth 1 https://github.com/TartanLlama/expected.git tl_expected
```

### Compile Example

```bash
hipcc -std=c++17 -O2 \
    -I/usr/local/kokkos-hip/include \
    -I/tmp/tl_expected/include \
    your_test.cpp \
    -L/usr/local/kokkos-hip/lib \
    -lkokkoscore -lkokkoscontainers -lkokkossimd -ldl \
    -o your_test
```

### Code Modifications for tl::expected

Replace `std::expected` and `std::unexpected` with `tl::expected` and `tl::unexpected`:

```cpp
#include "tl/expected.hpp"

// Use tl:: namespace
tl::expected<Result, Error> my_function() {
    if (error_condition)
        return tl::unexpected(Error::SomeError);
    return Result{...};
}
```

## Validation Results

### Test Environment

- CPU: AMD Ryzen 9 9955HX
- GPU: AMD Radeon Graphics (gfx1036, integrated RDNA2)
- ROCm: 6.0
- Kokkos: 4.3.00

### Basic Validation (Vector Add)

```
Execution space: HIP
Sum: 3000 (expected: 3000)
Kokkos HIP test: PASSED
```

### Thomas Algorithm (Tridiagonal Solver)

```
CPU Reference: x[0]=50, x[N/2]=1275, x[N-1]=50, Sum=85850
Kokkos HIP:    x[0]=50, x[N/2]=1275, x[N-1]=50, Sum=85850
Max difference: 0
Result: IDENTICAL
```

### FDM American Option Pricing

```
Execution space: HIP

--- American Put Pricing ---
  Price: 6.6481
  Delta: -0.4402
  Time:  34.38 ms

--- FDM IV Solve ---
  Implied Vol: 20.0000%
  True Vol:    20.0000%
  IV Error:    0.0000%
  Iterations:  5
  Time:        312.12 ms
```

### IV Solve Accuracy (Multiple Volatilities)

| True Vol | Recovered IV | Error | Iterations | Time |
|----------|--------------|-------|------------|------|
| 15% | 15.00% | 0.0000% | 5 | 355ms |
| 20% | 20.00% | 0.0000% | 5 | 312ms |
| 25% | 25.00% | 0.0000% | 4 | 259ms |
| 30% | 30.00% | 0.0000% | 5 | 268ms |
| 35% | 35.00% | 0.0000% | 5 | 258ms |

All IV recoveries are exact within solver tolerance.

## Architecture Notes

The Kokkos implementation uses a hybrid CPU/GPU approach:

1. **GPU-parallel operations**: `parallel_for`, `parallel_reduce` for:
   - Payoff initialization
   - Black-Scholes operator application
   - Jacobian assembly
   - Boundary condition application
   - Obstacle computation

2. **Host-side operations** (inherently serial):
   - Thomas algorithm (tridiagonal solve)
   - Brent's method for IV root-finding

Data is transferred via `Kokkos::create_mirror_view_and_copy()` for host-side operations.

## Troubleshooting

### "HIP enabled but no automatically detected AMD GPU architecture is supported"

Specify the architecture explicitly:
```bash
-DKokkos_ARCH_AMD_GFX1030=ON  # or appropriate architecture
```

### "rocthrust not found"

Disable rocthrust (not required for core functionality):
```bash
-DKokkos_ENABLE_ROCTHRUST=OFF
```

### std::unexpected conflict

The C++ standard library has a deprecated `std::unexpected()` function that conflicts with `std::unexpected<E>`. Use `tl::unexpected` from TartanLlama/expected.

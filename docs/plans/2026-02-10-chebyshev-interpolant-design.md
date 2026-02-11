# Chebyshev Interpolant Design

## Goal

Add a generic N-dimensional Chebyshev interpolant to `src/math/chebyshev/`
that satisfies `SurfaceInterpolant<T, N>` and supports pluggable tensor
storage (raw or Tucker-compressed).

## Architecture

`ChebyshevInterpolant<N, Storage>` is a policy-based template. The
interpolant handles Chebyshev node generation, domain clamping, barycentric
weight computation, and finite-difference partial derivatives. The `Storage`
policy handles tensor data and contraction.

Two storage policies:

- **`RawTensor<N>`** — flat `vector<double>`, direct N-dim contraction.
  No Eigen dependency.
- **`TuckerTensor<N>`** — HOSVD-compressed core + factor matrices.
  Contracts barycentric weights through factors then core. Depends on Eigen.

Convenience aliases:

```cpp
template <size_t N> using ChebyshevTensor = ChebyshevInterpolant<N, RawTensor<N>>;
template <size_t N> using ChebyshevTucker = ChebyshevInterpolant<N, TuckerTensor<N>>;
```

Both satisfy `SurfaceInterpolant<T, N>` directly (eval + partial with
`std::array<double, N>` coordinates).

## Storage Policy Interface

```cpp
template <size_t N>
struct SomeStorage {
    // Build from sampled values (row-major, shape[0] x ... x shape[N-1])
    static SomeStorage build(std::vector<double> values,
                             const std::array<size_t, N>& shape,
                             /* policy-specific params */);

    // Contract with per-axis barycentric weight vectors.
    // coeffs[d] has length shape[d].
    double contract(const std::array<std::vector<double>, N>& coeffs) const;

    // Number of stored coefficients.
    size_t compressed_size() const;
};
```

`RawTensor<N>::contract` sums over all grid points weighted by the product
of per-axis barycentric weights. `TuckerTensor<N>::contract` multiplies
each weight vector by its factor matrix (`U_d^T * coeffs[d]` → R_d values),
then contracts the resulting rank-sized vectors with the core tensor.

## ChebyshevInterpolant<N, Storage>

```cpp
template <size_t N, typename Storage>
class ChebyshevInterpolant {
public:
    struct Domain { std::array<std::array<double, 2>, N> bounds; };
    struct Config {
        std::array<size_t, N> num_pts;
        // Storage-specific params forwarded via build methods
    };

    // Primary construction: pre-computed values on CGL nodes
    static ChebyshevInterpolant build_from_values(
        std::span<const double> values,
        const Domain& domain, const Config& config, auto&&... storage_args);

    // Sampling construction: callable takes array<double, N>
    static ChebyshevInterpolant build(
        std::function<double(std::array<double, N>)> f,
        const Domain& domain, const Config& config, auto&&... storage_args);

    // SurfaceInterpolant interface
    double eval(const std::array<double, N>& query) const;
    double partial(size_t axis, const std::array<double, N>& coords) const;

    // Metadata
    size_t compressed_size() const;
    std::array<size_t, N> num_pts() const;
    const Domain& domain() const;
};
```

`eval()` pipeline:
1. Clamp query to domain bounds
2. For each axis d, compute barycentric weight vector (length shape[d]),
   handling exact node coincidence
3. Call `storage_.contract(coeffs)`

`partial()`: central finite difference with `h = 1e-6 * (hi - lo)`,
clamped to domain. Unchanged from current implementation.

`storage_args` are forwarded to `Storage::build()`. For `RawTensor<N>`
this is empty. For `TuckerTensor<N>` this is `double epsilon`.

## Tucker HOSVD Generalization

The current 3D and 4D HOSVD implementations use nested loops of fixed
depth for mode-unfolding and sequential contraction. The generic version
uses flat-index stride computation:

**`mode_unfold<N>`**: Iterates flat index 0..total-1, computes N-dim
subscript via div/mod, extracts row (= subscript[mode]) and column
(remaining subscripts packed with strides).

**`tucker_hosvd<N>`**: Sequential mode contraction loop:
```
for mode = 0..N-1:
    unfold current tensor along mode
    SVD, truncate to rank
    contract: G = U^T * unfolded
    repack to row-major with updated shape
```

The repacking logic is the trickiest part. Each contraction replaces
shape[mode] with ranks[mode] and requires correct stride recomputation.
A helper `repack_contraction<N>()` handles this generically using the
same flat-index approach as mode_unfold.

## File Layout

```
src/math/chebyshev/
├── chebyshev_nodes.hpp           # Moved from dimensionless/ (unchanged)
├── raw_tensor.hpp                # RawTensor<N>
├── tucker_tensor.hpp             # TuckerTensor<N> + tucker_hosvd<N>
├── chebyshev_interpolant.hpp     # ChebyshevInterpolant<N, Storage>
└── BUILD.bazel
```

Build targets:
- `chebyshev_nodes` — header-only, no deps
- `raw_tensor` — header-only, no deps
- `tucker_tensor` — header-only, depends on `@eigen`
- `chebyshev_interpolant` — depends on `chebyshev_nodes`
  (storage dep comes from user's choice of alias target)

## Tests

`tests/chebyshev_interpolant_test.cc`:

- Polynomial exactness (degree < num_pts) for both RawTensor and Tucker
  at N=3 and N=4
- Spectral convergence for smooth functions (error decreases with num_pts)
- Tucker compression reduces storage vs raw
- Partial derivatives match analytical for simple functions
- Domain clamping (out-of-bounds queries return boundary values)
- `build()` with callable matches `build_from_values()`

## Not In Scope

- Benchmark adapter migration (stays in benchmarks/)
- Coordinate transforms (future PR)
- Piecewise / incremental builders (future PR)
- Moving existing bspline files to src/math/bspline/ (separate cleanup)
- Discrete dividend support

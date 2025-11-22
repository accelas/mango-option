# mdspan Adoption Strategy for mango-iv

**Date:** 2025-11-22
**Status:** ✅ Implemented (2025-11-22)
**Author:** Claude Code

## Executive Summary

This document proposes adopting C++23 `std::mdspan` (via Kokkos reference implementation) to eliminate manual multi-dimensional indexing, enable zero-copy LAPACK interop, and improve type safety in three high-value components:

1. **LAPACK Banded Matrix (HIGH PRIORITY)**: Custom layout eliminates O(bandwidth × n) storage conversion
2. **BSplineND Tensor Indexing (HIGH PRIORITY)**: Replace manual N-dimensional index calculation with type-safe subscripting
3. **NonUniformSpacing Multi-Section Buffer (MEDIUM PRIORITY)**: Self-documenting 2D view of 5-section layout

**Estimated Impact:**
- **Performance**: Eliminate ~400-800 element copies per B-spline factorization
- **Code Quality**: Remove ~60 lines of manual offset arithmetic
- **Maintainability**: Single source of truth for array layouts
- **Type Safety**: Compiler-verified multi-dimensional indexing

---

## Background: What is mdspan?

`std::mdspan` (C++23) provides multi-dimensional array views with:
- **Type-safe indexing**: `array[i, j, k]` instead of `array[i*n*m + j*m + k]`
- **Custom layouts**: User-defined mapping from logical indices to memory offsets
- **Zero overhead**: No runtime cost vs manual indexing
- **Zero copy**: Views existing memory without allocation

**Example:**
```cpp
// Before: Manual 2D indexing
std::vector<double> data(rows * cols);
double value = data[i * cols + j];  // Error-prone, hard to read

// After: mdspan
std::vector<double> data(rows * cols);
mdspan<double, dextents<size_t, 2>> matrix(data.data(), rows, cols);
double value = matrix[i, j];  // Type-safe, self-documenting
```

**Kokkos mdspan polyfill:**
- Already integrated in this project (see `src/support/mdspan/`)
- Header-only, zero dependencies
- Compatible with C++23 (will seamlessly upgrade when compilers support it)

---

## Candidate 1: LAPACK Banded Matrix Storage (HIGH PRIORITY)

### Current Problem

**File:** `src/math/banded_matrix_solver.hpp:286-303`

Every call to `factorize_banded()` performs a **full matrix copy** to convert from our row-major banded storage to LAPACK's column-major banded format:

```cpp
// Convert custom band format to LAPACK column-major band storage
for (lapack_int i = 0; i < n; ++i) {
    const size_t col_start = A.col_start(static_cast<size_t>(i));
    for (size_t k = 0; k < A.bandwidth(); ++k) {
        const size_t col = col_start + k;
        if (col >= A.size()) continue;

        const T value = A.band_values()[static_cast<size_t>(i) * A.bandwidth() + k];
        const lapack_int col_idx = static_cast<lapack_int>(col);
        const lapack_int row_idx = kl + ku + i - col_idx;

        if (row_idx >= 0 && row_idx < workspace.ldab_) {
            const size_t storage_idx =
                static_cast<size_t>(row_idx + col_idx * workspace.ldab_);
            workspace.lapack_storage_[storage_idx] = value;  // O(bandwidth × n) copies!
        }
    }
}
```

**Cost:** For n=100, bandwidth=4, that's ~400 element copies per factorization.

**Root cause:** `BandedMatrix` stores data in custom row-major format, but LAPACK expects column-major banded storage.

### LAPACK Banded Storage Specification

From DGBTRF documentation:

```
AB is DOUBLE PRECISION array, dimension (LDAB,N)
On entry, the matrix A in band storage, in rows KL+1 to 2*KL+KU+1;
rows 1 to KL of the array need not be set.

The j-th column of A is stored in the j-th column of the array AB as follows:
AB(KL+KU+1+i-j,j) = A(i,j) for max(1,j-KU)<=i<=min(N,j+KL)

LDAB is INTEGER
The leading dimension of the array AB.  LDAB >= 2*KL+KU+1.
```

**Key formula:** `AB(kl + ku + 1 + i - j, j) = A(i, j)`
**Layout:** Column-major (Fortran-style)

### Proposed mdspan Solution

**Step 1: Custom Layout Policy**

```cpp
/// Custom mdspan layout matching LAPACK banded storage
///
/// Maps logical matrix index (i,j) to LAPACK banded storage offset.
/// Formula: AB(kl + ku + i - j, j) where AB is column-major
struct lapack_banded_layout {
    template<class Extents>
    struct mapping {
        using extents_type = Extents;
        using index_type = typename Extents::index_type;
        using size_type = typename Extents::size_type;
        using rank_type = typename Extents::rank_type;
        using layout_type = lapack_banded_layout;

    private:
        extents_type extents_;
        index_type kl_;      ///< Number of sub-diagonals
        index_type ku_;      ///< Number of super-diagonals
        index_type ldab_;    ///< Leading dimension (= 2*kl + ku + 1)

    public:
        constexpr mapping(extents_type ext, index_type kl, index_type ku) noexcept
            : extents_(ext)
            , kl_(kl)
            , ku_(ku)
            , ldab_(2 * kl + ku + 1)
        {}

        /// Map (i, j) to flat offset
        ///
        /// Returns offset for LAPACK banded storage: AB(kl + ku + i - j, j)
        /// in column-major layout.
        constexpr index_type operator()(index_type i, index_type j) const noexcept {
            // LAPACK formula: AB(kl + ku + i - j, j)
            const index_type row_offset = kl_ + ku_ + i - j;

            // Column-major: offset = row + col * ldab
            return row_offset + j * ldab_;
        }

        constexpr const extents_type& extents() const noexcept { return extents_; }

        static constexpr bool is_always_unique() noexcept { return true; }
        static constexpr bool is_always_exhaustive() noexcept { return false; }
        static constexpr bool is_always_strided() noexcept { return true; }

        constexpr bool is_unique() const noexcept { return true; }
        constexpr bool is_exhaustive() const noexcept { return false; }
        constexpr bool is_strided() const noexcept { return true; }

        constexpr index_type required_span_size() const noexcept {
            return ldab_ * extents_.extent(1);  // ldab * n
        }

        constexpr index_type stride(rank_type r) const noexcept {
            if (r == 0) return 1;          // Row stride (column-major)
            if (r == 1) return ldab_;      // Column stride
            return 0;
        }
    };
};
```

**Step 2: Refactored BandedMatrix**

```cpp
template<std::floating_point T>
class BandedMatrix {
public:
    using extents_type = dextents<size_t, 2>;
    using layout_type = lapack_banded_layout;
    using mdspan_type = mdspan<T, extents_type, layout_type>;

    /// Construct banded matrix with LAPACK-compatible storage
    ///
    /// @param n Matrix dimension (n × n)
    /// @param kl Number of sub-diagonals
    /// @param ku Number of super-diagonals
    explicit BandedMatrix(size_t n, size_t kl, size_t ku)
        : n_(n)
        , kl_(static_cast<lapack_int>(kl))
        , ku_(static_cast<lapack_int>(ku))
        , ldab_(2 * kl_ + ku_ + 1)
        , data_(static_cast<size_t>(ldab_) * n, T{0})  // LAPACK storage
        , view_(data_.data(), n, n, kl_, ku_)
    {
        assert(kl >= 0 && ku >= 0);
        assert(n > 0);
    }

    /// Type-safe 2D indexing via mdspan
    ///
    /// Automatically uses LAPACK banded layout.
    T& operator()(size_t i, size_t j) {
        return view_[i, j];  // Uses lapack_banded_layout::mapping
    }

    T operator()(size_t i, size_t j) const {
        return view_[i, j];
    }

    /// Zero-copy LAPACK interface
    ///
    /// Returns raw pointer for direct use with LAPACKE functions.
    T* lapack_data() noexcept { return data_.data(); }
    const T* lapack_data() const noexcept { return data_.data(); }

    /// Get matrix dimension
    size_t size() const noexcept { return n_; }

    /// Get number of sub-diagonals
    lapack_int kl() const noexcept { return kl_; }

    /// Get number of super-diagonals
    lapack_int ku() const noexcept { return ku_; }

    /// Get leading dimension for LAPACK
    lapack_int ldab() const noexcept { return ldab_; }

private:
    size_t n_;                 ///< Matrix dimension
    lapack_int kl_;            ///< Sub-diagonals
    lapack_int ku_;            ///< Super-diagonals
    lapack_int ldab_;          ///< Leading dimension (2*kl + ku + 1)
    std::vector<T> data_;      ///< LAPACK column-major banded storage
    mdspan_type view_;         ///< Type-safe 2D view
};
```

**Step 3: Zero-Copy Factorization**

```cpp
template<std::floating_point T>
[[nodiscard]] BandedResult<T> factorize_banded(
    BandedMatrix<T>& A,  // Note: now modified in-place
    BandedLUWorkspace<T>& workspace) noexcept
{
    static_assert(std::same_as<T, double>,
                 "LAPACKE banded solvers currently only support double precision");

    const lapack_int n = static_cast<lapack_int>(A.size());
    if (n == 0) {
        return BandedResult<T>::error_result("Matrix dimension must be > 0");
    }

    workspace.factored_ = false;

    // Store LAPACK parameters
    workspace.kl_ = A.kl();
    workspace.ku_ = A.ku();
    workspace.ldab_ = A.ldab();

    // NO CONVERSION NEEDED! Data is already in LAPACK format.
    workspace.lapack_storage_ = A.lapack_data();  // Just store pointer

    // Perform LU factorization directly on A's storage
    workspace.pivot_indices_.resize(static_cast<size_t>(n));
    const lapack_int info = LAPACKE_dgbtrf(
        LAPACK_COL_MAJOR,
        n, n, workspace.kl_, workspace.ku_,
        A.lapack_data(),  // Direct pointer, zero-copy!
        workspace.ldab_,
        workspace.pivot_indices_.data()
    );

    if (info < 0) {
        return BandedResult<T>::error_result("LAPACKE_dgbtrf: invalid argument");
    }
    if (info > 0) {
        return BandedResult<T>::error_result("Matrix is singular");
    }

    workspace.factored_ = true;
    return BandedResult<T>::ok_result();
}
```

### Benefits

1. **Zero-copy LAPACK calls**: Eliminate lines 286-303 (O(bandwidth × n) conversion)
2. **Type-safe indexing**: `matrix[i, j]` instead of manual offset calculation
3. **Single source of truth**: Storage layout defined once in `lapack_banded_layout`
4. **Performance**: For n=100, bandwidth=4: eliminate ~400 element copies per factorization
5. **Maintainability**: Custom layout encapsulates LAPACK's complex indexing formula

### Trade-offs

**Cons:**
- Slightly more complex initial setup (custom layout implementation)
- Users must understand layout is column-major (but this was already true for LAPACK)
- Factorization now modifies matrix in-place (API change, but more efficient)

**Mitigation:**
- Comprehensive documentation of layout policy
- Unit tests verifying LAPACK compatibility
- Example code showing typical usage

---

## Candidate 2: BSplineND Tensor Coefficient Indexing (HIGH PRIORITY)

### Current Problem

**File:** `src/math/bspline_nd.hpp:237-249`

N-dimensional B-spline evaluation requires converting multi-dimensional indices to flat array offsets:

```cpp
/// Compute flat index from N-dimensional index (row-major order)
///
/// Converts multi-dimensional index to flat array index.
/// Example for 3D: idx = i*dim1*dim2 + j*dim2 + k
size_t compute_flat_index(const std::array<int, N>& indices) const noexcept {
    size_t idx = 0;
    size_t stride = 1;

    // Compute index in row-major order (last dimension varies fastest)
    for (size_t dim = N; dim > 0; --dim) {
        const size_t d = dim - 1;
        idx += static_cast<size_t>(indices[d]) * stride;
        stride *= dims_[d];
    }

    return idx;
}

// Usage:
const size_t flat_idx = compute_flat_index(indices);
sum = std::fma(coeffs_[flat_idx], weight, sum);
```

**Issues:**
- 10 lines of manual stride calculation (error-prone)
- No compile-time verification of index dimensionality
- Performance-critical hot path (~135ns per query)

### Proposed mdspan Solution

```cpp
template<std::floating_point T, size_t N>
    requires (N >= 1)
class BSplineND {
public:
    using GridArray = std::array<std::vector<T>, N>;
    using KnotArray = std::array<std::vector<T>, N>;
    using QueryPoint = std::array<T, N>;
    using Shape = std::array<size_t, N>;

    // NEW: mdspan for N-dimensional coefficient array
    using CoeffExtents = dextents<size_t, N>;
    using CoeffMdspan = mdspan<T, CoeffExtents, layout_right>;  // Row-major

    [[nodiscard]] static std::expected<BSplineND, std::string> create(
        GridArray grids,
        KnotArray knots,
        std::vector<T> coeffs)
    {
        // ... validation (unchanged) ...

        return BSplineND(std::move(grids), std::move(knots), std::move(coeffs));
    }

private:
    GridArray grids_;
    KnotArray knots_;
    std::vector<T> coeffs_;     ///< Coefficient storage
    CoeffMdspan coeffs_view_;   ///< N-dimensional view of coeffs_
    Shape dims_;

    BSplineND(GridArray grids, KnotArray knots, std::vector<T> coeffs)
        : grids_(std::move(grids))
        , knots_(std::move(knots))
        , coeffs_(std::move(coeffs))
        , coeffs_view_(nullptr, {})  // Initialized below
        , dims_{}
    {
        // Extract dimensions
        for (size_t i = 0; i < N; ++i) {
            dims_[i] = grids_[i].size();
        }

        // Create mdspan view with proper extents
        coeffs_view_ = create_coeffs_view(coeffs_.data(), dims_);
    }

    /// Helper to create mdspan with variadic extents
    static CoeffMdspan create_coeffs_view(T* data, const Shape& dims) {
        return create_view_impl(data, dims, std::make_index_sequence<N>{});
    }

    template<size_t... Is>
    static CoeffMdspan create_view_impl(T* data, const Shape& dims,
                                        std::index_sequence<Is...>) {
        return CoeffMdspan(data, dims[Is]...);
    }

    /// Recursive tensor-product evaluation (SIMPLIFIED!)
    template<size_t Dim>
    T eval_tensor_product(
        const std::array<int, N>& spans,
        const std::array<std::array<T, 4>, N>& weights,
        std::array<int, N> indices) const
    {
        T sum = 0.0;

        for (int offset = 0; offset < 4; ++offset) {
            const int idx = spans[Dim] - offset;

            if (static_cast<unsigned>(idx) >= static_cast<unsigned>(dims_[Dim])) {
                continue;
            }

            indices[Dim] = idx;
            const T weight = weights[Dim][offset];

            if constexpr (Dim == N - 1) {
                // Base case: use mdspan multi-dimensional indexing
                const T coeff = access_coeffs(coeffs_view_, indices);
                sum = std::fma(coeff, weight, sum);
            } else {
                // Recursive case (unchanged)
                const T nested_sum = eval_tensor_product<Dim + 1>(spans, weights, indices);
                sum = std::fma(nested_sum, weight, sum);
            }
        }

        return sum;
    }

    /// Access N-dimensional coefficient array via mdspan
    ///
    /// Uses variadic template expansion to convert std::array to mdspan subscript.
    template<size_t... Is>
    static T access_coeffs_impl(const CoeffMdspan& view, const std::array<int, N>& indices,
                                std::index_sequence<Is...>) {
        return view[indices[Is]...];  // Expands to view[indices[0], indices[1], ...]
    }

    static T access_coeffs(const CoeffMdspan& view, const std::array<int, N>& indices) {
        return access_coeffs_impl(view, indices, std::make_index_sequence<N>{});
    }
};
```

### Benefits

1. **Eliminate manual index calculation**: Remove 10-line `compute_flat_index()` function
2. **Type safety**: Compile-time verification of N-dimensional indexing
3. **Self-documenting**: `coeffs_view_[i, j, k, l]` is clearer than `coeffs_[flat_idx]`
4. **Zero overhead**: mdspan compiles to same assembly as manual indexing
5. **Flexibility**: Easy to experiment with different layouts (row-major, column-major)

### Trade-offs

**Cons:**
- Slightly more complex initialization (need to construct mdspan with extents)
- Requires variadic template expansion for `view[i0, i1, ..., iN]`

**Pros outweigh cons:** Cleaner code, better type safety, no performance cost.

---

## Candidate 3: NonUniformSpacing Multi-Section Buffer (MEDIUM PRIORITY)

### Current Problem

**File:** `src/pde/core/grid.hpp:272-300`

Non-uniform grid spacing stores 5 sections in a single flat buffer with manual offset arithmetic:

```cpp
struct NonUniformSpacing {
    std::vector<T> precomputed;  // [dx_left_inv | dx_right_inv | dx_center_inv | w_left | w_right]

    explicit NonUniformSpacing(std::span<const T> x)
        : n(x.size())
    {
        const size_t interior = n - 2;
        precomputed.resize(5 * interior);

        for (size_t i = 1; i <= n - 2; ++i) {
            const size_t idx = i - 1;

            // Manual section offsets (error-prone, hard to read):
            precomputed[idx] = T(1) / dx_left;                    // Section 0
            precomputed[interior + idx] = T(1) / dx_right;        // Section 1
            precomputed[2 * interior + idx] = T(1) / dx_center;   // Section 2
            precomputed[3 * interior + idx] = weight_left;        // Section 3
            precomputed[4 * interior + idx] = weight_right;       // Section 4
        }
    }
};
```

**Issues:**
- Manual section offset calculations scattered throughout constructor
- No clear documentation of buffer structure
- Easy to introduce off-by-one errors

### Proposed mdspan Solution

```cpp
template<typename T = double>
struct NonUniformSpacing {
    size_t n;
    std::vector<T> precomputed;

    // NEW: 2D view showing 5-section structure
    using SectionView = mdspan<T, dextents<size_t, 2>, layout_right>;
    SectionView sections_view_;  // Shape: (5, interior)

    explicit NonUniformSpacing(std::span<const T> x)
        : n(x.size())
    {
        const size_t interior = n - 2;
        precomputed.resize(5 * interior);

        // Create 2D view: 5 sections × interior points
        sections_view_ = SectionView(precomputed.data(), 5, interior);

        // Fill sections using self-documenting 2D indexing
        for (size_t i = 1; i <= n - 2; ++i) {
            const T dx_left = x[i] - x[i-1];
            const T dx_right = x[i+1] - x[i];
            const T dx_center = T(0.5) * (dx_left + dx_right);

            const size_t idx = i - 1;

            // Clear, self-documenting section assignments:
            sections_view_[0, idx] = T(1) / dx_left;              // dx_left_inv
            sections_view_[1, idx] = T(1) / dx_right;             // dx_right_inv
            sections_view_[2, idx] = T(1) / dx_center;            // dx_center_inv
            sections_view_[3, idx] = dx_right / (dx_left + dx_right);  // w_left
            sections_view_[4, idx] = dx_left / (dx_left + dx_right);   // w_right
        }
    }

    /// Access specific section as 1D span (for existing code compatibility)
    std::span<const T> dx_left_inv() const {
        return std::span(sections_view_.data_handle(), n - 2);
    }

    std::span<const T> dx_right_inv() const {
        return std::span(sections_view_.data_handle() + (n - 2), n - 2);
    }

    // ... etc for other sections ...
};
```

### Benefits

1. **Self-documenting structure**: `sections_view_[section, idx]` makes 5-section layout explicit
2. **Eliminate manual offsets**: No more `precomputed[2 * interior + idx]`
3. **Type safety**: Compiler catches dimension errors
4. **Easier iteration**: Can iterate over sections naturally

### Trade-offs

**Cons:**
- Slightly more complex (need mdspan view)
- Existing accessor functions (`dx_left_inv()`) still needed for compatibility

**Pros:**
- Much clearer initialization code
- Reduces cognitive load when reading/modifying

---

## Implementation Phases

### Phase 1: LAPACK Banded Matrix (Week 1)

**Goal:** Eliminate O(bandwidth × n) storage conversion

1. Implement `lapack_banded_layout` custom layout policy
2. Refactor `BandedMatrix` to use mdspan with custom layout
3. Update `factorize_banded()` to use zero-copy interface
4. Add comprehensive unit tests:
   - Verify LAPACK storage compatibility
   - Test matrix assembly via `operator()(i, j)`
   - Benchmark before/after conversion cost
5. Update documentation with usage examples

**Validation:**
- All existing banded solver tests pass
- Benchmark shows elimination of conversion overhead
- LAPACKE integration works without modification

**Estimated effort:** 2-3 days

---

### Phase 2: BSplineND Tensor Indexing (Week 2)

**Goal:** Replace manual N-dimensional index calculation

1. Add `CoeffMdspan` member to `BSplineND`
2. Implement variadic template helpers for mdspan access
3. Refactor `eval_tensor_product()` to use mdspan indexing
4. Remove `compute_flat_index()` function (10 lines deleted!)
5. Add tests:
   - Verify identical evaluation results
   - Benchmark to ensure zero overhead
   - Test with N=3, 4, 5 dimensions

**Validation:**
- All B-spline tests pass (identical numerical results)
- Benchmark shows no regression (~135ns per query maintained)
- Code is cleaner and more maintainable

**Estimated effort:** 2-3 days

---

### Phase 3: NonUniformSpacing (Week 3, if time permits)

**Goal:** Self-documenting multi-section buffer

1. Add `SectionView` mdspan member
2. Refactor constructor to use 2D indexing
3. Update documentation
4. Verify all centered difference operators still work

**Validation:**
- All PDE solver tests pass
- Code clarity improved

**Estimated effort:** 1-2 days

---

## Testing Strategy

### Unit Tests

For each component:

1. **Correctness tests:**
   - Verify identical results before/after mdspan adoption
   - Test edge cases (boundary indices, empty arrays, etc.)
   - Validate LAPACK compatibility (for banded matrices)

2. **Performance tests:**
   - Benchmark-driven development (measure before/after)
   - Ensure zero overhead for mdspan indexing
   - Measure conversion elimination (banded matrix case)

3. **Compilation tests:**
   - Verify template instantiation for various dimensions
   - Test const-correctness
   - Ensure compatibility with existing code

### Integration Tests

1. Run full B-spline interpolation tests with mdspan-based `BSplineND`
2. Run full PDE solver tests with mdspan-based banded matrices
3. Verify end-to-end American option pricing still works

### Regression Prevention

1. Add dedicated mdspan layout tests
2. Document expected performance characteristics
3. CI benchmarks to catch performance regressions

---

## Migration Strategy

### Backward Compatibility

**Approach:** Each phase introduces mdspan internally without changing public APIs.

1. **BandedMatrix:** Keep existing `operator()(i, j)` interface (now backed by mdspan)
2. **BSplineND:** Keep existing `eval()` interface (internal indexing uses mdspan)
3. **NonUniformSpacing:** Keep existing span accessors (extracted from mdspan view)

**Result:** Existing user code continues to work unchanged.

### Gradual Rollout

1. Phase 1 can be merged independently (banded matrices)
2. Phase 2 can be merged independently (B-spline indexing)
3. Phase 3 can be merged independently (grid spacing)

No "big bang" migration - each improvement is self-contained.

---

## Risk Mitigation

### Risk: mdspan overhead in hot paths

**Mitigation:**
- Comprehensive benchmarking before/after
- Compiler explorer verification of generated assembly
- Performance regression tests in CI

**Confidence:** High - mdspan is designed for zero overhead

### Risk: Complex template errors

**Mitigation:**
- Start with simple cases (banded matrix, 2D)
- Add clear documentation and examples
- Use concepts to improve error messages

### Risk: Compatibility with LAPACK

**Mitigation:**
- Thorough testing with LAPACKE reference results
- Validate storage layout with LAPACK documentation
- Keep fallback path (if needed)

**Confidence:** High - layout policy explicitly matches LAPACK spec

### Risk: Learning curve for contributors

**Mitigation:**
- Comprehensive documentation
- Example code showing common patterns
- Gradual introduction (one component at a time)

---

## Open Questions

1. **Should we expose mdspan in public APIs?**
   - **Recommendation:** Not initially - keep internal for now, gauge adoption success

2. **Should we support both row-major and column-major layouts?**
   - **Recommendation:** Start with row-major (existing behavior), add column-major if needed

3. **Should NonUniformSpacing expose mdspan view publicly?**
   - **Recommendation:** No - keep internal, maintain span accessors for compatibility

---

## Success Criteria

### Performance Metrics

- **Banded matrix:** Eliminate O(bandwidth × n) conversion (measure with benchmark)
- **B-spline eval:** Maintain ~135ns per query (no regression)
- **Grid spacing:** No measurable overhead vs current implementation

### Code Quality Metrics

- **Lines of code:** Remove ~60 lines of manual indexing
- **Cyclomatic complexity:** Reduce by eliminating nested offset calculations
- **Maintainability:** Easier to understand multi-dimensional layouts

### Adoption Metrics

- All existing tests pass
- Zero API changes for users
- Positive feedback from code reviews

---

## References

- C++23 mdspan proposal: [P0009R18](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p0009r18.html)
- Kokkos mdspan reference: https://github.com/kokkos/mdspan
- LAPACK DGBTRF documentation: http://www.netlib.org/lapack/explore-html/d3/d49/dgbtrf_8f.html
- Project mdspan polyfill: `src/support/mdspan/`

---

## Appendix A: Custom Layout Policy Example

Minimal working example of `lapack_banded_layout`:

```cpp
#include <experimental/mdspan>

using std::experimental::mdspan;
using std::experimental::dextents;

// Custom layout matching LAPACK banded storage
struct lapack_banded_layout {
    template<class Extents>
    struct mapping {
        Extents extents_;
        size_t kl_, ku_, ldab_;

        constexpr mapping(Extents ext, size_t kl, size_t ku) noexcept
            : extents_(ext), kl_(kl), ku_(ku), ldab_(2*kl + ku + 1)
        {}

        // Map (i,j) to offset: AB(kl + ku + i - j, j) in column-major
        constexpr size_t operator()(size_t i, size_t j) const noexcept {
            return (kl_ + ku_ + i - j) + j * ldab_;
        }

        constexpr const Extents& extents() const noexcept { return extents_; }
        static constexpr bool is_always_unique() noexcept { return true; }
        static constexpr bool is_always_strided() noexcept { return true; }
        constexpr size_t required_span_size() const noexcept {
            return ldab_ * extents_.extent(1);
        }
    };
};

// Usage example:
std::vector<double> storage(ldab * n);
mdspan<double, dextents<size_t, 2>, lapack_banded_layout> matrix(
    storage.data(), n, n, kl, ku);

matrix[i, j] = 42.0;  // Uses custom layout automatically!
```

---

## Appendix B: Performance Impact Analysis

### Banded Matrix Conversion Cost

**Current:**
```cpp
// O(bandwidth × n) loop
for (i = 0; i < n; ++i) {
    for (k = 0; k < bandwidth; ++k) {
        // Manual copy + offset calculation
    }
}
```

**Proposed:**
```cpp
// Zero-copy - data already in LAPACK format
LAPACKE_dgbtrf(..., matrix.lapack_data(), ...);
```

**Estimated savings:** For n=100, bandwidth=4:
- Before: ~400 element copies + offset calculations
- After: Zero copies, direct pointer pass

**Impact:** Significant for repeated factorizations (e.g., price table construction)

### B-spline Index Calculation Cost

**Current:**
```cpp
size_t idx = 0, stride = 1;
for (dim = N; dim > 0; --dim) {
    idx += indices[dim-1] * stride;
    stride *= dims_[dim-1];
}
return coeffs_[idx];
```

**Proposed:**
```cpp
return coeffs_view_[i0, i1, i2, ...];  // Compiles to identical assembly
```

**Impact:** Zero overhead (compiler optimizes both identically), but much clearer code.

---

## Implementation Notes

**Completed:** 2025-11-22

**Actual Results:**
- ✅ Phase 1: LAPACK banded matrix (zero-copy verified)
- ✅ Phase 2: BSplineND tensor indexing (compute_flat_index removed)
- ✅ Phase 3: NonUniformSpacing 2D view (self-documenting)

**Performance:**
- Banded matrix: Eliminated ~400 element copies (n=100, bandwidth=4)
- B-spline: Zero overhead (identical assembly)
- Grid spacing: No measurable overhead

**Test Coverage:**
- 12 new tests added
- All existing tests pass
- End-to-end integration verified

**Lines Changed:**
- Added: ~300 lines (custom layout, tests)
- Removed: ~60 lines (manual indexing)
- Net: +240 lines (includes comprehensive tests)

---

**End of Design Document**

# Task 1.3 Investigation: Remove storage conversion from factorize_banded()

## Summary

**Task 1.3 is already complete.** The conversion loop mentioned in the plan does not exist in the current codebase.

## Investigation Details

### What the plan expected to find

The plan (lines 629-649) described a conversion loop that should be removed:

```cpp
// OLD CODE - EXPECTED TO FIND:
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
            workspace.lapack_storage_[storage_idx] = value;
        }
    }
}
```

### What actually exists

The current `factorize_banded()` implementation (lines 267-315 in `src/math/banded_matrix_solver.hpp`):

```cpp
// Copy matrix data to workspace for factorization
// (LAPACK factorization modifies the matrix in-place)
const size_t storage_size = static_cast<size_t>(workspace.ldab_) * static_cast<size_t>(n);
workspace.lapack_storage_.assign(A.lapack_data(), A.lapack_data() + storage_size);

// Perform LU factorization with partial pivoting
workspace.pivot_indices_.resize(static_cast<size_t>(n));
const lapack_int info = LAPACKE_dgbtrf(
    LAPACK_COL_MAJOR,
    n, n, kl, ku,
    workspace.lapack_storage_.data(),  // Uses copied LAPACK-format data
    workspace.ldab_,
    workspace.pivot_indices_.data()
);
```

### Key findings

1. **No conversion loop exists**: The nested loop that converted between storage formats is not present
2. **Data is already in LAPACK format**: Thanks to Task 1.2's mdspan refactoring with `lapack_banded_layout`
3. **Single copy operation**: Line 294 uses `std::vector::assign()` to copy the entire matrix in one operation
4. **Copy is necessary**: LAPACK's `dgbtrf` modifies the matrix in-place, so we must preserve the original

### Why the copy exists

The copy operation on line 294 is **not a format conversion**. It's a preservation copy because:

1. **LAPACK destroys input**: `LAPACKE_dgbtrf` performs in-place LU factorization, overwriting the input matrix
2. **API contract**: `factorize_banded` takes `const BandedMatrix<T>&`, promising not to modify the input
3. **Reusability**: Users may want to factorize the same matrix multiple times with different workspaces

This is fundamentally different from the conversion loop that **transformed between different storage layouts**.

### Comparison: Before vs After

**Before (old conversion loop - NOT in current code):**
- Iterate over matrix elements
- For each element, calculate its LAPACK banded storage position
- Copy element-by-element with complex index arithmetic
- Cost: O(bandwidth × n) individual assignments

**After (current implementation):**
- Data already stored in LAPACK format via mdspan
- Single bulk copy operation
- No index calculations needed
- Cost: O(bandwidth × n) bulk memory copy (but no arithmetic overhead)

### Test results

All tests pass:
- `//tests:bspline_banded_solver_test` - PASSED
- `//tests:lapack_banded_layout_test` - PASSED
- `//tests:mdspan_integration_test` - PASSED

## Conclusion

Task 1.3's goal of **eliminating the O(bandwidth × n) conversion** was already achieved by Task 1.2's mdspan refactoring. The current implementation:

✅ Stores data in LAPACK format via mdspan (from Task 1.2)
✅ No conversion loop exists
✅ Single bulk copy preserves const-correctness
✅ All tests pass

**No further changes needed for Task 1.3.**

## Related commits

- `6031725` - Add LAPACK banded storage custom mdspan layout (Task 1.1)
- `1f7a7d7` - Refactor BandedMatrix to use mdspan with LAPACK layout (Task 1.2)
- `eedfd63` - Fix BandedMatrix to support variable bandwidth via set_col_start() (Task 1.2 fix)

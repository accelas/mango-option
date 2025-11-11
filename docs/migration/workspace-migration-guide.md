# Workspace Migration Guide

## Overview

The workspace memory management has been refactored to use modern C++23 features.
Old workspace classes are deprecated in favor of new PMR-based implementations.

## Migration Path

### WorkspaceStorage → PDEWorkspace

**Before:**
```cpp
#include "src/workspace.hpp"
WorkspaceStorage workspace(n, grid);
auto u = workspace.u_current();
```

**After:**
```cpp
#include "src/memory/pde_workspace.hpp"
PDEWorkspace workspace(n, grid);
auto u = workspace.u_current();  // Same API!
```

### For SIMD Kernels

New: Use padded accessors for vectorized operations:

```cpp
auto u_padded = workspace.u_current_padded();
auto lu_padded = workspace.lu_padded();
stencil.compute_second_derivative_tiled(u_padded, lu_padded, 1, n-1);
```

## API Compatibility

PDEWorkspace is API-compatible with WorkspaceStorage for:
- `u_current()`, `u_next()`, `u_stage()`, `rhs()`, `lu()`, `psi_buffer()`
- `dx()` - precomputed grid spacing

New accessors:
- `u_current_padded()` - SIMD-friendly padded span
- `dx_padded()` - SIMD-friendly grid spacing
- `tile_info(idx, num_tiles)` - operator tiling metadata

## Deprecation Timeline

- v2.5 (current): WorkspaceStorage marked deprecated, warnings issued
- v2.6: Remove WorkspaceStorage, PDEWorkspace becomes default
- v3.0: Remove all deprecated workspace code

## Questions?

See `docs/plans/2025-11-10-unified-memory-management-c++23-refactor.md`

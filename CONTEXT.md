# Mango Option

Mango Option is a pricing and implied-volatility library whose public surface spans a native C++ API, a parity-oriented Python API, and a focused Rust binding over the same option-pricing core.

## Language

**Python API parity**:
Every supported C++ capability is reachable from Python through an idiomatic Python surface, even when the Python shape does not mirror C++ templates or helper types one-to-one.
_Avoid_: Binding parity, wrapper parity, pybind coverage

**Rust binding (core)**:
A safe Rust surface (the `mango-option` crate over `mango-option-sys` and an `extern "C"` shim) exposing American option pricing with Greeks and FDM implied volatility, with full constant-rate/yield-curve and continuous/discrete-dividend fidelity. Unlike the Python API, it is a deliberately focused subset rather than full parity: price tables, interpolated IV, and batch solving are out of scope.
_Avoid_: Rust parity, full Rust API, GPU rewrite

**C++ capability**:
A documented, tested, or example-backed option-pricing, implied-volatility, interpolation, grid-control, or serialization workflow that the C++ API supports for library users.
_Avoid_: Public header symbol, internal type, implementation detail

**Price-table persistence**:
The ability to serialize and reload a precomputed interpolation surface so it can be reused without rebuilding the price table.
_Avoid_: Workspace save/load, solver checkpointing

**Reusable price table**:
A Python-facing object that contains a precomputed interpolation surface and can evaluate prices, Greeks, and implied volatility without rebuilding that surface.
_Avoid_: PriceTableData, workspace, solver checkpoint

**Price table factory**:
A Python-facing construction workflow that builds a reusable price table from one configuration object selecting backend, grid, adaptive refinement, and dividend behavior.
_Avoid_: Builder parity, template builder, per-backend constructor

**4D B-spline price table**:
A precomputed interpolation surface over log-moneyness, maturity, volatility, and rate using B-spline interpolation. This is a first-class price-table workflow and is not interchangeable with Chebyshev or dimensionless interpolation for parity acceptance.
_Avoid_: Generic interpolation backend, substitute backend, 3D table

**Binding contract**:
The Python-layer guarantee that stable C++ capabilities are reachable with correct object conversion, dispatch, persistence plumbing, and error mapping.
_Avoid_: Numerical validation, model correctness

**Binding reachability matrix**:
A Python test inventory that proves every stable parity method, config field, enum, persistence path, and error mapping is reachable from Python.
_Avoid_: Numerical benchmark, pricing validation suite

**Conversion coverage**:
Python binding tests that prove Python values convert to and from the expected C++ domain/config types at API boundaries.
_Avoid_: Algorithm validation, numerical tolerance suite

**Automatic conversion**:
Python-native values are accepted at API boundaries and converted into C++ domain/config types without requiring explicit wrapper construction for common cases.
_Avoid_: Manual wrapper-only API, C++-shaped ceremony

**Simple API**:
An experimental C++ convenience layer for market-data-oriented option chain and volatility-surface workflows.
_Avoid_: Stable pricing API, parity baseline

## Relationships

- **Python API parity** is measured against **C++ capabilities**, not against every C++ symbol.
- A **C++ capability** is established by public documentation, tests, or examples; public headers are used to discover possible gaps, not to define parity by themselves.
- A **C++ capability** may be represented by a different Python class or function when the C++ shape is template-heavy or otherwise non-idiomatic in Python.
- Runnable tests and examples take precedence over prose documentation when determining whether a workflow is currently supported; stale documentation is corrected rather than treated as a parity requirement.
- **Price-table persistence** is a current **C++ capability** because the data round-trip and Parquet file round-trip are covered by runnable tests.
- Python exposes **price-table persistence** through a **reusable price table** API, not by making `PriceTableData` the primary user-facing workflow.
- A **reusable price table** derives Greeks from the reconstructed interpolation surface; Greeks are not separately stored in persisted data.
- Python creates a **reusable price table** through a **price table factory**, not through direct one-to-one bindings for every C++ builder type.
- The **4D B-spline price table** path requires explicit C++ and Python reachability coverage, plus user-facing docs that name its four axes.
- The **price table factory** is the primary Python construction API; direct interpolated-IV solver construction remains as a backward-compatible convenience wrapper.
- A **reusable price table** supports both direct IV solving for convenience and creation of a reusable interpolated-IV solver for configured or batch workflows.
- New Python parity APIs raise typed module exceptions for construction, validation, persistence, and pricing failures; tuple-style result APIs are preserved only for backward compatibility.
- Numerical and model correctness belong to C++ tests; Python tests verify the **binding contract** and should not become the primary place to validate pricing or Greek algorithms.
- Python parity acceptance is measured with a **binding reachability matrix**, using cheap assertions about construction, field conversion, dispatch, persistence plumbing, result shape, and typed exceptions.
- **Conversion coverage** is required for stable parity APIs, especially unions/variants, optional fields, Python-native sequences, enums, nested configs, and persisted artifact boundaries.
- Stable Python APIs prefer **automatic conversion** for common Python-native values where pybind can support it cleanly.
- Dividend schedules accept both explicit `Dividend` objects and Python `(time, amount)` pairs.
- Top-level factory configuration uses explicit Python config objects; broad dict-based config construction is deferred.
- Persistence APIs accept both string paths and Python path-like objects.
- Stable Python parity does not require a hard numpy, pandas, pyarrow, or dataframe dependency.
- Removing the current hard numpy dependency is part of the Python parity cleanup unless a real bulk numeric API later justifies it.
- Experimental C++ APIs are excluded from the stable **Python API parity** baseline unless explicitly promoted.
- The **Simple API** is experimental and is excluded from the stable **Python API parity** baseline.

## Example Dialogue

> **Dev:** "Should Python expose every C++ template instantiation directly?"
> **Domain expert:** "No. We need **Python API parity**: Python users must be able to perform the same supported workflows through a **reusable price table**, but the API can be Pythonic."

## Flagged Ambiguities

- "feature parity" was used ambiguously between direct one-to-one pybind exposure and workflow-level parity. Resolved: use **Python API parity**.
- Documentation and executable behavior can disagree. Resolved: executable tests/examples establish current support; prose documentation is reconciled to match.
- `mango::simple` has public C++ targets and tests, but the intended status is experimental. Resolved: treat the **Simple API** as outside the stable parity baseline.
- Existing Python IV solvers use tuple-style `(success, result, error)` returns while new Pythonic APIs should raise typed module exceptions. Resolved: preserve existing tuple APIs for compatibility, but use exceptions for new parity APIs.
- Python persistence and Greek tests could duplicate numerical assertions owned by C++. Resolved: C++ tests own mathematical correctness; Python tests own binding reachability and plumbing.

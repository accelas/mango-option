# Opt-in modal interpolant checkpoint

`ChebyshevModalInterpolant<N>` is a typed adapter over the explicit
`detail::ChebyshevPolynomial`. It supports ranks 1–4, 2–257 points per axis,
and the polynomial's existing total coefficient capacity. The existing
financial aliases, factories, and certificate activation are unchanged.

`build_from_values(values, Domain<N>, num_pts)` explicitly converts ascending
CGL tensor samples using DCT-I. This defines a stored binary64 polynomial;
it does not promise exact equality to the old rounded-node barycentric
expression. `eval`, `partial`, and `eval_second_partial` all evaluate that
same polynomial. Finite coordinates follow its clamping convention, endpoint
derivatives use the polynomial's interior limit, and invalid coordinates or
derivative axes return NaN according to the polynomial interface.

`build_from_coefficients` restores modal coefficients verbatim, with no DCT,
resampling, rank reduction, or pruning of zero high modes. Coefficients
multiply `T_0, ..., T_n` without half factors. `domain()` and `num_pts()` retain
the supplied dimensions and bounds. `polynomial()` exposes a const reference
for the proof layer. All storage is owned; copying the adapter detaches the
coefficient vector and requires no numeric-clone specialization.

## Persistence

The existing `PriceTableData::Segment::interp_type` field has an explicit
`chebyshev_modal` tag. Its `values` array contains actual modal coefficients,
with full shape and original domain. The existing `chebyshev` tag continues
to contain nodal samples. `make_chebyshev_modal` and
`reconstruct_chebyshev_modal_leaf` accept only the modal tag; the existing
nodal reader rejects it. Conversely, the modal reader rejects nodal payloads.
Malformed shape/domain, nonfinite coefficients, and contradictory grids or
knots are refused.

Generic format-4 Parquet I/O already stores and checksums the representation
tag and arrays without numerical conversion. This checkpoint adds no schema
field and changes no existing tag's meaning, so the format remains 4.0.
Round-trip tests preserve coefficient bits, original ratio endpoints, strike
bounds, and fixed-expiry/dividend metadata. The original format-4 implementation
at `e9e10221` already makes `make_chebyshev` reject any tag other than exactly
`chebyshev`; all old 3D, 4D, and segmented Chebyshev `from_data` routes call that
helper. Generic `read_parquet` reads an opaque tagged `PriceTableData` without
interpreting its arrays, and subsequent old typed reconstruction explicitly
rejects the modal tag. There is no non-B-spline fallback to nodal interpretation
and no new top-level financial `from_data` dispatch;
callers must explicitly select the modal segment/leaf helper.

## Validation and remaining activation

The adapter and persistence seams have RED/GREEN tests for a tensor polynomial
and its physical derivatives, independent copy ownership, full 257-point shape,
and modal versus nodal restoration. Malformed payload guards and a format-4
Parquet round-trip are tested separately. The proof kernel continues to receive
the exact owned polynomial through its existing API.

Selecting this representation in financial surface aliases or factories,
publishing certificates, and qualifying financial accuracy/performance remain
separate integration work. This checkpoint supplies an unused coherent adapter
and persistence path, not a backend promotion or proof of nodal equivalence.

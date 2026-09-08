# Reference domain and numerical model metadata (#460)

A segmented table has a physical query domain in log-moneyness and absolute
strike. `SurfaceBounds::strike_bounds` is an inclusive `StrikeBounds {min,max}`;
a singleton is valid. Homogeneous continuous tables may omit this interval.
Segmented publication requires an explicit resolved interval. Extra reference
strikes are support, not permission to widen that interval.

Factory defaults derive the interval from the requested spot and S/K endpoints,
`[spot/max_ratio, spot/min_ratio]`, before PDE/dividend/headroom expansion.
`DiscreteDividendConfig::strike_bounds` and `SegmentedAdaptiveConfig::strike_bounds`
allow an independently requested interval. Explicit values are preserved after
validation. Reference-set validation and measured density selection use this
same resolved domain in the subsequent #460 selection slice.

At reference Ki, MultiKRefSplit maps `(S,K)` to `(S*Ki/K,Ki)`, preserving S/K.
Positive weights remain linear in absolute K and independent of sigma and S.
The split's linear spot Jacobian multiplies delta by Ki/K and gamma by its
square before value normalization. Vega, theta and rho receive no spot factor.
An exact reference query reads only that reference, avoiding zero-times-NaN
contamination from an inactive neighbor.

## Endpoint arithmetic and checked admission

PriceTable caches `exp(m_min)` and `exp(m_max)` and checks the physical spot
against `K*ratio_min` and `K*ratio_max`. The endpoint is that rounded quote-space
product. No epsilon expands the interval; the next representable spot outside
it is rejected. This avoids division/log round-trip artifacts for S=m*K at
published endpoints. Absolute strike comparisons are also inclusive and strict;
`nextafter` outside the stored interval is rejected.

AnyPriceTable validation and interpolated-IV admission use these same methods.
Proof/certification must cover the actual admitted quote-space domain, including
exp/log and product rounding, rather than assume a narrower exact-real log box.
Price and vega remain unchecked hot primitives with an admitted-query precondition.
Typed Greeks reject unsupported strike/moneyness/maturity inputs.

## Fixed-expiry provenance owner

PriceTable owns optional `FixedExpiryMetadata {reference_maturity,
discrete_dividends}`. The vector contains the canonical positive cash-dividend
offsets from this numerical anchor. It is not a wall-clock expiry. The anchor
may exceed the largest published query maturity. Segmented factories and the
Chebyshev manual builder populate it; the raw B-spline adaptive result carries
it for callers wrapping the raw inner surface in a PriceTable.

`PriceTable::fixed_expiry()` and `AnyPriceTable::fixed_expiry()` expose this
metadata. AnyPriceTable::Impl no longer owns an alternate dividend schedule.
InterpolatedIVSolver construction reads the table's anchor and schedule;
an explicit schedule override must match the known canonical model. A query's
schedule is rolled from reference_maturity, never guessed from tau_max.
Recognized segmented IV surfaces lacking fixed-expiry metadata are refused.

Serialization must copy both optional strike bounds and fixed-expiry metadata
through direct PriceTable to_data/from_data as well as AnyPriceTable save/load.
The persistence slice must reject missing/invalid segmented metadata, preserve
the numeric anchor even when tau_max is smaller, version the corrected mapping,
and bind all metadata to integrity checks. It must not infer metadata from the
reference hull or published maturity. Parquet format 3.0 stores both metadata fields with explicit presence markers.
Its payload CRC includes the presence markers, strike endpoints, anchor,
dividend count, and every canonical dividend time/amount. Older formats are
refused. The in-memory reconstruction and Parquet writer/reader share metadata
validation: segmented tables require both fields, a valid canonical schedule
and anchor, and reference support covering the declared strike interval.
The interval may be a singleton and may be narrower than the reference hull.
No clock timestamps or inferred anchors are introduced.

## Focused evidence

Public RED-to-GREEN cases cover reference endpoints/midpoints, moving spot and
strike, equal S/K at different K, exact and finite-difference Greeks, inactive
neighbors, absolute-domain admission, one-ULP endpoint rejection, and a model
anchor of2 with published tau_max1. The latter admits the correctly rolled
schedule and rejects a contradictory construction override. Python and Rust
binding builds pass at this checkpoint; final integrated checks follow the
remaining reference-selection and persistence slices.

## Shared reference request controls

`MultiKRefConfig` now carries exact optional `K_refs`, `max_references` and
`max_selection_rounds`. Limits include explicit sets and the seed candidate.
The shared resolver refuses nonfinite, nonpositive, duplicate, over-budget or
insufficient-coverage arrays before any PDE solve. It sorts a copy without
changing values. Automatic requests receive a covering seed derived from the
requested absolute interval, with no fixed count/span domain promise. A singleton
interval needs one reference; a nondegenerate interval needs at least two.
Build-time density measurement is a separate step from this cheap validation.

The retired count/span fields are removed from core C++/Python and the safe
Rust configuration. Until the coordinated C ABI revision, the existing ABI
slots retain their layout: automatic mode accepts only the legacy defaults
(count0 or11, span0.3), and rejects nondefault controls explicitly. Explicit
arrays do not use those old automatic-only fields. No slot is reinterpreted
as a new resource limit.

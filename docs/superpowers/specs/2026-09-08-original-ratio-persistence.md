# Original moneyness ratio persistence

The numerical query domain retains the caller's original S/K endpoints in
`MoneynessBounds`, separately from the log interpolation coordinates. For
example, `.1` does not survive `exp(log(.1))` bit-for-bit; reconstructing the
ratio from log metadata can reject the requested `S=10, K=100` endpoint.

`PriceTableData::ratio_bounds` is required for reconstructed tables. Tables
created directly from log domains resolve those ratios once, then persist the
resolved values explicitly. Parquet format 4.0 stores `mango.ratio_min` and
`mango.ratio_max`; both enter the metadata checksum. Missing/invalid ratio
metadata and older formats are rejected, following the approved #483 contract's
explicit compatibility policy. The domain is not guessed from support knots,
reference strikes, or repeated logarithm/exponential conversions.

The admission consistency enclosure is recomputed from the requested ratios
using `MoneynessDomain`. Its exact binary-product comparisons prevent extreme
quote rounding from admitting unrelated real S/K values. Numerical support and
requested-domain ownership remain separate. This metadata change does not
activate a financial monotonicity certificate; that proof and load-time
recertification are subsequent #459 work.

Both in-memory and Parquet tests pin original endpoint admission, missing
metadata rejection, and checksum coverage of valid-looking ratio mutations.

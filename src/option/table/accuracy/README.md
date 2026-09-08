# Internal measured-accuracy acceptance

`assess` is the private Gate 7 policy seam. It receives final-surface measured
errors, explicit targets/policy, and independently established viability and
physical-price proof status. It does not build, certify, or publish a table.

The default request requires maximum absolute price error <= 0.01 quote units
and actual IV error <= 2e-5 decimal volatility (0.2 absolute-IV basis points).
`Targets::iv = std::nullopt` requests price-only acceptance. RMS is reported
separately and never substitutes for the maximum. Evaluators must qualify
references and retain unresolved/refused/untested rows before calling policy.

The shared #460 channel vocabulary is reused, but final accuracy cannot count
ideal structural identities as measurements. `IvMetricKind` preserves whether
IV statistics are actual absolute-IV error, a price/vega proxy, or unknown.
Only the first can satisfy an IV target. An entirely filtered IV population
remains unmeasured. Mixed filtered/measured IV populations are allowed if all
requested rows are accounted for and the measurable rows meet the target.
Every requested price row still needs a measurement.

Strict requests refuse measured misses. Explicit best effort can accept those
misses with unmet-target flags, but cannot bypass missing requested evidence,
hard viability, or certification. `PriceProofStatus::NotRun`, `Indeterminate`,
and `NegativeWitness` remain distinct from `Certified`. A proof-status enum is
diagnostic information; it does not create a payload-bound admission token.

`Assessment` owns an immutable compact snapshot, including evidence on
refusal. There are no borrowed pointers or iteration histories. This C++ value
is not a wire format: persistence/ABI adapters must use versioned field-wise
encoding and bind historical measurements to their population, model, domain,
and coefficient payload. Loading must recompute the certificate.

Remaining activation seams:

- Route public factories/builders through final composed price and actual-IV
  evaluation on the frozen required population, with qualified independent
  references and per-stratum reports. Never promote #460's price/vega proxy.
- Supply real hard-viability results and the final payload-bound certificate;
  attach the resulting snapshot only after the public admission decision.
- Coordinate default target/policy settings and diagnostics with the active
  builders. No factory, adaptive-parameter, or BuildDiagnostics change is
  activated by this private module.
- Add versioned persistence and C/Rust copy-out adapters. Missing historical
  measurement remains unknown; no coefficient/domain identity is guessed.
- Run the fixed cohort and tune only after those evaluation/proof seams exist.

The focused tests exercise policy through `assess`, using declared measured
summaries. They do not establish numerical accuracy or certify any backend.

# Shared contract for #483

Read this file before implementing any sibling handoff. These are the agreed
policies for the epic's final state; each handoff identifies which gate owns
activation. An upstream gate is not required to implement downstream
certification, strict acceptance, or ABI publication merely to merge.

## 1. Product and ownership

The library accelerates trading-hour pricing/IV queries using tables prepared
after hours. Ordinary callers provide option/model specifications and the
requested query domain; numerical grids and default reference selection
belong to the numerical builders.

The core C++ contract is primary. The Simple API is a downstream convenience
layer and can be redesigned after the core settles. Fleet scheduling,
underlying counts, deployment hardware, expiry collections, wall-clock
conversion, cache refresh, and fallback decisions are caller concerns.

The engine returns explicit configuration, domain, validity, accuracy, or
identifiability failures. It does not silently start a new table build or a
PDE fallback inside an interpolation query.

## 2. Time semantics and reuse

Remaining maturity tau is time until expiry at the query's valuation point.

For time-homogeneous Black-Scholes with constant sigma, rate, and continuous
dividend yield, one PDE solve to the largest maturity per parameter
combination can serve all shorter maturities through snapshots. American
early exercise does not break that invariance. Calendar-dependent inputs
must not be treated as time-homogeneous merely because an interval contains
no event.

A dated-cash-dividend table represents a fixed expiry across its remaining
life. The caller selects the appropriate expiry-specific table and supplies
remaining maturity. The existing MultiKRefSplit and TauSegmentSplit remain
the composition mechanism. A core expiry-bank module or absolute clock input
is not required.

Let T0 be the remaining maturity at the table's numerical anchor, d_i a
dividend offset from that anchor, and tau a query's remaining maturity.
Elapsed time is T0 - tau. The query-relative dividend offset is:

    d_query_i = d_i - (T0 - tau)

Equivalently, the fixed dividend event coordinate is tau_i = T0 - d_i;
a future dividend appears tau - tau_i after query valuation. Filter and merge
the rolled schedule using the same canonical rules as direct pricing.
Keep the anchored schedule/reference horizon available where query validation
and serialization need this numerical contract.

An exact dividend-instant query uses the post-dividend side in calendar time.
Before/after calendar-event values are distinct; document their relation to
the backward solver's before/after callback order. Numerical gaps may not
silently replace the requested time with a neighbor's price. An unsupported
boundary/topology produces an explicit error.

Every stored sample must represent its labelled remaining maturity and event
side. Request exact mandatory sample times or verify the returned labels;
nearest-step snapping or deduplication is not an exact-node sampling policy.
A payoff row at tau=0 may be filled analytically for construction. This packet
does not add a new exact-expiry query API: current positive-maturity validation
and IV identifiability remain separate from that mathematical limit.

Automatic maturity partitioning and short-versus-long-horizon optimization are
future work. Current changes retain mathematically valid reuse and assess
the declared domain honestly.

## 3. Strike semantics

A segmented table supports variable spot over its declared log-moneyness
domain and an explicit supported absolute-strike interval. Build-time
coverage and query admission must use the same domain.

For a query (S,K) evaluated through reference strike K_i, preserve moneyness:

    S_i = S * K_i / K
    local strike = K_i

Blend normalized values using positive, sigma-independent weights linear in
strike, then restore quote units. Delta and gamma must include the spot-map
chain factors K_i/K and (K_i/K)^2. Cash dividends still break exact strike
homogeneity; the reference blend is an approximation subject to measurement.

Automatic K_refs derive their coverage from the requested strike domain and
their density from accuracy criteria within resource ceilings. Legacy fixed
count/span heuristics are not the domain contract; initial counts are
implementation tuning. Establish one resolver/validator across backends and
manual/adaptive entry points.

Explicit K_refs are used exactly after validation. Reject nonfinite,
nonpositive, duplicate, insufficient-coverage, or inadequate configurations;
do not silently substitute a different set.

## 4. Numerical defaults, explicit settings, and ceilings

Default density and clustering may adapt when measurements justify it.
Explicit manual grids, levels, and clustering overrides are constraints;
refuse when the requested criteria cannot be met. Adaptive seed knots may
receive additional knots within their stated ceilings.

Aggregation preserves actual refined positions. Use a sorted/deduplicated
union when it fits. On overflow, select deterministically within the cap,
preserving endpoints and required seed/event nodes, and validate bounded
candidates against the same reference data. Refuse if none is viable.
Initial assembly and retries obey the same rule.

MAX_WIDTH=5.8 is an optimization-routing heuristic, not a universal validity
limit. Assess the actual resolved grid and applicable numerical/accuracy
criteria. Preserve required coverage. Explicit maximum point counts are
ceilings, including odd-grid adjustments.

## 5. Price accuracy, IV accuracy, and identifiability

Standard proposed defaults agreed in the interview:

- Absolute price error: 0.01 in quote units, configurable.
- Absolute IV error: 2e-5 in decimal volatility, configurable.
- One absolute-IV basis point is 1e-4 in decimal volatility. Thus 2e-5 is
  0.2 bp, not 2 bps.

Use maximum absolute error over a predeclared validation population for
acceptance; report RMS separately. This is measured accuracy, not a uniform
mathematical error bound. Price checks apply even where IV metrics are
filtered. A root residual is not total price error or IV uncertainty.

Preserve the established short-dated policy from the Aug-29 adaptive-safety
design: TV/K below 1e-4 or reference vega below its floor excludes a point
from IV-error measurement. A filtered point is not a zero-error measurement.
Short positive maturity can be valid for pricing without supporting a
trustworthy or identifiable IV.

An adaptive build fails by default when it exhausts its budget without
meeting requested targets. Explicit best-effort policy may return a table
above a requested target only when hard viability and certification still
pass, with achieved errors/unmet targets clearly reported.

A price-only build may succeed when all IV probes are filtered if price
accuracy and certification pass. Mark IV accuracy unmeasured. An IV-targeted
build refuses an unmeasurable IV target. This distinction feeds the planned
accuracy-reporting/notification contract; a new per-query uncertainty interval
is outside this epic.

## 6. Certification and IV construction

Public price-table factory and loading paths, including manual builds,
enforce certification of the final physical price as nondecreasing in sigma
over the published domain. Raw mathematical interpolants can remain
unconstrained. Public interpolated-IV solver construction requires a
certified price surface.

Certify after all coordinate transforms, EEP add-backs/floors, temporal
routing, and reference-strike blending. A denser finite sample scan is not a
certificate. Distinguish a witnessed violation from failure to prove the
property within a budget.

Nondecreasing prices may have legitimate flat exercise regions. Preserve
identifiability/refusal checks and add a sensitivity check at the returned
IV. The current bracket-quartile maximum is not a final-root conditioning
check. Nonfinite sensitivities and insufficient root sensitivity are explicit
failures; no accuracy/confidence claim follows merely from successful root
finding.

Start #459 with post-fit certification/refusal. If corrected existing fits
cannot serve the required workload, a separate shape-constrained fitting
phase is authorized under the same price/IV criteria. Keep the full requested
domain; do not replace proof with a heuristic or silently narrow the domain.

Once mandatory certification and identifiability checks cover public IV
construction, retire the 17-point multiple-root screen and its
detect_multiple_roots option. Certification and final-root sensitivity are
mandatory behavior. Remove the toggle from the final C++/Python/C/Rust
contract as applicable; do not add it to the new C ABI.

## 7. Required and exploratory validation populations

Declare a fixed required supported cohort before tuning. Every required
pricing case meets its price target; every independently IV-measurable
required case meets its IV target. Expected unmeasurable/invalid cases report
the correct status. A backend's documented model restrictions remain explicit.

Use a separate exploratory stress cohort where refusals are permitted and
counted. Neither filtered points nor failed builds/queries may disappear
from denominators. Include endpoints, off-node values, off-K_ref strikes,
low volatility, maturity strata, both option types, supported rate regimes,
and both sides of relevant dividend events.

Use public price/Greek/query methods and independent or converged references.
Align the physical contract, event side, and maturity before comparing.
Report supported Greeks against their own declared regression criteria;
monotonicity does not certify Greek accuracy. Preserve existing tighter
regression tolerances. Numerical Greek tolerances not already pinned must
be established by reference convergence, not by fitting the threshold to
the implementation's error.

After accuracy passes, trading-hour query latency is the primary performance
objective. Report build time, memory, reference density, PDE counts, achieved
accuracy, and refusal rates together. Comparative backend promotion follows
measured, comparable workloads; no unapproved universal speed/error ratio
or deployment-specific deadline is inherited from older drafts.

## 8. Persistence and ABI

Compatibility is not a constraint for the coordinated C ABI revision or
saved-table changes needed by this contract. Update C/C++ and Rust together
with explicit format/ABI versions and layout checks. Older saved files missing
required information are rejected rather than guessed.

Persist necessary numerical model/domain metadata, including supported strike
bounds and fixed-expiry dividend offsets. No engine wall clock or fleet
configuration is required.

Persist a compact measured-accuracy summary tied to the stored model, domain,
and coefficient payload. Identify it as historical measurement; missing
evidence is explicitly unknown. Recompute monotonicity certification on load.
Full adaptive iteration histories may remain outside persistence.

Expose C diagnostics through caller-owned copy-out: a summary getter and
caller-buffer iteration copy with required-capacity reporting. No borrowed
pointers into temporary C++ results. Missing adaptive history is normal for
manual/loaded objects and is represented explicitly. Apply the same semantics
to table and IV-solver handles.

Keep the C/Rust backend scope focused on the currently supported backend;
this diagnostics revision does not silently add backend parity work.

## 9. Work discipline

Use TDD at the agreed public seams. Match each RED case to an independent
contract/reference or an exact mathematical invariant. Separate raw-sample,
fitting, composition, and IV errors so a downstream approximation does not
conceal a wrong contract.

The exact algorithm, private helper shape, and resource-efficient proof
method are agent decisions when they preserve this contract. Failed
characterization does not authorize weakening it.

Complete relevant tests for each slice. Before an implementation PR, run the
full suite and binding builds required by repository guidance, explicitly
build affected manual-tagged benchmarks, and report actual failures/warnings.
Measure performance on matched revisions/configurations when changing a hot
path. Test green does not establish that the measured population was
representative.

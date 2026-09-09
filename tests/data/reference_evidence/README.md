# Completed reference evidence regression

`earlier_prefixes.json` contains assessment-only extracts for the twenty frozen
#462 IDs whose v3 OR-filter replay stopped on an earlier qualified price while
the original v2 run had completed later, unresolved price evidence. Each pair
contains the early v3 record and the complete v2 record. Raw PDE observations,
vega bump observations and timing are omitted; source paths and SHA-256 hashes
identify the unchanged full source records. These fixtures do not regenerate or
change the 8,628-ID production population.

The original worker is d306ec1f; its effective binary/library hash is recorded in
each policy. The v3 replay driver is 523834a4. The numerical budgets are unchanged.
The three C2/CS/DQ PUT IDs ending `8c0e3eb1686ff95288c5`,
`db700bb90c1b614326f8`, and `3d3dc231f5341828292e` have later point observations
outside the earlier price allowance. The remaining seventeen have point
agreement but unresolved later convergence. Neither category may be consumed as
a qualified reference by selecting its earlier prefix.

`tools.reference_consumption.reconcile_reference` is a separate, in-memory
admission step for all completed direct-FDE records of one frozen ID. It does not
run a worker, change the qualification driver, or rewrite any ledger. Callers
must supply all completed evidence and verify source file hashes when loading
it. It refuses different frozen rows, worker/library identities, numerical
policies, and conflicting assessments for the same round. Missing future
evidence is not fabricated and the API does not require extra rounds.

The latest completed price assessment governs admission. If it is qualified,
its price and vega re-enter the existing IV OR rule. Earlier unresolved evidence
can recover; disjoint qualified allowances block consumption and require
investigation. A later unqualified point outside an earlier allowance is labeled
an observational contradiction, not a proof of the limiting PDE value.

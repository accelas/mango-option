# SPDX-License-Identifier: MIT
"""Frozen-output price feasibility only; no PDE, table fit, IV or Greek claim."""
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

root=Path(sys.argv[1]).resolve()
snapshot=root/'.cache/483-research/459-phase-b'
output=root/'.cache/483-research/459-zero-eep-floor'
output.mkdir(exist_ok=True)
original=json.loads((snapshot/'reference-hashes.json').read_text())
rows=[]
for identifier, expected in sorted(original.items()):
    path=snapshot/'references'/(identifier+'.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    reference=json.loads(path.read_text())
    row=reference['row']
    assert row['option_type']=='CALL' and row['dividend_yield']==0 and not row['rolled_dividends']
    S,K,T,sigma,r=(row[key] for key in ['spot','strike','maturity','volatility','rate'])
    d1=(math.log(S/K)+(r+.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    d2=d1-sigma*math.sqrt(T)
    euro=S*.5*math.erfc(-d1/math.sqrt(2))-K*math.exp(-r*T)*.5*math.erfc(-d2/math.sqrt(2))
    intrinsic=max(S-K,0.)
    candidate=max(euro,intrinsic)
    measured=reference['price_status']=='qualified'
    rows.append({'id':identifier,'physical':row,'candidate_price':candidate,'european_price':euro,
        'intrinsic':intrinsic,'active_branch':'intrinsic' if intrinsic>euro else 'European',
        'rate_regime':'negative' if r<0 else 'nonnegative',
        'ratio_regime':'below_.95' if S/K<.95 else 'above_1.05' if S/K>1.05 else 'central',
        'sigma_regime':'at_most_.1' if sigma<=.1 else 'above_.1',
        'tau_regime':'at_most_30days' if T<=30/365 else 'at_most_1year' if T<=1 else 'above_1year',
        'reference_price_status':reference['price_status'],'reference_iv_status':reference['iv']['status'],
        'reference_price':reference.get('price'),'reference_uncertainty':reference.get('price_uncertainty'),
        'signed_error':candidate-reference['price'] if measured else None,
        'absolute_error':abs(candidate-reference['price']) if measured else None,
        'target_miss':abs(candidate-reference['price'])>.01 if measured else None})
assert len(rows)==772

def summarize(population):
    observed=[row for row in population if row['absolute_error'] is not None]
    errors=[row['absolute_error'] for row in observed]
    return {'requested':len(population),'measured':len(observed),'unresolved':len(population)-len(observed),
        'target_misses':sum(row['target_miss'] for row in observed),
        'max_error':max(errors) if errors else None,
        'rms_error':math.sqrt(math.fsum(e*e for e in errors)/len(errors)) if errors else None,
        'max_signed_error':max((r['signed_error'] for r in observed),default=None),
        'min_distance_from_.01_threshold':min((abs(e-.01) for e in errors),default=None)}
report={'scope':'max(European,intrinsic) with zero fitted EEP; price-only frozen-output feasibility',
    'formula':'binary64 Black-Scholes via stdlib erfc; finite ordinary-scale physical rows',
    'targets_unchanged':{'price':.01,'actual_iv':2e-5},'full':summarize(rows),
    'reference_iv_counts':dict(collections.Counter(row['reference_iv_status'] for row in rows)),
    'groups':{key:{value:summarize([r for r in rows if r[key]==value]) for value in sorted({r[key] for r in rows})}
              for key in ['rate_regime','active_branch','ratio_regime','sigma_regime','tau_regime','reference_iv_status']},
    'worst_rows':sorted((r for r in rows if r['absolute_error'] is not None),key=lambda r:r['absolute_error'],reverse=True)[:20],
    'no_pde_solves':True,'no_candidate_build':True,'no_actual_iv_or_greek_or_whole_shape_claim':True,
    'all772_original_reference_hashes_match':True}
(output/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='worst_rows'},indent=2))
print('worst',report['worst_rows'][0]['id'],report['worst_rows'][0]['signed_error'])

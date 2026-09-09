# SPDX-License-Identifier: MIT
"""Summarize the fixed21 reference rows and a separate rate-basis identity."""
import collections
import json
import math
from pathlib import Path
import sys
import numpy as np
root=Path(sys.argv[1]);p=root/'.cache/483-research/459-rate-join-21'
rows=[json.loads(f.read_text()) for f in sorted((p/'cases').glob('*.json'))]
assert len(rows)==21
summary={'requested':21,'price_status':dict(collections.Counter(r['price_status'] for r in rows)),
         'iv_status':dict(collections.Counter(r['iv']['status'] for r in rows)),
         'primary_required_population':772,'primary_population_modified':False,'rho_secants':{}}
for anchor in ['ATM','ITM_LOW','ITM_MID']:
    indexed={r['row']['rate']:r for r in rows if r['row']['anchor']==anchor}
    zero=indexed[0.]
    secants=[]
    for h in [.001,.0001,.00001,.000001,.0000001]:
        ref=indexed[-h]
        if ref['price_status']!='qualified':
            secants.append({'h':h,'status':'reference-unresolved','value':None,'uncertainty':None})
            continue
        secants.append({'h':h,'status':'qualified-endpoint-secant-not-point-rho',
            'value':(zero['price']-ref['price'])/h,
            'uncertainty':(zero['price_uncertainty']+ref['price_uncertainty'])/h})
    summary['rho_secants'][anchor]={'analytic_right_rho':zero['greeks']['rho'],'secants':secants,
        'point_left_rho_qualification':'not_established_by_this_stencil'}
# Exact cubic-Hermite identity at the midpoint of the unsampled last cell.
# Source rows are the already sampled four negative rates. No new FDE rows.
samples=np.loadtxt(root/'.cache/483-research/459-constrained-c0/candidate.samples.tsv')
axes=np.loadtxt(root/'.cache/483-research/459-constrained-c0/candidate.axes.tsv')
gx=axes[axes[:,0]==0,2];gt=axes[axes[:,0]==1,2];gs=axes[axes[:,0]==2,2]
i=int(np.argmin(abs(gx-math.log(1.3))));j=int(np.argmin(abs(gt-2)));k=int(np.argmin(abs(gs-.05)))
assert abs(math.exp(gx[i])-1.3)<1e-14 and gt[j]==2 and gs[k]==.05
block=samples[(samples[:,0]==i)&(samples[:,1]==j)&(samples[:,2]==k)]
assert len(block)==4
weights=np.array([-2.,9.,-18.,35.])/48.
def european(rate):
    S,K,T,sigma=130.,100.,2.,.05
    d1=(math.log(S/K)+(rate+.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    d2=d1-sigma*math.sqrt(T)
    return S*.5*math.erfc(-d1/math.sqrt(2))-K*math.exp(-rate*T)*.5*math.erfc(-d2/math.sqrt(2))
rates=[-.05,-.0375,-.025,-.0125]
q=-.00625
price=european(q)+max(0.,float(weights@block[:,4]))
ref=next(r for r in rows if r['row']['anchor']=='ITM_LOW' and r['row']['rate']==q)
zero=next(r for r in rows if r['row']['anchor']=='ITM_LOW' and r['row']['rate']==0)
# q0 discounted-payoff coupling makes American price nondecreasing in r.
# Thus for all these negative rates, intrinsic<=American(r)<=European(0).
# Every acceptable table source quote implies a corresponding EEP interval.
intrinsic=30.
upper=zero['price']+zero['price_uncertainty']
target=.01
lo=np.array([intrinsic-european(r)-target for r in rates])
hi=np.array([upper-european(r)+target for r in rates])
premium_lo=float(np.sum(np.where(weights>=0,weights*lo,weights*hi)))
premium_hi=float(np.sum(np.where(weights>=0,weights*hi,weights*lo)))
assert premium_lo>0
bound={'scope':'Separate mathematical capacity diagnostic, not a candidate table or sigma certificate',
       'composition':'European + max(0, raw EEP); no physical intrinsic floor',
       'intrinsic_floor_limitation':'A physical intrinsic floor would evade this specific obstruction; not analyzed as a new candidate',
       'physical_anchor':{'spot':130,'strike':100,'tau':2,'sigma':.05,'q':0,'type':'CALL'},
       'existing_source_rates':rates,'query_rate':q,'weights':weights.tolist(),
       'sum_abs_weights':float(abs(weights).sum()),
       'line_interpolation_price':price,'qualified_reference_price':ref['price'],
       'reference_uncertainty':ref['price_uncertainty'],'line_error':price-ref['price'],
       'source_true_price_enclosure':[intrinsic,upper],
       'allowed_source_price_error':target,
       'implied_query_price_interval_if_all_source_targets_met':[european(q)+premium_lo,european(q)+premium_hi],
       'reference_target_interval':[ref['price']-ref['price_uncertainty']-target,ref['price']+ref['price_uncertainty']+target],
       'European_math':'ordinary binary64 Black-Scholes add-back for capacity arithmetic; not a new American oracle',
       'feasible_intersection':european(q)+premium_hi>=ref['price']-ref['price_uncertainty']-target}
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(p/'minimal-basis-capacity.json').write_text(json.dumps(bound,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rho_secants'},indent=2))
print(json.dumps(bound,indent=2))

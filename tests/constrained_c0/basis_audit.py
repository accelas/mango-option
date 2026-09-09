# SPDX-License-Identifier: MIT
"""Capacity-only audit of existing samples, not another constructed candidate."""
import json
import math
from pathlib import Path
import sys
import numpy as np

artifact = Path(sys.argv[1])
plan = json.loads((artifact/'basis-audit-plan.json').read_text())
samples = np.loadtxt(artifact/plan['source_samples'])
axes = np.loadtxt(artifact/'candidate.axes.tsv')
grids = [axes[axes[:,0]==d,2] for d in range(3)]
sigma_matrix = np.loadtxt(artifact/'candidate.matrix2.tsv')
rates = [-.05,-.0375,-.025,-.0125]

def basis(knots, i, degree, x):
    if degree == 0:
        return float(knots[i] <= x < knots[i+1])
    left = right = 0.
    if knots[i+degree] > knots[i]:
        left = (x-knots[i])/(knots[i+degree]-knots[i])*basis(knots,i,degree-1,x)
    if knots[i+degree+1] > knots[i+1]:
        right = (knots[i+degree+1]-x)/(knots[i+degree+1]-knots[i+1])*basis(knots,i+1,degree-1,x)
    return left+right

results = []
for variant in plan['variants']:
    dimension = variant['negative_free_coefficients']
    knots = [-.05]*4+[plan['fixed_negative_knot']]*variant['new_knot_multiplicity']+[0]*2+[.1]*4
    rate_matrix = np.array([[basis(knots,i,3,r) for i in range(dimension)] for r in rates])
    matrix = np.array([np.kron(sigma_matrix[s], rate_matrix[r]) for s in range(5) for r in range(4)])
    records, residual_rows = [], []
    for i in range(len(grids[0])):
        for j in range(len(grids[1])):
            block = samples[(samples[:,0]==i)&(samples[:,1]==j)]
            coefficients = np.linalg.lstsq(matrix,block[:,4],rcond=None)[0]
            residual = matrix@coefficients-block[:,4]
            ratio = math.exp(grids[0][i])
            inside = .7 <= ratio <= 1.3
            records.append({'x_index':i,'tau_index':j,'ratio':ratio,'tau':float(grids[1][j]),
                            'inside_domain':inside,'rms':float(np.sqrt(np.mean(residual**2))),
                            'max':float(abs(residual).max())})
            for sample, error in zip(block,residual):
                if inside:
                    residual_rows.append({'ratio':ratio,'tau':float(grids[1][j]),
                        'sigma':float(grids[2][int(sample[2])]),'rate':rates[int(sample[3])],
                        'error':float(error)})
    residual = np.array([r['error'] for r in residual_rows])
    results.append({'variant':variant,'rate_knots':knots,'rank':int(np.linalg.matrix_rank(matrix)),
                    'condition':float(np.linalg.cond(matrix)),
                    'inside_count':len(residual_rows),'inside_rms':float(np.sqrt(np.mean(residual**2))),
                    'inside_max':float(abs(residual).max()),
                    'worst_inside_block':max((r for r in records if r['inside_domain']),key=lambda r:r['rms']),
                    'worst_inside_row':max(residual_rows,key=lambda r:abs(r['error']))})
(artifact/'basis-audit.json').write_text(json.dumps({'scope':plan['scope'],'results':results},indent=2)+'\n')
print(json.dumps(results,indent=2))

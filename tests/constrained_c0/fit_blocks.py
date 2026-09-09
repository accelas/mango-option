# SPDX-License-Identifier: MIT
"""Private bounded research solver; no production dependency."""
import numpy as np

def constrained_least_squares(matrix, values, cuts, lower, max_iterations=100):
    matrix, values, cuts, lower = map(np.asarray, (matrix, values, cuts, lower))
    n = matrix.shape[1]
    if not all(np.isfinite(x).all() for x in (matrix, values, cuts, lower)):
        return {'status': 'nonfinite_input'}
    if np.linalg.matrix_rank(matrix) != n:
        return {'status': 'rank_deficient'}
    # Zero premium has nonnegative total European vega, hence is the fixed
    # feasible starting point for these physical derivative constraints.
    if (lower > 0).any():
        return {'status': 'unsupported_initial_feasibility'}
    norms = np.linalg.norm(cuts, axis=1)
    keep = norms > 0
    g = cuts[keep] / norms[keep, None]
    b = lower[keep] / norms[keep]
    h = matrix.T @ matrix
    f = matrix.T @ values
    x = np.zeros(n)
    active = []
    tolerance = 1e-10
    for iteration in range(max_iterations):
        gradient = h @ x - f
        ga = g[active]
        kkt = np.block([[h, -ga.T], [ga, np.zeros((len(active),len(active)))]])
        try:
            direction = np.linalg.solve(kkt, np.r_[-gradient, np.zeros(len(active))])
        except np.linalg.LinAlgError:
            return {'status': 'singular_active_set', 'iterations': iteration+1}
        step, multipliers = direction[:n], direction[n:]
        if np.linalg.norm(step, np.inf) <= tolerance * max(1.,np.linalg.norm(x,np.inf)):
            if len(active) and multipliers.min() < -tolerance:
                active.pop(int(multipliers.argmin()))
                continue
            residual = g @ x - b
            if residual.size and residual.min() < -tolerance:
                return {'status':'infeasible_numerical_result','iterations':iteration+1,
                        'min_normalized_slack': float(residual.min())}
            return {'status':'solved','coefficients':x,'iterations':iteration+1,
                    'min_normalized_slack': float(residual.min()) if residual.size else 0.,
                    'active_constraints':len(active),
                    'condition':float(np.linalg.cond(matrix)),
                    'sample_rms':float(np.linalg.norm(matrix@x-values)/np.sqrt(len(values)))}
        slack = g @ x - b
        velocity = g @ step
        alpha, blocker = 1., None
        for i in range(len(b)):
            if i in active or velocity[i] >= 0: continue
            candidate = max(0.,slack[i]) / -velocity[i]
            if candidate < alpha:
                alpha, blocker = candidate, i
        x += alpha*step
        if blocker is not None:
            proposed = g[active+[blocker]]
            if np.linalg.matrix_rank(proposed) != len(active)+1:
                return {'status':'dependent_blocking_constraints','iterations':iteration+1}
            active.append(blocker)
    return {'status':'iteration_limit','iterations':max_iterations}

if __name__ == '__main__':
    import json
    import sys
    import time
    from pathlib import Path
    started = time.monotonic()
    prefix = Path(sys.argv[1])
    def path(suffix): return Path(str(prefix)+suffix)
    samples = np.loadtxt(path('.samples.tsv'))
    cuts_data = np.loadtxt(path('.cuts.tsv'))
    sigma_data = np.loadtxt(path('.sigma_cuts.tsv'))
    matrices = [np.loadtxt(path('.matrix'+str(d)+'.tsv')) for d in range(3)]
    nx, nt, ns = [m.shape[0] for m in matrices]
    assert ns == 5
    def rate_basis(rate):
        t = -rate/.05
        return np.array([t**3, 3*t*t*(1-t)])
    rates = [-.05,-.0375,-.025,-.0125]
    matrix = np.array([np.kron(matrices[2][s], rate_basis(r))
                       for s in range(ns) for r in rates])
    block_coefficients = np.zeros((nx,nt,ns,2))
    rows = []
    failures = 0
    for i in range(nx):
        for j in range(nt):
            observations = samples[(samples[:,0]==i)&(samples[:,1]==j)]
            constraints = cuts_data[(cuts_data[:,0]==i)&(cuts_data[:,1]==j)]
            assert len(observations)==20 and len(constraints)==81
            g = np.array([np.kron(sigma_data[int(c[2]),1+ns:],rate_basis(c[4]))
                          for c in constraints])
            result = constrained_least_squares(matrix,observations[:,4],g,-constraints[:,5])
            coefficients = result.pop('coefficients',None)
            if coefficients is not None:
                block_coefficients[i,j] = coefficients.reshape(ns,2)
                result['min_physical_vega_cut'] = float((g@coefficients+constraints[:,5]).min())
            else:
                failures += 1
            rows.append({'x_index':i,'tau_index':j,**result})
    report = {'blocks_requested':nx*nt,'blocks_failed':failures,'blocks':rows,
              'rate_grid':[-.05,-.0375,-.025,0,.05,.1],
              'rate_knots':[-.05]*4+[0,0]+[.1]*4,
              'constraints_per_block':81,'unknowns_per_block':10,
              'samples_per_block':20,'fit_seconds':time.monotonic()-started}
    path('.fit.json').write_text(json.dumps(report,indent=2)+'\n')
    if failures:
        print(json.dumps({k:v for k,v in report.items() if k!='blocks'}))
        sys.exit(3)
    # Linear remaining-axis fits preserve the zero rate coefficient planes.
    values = np.linalg.solve(matrices[0], block_coefficients.reshape(nx,-1)).reshape(nx,nt,ns,2)
    values = np.linalg.solve(matrices[1],values.transpose(1,0,2,3).reshape(nt,-1)).reshape(nt,nx,ns,2).transpose(1,0,2,3)
    coefficients = np.zeros((nx,nt,ns,6))
    coefficients[:,:,:,:2] = values
    assert np.isfinite(coefficients).all()
    np.savetxt(path('.coefficients.tsv'),coefficients.ravel(),fmt='%.17g')
    report['coefficients'] = int(coefficients.size)
    report['fit_seconds'] = time.monotonic()-started
    report['remaining_axis_condition'] = [float(np.linalg.cond(x)) for x in matrices[:2]]
    path('.fit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='blocks'}))

# SPDX-License-Identifier: MIT
"""One fixed candidate, with a total construction execution limit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
artifact = root / '.cache/483-research/459-constrained-c0'
if (artifact / 'execution.json').exists():
    raise SystemExit('Preserved attempt exists; use a separately authorized versioned experiment.')
prefix = artifact / 'candidate'
baseline = root / '.cache/483-research/459-phase-a/C0-CALL.parquet'
worker = Path('bazel-bin/tests/constrained_c0/worker').resolve()
commands = [('sampling', [str(worker),'sample',str(baseline),str(prefix)]),
            ('fitting', [sys.executable,'tests/constrained_c0/fit_blocks.py',str(prefix)]),
            ('composition',[str(worker),'compose',str(baseline),str(prefix),str(prefix)+'.coefficients.tsv'])]
env = dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
started = time.monotonic()
results = []
for stage, command in commands:
    before = time.monotonic()
    remaining = 600-(before-started)
    if remaining <= 0:
        results.append({'stage':stage,'status':'unfinished_execution_limit'})
        break
    with (artifact/(stage+'.log')).open('w') as log:
        try:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=remaining)
            status = 'completed' if result.returncode==0 else 'failed'
            row = {'stage':stage,'status':status,'exit_code':result.returncode}
        except subprocess.TimeoutExpired:
            row = {'stage':stage,'status':'unfinished_execution_limit'}
    row['seconds'] = time.monotonic()-before
    row['command'] = command
    results.append(row)
    (artifact/'execution.json').write_text(json.dumps({'stages':results,'seconds':time.monotonic()-started},indent=2)+'\n')
    print(json.dumps(row),flush=True)
    if row['status']!='completed': break
(artifact/'execution.json').write_text(json.dumps({'stages':results,'seconds':time.monotonic()-started},indent=2)+'\n')
sys.exit(0 if results[-1]['stage']=='composition' and results[-1]['status']=='completed' else 1)

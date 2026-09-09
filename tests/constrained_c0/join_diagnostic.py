# SPDX-License-Identifier: MIT
"""Independent exploratory rate-join observations using the approved oracle."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

root=Path(sys.argv[1]).resolve()
artifact=root/'.cache/483-research/459-rate-join-21'
if (artifact/'execution.json').exists():
    raise SystemExit('Completed study is immutable; use a separately versioned study.')
manifest=json.loads((artifact/'manifest.json').read_text())
driver=root/'.cache/483-research/462-references-v3-or-filter/driver/reference_qualification.py'
worker_path=(root/'.worktrees/483-462-centered-references/bazel-bin/benchmarks/reference_oracle_worker').resolve()
for name,path in [('driver',driver),('worker',worker_path)]:
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['reference'][name+'_sha256']
spec=importlib.util.spec_from_file_location('join_reference_qualification',driver)
qualification=importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)
lib_hashes=qualification.runtime_libraries(str(worker_path))
effective_hash=qualification.digest(qualification.encoded({'binary':manifest['reference']['worker_sha256'],'libraries':lib_hashes}).encode())
meta={'effective_worker_hash':effective_hash,'runtime_libraries':lib_hashes,
      'worker_version':json.loads(subprocess.check_output([worker_path,'--version'],text=True)),
      'driver':str(driver),'worker':str(worker_path),'settings':manifest['reference']}
qualification.atomic_json(artifact/'metadata.json',meta)
cache=sqlite3.connect(artifact/'samples.sqlite',check_same_thread=False)
cache.execute('CREATE TABLE IF NOT EXISTS samples(key TEXT PRIMARY KEY, request TEXT, response TEXT)')
lock=threading.Lock()
args=SimpleNamespace(rounds=3,quantlib=False,vega_bump_fraction=.04,audit_analytic=False)
started=time.monotonic()

def assess(row):
    worker=qualification.Worker(str(worker_path),cache,lock,effective_hash)
    try:
        result=qualification.qualify(worker,row,args)
    except Exception as error:
        result={'id':row['id'],'row':row,'status':'oracle-unresolved','price_status':'oracle-unresolved',
                'iv':{'status':'oracle-unresolved'},'error':str(error),'sample_keys':sorted(worker.sample_keys)}
    finally:
        worker.close()
    result['effective_worker_hash']=effective_hash
    qualification.atomic_json(artifact/'cases'/(row['id']+'.json'),result)
    return {'id':row['id'],'status':result['status'],'price_status':result['price_status'],
            'iv':result['iv']['status'],'price':result.get('price'),'uncertainty':result.get('price_uncertainty')}

with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(assess,row) for row in manifest['rows']]
    for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
cache.close()
qualification.atomic_json(artifact/'execution.json',{'status':'complete','rows':21,'seconds':time.monotonic()-started})

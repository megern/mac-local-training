"""Frozen three-seed adapter residency study; random tiny models are software checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    import psutil
    root=Path(__file__).resolve().parent
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    cells=[{'seed':seed,'strategy':strategy} for seed in (11,29,47) for strategy in ('all','rotating')]
    random.Random(20261011).shuffle(cells)
    sources=[root/'mac_local_training/engine.py',root/'mac_local_training/cli.py',Path(__file__).resolve(),root/'examples/train.jsonl',root/'examples/valid.jsonl']
    protocol={'planned_at_unix':time.time(),'cells':cells,'steps':16,'accumulation_steps':2,'max_seq_length':32,
              'source_sha256':{str(path.relative_to(root)):sha(path) for path in sources},
              'scope':'Three fixed seeds, random tiny models and synthetic token data. Matched data/steps, different layer update coverage. No useful model-quality, generalization or total-RAM savings claim.',
              'guards':{'process_tree_rss_gib':6,'system_swap_growth_mib':1536,'timeout_seconds':180}}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    results=[]
    for cell in cells:
        name=f"seed{cell['seed']}-{cell['strategy']}";cell_output=output/name
        command=[sys.executable,'-m','mac_local_training.cli','train','--tiny','--train-data',str(root/'examples/train.jsonl'),'--valid-data',str(root/'examples/valid.jsonl'),'--output',str(cell_output),'--strategy',cell['strategy'],'--seed',str(cell['seed']),'--steps','16','--accumulation-steps','2','--max-seq-length','32','--checkpoint-every','16']
        environment=dict(os.environ,PYTHONPATH=str(root),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        baseline=psutil.swap_memory().used;started=time.monotonic();peak=0;guard=None
        with (output/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
            while child.poll() is None:
                try:
                    process=psutil.Process(child.pid);rss=sum(p.memory_info().rss for p in [process]+process.children(recursive=True));peak=max(peak,rss)
                except psutil.Error:rss=0
                if rss>6*1024**3:guard='process tree RSS above 6 GiB'
                elif psutil.swap_memory().used-baseline>1536*1024**2:guard='whole-system swap growth over 1536 MiB; not causal attribution'
                elif time.monotonic()-started>180:guard='180 second timeout'
                if guard:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                time.sleep(.2)
        row={'cell':cell,'exit_code':child.returncode,'guard_stop':guard,'process_tree_rss_peak_bytes':peak,'wall_seconds':time.monotonic()-started}
        if child.returncode==0:
            report_path=cell_output/'report.json';row['report']=json.loads(report_path.read_text());row['report_sha256']=sha(report_path)
        results.append(row)
        (output/'study.json').write_text(json.dumps({'protocol_sha256':sha(output/'protocol.json'),'results':results,'scope':protocol['scope']},indent=2)+'\n')
        if child.returncode!=0:raise RuntimeError('Study stopped; retain journal and inspect '+name)
    if any(sha(root/name)!=expected for name,expected in protocol['source_sha256'].items()):raise ValueError('Frozen source changed')
    return {'completed':len(results),'output':str(output)}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    print(json.dumps(run(parser.parse_args().output)))

"""Future-run supervisor: preserve device failures, logs and whole-process time.

Only the specific Metal interactivity failure is retryable, at most once.
Numerical, source-integrity and other errors stop the study without a retry.
This is a transparent execution addendum, never a change to acceptance criteria.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from endpoint_study import digest, verify, write

SIGNATURE = 'kIOGPUCommandBufferCallbackErrorImpactingInteractivity'


def retryable(returncode, stderr, attempts, max_retries):
    return returncode != 0 and SIGNATURE in stderr and attempts <= max_retries


def supervise(root, worker, max_retries=1):
    root, worker = root.resolve(), worker.resolve()
    if max_retries not in (0, 1):
        raise ValueError('Retry bound must be zero or one')
    protocol_path = root / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    verify(protocol)
    expected = protocol['sources_sha256'].get(f'studies/{worker.name}')
    if expected != digest(worker):
        raise ValueError('Worker is not a frozen study source')
    logs = root / 'supervisor'
    logs.mkdir(exist_ok=True)
    journal = logs / 'attempts.json'
    if journal.exists():
        ledger = json.loads(journal.read_text())
        if ledger['protocol_sha256'] != digest(protocol_path) or ledger['max_retries'] != max_retries:
            raise ValueError('Supervisor identity changed')
    else:
        ledger = {'protocol_sha256': digest(protocol_path), 'max_retries': max_retries,
                  'rule': 'Retry only the exact Metal interactivity error. Keep all failed attempts; do not retry unfavorable results.',
                  'worker_sha256': digest(worker), 'attempts': []}
        write(journal, ledger)
    for item in protocol['runs']:
        directory = root / 'runs' / item['id']
        if (directory / 'result.json').exists():
            result = json.loads((directory / 'result.json').read_text())
            if result['protocol_sha256'] != digest(protocol_path) or result['run'] != item:
                raise ValueError('Completed result identity changed')
            continue
        previous = [row for row in ledger['attempts'] if row['run_id'] == item['id']]
        if directory.exists():
            raise ValueError('Unjournaled partial run exists; preserve it and use a new study')
        attempt = len(previous) + 1
        if attempt > max_retries + 1:
            raise ValueError('Retry budget already exhausted')
        while True:
            prefix = f"{item['id']}-attempt-{attempt}"
            stdout_path, stderr_path = logs / f'{prefix}.stdout.log', logs / f'{prefix}.stderr.log'
            started = time.perf_counter()
            with stdout_path.open('x') as out, stderr_path.open('x') as err:
                child = subprocess.run([sys.executable, str(worker), 'worker', '--output', str(root), '--run-id', item['id']], stdout=out, stderr=err)
            elapsed = time.perf_counter() - started
            stderr = stderr_path.read_text()
            row = {'run_id': item['id'], 'attempt': attempt, 'returncode': child.returncode,
                   'whole_process_seconds': elapsed,
                   'stdout_sha256': digest(stdout_path), 'stderr_sha256': digest(stderr_path),
                   'retryable_metal_signature': SIGNATURE in stderr}
            ledger['attempts'].append(row)
            write(journal, ledger)
            print(json.dumps(row), flush=True)
            if child.returncode == 0:
                if not (directory / 'result.json').exists():
                    raise ValueError('Successful process omitted result')
                break
            if not retryable(child.returncode, stderr, attempt, max_retries):
                raise RuntimeError(f"Study stopped at {item['id']}; preserved logs in {logs}")
            archive = root / 'failures' / prefix
            archive.parent.mkdir(exist_ok=True)
            if directory.exists():
                directory.rename(archive)
            # Cooldown happens outside every measured worker interval.
            time.sleep(10)
            attempt += 1
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--max-retries', type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    supervise(args.output, args.worker, args.max_retries)


if __name__ == '__main__':
    main()

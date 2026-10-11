"""Audit bounded retries and all whole-process durations for supervised studies."""
import argparse
import json
import math
from pathlib import Path
from endpoint_study import digest, verify, write


def audit(root):
    protocol_path = root / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    verify(protocol)
    policy = protocol['execution_policy']
    journal = root / 'supervisor/attempts.json'
    ledger = json.loads(journal.read_text())
    if ledger['protocol_sha256'] != digest(protocol_path) or ledger['max_retries'] != policy['max_retries']:
        raise ValueError('Execution policy/protocol mismatch')
    worker = policy['supervisor']  # Both supervisor and worker must be hash pinned.
    if worker not in protocol['sources_sha256']:
        raise ValueError('Supervisor was not frozen')
    workers = [name for name, value in protocol['sources_sha256'].items() if value == ledger['worker_sha256'] and name.endswith('_study.py') and not Path(name).name.startswith('analyze_')]
    if len(workers) != 1:
        raise ValueError('Execution worker hash is not uniquely frozen')
    expected = [run['id'] for run in protocol['runs']]
    completed = []
    attempts = []
    retries = 0
    for run_id in expected:
        rows = [row for row in ledger['attempts'] if row['run_id'] == run_id]
        if not 1 <= len(rows) <= policy['max_retries'] + 1:
            raise ValueError('Missing run or excessive attempts')
        for index, row in enumerate(rows, 1):
            if row['attempt'] != index or not math.isfinite(row['whole_process_seconds']) or row['whole_process_seconds'] <= 0:
                raise ValueError('Invalid attempt identity/time')
            prefix = f'{run_id}-attempt-{index}'
            for kind in ('stdout', 'stderr'):
                log = root / 'supervisor' / f'{prefix}.{kind}.log'
                if digest(log) != row[f'{kind}_sha256']:
                    raise ValueError('Changed execution log')
            stderr = (root / 'supervisor' / f'{prefix}.stderr.log').read_text()
            signature = policy['only_retry_signature'] in stderr
            if signature != row['retryable_metal_signature']:
                raise ValueError('Error signature mismatch')
            if index < len(rows):
                if row['returncode'] == 0 or not signature:
                    raise ValueError('Unauthorized retry')
                retries += 1
            elif row['returncode'] != 0:
                raise ValueError('Incomplete study')
            attempts.append(row)
        result = json.loads((root / 'runs' / run_id / 'result.json').read_text())
        if result['protocol_sha256'] != digest(protocol_path):
            raise ValueError('Completed result protocol mismatch')
        completed.append(run_id)
    if [row['run_id'] for row in ledger['attempts']] != [row['run_id'] for row in attempts]:
        raise ValueError('Unplanned runs or execution ordering changed')
    result = {'status': 'all execution-policy checks passed',
              'protocol_sha256': digest(protocol_path), 'journal_sha256': digest(journal),
              'worker_source': workers[0], 'worker_sha256': ledger['worker_sha256'],
              'completed_runs': len(completed), 'attempts_count': len(attempts),
              'failed_attempts_count': retries,
              'sum_whole_process_seconds_including_failures': sum(row['whole_process_seconds'] for row in attempts),
              'attempts': attempts,
              'limitations': ['Whole-process sums include model/data/source hash checks, loading, warmup, validation and checkpoints.',
                              'They exclude supervisor cooldown and between-process Python work; not total job wall time.',
                              'This audit does not independently authenticate the machine; numerical checkpoints are audited separately.']}
    write(root / 'execution-audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output.resolve())
    print(json.dumps({key: value for key, value in result.items() if key != 'attempts'}, indent=2))


if __name__ == '__main__':
    main()

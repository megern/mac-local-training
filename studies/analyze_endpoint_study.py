"""Audit every frozen run, checkpoint, numerical comparison and paired metric."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from endpoint_study import digest, verify, write


def geometric_interval(ratios):
    import numpy as np
    values = np.log(np.asarray(ratios, dtype=float))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid paired ratios')
    rng = np.random.default_rng(20261012)
    draws = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return {'geometric_mean_ratio': float(np.exp(values.mean())),
            'paired_bootstrap_95_ci': [float(x) for x in np.exp(np.quantile(draws, [.025, .975]))],
            'ratios': ratios}


def sign_test(ratios):
    below = sum(x < 1 for x in ratios)
    above = sum(x > 1 for x in ratios)
    n = below + above
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(below, above) + 1)) / 2 ** n) if n else 1.0
    return {'below_one': below, 'above_one': above, 'ties': len(ratios) - n, 'exact_two_sided_p': p}


def checkpoint(root, result):
    manifest_path = root / 'training/latest.json'
    if digest(manifest_path) != result['checkpoint_manifest_sha256']:
        raise ValueError('Changed checkpoint manifest')
    manifest = json.loads(manifest_path.read_text())
    directory = (root / 'training' / manifest['checkpoint']).resolve()
    if not directory.is_relative_to((root / 'training').resolve()):
        raise ValueError('Checkpoint leaves run directory')
    for name, expected in manifest['files_sha256'].items():
        if Path(name).name != name or digest(directory / name) != expected:
            raise ValueError('Changed checkpoint data')
    state = json.loads((directory / 'state.json').read_text())
    for left, right in [('identity', 'identity'), ('coverage', 'coverage'), ('cursor', 'examples'), ('training_tokens', 'targets'), ('history', 'history')]:
        if state[left] != result[right]:
            raise ValueError(f'Report/checkpoint mismatch: {left}')
    return directory, state


def array_error(left, right):
    import numpy as np
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError('Array shape/dtype mismatch')
    l, r = left.astype(np.float64), right.astype(np.float64)
    if not (np.isfinite(l).all() and np.isfinite(r).all()):
        raise ValueError('Nonfinite checkpoint arrays')
    return float(np.max(np.abs(l - r)))


def compare_arrays(reference, candidate):
    from safetensors import safe_open
    errors = []
    with safe_open(reference / 'adapters.safetensors', framework='numpy') as a, safe_open(candidate / 'adapters.safetensors', framework='numpy') as b:
        if a.keys() != b.keys():
            raise ValueError('Adapter keys differ')
        for name in a.keys():
            errors.append(array_error(a.get_tensor(name), b.get_tensor(name)))
    moment_errors = []
    seen = set()
    candidate_state = json.loads((candidate / 'state.json').read_text())
    with safe_open(reference / 'optimizer-0.safetensors', framework='numpy') as a:
        for filename in candidate_state['optimizer_files'].values():
            with safe_open(candidate / filename, framework='numpy') as b:
                for name in b.keys():
                    if name not in a.keys():
                        raise ValueError('Candidate optimizer contains unknown key')
                    if name not in ('step', 'learning_rate'):
                        if name in seen:
                            raise ValueError('Optimizer moment appears in multiple groups')
                        seen.add(name)
                    moment_errors.append(array_error(a.get_tensor(name), b.get_tensor(name)))
        if seen != set(a.keys()) - {'step', 'learning_rate'}:
            raise ValueError('Missing optimizer moments')
    return {'max_adapter_error': max(errors), 'max_optimizer_moment_error': max(moment_errors)}


def analyze(root):
    protocol_path = root / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    verify(protocol)
    expected = {row['id'] for row in protocol['runs']}
    if len(expected) != len(protocol['runs']):
        raise ValueError('Duplicate run IDs')
    results = {}
    checkpoints = {}
    for row in protocol['runs']:
        directory = root / 'runs' / row['id']
        result = json.loads((directory / 'result.json').read_text())
        if result['run'] != row or result['protocol_sha256'] != digest(protocol_path):
            raise ValueError('Run identity/protocol mismatch')
        for field in ('elapsed_seconds', 'mlx_measured_peak_bytes', 'optimizer_resident_peak_bytes'):
            if not math.isfinite(result[field]) or result[field] <= 0:
                raise ValueError(f'Invalid measurement: {field}')
        total_steps = protocol['warmup_steps'] + protocol['measured_steps']
        if result['coverage'] != [total_steps] * 28 or result['examples'] != total_steps * 2:
            raise ValueError('Unequal layer/data exposure')
        if len(result['history']) != total_steps or [h['step'] for h in result['history']] != list(range(1, total_steps + 1)):
            raise ValueError('Missing update records')
        identity = result['identity']
        for name in ('train', 'valid'):
            if identity[f'{name}_sha256'] != protocol[name]['sha256']:
                raise ValueError('Changed dataset identity')
        if identity['model_files'] != protocol['model_files']:
            raise ValueError('Changed model identity')
        expected_config = dict(protocol['config'], rank=row['rank'], seed=row['seed'],
                               tiny_model=False, model_path=protocol['model_path'],
                               train_path=protocol['train']['path'], valid_path=protocol['valid']['path'],
                               strategy='all' if row['method'] == 'all' else 'rotating')
        if identity['config'] != expected_config:
            raise ValueError('Changed training config')
        cp, state = checkpoint(directory, result)
        if state['step'] != total_steps:
            raise ValueError('Checkpoint has different step count')
        results[row['id']] = result
        checkpoints[row['id']] = cp
    if len({json.dumps(r['runtime'], sort_keys=True) for r in results.values()}) != 1:
        raise ValueError('Runtime versions differ')
    comparisons = []
    gates = []
    for rank in (4, 256):
        seed_rows = []
        for seed in (17, 23, 41, 59, 73, 89):
            names = {m: f'rank-{rank}-seed-{seed}-{m}' for m in ('all', 'fixed', 'endpoint')}
            pair = {m: results[n] for m, n in names.items()}
            errors = {m: compare_arrays(checkpoints[names['all']], checkpoints[names[m]]) for m in ('fixed', 'endpoint')}
            for method in ('fixed', 'endpoint'):
                candidate = pair[method]
                reference = pair['all']
                if candidate['targets'] != reference['targets']:
                    raise ValueError('Supervised target counts differ')
                if not math.isclose(candidate['nll_before'], reference['nll_before'], rel_tol=0, abs_tol=1e-5):
                    raise ValueError('Initialization NLL differs')
                errors[method]['validation_nll_difference'] = candidate['nll_after'] - reference['nll_after']
                passed = errors[method]['max_adapter_error'] <= protocol['acceptance']['max_adapter_error'] and errors[method]['max_optimizer_moment_error'] <= protocol['acceptance']['max_optimizer_moment_error'] and abs(errors[method]['validation_nll_difference']) <= protocol['acceptance']['max_validation_nll_difference']
                gates.append(passed)
                expected_ops = protocol['measured_steps'] * (4 if method == 'fixed' else 3)
                if any(candidate['io'][field] != expected_ops for field in ('switches', 'reads', 'writes')):
                    raise ValueError('Observed state-operation count contradicts scheduling theorem')
            row = {'seed': seed, 'equivalence': errors,
                   'elapsed_endpoint_over_fixed': pair['endpoint']['elapsed_seconds'] / pair['fixed']['elapsed_seconds'],
                   'peak_endpoint_over_fixed': pair['endpoint']['mlx_measured_peak_bytes'] / pair['fixed']['mlx_measured_peak_bytes'],
                   'logical_read_fraction': pair['endpoint']['io']['logical_read_bytes'] / pair['fixed']['io']['logical_read_bytes'],
                   'logical_write_fraction': pair['endpoint']['io']['logical_write_bytes'] / pair['fixed']['io']['logical_write_bytes'],
                   'elapsed_fixed_over_all': pair['fixed']['elapsed_seconds'] / pair['all']['elapsed_seconds'],
                   'elapsed_endpoint_over_all': pair['endpoint']['elapsed_seconds'] / pair['all']['elapsed_seconds'],
                   'peak_fixed_over_all': pair['fixed']['mlx_measured_peak_bytes'] / pair['all']['mlx_measured_peak_bytes'],
                   'peak_endpoint_over_all': pair['endpoint']['mlx_measured_peak_bytes'] / pair['all']['mlx_measured_peak_bytes']}
            seed_rows.append(row)
        ratios = [row['elapsed_endpoint_over_fixed'] for row in seed_rows]
        stats = geometric_interval(ratios)
        comparisons.append({'rank': rank, 'paired_seed_blocks': seed_rows, 'elapsed': stats,
                            'sign_test': sign_test(ratios),
                            'speed_gate_passed': stats['geometric_mean_ratio'] <= 1 - protocol['acceptance']['speed_gain_threshold'] and stats['paired_bootstrap_95_ci'][1] < protocol['acceptance']['speed_ci_upper_ratio'],
                            'memory_nonincrease_gate_passed': all(row['peak_endpoint_over_fixed'] <= protocol['acceptance']['memory_ratio_upper'] for row in seed_rows)})
    primary = next(row for row in comparisons if row['rank'] == protocol['statistics']['primary_rank'])
    result = {'schema_version': 1, 'protocol_sha256': digest(protocol_path), 'runs_audited': len(results),
              'all_numerical_equivalence_gates_passed': all(gates),
              'state_switch_reduction_fraction': 0.25,
              'primary_joint_gate_passed': all(gates) and primary['speed_gate_passed'] and primary['memory_nonincrease_gate_passed'],
              'comparisons': comparisons, 'limitations': protocol['limitations'],
              'novelty_status': protocol['novelty_status'],
              'source_result_sha256': {name: digest(root / 'runs' / name / 'result.json') for name in results}}
    write(root / 'analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.output.resolve())
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_result_sha256', 'comparisons')}, indent=2))


if __name__ == '__main__':
    main()

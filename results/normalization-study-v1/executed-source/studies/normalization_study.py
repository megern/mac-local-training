"""Offline, counterbalanced group-local gradient-normalization study.

Freeze a protocol with `prepare`, execute all fresh-process runs with `run`,
then audit with `analyze_normalization_study.py`. No network access or test scoring.
"""
from dataclasses import replace
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
SOURCES = ['mac_local_training/engine.py', 'mac_local_training/synchronous.py',
           'mac_local_training/offloaded.py', 'mac_local_training/endpoint.py',
           'mac_local_training/normalization.py', 'studies/endpoint_study.py',
           'studies/normalization_study.py', 'studies/analyze_normalization_study.py']


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify(protocol):
    for name, expected in protocol['sources_sha256'].items():
        if digest(PACKAGE / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    for field in ('train', 'valid'):
        if digest(protocol[field]['path']) != protocol[field]['sha256']:
            raise ValueError(f'Frozen {field} data changed')
    for name, expected in protocol['model_files'].items():
        if digest(Path(protocol['model_path']) / name) != expected:
            raise ValueError(f'Frozen model file changed: {name}')


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        raise ValueError('Protocol exists; never replace a frozen study')
    inputs = {}
    for name, source, count in [('train', args.train_data, 16), ('valid', args.valid_data, 8)]:
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        eligible = [row for row in rows if 'tokens' in row and 'loss_start' in row
                    and 2 <= len(row['tokens']) <= 97]
        if len(eligible) < count:
            raise ValueError('Insufficient completion rows of at most 97 tokens')
        chosen = eligible[:count]
        target = output / f'{name}-local.jsonl'
        target.write_text(''.join(json.dumps(row) + '\n' for row in chosen))
        inputs[name] = {'path': str(target), 'sha256': digest(target),
                        'source_sha256': digest(source), 'selected_ids': [row['id'] for row in chosen],
                        'selection': f'First {count} eligible source rows, length <=97, no truncation or label balancing'}
    model = args.model.resolve()
    model_config = json.loads((model / 'config.json').read_text())
    if model_config.get('num_hidden_layers') != 28 or model_config.get('model_type') != 'qwen3' or model_config.get('torch_dtype') != 'bfloat16':
        raise ValueError('Frozen study requires the 28-layer original Qwen3 BF16 model')
    if 'quantization' in model_config:
        raise ValueError('This study requires original, non-quantized weights')
    seeds = [17, 23, 41, 59, 73, 89]
    permutations = list(itertools.permutations(['all', 'endpoint', 'normalized']))
    # Reverse rank order in alternate blocks to spread time/thermal effects.
    runs = []
    for block, seed in enumerate(seeds):
        for rank in ([4, 256] if block % 2 == 0 else [256, 4]):
            for position, method in enumerate(permutations[block]):
                runs.append({'id': f'rank-{rank}-seed-{seed}-{method}', 'rank': rank,
                             'seed': seed, 'method': method, 'block': block, 'position': position})
    protocol = {'schema_version': 1, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'hypothesis': 'Group-local token normalization reduces measured peak MLX allocation at rank 256 by at least 3% versus endpoint offload, preserving updates and limiting elapsed overhead to 10%.',
                'candidate': 'Accumulate and check all raw gradients, divide by the same target count only for the current group, release consumed raw gradients, retain endpoint state.',
                'novelty_status': 'Original MLX implementation experiment; gradient buffer lifetime optimization is established, so algorithm priority is not claimed.',
                'model_path': str(model),
                'model_files': {str(p.relative_to(model)): digest(p) for p in sorted(model.rglob('*'))
                                if p.is_file() and p.suffix in {'.json', '.safetensors', '.model', '.txt'}},
                **inputs, 'runs': runs, 'warmup_steps': 2, 'measured_steps': 4,
                'config': {'microbatch_size': 1, 'accumulation_steps': 2, 'window_layers': 7,
                           'window_updates': 1, 'learning_rate': 0.0002, 'lora_scale': 1.0,
                           'max_seq_length': 96, 'memory_budget_gib': 6.0},
                'acceptance': {'max_adapter_error': 1e-5, 'max_optimizer_moment_error': 1e-5,
                               'max_validation_nll_difference': 1e-5,
                               'endpoint_warm_switches_per_step': 3, 'normalized_warm_switches_per_step': 3,
                               'same_examples_targets_coverage': True,
                               'memory_reduction_threshold': 0.03, 'memory_ci_upper_ratio': 1.0,
                               'elapsed_ci_upper_ratio': 1.10},
                'statistics': {'unit': 'paired seed block, not individual steps',
                               'primary_rank': 256, 'control_rank': 4,
                               'confidence': '95% paired bootstrap of geometric mean memory and elapsed ratios, 20000 draws, seed 20261012',
                               'sign_test': 'Exact two-sided paired sign test over six seed blocks',
                               'no_optional_stopping': True},
                'timing': 'Synchronized external update loop including group activation, GC, cache clearing, state I/O; excludes load, warmup, validation, checkpoints.',
                'reserved_test_scoring': False,
                'limitations': ['Single machine and single model; six seeds are not six machines.',
                                'Rank 256 intentionally stresses optimizer state and is not a typical small-rank LoRA workload.',
                                'Logical file bytes are not physical SSD traffic; OS page cache remains active.',
                                'Existing training/validation data are reused for numerical equivalence, not independent task-quality claims.',
                                'Shared-machine background and thermal state are observed, not controlled.'],
                'sources_sha256': {name: digest(PACKAGE / name) for name in SOURCES}}
    write(output / 'protocol.json', protocol)
    source_dir = output / 'source'
    source_dir.mkdir()
    import shutil
    for name in SOURCES:
        target = source_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PACKAGE / name, target)
    print(json.dumps({'protocol_sha256': digest(output / 'protocol.json'), 'runs': len(runs)}, indent=2))


def snapshot():
    import psutil
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {'available_memory_bytes': memory.available, 'total_memory_bytes': memory.total,
            'swap_used_bytes': swap.used, 'load_average': list(os.getloadavg()),
            'process_rss_bytes': psutil.Process().memory_info().rss}


class IOMetrics:
    """Same observation overhead for both offloading schedules."""
    def reset_io(self):
        self.io = {'switches': 0, 'reads': 0, 'writes': 0,
                   'logical_read_bytes': 0, 'logical_write_bytes': 0}

    def _save_optimizer(self, path):
        super()._save_optimizer(path)
        self.io['writes'] += 1
        self.io['logical_write_bytes'] += path.stat().st_size

    def activate_group(self, group):
        if self.active_group == group:
            return super().activate_group(group)
        if self.active_group is not None:
            self.io['switches'] += 1
        path = self.state_paths.get(group)
        if path is not None:
            self.io['reads'] += 1
            self.io['logical_read_bytes'] += path.stat().st_size
        return super().activate_group(group)


def worker(args):
    protocol_path = args.output / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    verify(protocol)
    item = next(row for row in protocol['runs'] if row['id'] == args.run_id)
    from mac_local_training.engine import TrainingConfig, TrainingEngine
    from mac_local_training.offloaded import OffloadedTrainingEngine
    from mac_local_training.endpoint import EndpointTrainingEngine
    from mac_local_training.normalization import StreamingNormalizationEngine
    import mlx.core as mx
    classes = {'all': TrainingEngine, 'endpoint': EndpointTrainingEngine, 'normalized': StreamingNormalizationEngine}
    cls = type('ObservedEngine', (IOMetrics, classes[item['method']]), {})
    total_steps = protocol['warmup_steps'] + protocol['measured_steps']
    run_dir = args.output / 'runs' / item['id']
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg = TrainingConfig(train_path=protocol['train']['path'], valid_path=protocol['valid']['path'],
                         output_path=str(run_dir / 'training'), model_path=protocol['model_path'],
                         strategy='all' if item['method'] == 'all' else 'rotating',
                         steps=total_steps, checkpoint_every=total_steps, rank=item['rank'], seed=item['seed'],
                         **protocol['config'])
    before = snapshot()
    loading_started = time.perf_counter()
    engine = cls(cfg)
    engine.reset_io()
    loading_seconds = time.perf_counter() - loading_started
    nll_before = engine.evaluate()
    for _ in range(protocol['warmup_steps']):
        engine.update_once()
    mx.eval(engine.model.parameters(), engine.optimizer.state)
    engine.reset_io()
    mx.reset_peak_memory()
    before_measurement = snapshot()
    started = time.perf_counter()
    for _ in range(protocol['measured_steps']):
        engine.update_once()
    mx.eval(engine.model.parameters(), engine.optimizer.state)
    elapsed = time.perf_counter() - started
    peak = int(mx.get_peak_memory())
    measured_io = dict(engine.io)
    after_measurement = snapshot()
    nll_after = engine.evaluate()
    engine.pre_nll = nll_before
    engine.checkpoint()
    manifest_path = run_dir / 'training/latest.json'
    result = {'run': item, 'protocol_sha256': digest(protocol_path),
              'elapsed_seconds': elapsed, 'loading_seconds': loading_seconds,
              'mlx_measured_peak_bytes': peak, 'io': measured_io,
              'nll_before': nll_before, 'nll_after': nll_after,
              'coverage': engine.coverage, 'examples': engine.cursor, 'targets': engine.training_tokens,
              'optimizer_resident_peak_bytes': max(x['optimizer_resident_bytes'] for x in engine.history[-protocol['measured_steps']:]),
              'full_gradient_resident_bytes': max(x.get('full_gradient_resident_bytes', 0) for x in engine.history),
              'resources': {'before_loading': before, 'before_measurement': before_measurement, 'after_measurement': after_measurement},
              'runtime': engine.identity['versions'], 'identity': engine.identity,
              'checkpoint_manifest_sha256': digest(manifest_path), 'history': engine.history}
    write(run_dir / 'result.json', result)
    print(json.dumps({'completed': item['id'], 'seconds': round(elapsed, 3), 'switches': measured_io['switches']}), flush=True)


def run(args):
    protocol_path = args.output / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    verify(protocol)
    for row in protocol['runs']:
        result = args.output / 'runs' / row['id'] / 'result.json'
        if result.exists():
            saved = json.loads(result.read_text())
            if saved['protocol_sha256'] != digest(protocol_path):
                raise ValueError('Existing result has a different protocol')
            continue
        subprocess.run([sys.executable, str(Path(__file__).resolve()), 'worker',
                        '--output', str(args.output), '--run-id', row['id']], check=True)
    print('All frozen runs completed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--train-data', type=Path)
    parser.add_argument('--valid-data', type=Path)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.action == 'prepare':
        if not all((args.model, args.train_data, args.valid_data)):
            parser.error('prepare requires model, train-data and valid-data')
        prepare(args)
    elif args.action == 'worker':
        worker(args)
    else:
        run(args)


if __name__ == '__main__':
    main()

"""Single original-BF16 equivalence check against the captured all-adapter pilot.

Fixed acceptance: maximum adapter-array error <= 1e-5 and validation NLL
difference <= 1e-5, with equal supervised targets and per-layer coverage.
No reserved test is loaded. This checks implementation equivalence, not novelty.
"""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[23, 41, 59],
                        help='Use one of the frozen confirmation all-adapter references instead of pilot17')
    args = parser.parse_args()
    root = Path.cwd()
    sys.path.insert(0, str(root / 'outputs/mac-local-training'))
    from mac_local_training.engine import TrainingConfig
    from mac_local_training.synchronous import SynchronousTrainingEngine
    import mlx.core as mx
    baseline_dir = (root / 'work/runs/completion-confirmation-v1' / f'all-8-seed-{args.seed}'
                    if args.seed else root / 'work/runs/completion-pilot-v1/all')
    baseline = json.loads((baseline_dir / 'report.json').read_text())
    manifest = json.loads((baseline_dir / 'latest.json').read_text())
    checkpoint = baseline_dir / manifest['checkpoint']
    expected_path = checkpoint / 'adapters.safetensors'
    if hashlib.sha256(expected_path.read_bytes()).hexdigest() != manifest['files_sha256']['adapters.safetensors']:
        raise ValueError('Baseline checkpoint changed')
    output = (root / 'work/runs/synchronous-bf16-confirmation-v1' / f'seed-{args.seed}'
              if args.seed else root / 'work/runs/synchronous-bf16-equivalence-v1')
    cfg = TrainingConfig(**baseline['identity']['config'], steps=8, output_path=str(output), checkpoint_every=8)
    engine = SynchronousTrainingEngine(replace(cfg, strategy='rotating'))
    report = engine.run()
    for field in ('train_sha256', 'valid_sha256', 'model_files', 'versions'):
        if report['identity'][field] != baseline['identity'][field]:
            raise ValueError('Baseline and candidate input identities disagree')
    expected = mx.load(str(expected_path))
    actual = engine.adapter_arrays()
    if set(expected) != set(actual):
        raise ValueError('Adapter keys disagree')
    differences = {name: float(mx.max(mx.abs(actual[name].astype(mx.float32) - expected[name].astype(mx.float32))).item())
                   for name in actual}
    max_error = max(differences.values())
    nll_difference = report['heldout_nll_after'] - baseline['heldout_nll_after']
    coverage_equal = report['layer_update_coverage'] == baseline['layer_update_coverage']
    tokens_equal = report['tokens'] == baseline['tokens']
    passed = max_error <= 1e-5 and abs(nll_difference) <= 1e-5 and coverage_equal and tokens_equal
    result = {'status': 'passed fixed equivalence check' if passed else 'failed fixed equivalence check',
              'seed': cfg.seed,
              'max_adapter_absolute_error': max_error, 'validation_nll_difference': nll_difference,
              'same_layer_update_coverage': coverage_equal, 'same_supervised_target_count': tokens_equal,
              'optimizer_resident_fraction': report['peak_optimizer_resident_bytes'] / baseline['peak_optimizer_resident_bytes'],
              'mlx_peak_fraction': report['mlx_peak_bytes_this_process_segment'] / baseline['mlx_peak_bytes_this_process_segment'],
              'training_time_fraction': report['training_seconds'] / baseline['training_seconds'],
              'baseline_report_sha256': hashlib.sha256((baseline_dir / 'report.json').read_bytes()).hexdigest(),
              'candidate_report_sha256': hashlib.sha256((output / 'report.json').read_bytes()).hexdigest(),
              'baseline_adapter_sha256': manifest['files_sha256']['adapters.safetensors'],
              'per_array_max_errors': differences,
              'limitations': ['Single seed, eight updates, one original 0.6B model; no large-model fit demonstrated.',
                  'Equivalent only to the specified reference; not a task-quality improvement.',
                  'Extra recomputation and disk I/O; shared-machine times are uncontrolled.',
                  'No algorithm priority claim; optimizer offload and recomputation have prior art.']}
    (output / 'equivalence.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_array_max_errors'}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Audit all frozen runs and summarize paired test effects without selection."""
import hashlib
import json
from pathlib import Path
import statistics
import sys


def main():
    root = Path.cwd()
    frozen = root / 'work/reports/completion-confirmation-v1/protocol.json'
    protocol = json.loads(frozen.read_text())
    data = root / 'work/data/sms-completion-fresh-v1'
    data_protocol = json.loads((data / 'protocol.json').read_text())
    test_bytes = (data / 'test-reserved.jsonl').read_bytes()
    if hashlib.sha256(test_bytes).hexdigest() != data_protocol['splits']['test-reserved']['sha256']:
        raise ValueError('Test source changed')
    expected_cases = [(row['id'], row['label']) for row in
                      (json.loads(line) for line in test_bytes.decode().splitlines())]
    sys.path.insert(0, str(root / 'outputs/m1-memory-research/tools'))
    from public_sms_study import metrics
    records = {}
    for seed in protocol['seeds']:
        for strategy, steps in protocol['conditions']:
            name = f'{strategy}-{steps}-seed-{seed}'
            path = root / 'work/runs/completion-confirmation-v1' / name / 'confirmation.json'
            record = json.loads(path.read_text())
            train = record['training']
            if record['scored_split'] != 'test-reserved' or train['steps'] != steps:
                raise ValueError('Incomplete or wrong run')
            if train['loss_targets'] != {'train_completion_rows': 192, 'valid_completion_rows': 64}:
                raise ValueError('Wrong target objective')
            if train['examples_consumed'] != steps * 2 or train['tokens'] != steps * 2:
                raise ValueError('Unequal or wrong supervision budget')
            if train['layer_update_coverage'] != [steps if strategy == 'all' else steps // 4] * 28:
                raise ValueError('Unexpected layer coverage')
            if [(row['id'], row['label']) for row in record['rows']] != expected_cases:
                raise ValueError('Wrong test labels or identities')
            if metrics(record['rows']) != record['metrics']:
                raise ValueError('Metrics disagree with raw predictions')
            record['report_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            records[seed, strategy, steps] = record
    reference = records[protocol['seeds'][0], 'all', 8]
    identities = [(r['id'], r['label']) for r in reference['rows']]
    if len(identities) != 240 or len(set(x[0] for x in identities)) != 240:
        raise ValueError('Wrong independent case count')
    for record in records.values():
        if [(r['id'], r['label']) for r in record['rows']] != identities:
            raise ValueError('Unpaired test cases')
        for field in ('train_sha256', 'valid_sha256', 'model_files', 'versions'):
            if record['training']['identity'][field] != reference['training']['identity'][field]:
                raise ValueError('Different input identity')
    if reference['training']['identity']['model_files']['model.safetensors'] != 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b':
        raise ValueError('Wrong original BF16 model')
    per_seed = []
    for seed in protocol['seeds']:
        all_run = records[seed, 'all', 8]
        for steps in (8, 32):
            rotating = records[seed, 'rotating', steps]
            a, b = all_run['training'], rotating['training']
            if a['identity']['config']['seed'] != seed or b['identity']['config']['seed'] != seed:
                raise ValueError('Wrong paired seed')
            if any(a['identity']['config'][field] != b['identity']['config'][field]
                   for field in a['identity']['config'] if field != 'strategy'):
                raise ValueError('Unmatched paired configuration')
            per_seed.append({'seed': seed, 'rotating_steps': steps,
                  'macro_f1_all': all_run['metrics']['macro_f1'],
                  'macro_f1_rotating': rotating['metrics']['macro_f1'],
                  'macro_f1_difference': rotating['metrics']['macro_f1'] - all_run['metrics']['macro_f1'],
                  'false_positive_rate_difference': rotating['metrics']['false_positive_rate'] - all_run['metrics']['false_positive_rate'],
                  'optimizer_fraction': b['peak_optimizer_resident_bytes'] / a['peak_optimizer_resident_bytes'],
                  'mlx_peak_fraction': b['mlx_peak_bytes_this_process_segment'] / a['mlx_peak_bytes_this_process_segment'],
                  'end_to_end_fraction': b['end_to_end_seconds_this_process_segment'] / a['end_to_end_seconds_this_process_segment'],
                  'example_exposure_fraction': b['examples_consumed'] / a['examples_consumed']})
    import numpy as np
    rng = np.random.default_rng(20261012)
    labels = np.array([label == 'spam' for _, label in identities])
    pools = [np.flatnonzero(labels == value) for value in (False, True)]
    predicted = {key: np.array([r['prediction'] == 'spam' for r in value['rows']]) for key, value in records.items()}

    def macro(pred, sampled):
        gold, values = labels[sampled], pred[sampled]
        tp = int(np.sum(gold & values)); tn = int(np.sum(~gold & ~values))
        fp = int(np.sum(~gold & values)); fn = int(np.sum(gold & ~values))
        return .5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn))

    distributions = {8: [], 32: []}
    for _ in range(2000):
        sampled = np.concatenate([rng.choice(pool, len(pool), replace=True) for pool in pools])
        for steps in distributions:
            distributions[steps].append(statistics.mean(
                macro(predicted[seed, 'rotating', steps], sampled) - macro(predicted[seed, 'all', 8], sampled)
                for seed in protocol['seeds']))
    summary = {str(steps): {'mean_macro_f1_difference': statistics.mean(
                    row['macro_f1_difference'] for row in per_seed if row['rotating_steps'] == steps),
                    'conditional_case_bootstrap_95_percent_interval': np.quantile(distributions[steps], [.025, .975]).tolist(),
                    'positive_seed_count': sum(row['macro_f1_difference'] > 0 for row in per_seed if row['rotating_steps'] == steps)}
               for steps in distributions}
    result = {'status': 'all nine frozen runs audited', 'protocol_sha256': hashlib.sha256(frozen.read_bytes()).hexdigest(),
              'reports_sha256': {f'{seed}/{strategy}/{steps}': value['report_sha256']
                                 for (seed, strategy, steps), value in records.items()},
              'per_seed': per_seed, 'summary': summary,
              'bootstrap': '2000 stratified paired case resamples; each sample shared across all models; mean across three fixed seeds. Conditional on this dataset and these seeds, not training-seed or population uncertainty.',
              'limitations': protocol['limitations'] + ['Nine evaluations share the same 240 cases; they are not 2160 independent test cases.',
                   'Equal-layer-update comparator sees four times more examples.',
                   'No new algorithm, independent-corpus replication, practical large-model fit, or production adoption is established.']}
    output = root / 'work/reports/completion-confirmation-v1/analysis.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

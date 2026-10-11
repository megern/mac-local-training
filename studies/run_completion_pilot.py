"""Fresh-process original-BF16 pilot or frozen confirmation.

Default runs use validation only. Confirmation runs require the previously
frozen source/configuration protocol and score its reserved test once.
Neither mode establishes algorithm novelty or general model superiority.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', choices=['all', 'rotating'], required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--confirmation', action='store_true',
                        help='Require frozen confirmation protocol and score reserved test once')
    args = parser.parse_args()
    root = Path.cwd()
    sys.path.insert(0, str(root / 'outputs/mac-local-training'))
    from mac_local_training.engine import TrainingConfig, TrainingEngine
    import mlx.core as mx
    data = root / 'work/data/sms-completion-fresh-v1'
    protocol = json.loads((data / 'protocol.json').read_text())
    for name in ('train', 'valid'):
        if hashlib.sha256((data / (name + '.jsonl')).read_bytes()).hexdigest() != protocol['splits'][name]['sha256']:
            raise ValueError('Data changed')
    if args.confirmation:
        frozen = json.loads((root / 'work/reports/completion-confirmation-v1/protocol.json').read_text())
        if args.seed not in frozen['seeds'] or [args.strategy, args.steps] not in frozen['conditions']:
            raise ValueError('Run not allowed by frozen confirmation protocol')
        for relative, expected in frozen['source_sha256'].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                raise ValueError('Confirmation source changed')
        if hashlib.sha256((data / 'protocol.json').read_bytes()).hexdigest() != frozen['data_protocol_sha256']:
            raise ValueError('Data protocol changed')
    output = (root / 'work/runs/completion-confirmation-v1' / f'{args.strategy}-{args.steps}-seed-{args.seed}'
              if args.confirmation else root / 'work/runs/completion-pilot-v1' / args.strategy)
    cfg = TrainingConfig(train_path=str(data / 'train.jsonl'), valid_path=str(data / 'valid.jsonl'),
          output_path=str(output), model_path=str(root / 'work/models/qwen3-0.6b-original'),
          strategy=args.strategy, steps=args.steps, accumulation_steps=2, microbatch_size=1,
          window_layers=7, window_updates=1, learning_rate=0.0002, rank=4, seed=args.seed,
          max_seq_length=384, checkpoint_every=8, memory_budget_gib=6)
    engine = TrainingEngine(cfg)
    report = engine.run()
    split = 'test-reserved' if args.confirmation else 'valid'
    if hashlib.sha256((data / (split + '.jsonl')).read_bytes()).hexdigest() != protocol['splits'][split]['sha256']:
        raise ValueError('Scoring data changed')
    rows = [json.loads(line) for line in (data / (split + '.jsonl')).read_text().splitlines()]
    predictions = []
    started = time.monotonic()
    for row in rows:
        prefix = row['tokens'][:row['loss_start']]
        logits = engine.model(mx.array(prefix, dtype=mx.int32)[None, :])[:, -1, :].astype(mx.float32)
        margin = float((logits[0, 75545] - logits[0, 5604]).item())
        predictions.append({'id': row['id'], 'label': row['label'],
                            'prediction': 'spam' if margin > 0 else 'ham', 'margin': margin})
    sys.path.insert(0, str(root / 'outputs/m1-memory-research/tools'))
    from public_sms_study import metrics
    result = {'status': 'completed frozen confirmation' if args.confirmation else 'completed validation-only engineering pilot',
              'strategy': args.strategy, 'scored_split': split,
              'training': report, 'rows': predictions, 'metrics': metrics(predictions),
              'classification_seconds': time.monotonic() - started,
              'limitations': ['Short local adapter experiment; not evidence of algorithm novelty.',
                              'Equal example exposure does not mean equal layer update coverage.',
                              'Equal layer coverage requires four times the data exposure for rotating.']}
    (output / ('confirmation.json' if args.confirmation else 'validation-pilot.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'strategy': args.strategy, 'nll_change': report['heldout_nll_change'],
                     'metrics': result['metrics'], 'split': split,
                     'mlx_peak_bytes': report['mlx_peak_bytes_this_process_segment'],
                     'optimizer_bytes': report['peak_optimizer_resident_bytes']}, indent=2), flush=True)


if __name__ == '__main__':
    main()

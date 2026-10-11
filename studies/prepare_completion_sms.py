"""Prepare a fresh completion-only study, excluding every previously used group.

Run from the portfolio workspace. Raw texts and tokenized records stay in work/.
This prepares data; it does not train, inspect scores, or evaluate reserved tests.
"""
import hashlib
import json
from pathlib import Path
import random
import sys
from collections import Counter


def main():
    root = Path.cwd()
    sys.path.insert(0, str(root / 'outputs/m1-memory-research/tools'))
    from prepare_public_sms import groups, messages
    from transformers import AutoTokenizer
    source = root / 'work/data/sms-public-v1/SMSSpamCollection'
    raw = source.read_bytes()
    expected = json.loads(source.with_name('source.json').read_text())['member_sha256']
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Source hash changed')
    used = set()
    prior_hashes = {}
    for name in ('sms-group-study-v1', 'sms-group-followup-v1'):
        path = root / 'work/data' / name / 'protocol.json'
        prior_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for split in json.loads(path.read_text())['splits'].values():
            for members in split['source_group_ids'].values():
                used.update(members)
    rows = []
    for i, line in enumerate(raw.decode().splitlines()):
        label, text = line.split('\t', 1)
        rows.append(dict(id=f'uci-{i:05d}', label=label, text=text))
    clusters, pairs = groups(rows)
    pool = {'ham': [], 'spam': []}
    membership = {}
    for cluster in clusters:
        if len({r['label'] for r in cluster}) != 1 or any(r['id'] in used for r in cluster):
            continue
        row = min(cluster, key=lambda r: r['id'])
        pool[row['label']].append(row)
        membership[row['id']] = sorted(r['id'] for r in cluster)
    rng = random.Random(20261011)
    available = {k: len(v) for k, v in pool.items()}
    for values in pool.values():
        values.sort(key=lambda r: r['id'])
        rng.shuffle(values)
    output = root / 'work/data/sms-completion-fresh-v1'
    if output.exists():
        raise ValueError('Fresh output required')
    output.mkdir()
    tokenizer = AutoTokenizer.from_pretrained(root / 'work/models/qwen3-0.6b-original',
                                             local_files_only=True, trust_remote_code=False)
    candidates = {'ham': 5604, 'spam': 75545}
    protocol = {'status': 'prepared; reserved test not evaluated', 'seed': 20261011,
                'source_sha256': expected, 'prior_protocol_sha256': prior_hashes,
                'excluded_source_ids': len(used), 'fresh_pool': available, 'near_pairs': pairs,
                'objective': 'single true label token; prompt targets masked',
                'context_limit': 384, 'splits': {},
                'limitations': ['Public 2011 English SMS; foundation-model contamination unknown.',
                                'Lexical grouping does not guarantee semantic independence.',
                                'Test prevalence is fixed by design, not population prevalence.']}
    for name, counts in [('train', {'ham': 96, 'spam': 96}),
                         ('valid', {'ham': 48, 'spam': 16}),
                         ('test-reserved', {'ham': 192, 'spam': 48})]:
        selected = []
        for label, count in counts.items():
            if len(pool[label]) < count:
                raise ValueError('Insufficient independent groups')
            selected.extend(pool[label][:count])
            del pool[label][:count]
        rng.shuffle(selected)
        records = []
        for row in selected:
            prefix = tokenizer.apply_chat_template(messages(row['text']), tokenize=False,
                         add_generation_prompt=True, enable_thinking=False) + '{"label":"'
            tokens = tokenizer.encode(prefix, add_special_tokens=False)
            target = candidates[row['label']]
            if tokenizer.encode(prefix + row['label'], add_special_tokens=False) != tokens + [target]:
                raise ValueError('Candidate boundary changed')
            if len(tokens) > 384:
                raise ValueError('Context too long; do not silently truncate or resample')
            records.append(dict(id=row['id'], label=row['label'], tokens=tokens + [target],
                                loss_start=len(tokens)))
        payload = ''.join(json.dumps(r) + '\n' for r in records).encode()
        (output / (name + '.jsonl')).write_bytes(payload)
        protocol['splits'][name] = {'rows': len(records), 'labels': dict(Counter(r['label'] for r in records)),
             'sha256': hashlib.sha256(payload).hexdigest(),
             'source_group_ids': {r['id']: membership[r['id']] for r in selected}}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({k: {n: v for n, v in x.items() if n != 'source_group_ids'}
                      for k, x in protocol['splits'].items()}, indent=2))


if __name__ == '__main__':
    main()

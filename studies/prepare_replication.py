"""Freeze a new local protocol from an exported design and reconstructed rows.

Path substitution changes the protocol hash. This prepares an independent
replication; it never replaces or impersonates the original experiment.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
from endpoint_study import PACKAGE, digest, write


def prepare(design_path, model, data, output):
    design = json.loads(design_path.read_text())
    protocol = copy.deepcopy(design)
    original_hash = protocol.pop('recorded_protocol_sha256')
    protocol.pop('export_note', None)
    model, data, output = model.resolve(), data.resolve(), output.resolve()
    if output.is_relative_to(model) or output.is_relative_to(data):
        raise ValueError('Study output must be outside model and reconstructed-data directories')
    if output.exists():
        raise ValueError('Use a new study directory')
    for name, expected in protocol['model_files'].items():
        path = model / name
        if not path.resolve().is_relative_to(model) or digest(path) != expected:
            raise ValueError(f'Original model differs: {name}')
    for split in ('train', 'valid'):
        path = data / f'{split}.jsonl'
        if digest(path) != protocol[split]['sha256']:
            raise ValueError(f'Reconstructed {split} differs')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if [row['id'] for row in rows] != protocol[split]['selected_ids']:
            raise ValueError('Selected IDs/order differ')
        protocol[split]['path'] = str(path)
        protocol[split]['replication_selection'] = 'Exact reconstructed exported IDs in the original order; no selection rerun.'
    for name, expected in protocol['sources_sha256'].items():
        path = PACKAGE / name
        if not path.resolve().is_relative_to(PACKAGE.resolve()) or digest(path) != expected:
            raise ValueError(f'Executed source differs: {name}; use the recorded source version')
    protocol['sources_sha256']['studies/prepare_replication.py'] = digest(Path(__file__))
    protocol['model_path'] = str(model)
    protocol['created_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    protocol['replication_of_protocol_sha256'] = original_hash
    protocol['replication_note'] = 'Independent run with local paths and reconstructed exact data; the original protocol hash is provenance only.'
    output.mkdir(parents=True)
    write(output / 'protocol.json', protocol)
    for name in protocol['sources_sha256']:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PACKAGE / name, target)
    return {'protocol_sha256': digest(output / 'protocol.json'), 'runs': len(protocol['runs']),
            'replication_of_protocol_sha256': original_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.design, args.model, args.data, args.output), indent=2))


if __name__ == '__main__':
    main()

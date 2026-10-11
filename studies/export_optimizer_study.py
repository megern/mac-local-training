"""Export aggregate evidence and executed source; never export datasets or weights."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from endpoint_study import digest, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    root = args.study.resolve()
    destination = args.destination.resolve()
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError('Export must be outside the local study tree')
    analysis = json.loads((root / 'analysis.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    if analysis['protocol_sha256'] != digest(root / 'protocol.json'):
        raise ValueError('Analysis/protocol mismatch')
    if analysis['runs_audited'] != len(protocol['runs']):
        raise ValueError('Incomplete analysis')
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'export-receipt.json').exists():
        raise ValueError('Export already exists; use a fresh destination')
    summaries = []
    fields = ['run', 'elapsed_seconds', 'loading_seconds', 'mlx_measured_peak_bytes', 'io',
              'nll_before', 'nll_after', 'coverage', 'examples', 'targets',
              'optimizer_resident_peak_bytes', 'full_gradient_resident_bytes',
              'resources', 'runtime', 'checkpoint_manifest_sha256', 'history']
    for item in protocol['runs']:
        path = root / 'runs' / item['id'] / 'result.json'
        if digest(path) != analysis['source_result_sha256'][item['id']]:
            raise ValueError('Result changed since audit')
        result = json.loads(path.read_text())
        summary = {field: result[field] for field in fields}
        summary['local_complete_result_sha256'] = digest(path)
        summary['full_gradient_residency_recorded'] = any('full_gradient_resident_bytes' in row for row in result['history'])
        if not summary['full_gradient_residency_recorded']:
            summary['gradient_storage_note'] = 'The frozen worker emitted zero as a missing-field placeholder; this is not a measured zero gradient allocation.'
        summaries.append(summary)
    design = copy.deepcopy(protocol)
    design['model_path'] = '<local-original-model>'
    for split in ('train', 'valid'):
        design[split]['path'] = f'<local-reconstructed-data>/{split}.jsonl'
    design['recorded_protocol_sha256'] = digest(root / 'protocol.json')
    design['export_note'] = 'Paths replaced for sharing. This is a protocol design, not the original hash-identical execution protocol. Use prepare to freeze a new local replication.'
    write(destination / 'protocol-design.json', design)
    write(destination / 'analysis.json', analysis)
    write(destination / 'run-measurements.json', summaries)
    for name in ['host-environment.json', 'prior-art.json', 'measurement-errata.json', 'failure-ledger.json', 'execution-audit.json', 'execution-context.json']:
        if (root / name).exists():
            shutil.copyfile(root / name, destination / name)
    for name, expected in protocol['sources_sha256'].items():
        source = root / 'source' / name
        if digest(source) != expected:
            raise ValueError('Frozen source snapshot changed')
        target = destination / 'executed-source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if (root / 'comparison.png').exists():
        shutil.copyfile(root / 'comparison.png', destination / 'comparison.png')
    write(destination / 'export-receipt.json', {
        'status': 'local review artifact; not published', 'runs': len(summaries),
        'no_datasets_or_weights_exported': True,
        'original_protocol_sha256': digest(root / 'protocol.json'),
        'note': 'Local complete records/checkpoints remain private. Hashes and aggregate exports provide provenance, not independent authentication of the measurements.',
        'exported_files_sha256': {str(path.relative_to(destination)): digest(path)
                                  for path in sorted(destination.rglob('*')) if path.is_file()}})
    print(json.dumps({'exported_runs': len(summaries), 'destination': str(destination)}, indent=2))


if __name__ == '__main__':
    main()

"""Audit shared study exports without loading models, data or checkpoint arrays."""
import json
import math
from pathlib import Path
from .report_bundle import digest

def audit_export(root):
    root = Path(root).resolve()
    receipt = json.loads((root / 'export-receipt.json').read_text())
    for name, expected in receipt['exported_files_sha256'].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink() or digest(path) != expected:
            raise ValueError('Changed or escaped export artifact: ' + name)
    analysis = json.loads((root / 'analysis.json').read_text())
    design = json.loads((root / 'protocol-design.json').read_text())
    rows = json.loads((root / 'run-measurements.json').read_text())
    expected = {row['id'] for row in design['runs']}
    if len(expected) != len(design['runs']) or len(rows) != len(expected) or {row['run']['id'] for row in rows} != expected or (analysis['runs_audited'] != len(expected)):
        raise ValueError('Incomplete or duplicate run coverage')
    if analysis['protocol_sha256'] != receipt['original_protocol_sha256'] or analysis['protocol_sha256'] != design['recorded_protocol_sha256']:
        raise ValueError('Protocol/export identity differs')
    for row in rows:
        if row['local_complete_result_sha256'] != analysis['source_result_sha256'][row['run']['id']]:
            raise ValueError('Run provenance differs from analysis')
        for field in ('elapsed_seconds', 'mlx_measured_peak_bytes'):
            value = row[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid measurement')
    for field in ('all_numerical_equivalence_gates_passed', 'primary_joint_gate_passed'):
        if type(analysis[field]) is not bool:
            raise ValueError('Missing boolean acceptance decision')
    if analysis['primary_joint_gate_passed'] and (not analysis['all_numerical_equivalence_gates_passed']):
        raise ValueError('Positive primary decision contradicts numerical rejection')
    measurements = {(row['run']['rank'], row['run']['seed'], row['run']['method']): row for row in rows}
    for comparison in analysis['comparisons']:
        for block in comparison['paired_seed_blocks']:
            for field, value in block.items():
                if not field.startswith(('elapsed_', 'peak_')) or '_over_' not in field:
                    continue
                prefix, rest = field.split('_', 1)
                numerator, denominator = rest.split('_over_', 1)
                metric = 'elapsed_seconds' if prefix == 'elapsed' else 'mlx_measured_peak_bytes'
                computed = measurements[comparison['rank'], block['seed'], numerator][metric] / measurements[comparison['rank'], block['seed'], denominator][metric]
                if type(value) not in (int, float) or not math.isclose(value, computed, rel_tol=1e-09, abs_tol=1e-12):
                    raise ValueError('Analysis paired ratio differs from recorded measurements')
    return {'study': root.name, 'export_files_verified': len(receipt['exported_files_sha256']), 'completed_runs': len(rows), 'primary_joint_gate_passed': analysis['primary_joint_gate_passed'], 'numerical_gates_passed': analysis['all_numerical_equivalence_gates_passed'], 'comparisons': analysis['comparisons'], 'novelty_status': analysis['novelty_status'], 'scope': 'Shared file consistency and coverage only. Original checkpoint-array audits are reported evidence, not independently repeated by this command.'}

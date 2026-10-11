"""Append a completed independent memory/time comparison to the review note."""
import argparse
import json
from pathlib import Path


def interval(stats):
    lo, hi = stats['paired_bootstrap_95_ci']
    return f"{stats['geometric_mean_ratio']:.4f} [{lo:.4f}, {hi:.4f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--note', type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads((args.study / 'analysis.json').read_text())
    execution = json.loads((args.study / 'execution-audit.json').read_text())
    if analysis['runs_audited'] != 36 or execution['completed_runs'] != 36:
        raise ValueError('Complete numerical and execution audits are required')
    primary = next(row for row in analysis['comparisons'] if row['rank'] == 256)
    maximum_adapter = max(error['max_adapter_error'] for row in analysis['comparisons'] for block in row['paired_seed_blocks'] for error in block['equivalence'].values())
    maximum_moment = max(error['max_optimizer_moment_error'] for row in analysis['comparisons'] for block in row['paired_seed_blocks'] for error in block['equivalence'].values())
    maximum_nll = max(abs(error['validation_nll_difference']) for row in analysis['comparisons'] for block in row['paired_seed_blocks'] for error in block['equivalence'].values())
    rows = '\n'.join(f"| {row['rank']} | {interval(row['memory'])} | {interval(row['elapsed'])} | {'passed' if row['memory_reduction_gate_passed'] else 'failed'} | {'passed' if row['runtime_noninferiority_gate_passed'] else 'failed'} |" for row in analysis['comparisons'])
    benefit = 100 * (1 - primary['memory']['geometric_mean_ratio'])
    if analysis['primary_joint_gate_passed']:
        decision = f'The independently frozen rank-256 comparison meets its local memory/time tradeoff gate: observed geometric-mean allocator-peak reduction {benefit:.2f}%, with the paired interval excluding a ratio of one. This supports an empirical implementation result in the tested workload. It does not establish a novel recomputation algorithm, a speed improvement, or superiority over complete offload systems in the literature.'
    else:
        decision = 'The primary joint memory/time gate did not pass. The complete comparison remains a negative or mixed result, and no overall efficiency benefit should be claimed.'
    marker = '\n## Independent deferred-recomputation confirmation\n'
    text = args.note.read_text().split(marker)[0]
    text += marker + f'''

A third frozen study adds 36 completed fresh-process runs, bringing this review artifact to 108 completed runs. It was prepared after the first two studies and a post-hoc root-lifetime diagnostic, but before any of its own confirmation measurements. Seeds 101/113/127/139/151/163 and the next eligible 16/8 training/validation rows are disjoint from the two preceding mechanical studies. They are not fresh independent task-quality data.

The compared methods are full Adam, single-pass endpoint retention, and deferred group-gradient recomputation. The latter differentiates one group's adapters at a time, stages every group's gradient on local disk before any update, then updates all groups. Each logical update covers every layer with identical examples, masks and pre-update weights. It uses four backward passes per microbatch rather than one. Recomputation is established prior art; this is a specific MLX/M1 empirical comparison with a complete state audit.

Rank 256 is the prespecified primary condition. Success requires observed geometric-mean peak reduction >=3%, a 95% paired-bootstrap memory interval upper bound <1, numerical equivalence, and elapsed-ratio interval upper bound <=5. The fivefold ceiling explicitly accepts a costly memory-saving tradeoff; it is not the earlier normalization experiment's 10% time tolerance. Both designs and their different criteria were frozen before their respective runs, and earlier failures remain in this report. No measured confirmation result was used to stop early or change these thresholds.

| Rank | MLX peak recomputed / endpoint, mean [95% interval] | Elapsed recomputed / endpoint, mean [95% interval] | Memory gate | Time-cost gate |
|---|---|---|---|---|
{rows}

{decision}

All candidate states were compared against the paired full-Adam oracle. Numerical gates {'passed' if analysis['all_numerical_equivalence_gates_passed'] else 'failed'}; maximum adapter error {maximum_adapter:g}, maximum Adam-state error {maximum_moment:g}, maximum completion-NLL difference {maximum_nll:g}. Agreement is limited to six-step tested configurations, not long-run convergence.

The warm measured loop has 32 optimizer-state reads and writes for deferred recomputation versus 12 for endpoint retention. Temporary gradient files add separate reads/writes; these are not included in the `io` state-operation dictionary. Whole-loop time includes both types of I/O. The frozen worker's zero `full_gradient_resident_bytes` for engines without that history field is a missing-measurement placeholder, not a zero allocation. MLX peak, not that placeholder, is the primary memory endpoint.

### Execution accounting

The execution policy was frozen before starting. A supervisor records every attempt's whole-process time and log hashes and permits at most one retry for the exact Metal interactivity signature. The completed journal contains {execution['attempts_count']} attempts, {execution['failed_attempts_count']} failed attempts, and {execution['sum_whole_process_seconds_including_failures']:.2f} summed whole-process seconds including failures. That sum includes hashing, load, warmup, validation and checkpoints but excludes supervisor cooldown and between-process work; it is not total job wall time. Only the four-update synchronized loop is used for the paired timing endpoint. No unfavorable numerical or timing result is retried.

Concurrent document editing, lightweight CPU checks, token reconstruction and plotting of earlier studies took place on the shared machine; no second project MLX training job ran concurrently. User applications remained open, and thermal state was not measured. Timing intervals are descriptive and conditional on these blocks and this machine, not a causal general speed guarantee.

### Why the preceding buffer changes were insufficient

A separate post-hoc seed-17/rank-256 trace observed the peak during the second gradient accumulation, before whole-tree normalization. Dropping the extra Python gradient root before evaluation reduced active allocation after that accumulation by 146,800,640 bytes, while cumulative peak remained 1,894,636,028 bytes in both single-update traces. Thus lower phase-specific live allocation did not imply lower whole-loop peak. The traces are diagnostic, not confirmation runs or a complete allocator timeline. Their source and limitations are preserved in `results/gradient-root-phase-pilot-v1/`.

The strongest current contribution is the exact-state, reproducible empirical artifact and its measured memory/time boundary, including retained negative results. Novel algorithmic priority remains unestablished. The practical next comparison should expand models, context lengths, ranks and independent replications rather than relabel established offloading as a discovery.

Follow [independent-recomputation reproduction](../docs/recomputation-reproduction.md). All 36 measurements, exact executed source, sanitized protocol design, data reconstruction manifest and execution audit are in `results/recomputation-study-v1/`. No raw texts, token rows, model weights or adapters are distributed. This remains a local review draft; external replication and scientific review have not occurred.
'''
    args.note.write_text(text)
    print(json.dumps({'runs_added':36, 'primary_joint_gate_passed':analysis['primary_joint_gate_passed'], 'observed_primary_peak_reduction_percent':benefit}, indent=2))


if __name__ == '__main__':
    main()

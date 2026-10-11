"""Build a review draft from completed audits, preserving failed success gates."""
import argparse
import json
from pathlib import Path


def interval(stats):
    lo, hi = stats['paired_bootstrap_95_ci']
    return f"{stats['geometric_mean_ratio']:.4f} [{lo:.4f}, {hi:.4f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint-study', type=Path, required=True)
    parser.add_argument('--normalization-study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    e = json.loads((args.endpoint_study / 'analysis.json').read_text())
    n = json.loads((args.normalization_study / 'analysis.json').read_text())
    if e['runs_audited'] != 36 or n['runs_audited'] != 36:
        raise ValueError('Both frozen studies must have all 36 runs audited')
    primary = next(row for row in n['comparisons'] if row['rank'] == 256)
    numerical = e['all_numerical_equivalence_gates_passed'] and n['all_numerical_equivalence_gates_passed']
    max_adapter = max(error['max_adapter_error'] for study in (e, n)
                      for rank in study['comparisons'] for block in rank['paired_seed_blocks']
                      for error in block['equivalence'].values())
    max_moment = max(error['max_optimizer_moment_error'] for study in (e, n)
                     for rank in study['comparisons'] for block in rank['paired_seed_blocks']
                     for error in block['equivalence'].values())
    max_nll = max(abs(error['validation_nll_difference']) for study in (e, n)
                  for rank in study['comparisons'] for block in rank['paired_seed_blocks']
                  for error in block['equivalence'].values())
    endpoint_table = '\n'.join(f"| {row['rank']} | {interval(row['elapsed'])} | {'passed' if row['speed_gate_passed'] else 'failed'} |"
                               for row in e['comparisons'])
    normalization_table = '\n'.join(f"| {row['rank']} | {interval(row['memory'])} | {interval(row['elapsed'])} | {'passed' if row['memory_reduction_gate_passed'] else 'failed'} | {'passed' if row['runtime_noninferiority_gate_passed'] else 'failed'} |"
                                    for row in n['comparisons'])
    if n['primary_joint_gate_passed']:
        decision = 'The rank-256 implementation meets the prespecified local memory/runtime gate. This supports a bounded implementation benefit on the measured machine, not an optimizer priority or general hardware claim.'
    else:
        decision = 'The rank-256 implementation does not meet the prespecified joint memory/runtime gate. It must not be presented as a demonstrated overall efficiency improvement. The numerical and tensor-lifetime results remain separate from this failed performance hypothesis.'
    ledger_path = args.normalization_study / 'failure-ledger.json'
    failure_text = 'No device failures recorded.'
    if ledger_path.exists():
        failure_text = ('One full-Adam rank-4/seed-23 attempt failed inside the measured loop with Metal `ImpactingInteractivity`. '
                        'The failed directory and exact signature were preserved. An explicit post-start execution addendum allowed one identical fresh-process retry; it completed. '
                        'The failed duration was not saved. Comparisons describe completed runs, not retry-inclusive job time. '
                        'The first recovery note mislabeled the stage as warmup; traceback inspection corrected it to the measured loop. '
                        'No driver timeout override, screen setting or algorithm configuration was changed.')
    text = f'''# Memory lifetimes in synchronous optimizer offloading on Apple M1

Local research note for review — project: megern/mac-local-training. Not published or peer reviewed.

## Abstract

We implement and audit synchronous grouped Adam variants on an Apple M1 MacBook Pro with 16 GiB unified memory using original Qwen3-0.6B BF16 weights. Two frozen studies contain 72 completed fresh-process runs: six seeds, ranks 4 and 256, three methods per study, balanced method ordering, two warmup and four measured updates. All numerical equivalence gates {'passed' if numerical else 'did not pass'}. Maximum final adapter error was {max_adapter:g}, maximum Adam-state error {max_moment:g}, and maximum completion-NLL difference {max_nll:g}. Replicated endpoint retention reduces warm state-read and state-write operation counts by 25%, but its speed gate {'passed' if e['primary_joint_gate_passed'] else 'failed'}. Group-local token normalization's primary joint memory/runtime gate {'passed' if n['primary_joint_gate_passed'] else 'failed'}. These are local implementation measurements, not a novel optimizer or improved task-quality result.

## Contribution and prior art

The contribution is an original reproducibility artifact: an MLX implementation, a complete Adam-state audit, balanced local measurements, hash-pinned data reconstruction, and explicit negative-result and failure reporting. Ascending/descending subgroup alternation is directly described in [MLP-Offload, section 3.2](https://arxiv.org/html/2509.02480v1) and is credited to that work. [ZeRO-Offload](https://arxiv.org/abs/2101.06840), [ZeRO-Infinity](https://arxiv.org/abs/2104.07857), [Deep Optimizer States](https://arxiv.org/html/2410.21316v2) and [MemAscend](https://arxiv.org/html/2505.23254v1) establish offloading, scheduling and memory-lifetime optimization as prior research directions. No claim that our method is first or superior to those complete systems follows from this port and benchmark. Group-local scalar normalization is an implementation hypothesis; broader algorithmic novelty has not been established.

## Methods

The frozen base and every q/v LoRA adapter remain resident. We compare full-adapter Adam, fixed-order moment offload and endpoint-retained offload in study one. Study two compares full Adam, endpoint-retained offload and group-local token normalization. The local variant retains raw accumulated gradients, checks them before any update, divides the current group by the same supervised-target count and releases consumed raw entries. It changes buffer lifetime rather than the gradient or Adam recurrence. It does not introduce global clipping, different loss scaling or base-weight quantization.

Hyperparameters are fixed: learning rate 0.0002, scale 1, one-example microbatches, two accumulation steps, seven layers per offload group, maximum input length 96, six total updates, memory allocation guard 6 GiB. Each seed/rank pair uses identical initialization, masks, examples, optimizer hyperparameters and per-layer exposure across methods. Rank 256 deliberately stresses optimizer-state costs and is not representative of typical small-rank LoRA.

The 16 training and eight validation examples are reconstructed by exact IDs from [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), Almeida and Hidalgo (2011), DOI 10.24432/C5CC84, CC BY 4.0. They are reused mechanical-validation data, not a fresh quality test. A single true label token is supervised for each example. No reserved classification test is loaded. Do not count repeated examples across seeds and methods as independent cases.

Each measured worker is a fresh sequential process. The external synchronized four-update timer includes gradients, group activation, GC, cache clearing and state I/O; it excludes loading, warmup, validation and checkpoints. MLX peak allocation is reset after warmup and read before validation. Operation counts and file sizes record logical I/O, not physical SSD traffic. Available system memory, swap, process RSS and load average are recorded around the loop. Thermal state was not logged, correcting imprecise wording in the original protocol limitation. No energy measurement was performed. Work ran on a shared machine, without closing user applications.

## Bounded arguments

Given fixed pre-update gradients and independent Adam state, group update functions commute. For one resident group, visiting G distinct groups requires at least G-1 switches per update. Alternating order starts with the previous endpoint and attains T(G-1) switches over T cold updates; fixed order uses TG-1 for G>1. After warmup with G=4, the measured counts are 12 versus 16 for four updates. This is a count bound, not a runtime guarantee, and the alternating-order principle has direct prior art.

For P bytes of raw gradients and largest group p bytes, full normalization temporarily retains two full gradient trees (2P). Group-local normalization bounds these phase-specific gradient tensors by P+p and releases consumed inputs. For four equal groups this is 1.25P. Backward activations and accumulation can set a larger whole-loop peak; a phase bound must not be reported as physical RAM savings. See [normalization argument](../docs/normalization-proof.md) and [scheduling proof](../docs/endpoint-proof.md).

## Numerical audit

All 72 completed runs are accounted for; candidate states are compared against the paired full-Adam oracle, including every adapter tensor, first/second moment, step scalar, target count and 28-layer update coverage. Checkpoint manifests and source/model/data hashes are verified. Acceptance is adapter/moment maximum absolute error <=1e-5 and NLL absolute difference <=1e-5. Observed maxima: adapters {max_adapter:g}, moments {max_moment:g}, NLL {max_nll:g}. This establishes implementation agreement in the tested six-step configurations, not long-run convergence or independent task utility.

## Scheduling results

Ratios below one favor endpoint retention. Each interval is a 95% paired bootstrap interval of the geometric mean over six seed blocks (20,000 draws, seed 20261012).

| Rank | Elapsed endpoint / fixed, mean [interval] | Speed gate |
|---|---|---|
{endpoint_table}

State-read and state-write operation counts decrease by 25% in all pairs. The prespecified primary speed gate requires an observed geometric mean reduction >=10% and an interval upper bound <1, with memory nonincrease within 5%. Savings in operations alone do not satisfy this gate. The whole Adam optimizer and the complete offload systems in the literature are not benchmarked here.

## Group-local normalization results

Ratios below one favor local normalization over the stronger endpoint-retained baseline. The primary rank-256 gate requires an observed geometric mean MLX-peak reduction >=3%, its interval upper bound <1, and elapsed interval upper bound <=1.10. Rank 4 is a control, not a second primary endpoint. Failure of a gate is retained rather than redefining the target.

| Rank | Peak allocation ratio, mean [interval] | Elapsed ratio, mean [interval] | Memory gate | Runtime gate |
|---|---|---|---|---|
{normalization_table}

{decision}

## Execution failures and scope

{failure_text}

An earlier scheduling pilot stopped after saving its first run because progress printing incorrectly passed `flush` to JSON encoding. That pilot is preserved and excluded from the fixed 36-run scheduling confirmation; the complete design was refrozen and rerun unchanged after the logging correction. No candidate outcome was used to change thresholds.

Six sequential paired blocks on one machine are not six independent machines. Balanced order reduces systematic ordering effects but does not control background activity, thermal drift or paging. Bootstrap intervals are descriptive conditional on this small block set. Numerical agreement, lower component byte counts, lower total allocation and shorter elapsed time are distinct claims. This study does not establish a first-of-kind method, a general large-model fit result, independent cybersecurity quality or production adoption.

## Reproduce and review

Follow the [reproduction guide](../docs/endpoint-reproduction.md) with locally obtained original model and UCI source files. The data manifest reconstructs identical token rows and rejects changed inputs. Frozen source snapshots, protocol designs, all completed measurements, analysis, host metadata and failure/errata ledgers are under `results/endpoint-study-v2/` and `results/normalization-study-v1/`. Raw SMS data and weights remain local. Shared aggregate hashes provide provenance, not independent authentication of recorded measurements. Human review and external replication remain necessary before stronger publication claims.
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(json.dumps({'completed_runs': 72, 'numerical_gates_passed': numerical,
                      'normalization_primary_joint_gate_passed': n['primary_joint_gate_passed']}, indent=2))


if __name__ == '__main__':
    main()

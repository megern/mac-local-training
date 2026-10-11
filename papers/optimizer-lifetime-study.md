# Memory lifetimes in synchronous optimizer offloading on Apple M1

Local research note for review — project: megern/mac-local-training. Not published or peer reviewed.

## Abstract

We implement and audit grouped Adam training on an Apple M1 MacBook Pro with 16 GiB unified memory and original Qwen3-0.6B BF16 weights. Three internally frozen studies contain 108 completed fresh-process runs, with six seeds, ranks 4 and 256, balanced method ordering, two warmup and four measured updates per study. Every completed numerical-equivalence gate passed, with zero observed final adapter, Adam-state and completion-NLL errors in the tested six-step configurations. Endpoint retention reduces optimizer-state operation counts by 25% but fails its speed gate; group-local normalization fails its memory/runtime gate. An independently frozen deferred-recomputation comparison reduces rank-256 MLX allocator peak by 7.11% versus single-pass endpoint retention, at 2.35 times the measured-loop duration. The respective 95% paired ratio intervals are [0.92847, 0.92917] and [1.97183, 2.88091]. Rank-4 allocation savings are only 0.81%, below the 3% memory gate. The contribution is the exact-state reproducible local empirical artifact and memory/time boundary, not a new optimizer, physical-RAM percentage, improved task quality or algorithmic-priority claim.

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

All 72 completed runs are accounted for; candidate states are compared against the paired full-Adam oracle, including every adapter tensor, first/second moment, step scalar, target count and 28-layer update coverage. Checkpoint manifests and source/model/data hashes are verified. Acceptance is adapter/moment maximum absolute error <=1e-5 and NLL absolute difference <=1e-5. Observed maxima: adapters 0, moments 0, NLL 0. This establishes implementation agreement in the tested six-step configurations, not long-run convergence or independent task utility.

## Scheduling results

Ratios below one favor endpoint retention. Each interval is a 95% paired bootstrap interval of the geometric mean over six seed blocks (20,000 draws, seed 20261012).

| Rank | Elapsed endpoint / fixed, mean [interval] | Speed gate |
|---|---|---|
| 4 | 0.8923 [0.7707, 1.0532] | failed |
| 256 | 0.9030 [0.7416, 1.0680] | failed |

State-read and state-write operation counts decrease by 25% in all pairs. The prespecified primary speed gate requires an observed geometric mean reduction >=10% and an interval upper bound <1, with memory nonincrease within 5%. Savings in operations alone do not satisfy this gate. The whole Adam optimizer and the complete offload systems in the literature are not benchmarked here.

## Group-local normalization results

Ratios below one favor local normalization over the stronger endpoint-retained baseline. The primary rank-256 gate requires an observed geometric mean MLX-peak reduction >=3%, its interval upper bound <1, and elapsed interval upper bound <=1.10. Rank 4 is a control, not a second primary endpoint. Failure of a gate is retained rather than redefining the target.

| Rank | Peak allocation ratio, mean [interval] | Elapsed ratio, mean [interval] | Memory gate | Runtime gate |
|---|---|---|---|---|
| 4 | 1.0001 [1.0000, 1.0004] | 1.3660 [1.0461, 1.6739] | failed | failed |
| 256 | 1.0000 [1.0000, 1.0000] | 1.1038 [0.6871, 1.8943] | failed | failed |

The rank-256 implementation does not meet the prespecified joint memory/runtime gate. It must not be presented as a demonstrated overall efficiency improvement. The numerical and tensor-lifetime results remain separate from this failed performance hypothesis.

## Execution failures and scope

One full-Adam rank-4/seed-23 attempt failed inside the measured loop with Metal `ImpactingInteractivity`. The failed directory and exact signature were preserved. An explicit post-start execution addendum allowed one identical fresh-process retry; it completed. The failed duration was not saved. Comparisons describe completed runs, not retry-inclusive job time. The first recovery note mislabeled the stage as warmup; traceback inspection corrected it to the measured loop. No driver timeout override, screen setting or algorithm configuration was changed.

An earlier scheduling pilot stopped after saving its first run because progress printing incorrectly passed `flush` to JSON encoding. That pilot is preserved and excluded from the fixed 36-run scheduling confirmation; the complete design was refrozen and rerun unchanged after the logging correction. No candidate outcome was used to change thresholds.

Six sequential paired blocks on one machine are not six independent machines. Balanced order reduces systematic ordering effects but does not control background activity, thermal drift or paging. Bootstrap intervals are descriptive conditional on this small block set. Numerical agreement, lower component byte counts, lower total allocation and shorter elapsed time are distinct claims. This study does not establish a first-of-kind method, a general large-model fit result, independent cybersecurity quality or production adoption.

## Reproduce and review

Follow the [reproduction guide](../docs/endpoint-reproduction.md) with locally obtained original model and UCI source files. The data manifest reconstructs identical token rows and rejects changed inputs. Frozen source snapshots, protocol designs, all completed measurements, analysis, host metadata and failure/errata ledgers are under `results/endpoint-study-v2/` and `results/normalization-study-v1/`. Raw SMS data and weights remain local. Shared aggregate hashes provide provenance, not independent authentication of recorded measurements. Human review and external replication remain necessary before stronger publication claims.

## Independent deferred-recomputation confirmation


A third frozen study adds 36 completed fresh-process runs, bringing this review artifact to 108 completed runs. It was prepared after the first two studies and a post-hoc root-lifetime diagnostic, but before any of its own confirmation measurements. Seeds 101/113/127/139/151/163 and the next eligible 16/8 training/validation rows are disjoint from the two preceding mechanical studies. They are not fresh independent task-quality data.

The compared methods are full Adam, single-pass endpoint retention, and deferred group-gradient recomputation. The latter differentiates one group's adapters at a time, stages every group's gradient on local disk before any update, then updates all groups. Each logical update covers every layer with identical examples, masks and pre-update weights. It uses four backward passes per microbatch rather than one. The computation-for-memory tradeoff is established, for example in [Chen et al. (2016)](https://arxiv.org/abs/1604.06174); that activation-rematerialization method differs from this repeated adapter-group differentiation. This is a specific MLX/M1 empirical comparison with a complete state audit.

Rank 256 is the prespecified primary condition. Success requires observed geometric-mean peak reduction >=3%, a 95% paired-bootstrap memory interval upper bound <1, numerical equivalence, and elapsed-ratio interval upper bound <=5. The fivefold ceiling explicitly accepts a costly memory-saving tradeoff; it is not the earlier normalization experiment's 10% time tolerance. Both designs and their different criteria were frozen before their respective runs, and earlier failures remain in this report. No measured confirmation result was used to stop early or change these thresholds.

| Rank | MLX peak recomputed / endpoint, mean [95% interval] | Elapsed recomputed / endpoint, mean [95% interval] | Memory gate | Time-cost gate |
|---|---|---|---|---|
| 4 | 0.9919 [0.9913, 0.9924] | 2.0670 [1.8567, 2.4055] | failed | passed |
| 256 | 0.9289 [0.9285, 0.9292] | 2.3502 [1.9718, 2.8809] | passed | passed |

The independently frozen rank-256 comparison meets its local memory/time tradeoff gate: observed geometric-mean allocator-peak reduction 7.11%, with the paired interval excluding a ratio of one. This supports an empirical implementation result in the tested workload. It does not establish a novel recomputation algorithm, a speed improvement, or superiority over complete offload systems in the literature.

All candidate states were compared against the paired full-Adam oracle. Numerical gates passed; maximum adapter error 0, maximum Adam-state error 0, maximum completion-NLL difference 0. Agreement is limited to six-step tested configurations, not long-run convergence.

The warm measured loop has 32 optimizer-state reads and writes for deferred recomputation versus 12 for endpoint retention. Temporary gradient files add separate reads/writes; these are not included in the `io` state-operation dictionary. Whole-loop time includes both types of I/O. The frozen worker's zero `full_gradient_resident_bytes` for engines without that history field is a missing-measurement placeholder, not a zero allocation. MLX peak, not that placeholder, is the primary memory endpoint.

### Execution accounting

The execution policy was frozen before starting. A supervisor records every attempt's whole-process time and log hashes and permits at most one retry for the exact Metal interactivity signature. The completed journal contains 36 attempts, 0 failed attempts, and 1028.34 summed whole-process seconds including failures. That sum includes hashing, load, warmup, validation and checkpoints but excludes supervisor cooldown and between-process work; it is not total job wall time. Only the four-update synchronized loop is used for the paired timing endpoint. No unfavorable numerical or timing result is retried.

Concurrent document editing, lightweight CPU checks, token reconstruction and plotting of earlier studies took place on the shared machine; no second project MLX training job ran concurrently. User applications remained open, and thermal state was not measured. Timing intervals are descriptive and conditional on these blocks and this machine, not a causal general speed guarantee.

### Why the preceding buffer changes were insufficient

A separate post-hoc seed-17/rank-256 trace observed the peak during the second gradient accumulation, before whole-tree normalization. Dropping the extra Python gradient root before evaluation reduced active allocation after that accumulation by 146,800,640 bytes, while cumulative peak remained 1,894,636,028 bytes in both single-update traces. Thus lower phase-specific live allocation did not imply lower whole-loop peak. The traces are diagnostic, not confirmation runs or a complete allocator timeline. Their source and limitations are preserved in `results/gradient-root-phase-pilot-v1/`.

The strongest current contribution is the exact-state, reproducible empirical artifact and its measured memory/time boundary, including retained negative results. Novel algorithmic priority remains unestablished. The practical next comparison should expand models, context lengths, ranks and independent replications rather than relabel established offloading as a discovery.

Follow [independent-recomputation reproduction](../docs/recomputation-reproduction.md). All 36 measurements, exact executed source, sanitized protocol design, data reconstruction manifest and execution audit are in `results/recomputation-study-v1/`. No raw texts, token rows, model weights or adapters are distributed. This remains a local review draft; external replication and scientific review have not occurred.


### Claim calibration

Protocols were internally frozen before their runs, not publicly registered with an independent registry. No family-wise multiple-hypothesis correction was applied across the three sequential studies or ranks. The third primary gate is a prespecified local engineering decision on its independent seed/row blocks, not an omnibus test of a general discovery. The narrower allocator-peak interval does not account for hardware, model or context-length generalization. No algorithmic-priority or universal-efficiency claim is established.

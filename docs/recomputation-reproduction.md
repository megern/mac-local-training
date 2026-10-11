# Deferred recomputation: independent mechanical confirmation

This local study tests an established technique, not a new optimizer. It asks whether recomputing selected adapter-gradient groups on an original-weight Qwen3 model has a measurable memory benefit on an Apple M1 with 16 GiB unified memory, and what time/I/O cost that benefit entails. Prior scheduling and normalization studies retain their negative performance gates.

## Frozen design

`studies/recomputation_study.py` freezes 36 sequential fresh-process runs: full Adam, single-pass endpoint-retained Adam, and deferred group-gradient recomputation; six new seeds 101/113/127/139/151/163; ranks 4 and 256; all six method-order permutations. Every run uses two warmup and four measured logical updates. Original BF16 base weights stay resident. Every adapter layer receives the same number of updates; inactive Adam moments and temporary group gradients are written to local storage.

The new 16 training and eight validation rows are the next eligible rows after the preceding studies' selections, and have no selected-ID overlap with those studies. They remain short, reused task data for numerical validation; there is no independent quality-test claim. No reserved test is scored.

The primary rank-256 hypothesis requires observed geometric-mean MLX-peak reduction at least 3% against endpoint retention, a 95% paired-bootstrap interval upper bound below 1, complete numerical-equivalence acceptance, and elapsed-ratio interval upper bound at most 5. This is an explicitly costly memory-saving tradeoff, not a speed improvement. Rank 4 is the control. The criteria are frozen before these runs; they do not replace the earlier studies' criteria.

All candidate adapter and Adam tensors, target counts, coverage, checkpoints and source/model/data hashes are audited against paired full Adam. Tolerances are 1e-5 for maximum absolute adapter/moment error and completion-NLL difference. Shared-machine memory, swap, load and RSS are observed; thermal state is not measured or controlled. MLX allocator peak is not physical RAM or SSD traffic. The `io` dictionary counts optimizer-state operations only; temporary gradient staging adds file reads/writes beyond that dictionary. A missing `full_gradient_resident_bytes` history field is emitted as a zero placeholder by the frozen worker and must not be interpreted as a measured zero allocation.

## Local replication

Install the optional `[train]` dependencies on an Apple Silicon Mac. Obtain the original model and public UCI SMS data from the links in the exported data manifest, and keep them local. The reconstruction command verifies tokenizer/data hashes before writing token rows.

```sh
python studies/reconstruct_endpoint_data.py \
  --manifest results/recomputation-study-v1/data-manifest.json \
  --source /local/SMSSpamCollection --model /local/Qwen3-0.6B \
  --output /local/reconstructed-new
```

Freeze a new replication from the exported design and the exact reconstructed rows. This validates model/data/source hashes, records the original protocol as provenance, and creates a different local protocol hash. It does not rerun the original skip-and-select routine on already selected rows.

```sh
python studies/prepare_replication.py \
  --design results/recomputation-study-v1/protocol-design.json \
  --model /local/Qwen3-0.6B --data /local/reconstructed-new \
  --output /local/new-study
python studies/supervise_optimizer_study.py \
  --output /local/new-study --worker studies/recomputation_study.py --max-retries 1
python studies/analyze_recomputation_study.py --output /local/new-study
python studies/audit_execution_journal.py --output /local/new-study
```

Use the recorded source version; the preparation helper refuses changed executed source. The original `recomputation_study.py prepare` command is retained for new full-input studies and selects the next eligible rows after the previous studies' selections.

The frozen execution policy permits only one retry for the exact Metal interactivity error, preserves failed directories and logs, and records whole-process time for every attempt. Unfavorable measurements, numerical disagreement and integrity failures are never retried. The measured-loop time remains separate from whole-process/retry-inclusive time. Other errors stop execution. All runs must finish before acceptance is evaluated.

## Correctness scope

Each group differentiates the same loss against the same pre-update weights, examples and masks. All group gradients are saved before any optimizer update. Independent Adam coordinates with the same update counters receive the same gradients as full Adam. Random state is replayed across group passes; this implementation attaches dropout-free adapters. The numerical audit checks the concrete implementation rather than relying on this argument alone.

Failures during partial application require discarding the in-memory engine and resuming the last committed checkpoint. The comparison does not establish long-run convergence, practical original-weight 70B training, general performance across models, or priority over existing offload/recomputation research.

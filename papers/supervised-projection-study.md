# Preserving update equivalence in supervision-aware MLX vocabulary projection

Research/development note, megern. Local original-weight M1 measurements; not peer reviewed. No new-optimizer or algorithmic-priority claim.

## Abstract

On an Apple M1 with 16 GiB unified memory, we test supervised-position projection for original Qwen3-0.6B BF16 LoRA training. Two independently frozen 36-run studies expose a numerical failure of a mathematically equivalent one-row implementation and evaluate a tile-preserving alternative. A post-hoc same-model diagnostic has identical scalar loss but maximum adapter-gradient error 0.03426027298 for one-row projection; padding to 16 rows has zero error in that diagnostic. The subsequent independent confirmation uses new seeds and selected rows. All final adapter, Adam-state and completion-NLL differences are zero in its tested six-step configurations. Its primary rank-256 allocator peak decreases 14.75% and measured-loop duration decreases 17.67% against a dense comparator that already releases redundant gradient roots. The time ratio is 0.823284 [0.783752, 0.866239]. This establishes a bounded implementation benefit, not task-quality improvement, physical-RAM savings or general large-model feasibility.

## Contribution and relationship to prior work

The original artifact is a controlled MLX/M1 implementation and full-state audit connecting loss-level agreement, gradient disagreement and a tested row-padding workaround. This is not an invention of sparse supervision, matrix tiling, optimizer offload or vocabulary-memory optimization. [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) establishes reduction of vocabulary-logit materialization; our simple gathered-position projection is not an implementation or superiority benchmark of CCE. [MLP-Offload](https://arxiv.org/html/2509.02480v1) describes alternating subgroup order, which this project credits as prior art. [MLX issue 1958](https://github.com/ml-explore/mlx/issues/1958) reports earlier batch-shape precision sensitivity and was closed. It is related context, not proof that the same closed bug caused the observed MLX 0.32.3 gradient discrepancy. We did not instrument internal kernel dispatch and do not establish an upstream regression or first discovery. Loss and gradient checks must be separate.

## Methods and frozen decisions

Original model storage is BF16 and not quantized. Frozen base weights and all q/v adapters remain resident. Dense all-adapter Adam is the numerical oracle. Endpoint offload updates all four seven-layer groups from the same pre-update gradients. The first candidate gathers only supervised hidden states before vocabulary projection. The second adds zeros until at least 16 rows are projected and falls back to the original dense loss whenever padding would eliminate the size benefit. The entire transformer context is processed in both cases.

Each study has six distinct seeds, ranks 4/256, all six permutations of three-method order, two warmup and four externally timed updates in fresh sequential processes. Learning rate is 0.0002, scale 1, microbatch one, accumulation two, context 256 and allocation guard 6 GiB. Each update applies to every adapter layer. The second dense comparator already releases redundant microbatch-gradient roots; the comparison therefore isolates padded projection. The all-Adam oracle retains its original dense code.

The first uses seeds 179/191/211/223/239/251 and source selections 32:48 training and 16:24 validation among eligible <=97-token completions. The second uses 263/277/293/307/317/331 and selections 48:64 and 24:32. The public [UCI SMS](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) completion labels are preserved, but token 1000 is repeated after the first token until each sequence has 257 tokens (256 model inputs and one supervised label target). This synthetically changes conditioning and is not a faithful new SMS-quality benchmark. No reserved task-quality test is evaluated.

Primary acceptance, fixed before each study, requires maximum adapter/moment/NLL errors <=1e-5, observed geometric-mean allocator-peak reduction >=3%, its paired-bootstrap 95% ratio interval upper <1 and elapsed-ratio interval upper <=1.10. Rank 4 is a control. The second design was frozen after the exploratory diagnostic but before its own confirmation; no confirmation outcome changed thresholds. An initial prepared tile design was not executed; before execution its comparator was strengthened to include the same root cleanup, and final v2 was frozen separately.

## Mathematical scope

For a row-wise vocabulary projection W and binary target mask m, L=sum(m[t] * CE(W h[t], y[t])). In real arithmetic with finite loss values, removing mask-zero rows leaves the loss and its derivative unchanged. Scattering the gathered hidden-state derivative back preserves conditioning dependencies through the full transformer. Added zero rows receive no loss and zero cotangent. Division uses the same supervised-target count. This reasoning does not prove floating-point equivalence: different operation shapes can select different numerical routes. The first failed study demonstrates why complete optimizer-state auditing is necessary.

## Results

Ratios below one mean lower allocator peak or shorter elapsed update loop. Each interval is a 20,000-draw paired-bootstrap interval over six seed blocks, seed 20261012. Array errors are maximum absolute final tensor differences; every Adam moment and update counter is compared with the paired full-Adam oracle.

| Method | Rank | MLX peak ratio [95% interval] | Elapsed ratio [95% interval] | Adapter error | Adam-state error | NLL difference magnitude |
|---|---|---|---|---|---|---|
| Single-row (failed) | 4 | 0.826099 [0.826099, 0.826099] | 0.825234 [0.820177, 0.829805] | 0.00807661 | 1.34818 | 0.226561 |
| Single-row (failed) | 256 | 0.852501 [0.852501, 0.852501] | 0.845877 [0.814429, 0.878806] | 0.00878523 | 0.498888 | 1.07812 |
| Tile-16 confirmation | 4 | 0.826112 [0.826112, 0.826112] | 0.817544 [0.812026, 0.823545] | 0 | 0 | 0 |
| Tile-16 confirmation | 256 | 0.852512 [0.852512, 0.852512] | 0.823284 [0.783752, 0.866239] | 0 | 0 | 0 |

The first primary gate failed and remains a negative result. The independent tile-16 gate passed. Allocation ratios are identical across six seeds at each rank, so those bootstrap intervals collapse; this does not imply zero uncertainty across machines, contexts or models. No family-wise correction across the project's sequential hypotheses was applied. These are conditional engineering acceptance decisions, not omnibus discovery or population-performance tests.

## Execution and reproducibility

Both new series completed 36 attempts each with no recorded process failures. A frozen supervisor retains logs and whole-process times and allows at most one retry only for the exact Metal interactivity signature. Unfavorable measurements are not retried. Measured-loop time excludes load, warmup, validation and checkpoints; full-process accounting is separate. Model/data/source hashes and checkpoint manifests are checked. The exploratory diagnostic has one seed and one pre-update example; its earlier startup attempt failed configuration validation before collecting measurements and is preserved locally.

Documentation, CPU installation/report checks and data reconstruction ran concurrently on the shared host; no second project MLX workload ran during confirmation. User apps remained open, thermal state and physical SSD traffic were not measured. The performance claim is MLX allocator peak, not total physical RAM or SSD consumption.

Follow [reproduction instructions](../docs/supervised-projection-reproduction.md). Both full aggregate exports, executed source snapshots, data manifests and execution audits are retained under `results/selected-head-study-v1/` and `results/tiled-head-study-v2/`. The diagnostic is under `results/head-gradient-diagnostic-v2/`. Raw SMS/token rows, model weights and adapters remain local. Hashes establish local consistency, not independent authenticity. External replication and other models, natural contexts, supervision densities and longer training remain outstanding.

Together with the three earlier studies, the project now accounts for 180 completed confirmation runs. Earlier speed/normalization failures and the original 7.11%-memory / 2.35-times-duration recomputation result remain in [the preceding note](optimizer-lifetime-study.md); they are not replaced or merged into the new projection claim. The v1.0 release's default supervised-head command uses the tile-16 path, while the failed one-row implementation is retained only as experimental source/evidence.

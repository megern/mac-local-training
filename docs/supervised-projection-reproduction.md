# Reproduce supervision-aware projection studies

All compute is local. The two studies use original Qwen3-0.6B BF16 storage, frozen base weights and q/v LoRA. They do not train all base parameters or demonstrate task-quality gains.

## Design and retained failure

The first 36-run study compares full Adam, dense endpoint offload and single-row supervised-position projection. Each public SMS completion is extended to a synthetic 256-token context by inserting token 1000 repeatedly after its first token. This is a controlled conditioning stress, not a faithful new SMS classification test. Seeds 179/191/211/223/239/251 and ranks 4/256 were fixed before execution.

Single-row projection failed the 1e-5 adapter/moment/NLL equivalence gate despite lower allocation and elapsed time. It must not be presented as an exact substitute. Complete measurements remain in `results/selected-head-study-v1`.

The post-hoc diagnostic differentiates the same initialized model on the same input. Loss values agree, but a one-row projection yields maximum LoRA-gradient error 0.03426027298. Padding that projection to 16/32/64/128/256 rows produced zero error in that single diagnostic. These are exploratory observations, not independent confirmation or a general numerical proof. Related batch-shape precision sensitivity was previously reported in [MLX issue 1958](https://github.com/ml-explore/mlx/issues/1958); that historical issue was closed, and this study does not prove a regression or establish an identical internal cause. Vocabulary-logit memory reduction has established prior art, including [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009).

The second confirmation freezes six new seeds 263/277/293/307/317/331 and new selected rows before execution. It pads supervised-position projection to at least 16 rows, with the unchanged dense loss when padded projection would not be smaller. The dense endpoint comparator also releases redundant gradient roots, so the comparison isolates the padded projection rather than crediting the prior root-lifetime change. Both methods update all adapter groups from the same pre-update gradients; full Adam is the independent state oracle.

Primary rank-256 acceptance is numeric maximum errors <=1e-5, observed geometric-mean allocator-peak reduction >=3%, paired 95% memory-ratio interval upper <1 and elapsed-ratio interval upper <=1.10. Rank 4 is a control. Each study has 36 fresh sequential processes, all six method permutations, two warmups and four measured updates. No reserved task-quality test is scored. Failed attempts and gates are preserved. The prepared but unexecuted first tile design was superseded before any confirmation measurement to strengthen the comparator; source and acceptance criteria in executed v2 are frozen.

## Reconstruct exact public-input rows

Use the optional local training dependencies and existing original model/tokenizer files and the local UCI source member. No dataset/model download occurs in the scripts. Tokenizer/model/source hashes are pinned by the data manifest.

```sh
python studies/reconstruct_endpoint_data.py \
  --manifest results/tiled-head-study-v2/data-manifest.json \
  --source /LOCAL/SMSSpamCollection --model /LOCAL/Qwen3-0.6B \
  --output /LOCAL/tile-data-new
python studies/prepare_replication.py \
  --design results/tiled-head-study-v2/protocol-design.json \
  --model /LOCAL/Qwen3-0.6B --data /LOCAL/tile-data-new \
  --output /LOCAL/tile-study-new
python studies/supervise_optimizer_study.py \
  --output /LOCAL/tile-study-new --worker studies/tiled_head_study.py --max-retries 1
python studies/analyze_tiled_head_study.py --output /LOCAL/tile-study-new
python studies/audit_execution_journal.py --output /LOCAL/tile-study-new
```

Replace tile paths/worker/analyzer with `selected-head-study-v1`, `selected_head_study.py`, and `analyze_selected_head_study.py` to reproduce the failed one-row comparison. Replications create new protocol hashes and retain original hashes as provenance. Use the exact recorded source version; changed sources are rejected. Only the precise Metal interactivity signature permits one retry; unfavorable results are never retried.

For the exploratory diagnostic, first prepare a replication of the one-row study, then run:

```sh
python studies/diagnose_head_gradients.py \
  --protocol /LOCAL/selected-study-new/protocol.json \
  --output /LOCAL/diagnostic-new
```

## Scope

MLX allocator peak is not physical RAM usage. Six blocks on one shared machine are not six independent machines. CPU documentation, installation and lightweight checks ran concurrently, while no second project MLX workload ran during the confirmation. Thermal state and physical SSD traffic were not measured. Repeated synthetic tokens do not establish natural-context generalization. External replication, other models/contexts and a systematic priority review remain outstanding. No publication to LinkedIn occurs through any command.

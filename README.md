# Mac Local Training

## Public preview v0.2.0

Python 3.11+; commands below use a local Mac/Linux shell. Source and reports are public; this is a research/development preview, not a production-security certification.

```sh
git clone --branch v0.2.0 https://github.com/megern/mac-local-training.git
cd mac-local-training
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install .
mac-local-training --help
```

The installed entry point was exercised outside the checkout in a separate local environment. This revision has 17 passing local tests. No hosted CI or cloud compute is used. Report an issue with the command, Python/OS versions, sanitized input and expected/actual behavior; remove credentials and private records first. Benchmarks describe the supplied fixtures and hardware.

An MLX experiment for Apple Silicon: keep base weights frozen, train small LoRA windows in turn, and move inactive Adam state to disk. It compares this schedule with updating all adapters on the same stream of tokens.

Measured runs use an Apple M1 with 16GB unified memory: a tiny, randomly initialized Qwen2 model for software checks, and a local 4-bit Qwen3 0.6B for a short paired LoRA experiment. These runs check optimization and memory behavior. They do **not** establish useful cybersecurity-model quality or that a huge base model fits this Mac.

## Run locally

Training requires an Apple Silicon Mac with a working Metal GPU; planning and doctor use standard Python on other platforms. No cloud training, inference API, hosted experiment logger, or hosted CI is used. Installing dependencies downloads public packages; actual training has no download path.

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m mac_local_training plan --parameters-billions 70 --bits 4
python3 -m mac_local_training train --tiny --train-data examples/train.jsonl --valid-data examples/valid.jsonl --output runs/rotating --steps 40 --checkpoint-every 10
python3 -m mac_local_training train --tiny --train-data examples/train.jsonl --valid-data examples/valid.jsonl --output runs/all --strategy all --steps 40 --checkpoint-every 10
python3 -m unittest discover -s tests -v
```

For a pretrained model, replace `--tiny` with `--model /absolute/local/model`. The directory must already contain compatible config, tokenizer and safetensors files. Supported architecture families are Qwen2, Qwen3, Llama and Mistral with standard attention projections. Custom remote code and Hub identifiers are refused. Quantized local weights follow the same adapter path. Use a much lower learning rate for a pretrained model and choose it from validation results; the default is for the tiny check.

The measured pretrained run uses [mlx-community/Qwen3-0.6B-4bit](https://huggingface.co/mlx-community/Qwen3-0.6B-4bit/tree/73e3e38d981303bc594367cd910ea6eb48349da8), pinned to revision `73e3e38d981303bc594367cd910ea6eb48349da8`, under Apache-2.0. Public model files were downloaded; all training and validation ran on the Mac. Base weights are not redistributed here. After obtaining those files locally, run these commands in separate processes:

```sh
python3 -m mac_local_training train --model /absolute/local/qwen3-0.6b-4bit --train-data examples/train.jsonl --valid-data examples/valid.jsonl --output runs/pretrained-all --strategy all --steps 8 --window-layers 4 --learning-rate 0.0001 --max-seq-length 96 --checkpoint-every 4
python3 -m mac_local_training train --model /absolute/local/qwen3-0.6b-4bit --train-data examples/train.jsonl --valid-data examples/valid.jsonl --output runs/pretrained-rotating --strategy rotating --steps 8 --window-layers 4 --learning-rate 0.0001 --max-seq-length 96 --checkpoint-every 4
```

## How the experiment is divided

### Completion-only targets (local development)

JSONL token rows may provide `loss_start`, the zero-based index of the first
token whose prediction contributes to the loss. For example,
`{"tokens":[1,5,8,13,2],"loss_start":4}` trains only the final token while
retaining all preceding tokens as context. Prompt and padding targets contribute
zero loss; accumulated gradients are divided by the number of supervised targets.
Rows without this field keep the original all-target behavior.

Completion rows exceeding `max_seq_length + 1` tokens are rejected instead of
silently chunking away prompt context. `loss_start` must be an integer between
1 and `len(tokens)-1`. This supports classification and answer-only fine-tuning;
it does not establish task quality. Development checks now include masked-loss
agreement with manually selected cross-entropy terms.

The workspace study scripts in `studies/` prepare fresh UCI SMS groups after
excluding both prior experiments, then run an original-BF16 validation-only
pilot. They depend on the sibling research checkout and existing local model
files. Tokenized data and weights remain local. The reserved test is never
loaded by the pilot runner. Pilot scores are exploratory and cannot establish
algorithm novelty or superiority.

1. Long token rows become causal windows with one shared boundary token. Every next-token target is retained, while attention context resets at the boundary.
2. Microbatches contribute summed token gradients. An update uses the total token count, avoiding a mean-of-means error for unequal lengths.
3. A rotating schedule activates `q_proj` and `v_proj` adapters for a small set of layers. Frozen base weights and every adapter weight remain resident.
4. When the active window changes, its Adam moments are saved locally and its optimizer is released. The next window's moments are restored from disk.
5. Atomic checkpoints retain all adapters, Adam state, the data cursor, RNG state, and configuration/model/data hashes. A resume rejects mismatched inputs or damaged files.

This is a local implementation and evaluation of block-coordinate adapter optimization. LoRA, quantization, gradient accumulation, and layer selection are established techniques. The project makes no priority claim for inventing them. Related work includes [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314), and [LISA](https://arxiv.org/abs/2403.17919).

## What the memory budget means

### Synchronous block Adam (experimental local development)

`train-synchronous` uses the same local training arguments and a rotating window
layout (`--strategy rotating`, the default, and `--window-updates 1`). It first
recomputes gradients for every window on the same examples and pre-update weights,
storing those gradients on disk, then applies all windows' Adam updates. Thus all
layers receive an update for each logical step. It avoids the rotating schedule's
unequal layer coverage by spending extra computation and I/O.

```sh
mac-local-training train-synchronous --tiny --train-data examples/train.jsonl --valid-data examples/valid.jsonl --output runs/synchronous --steps 8 --window-layers 1
```

Tiny-model tests compare adapter arrays, supervised target counts and validation
loss with ordinary all-adapter Adam, and verify checkpoint/resume equivalence.
Separate checkpoint identity prevents treating a rotating checkpoint as a
synchronous one. Gradients are temporary local files; weights and moments remain
in local checkpoints. If applying blocks fails partway, discard that engine and
resume its last committed checkpoint.

This is an experimental implementation of optimizer offload and recomputation,
not a new-algorithm claim or a way to fit a base model larger than memory. Its
training-time measurement includes block state/gradient I/O, unlike the original
runner's training-only measurement. Compare end-to-end measurements instead.

The default 6GiB MLX allocation guard leaves room on a 16GB Mac for the OS and other apps. It is not a hard process RSS limit or a promise that a run will fit. Local weight files exceeding the budget are rejected before loading. Train and validation JSONL files are capped at 64MiB combined because this first loader tokenizes them in memory.

Raw 4-bit weights alone need about 3.26GiB for 7B parameters and 32.60GiB for 70B. Metadata, unquantized tensors, adapters, gradients, activations and optimizer state add more. Splitting updates does not remove the base-weight requirement. This release has no disk-streamed base-model implementation.

## Reading the measurements

See [the measured comparison](results/README.md), [tiny-model JSON](results/benchmark.json), and [pretrained-model JSON with provenance](results/pretrained-benchmark.json). In the short pretrained run, rotating four of 28 layers reduced peak resident Adam state by 85.71%, while peak MLX allocation fell only 3.15% and process RSS increased 11.45%. All-adapter validation NLL fell from 4.2362 to 2.7878; rotating reached 4.1780. Each rotating layer received only one or two updates, versus eight for all-adapter training.

Compare optimizer-resident bytes separately from total MLX allocation and process RSS. Equal token budgets do not give equal parameter-update counts. Training-only throughput excludes state-swap I/O, checkpoint I/O and validation; a separate end-to-end metric includes them but starts after loading and tokenization. Run strategies in separate processes when comparing process high-water memory. The baseline ran first and kernel caches may favor the later run; these timings do not establish a controlled speedup.

The included corpus contains synthetic Arabic/English templates. It is a software check with overlapping templates across splits, not an independent task-generalization benchmark. A held-out NLL change is not a cybersecurity accuracy score.

## العربية

مختبر تدريب محلي على الماك يوزّع تحديث محوّلات LoRA بين مجموعات طبقات، ويحفظ حالة المحسّن للمجموعات غير النشطة على القرص. شغّلنا تجربة صغيرة فعلية على Qwen3 بحجم 0.6B: انخفضت ذاكرة حالة المحسّن مع التدوير، لكن ذاكرة العملية زادت وكان التحسن في فقد التحقق أقل. التجارب قصيرة وعلى بيانات اصطناعية؛ لا تثبت جودة نموذج للأمن السيبراني أو إمكانية تدريب نموذج يتجاوز ذاكرة الجهاز. التدريب والتحقق محليان بالكامل.

Primary implementation references: [MLX-LM LoRA](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) and [MLX memory management](https://ml-explore.github.io/mlx/build/html/python/memory_management.html).

## Adapter-state planning before a run

```sh
python3 -m mac_local_training.cli plan-adapters --trainable-parameters 655360 --active-fraction 0.25
python3 -m unittest discover -s tests
```

The planner separately reports resident FP32 adapter parameters, active gradients and two active Adam moments. For 655,360 parameters, the full state components total 10,485,760 bytes; a quarter-active model totals 4,587,520 bytes while all adapter parameters remain resident. This assumes inactive gradients and optimizer moments are released. These are raw component calculations, not measured RAM savings. Frozen base weights, activations, runtime overhead and OS headroom are excluded. Rotation does not make a 70B original model fit in 16 GiB.


### Portable planning, optional training and three-seed residency checks

```sh
mac-local-training doctor
python3 -m pip install '.[train]'   # Apple Silicon training dependencies only
python3 run_tiny_study.py --output /tmp/tiny-study-new
```

Planning and `doctor` install without MLX. `doctor` discovers platform/modules without initializing Metal or downloading a model. Training requires the optional `[train]` extra on an Apple Silicon Mac. `--seed` records initialization and data-order choices.

`results/tiny-residency-20261011/` contains six fresh-process runs with fixed seeds 11/29/47, 16 updates and matched synthetic data. Resident optimizer bytes are 14,348 for all-adapter updates and 3,596 for rotating updates for every seed. These are optimizer-state bytes, not total process memory. Rotating layers receive fewer updates; the NLL changes differ as expected, so this is not an equal-optimization or useful-model-quality comparison. Sources and protocol were frozen before execution. Weights/checkpoints remain local. See the per-run report for MLX allocations, RSS, token coverage and timing definitions.

Recreate the diagnostic figure with `python3 -m pip install '.[plots]'` and `python3 plot_tiny_study.py`. Plotting reads the published JSON and performs no training.

### Local experimental single-pass grouped Adam

`train-offloaded` uses the same arguments as `train-synchronous` (rotating layout,
window updates 1). It computes all adapter gradients against the same pre-update
weights once per microbatch, then updates groups with disk-backed inactive Adam
moments. The complete gradient tree remains in memory; reports explicitly record
`full_gradient_resident_bytes`. This exchanges gradient residency for avoiding
repeated backwards. It does not offload the frozen base model.

Checkpoint identity differs from ordinary rotating and recomputing synchronous
training. After a failed partial update, discard the object and resume the last
committed checkpoint. Tiny tests cover masked accumulation, frozen base weights,
reference equivalence, resume and incompatible checkpoint rejection. Original
BF16 validation is local and has not established speed gains or novelty.

### Controlled original-weight studies (local review branch)

This branch adds `train-endpoint` and `train-normalized`; both use the same
local arguments and rotating group layout as `train-offloaded`. All adapters
receive one update per logical step from the same pre-update gradients.

- `train-endpoint` alternates group order so the next step starts with the
  retained group. This scheduling idea is explicitly prior art in
  [MLP-Offload](https://arxiv.org/html/2509.02480v1), not an invention here.
- `train-normalized` retains raw accumulated gradients, divides only the current
  group by the total supervised-target count, and releases consumed raw values.
  It avoids a second complete normalized tree. This is a buffer-lifetime
  implementation experiment; overall memory savings must be measured.

Read the [bounded scheduling proof](docs/endpoint-proof.md),
[normalization argument](docs/normalization-proof.md),
[related-work review](docs/endpoint-prior-art.md) and
[reproduction guide](docs/endpoint-reproduction.md). The guide reconstructs exact
SMS token rows from local public inputs; no raw texts or weights are distributed.

Each study fixes 36 fresh-process runs before execution: six seeds, ranks 4/256,
three methods, all six method-order permutations, two warmups and four measured
updates. The external update timer includes state swapping for every method.
Final adapter arrays, Adam moments/counters, target counts and layer coverage
are compared with full Adam. Numerical agreement, operation savings, total
allocation and elapsed time are separate acceptance gates. Failed performance
gates and hardware failures are retained. These short mechanical checks do not
establish task quality, practical 70B fit or production readiness.


The two completed studies contain 72 audited runs. All numerical-equivalence
checks passed, but neither primary performance gate passed. See the
[research note](papers/optimizer-lifetime-study.md) and complete aggregate
[endpoint](results/endpoint-study-v2/analysis.json) and
[normalization](results/normalization-study-v1/analysis.json) evidence. These
local-review results retain every completed run and recorded device failure.

A [post-hoc root-lifetime diagnostic](results/gradient-root-phase-pilot-v1/README.md)
found lower active allocation after accumulation but unchanged global peak;
this is not a demonstrated peak-memory improvement.

A third [deferred-recomputation study](docs/recomputation-reproduction.md)
uses six new seeds and disjoint mechanical-validation selections. Its protocol
requires all 36 runs and a complete state audit before any positive decision.
It explicitly tests a memory/time tradeoff of established recomputation,
without claiming a new optimizer. The execution supervisor preserves every
attempt and allows only one retry for the exact documented Metal error.

`studies/prepare_replication.py` creates a new local protocol from a sanitized
exported design and exact reconstructed data, checking executed-source/model/
data hashes. Original execution and local replication protocols have different
hashes and remain distinguishable. No datasets or weights are distributed.


The third confirmation is complete: **36/36 audited runs**, plus the earlier
72, for **108 completed runs** in the research note. Its primary rank-256
condition lowers MLX allocator peak by **7.11%** versus endpoint retention
(95% paired ratio interval 0.92847–0.92917), while measured-loop time rises to
**2.35×** (interval 1.97–2.88×). All adapter/Adam tensor and completion-NLL errors
were zero in these tested configurations. Rank 4 saves only 0.81% and fails the
3% memory gate. These are single-model, shared-machine measurements; no new
algorithm, physical-RAM percentage or quality improvement is established.
See [complete third-study evidence](results/recomputation-study-v1/analysis.json)
and [execution accounting](results/recomputation-study-v1/execution-audit.json).

# Reproduce the local optimizer-state studies

All training, inference and auditing take place locally. No hosted CI, cloud
GPU or remote inference service is required. The code does not download models
or data. Obtain public files separately, then use existing local copies.

## Inputs and environment

- Apple Silicon and Python >=3.10. The measured host is an M1 with 16 GiB.
- MLX 0.32.3, MLX-LM 0.32.0, psutil 7.2.2; NumPy and safetensors for array audits,
  Transformers for reconstructing token rows, matplotlib for figures.
- Original [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) BF16 files, not an
  MLX-community quantized conversion. The data manifest pins the weight and
  tokenizer hashes. Follow the upstream model's license.
- Local `SMSSpamCollection` member from the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
  Almeida and Hidalgo (2011), DOI 10.24432/C5CC84, CC BY 4.0.

Install the repository and the pinned training extras locally. From this
repository's root, reconstruct the exact 16 training and 8 validation rows:

```sh
python studies/reconstruct_endpoint_data.py \
  --manifest results/endpoint-study-v2/data-manifest.json \
  --source /absolute/local/SMSSpamCollection \
  --model /absolute/local/Qwen3-0.6B \
  --output /absolute/local/reconstructed-data
```

No raw SMS texts, token rows or model/adapter weights are included in the public
artifact. Reconstruction fails if source, tokenizer, candidate boundary or final
tokenized hash differs. These are reused mechanical-validation data; do not use
their NLL as evidence of independent task-quality improvement.

## Freeze and execute the scheduling comparison

Use a fresh local study directory. Preparation records source/model/data hashes,
all six permutations of method order, six seeds, ranks 4 and 256, two warmup and
four measured updates. It refuses to replace an existing protocol.

```sh
python studies/endpoint_study.py prepare \
  --model /absolute/local/Qwen3-0.6B \
  --train-data /absolute/local/reconstructed-data/train.jsonl \
  --valid-data /absolute/local/reconstructed-data/valid.jsonl \
  --output /absolute/local/scheduling-study
python studies/endpoint_study.py run --output /absolute/local/scheduling-study
python studies/analyze_endpoint_study.py --output /absolute/local/scheduling-study
```

The runner uses separate sequential processes. Resuming skips only completed runs
whose protocol identity agrees. A partial failed run must be preserved. By
default restart in a new study directory; an explicit recovery addendum may instead
archive the failed directory, record the exact error/stage and allow a bounded
fresh-process retry without changing configuration. Every such deviation and
failure must appear in the report. Never retry an unfavorable measured result. Completed results and immutable
checkpoints are retained for array audits.

## Freeze and execute the normalization extension

```sh
python studies/normalization_study.py prepare \
  --model /absolute/local/Qwen3-0.6B \
  --train-data /absolute/local/reconstructed-data/train.jsonl \
  --valid-data /absolute/local/reconstructed-data/valid.jsonl \
  --output /absolute/local/normalization-study
python studies/normalization_study.py run --output /absolute/local/normalization-study
python studies/analyze_normalization_study.py --output /absolute/local/normalization-study
```

The comparison includes full Adam, endpoint-retained offload and group-local
normalization. Exact machine-independent times or allocation peaks are not
expected. New absolute paths and dependency/model metadata produce a new
protocol identity; data reconstruction hashes should match the published
manifest. Record OS, chip, available memory and background activity. Warm file
caches remain active; logical I/O counts are not physical SSD traffic.

Read `endpoint-prior-art.md` before citing this work. Alternating group order is
explicitly described in MLP-Offload (2025). The extension tests buffer lifetime
in MLX and makes no optimizer priority or 70B fit claim. Raw results must be
retained, including failures of the prespecified performance gates.

## Reading the sharing artifact

`protocol-design.json` has local paths replaced and records the original protocol
hash. It is not the original hash-identical execution protocol. `run-measurements.json`
contains all completed run measurements, histories and source hashes, without
raw texts, token rows or weights. `executed-source` preserves the prespecified
core files; run a replication from a complete repository checkout, not this
partial source snapshot. Local checkpoint audits verify arrays; the sharing
artifact does not authenticate measurements independently of the local machine.

Consult `measurement-errata.json`: memory/load were logged, but thermal state
was not. The normalization series also has `failure-ledger.json`, documenting
a Metal command-buffer failure during the measured loop and a bounded recovery
addendum. Its incomplete timing was not saved; comparisons are conditional on
completed runs and do not measure retry-inclusive job cost.

For future studies, `supervise_optimizer_study.py` can replace the runner's `run`
step. Invoke it with the frozen worker (`endpoint_study.py` or
`normalization_study.py`), `--output` and `--max-retries 0` or `1`. It journals
whole-process durations and exact stdout/stderr hashes, including failed
attempts. Only the specific Metal interactivity signature can receive one retry;
numerical and source-integrity failures stop the study. Specify this execution
policy before starting a replication. This supervisor was added after the local
series started; it was not used to collect those earlier measurements.


For an exact-data local replication from a sharing artifact, the alternative
`studies/prepare_replication.py` reads `protocol-design.json`, validates original
model/source/data hashes and creates a new protocol using reconstructed rows.
It records the original hash as provenance without impersonating the original
execution. This helper was added after the measured series. The completed
independent recomputation extension has its own
[reproduction guide](recomputation-reproduction.md).

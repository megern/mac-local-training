# Mac Local Training

An MLX experiment for Apple Silicon: keep base weights frozen, train small LoRA windows in turn, and move inactive Adam state to disk. It compares this schedule with updating all adapters on the same stream of tokens.

Measured runs use an Apple M1 with 16GB unified memory: a tiny, randomly initialized Qwen2 model for software checks, and a local 4-bit Qwen3 0.6B for a short paired LoRA experiment. These runs check optimization and memory behavior. They do **not** establish useful cybersecurity-model quality or that a huge base model fits this Mac.

## Run locally

Requires an Apple Silicon Mac with a working Metal GPU. No cloud training, inference API, hosted experiment logger, or hosted CI is used. Installing dependencies downloads public packages; actual training has no download path.

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

1. Long token rows become causal windows with one shared boundary token. Every next-token target is retained, while attention context resets at the boundary.
2. Microbatches contribute summed token gradients. An update uses the total token count, avoiding a mean-of-means error for unequal lengths.
3. A rotating schedule activates `q_proj` and `v_proj` adapters for a small set of layers. Frozen base weights and every adapter weight remain resident.
4. When the active window changes, its Adam moments are saved locally and its optimizer is released. The next window's moments are restored from disk.
5. Atomic checkpoints retain all adapters, Adam state, the data cursor, RNG state, and configuration/model/data hashes. A resume rejects mismatched inputs or damaged files.

This is a local implementation and evaluation of block-coordinate adapter optimization. LoRA, quantization, gradient accumulation, and layer selection are established techniques. The project makes no priority claim for inventing them. Related work includes [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314), and [LISA](https://arxiv.org/abs/2403.17919).

## What the memory budget means

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

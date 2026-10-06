# Measured local runs

[benchmark.json](benchmark.json) contains the first local software check on an Apple M1 with 16GB unified memory. Both strategies used the same seed, 40 updates, 80 microbatches, 7,718 target tokens, and separate fresh processes. The base was a randomly initialized four-layer Qwen2 with hidden size 32; this is deliberately small.

| Measurement | All adapters | Rotating one layer |
|---|---:|---:|
| Peak resident Adam state, bytes | 14,348 | 3,596 |
| Peak MLX allocation, bytes | 7,533,200 | 5,830,279 |
| Process RSS high-water, bytes | 105,185,280 | 110,051,328 |
| Validation NLL before | 5.932984 | 5.932984 |
| Validation NLL after | 4.891327 | 5.487117 |
| Updates received by each layer | 40 | 10 |
| End-to-end elapsed seconds | 1.400936 | 1.270076 |

Resident optimizer state fell by about 74.94% and peak MLX allocation by about 22.61%. Process RSS increased by about 4.63%. The all-adapter run reduced validation loss more. These measurements support a narrow optimizer-state tradeoff, not a blanket claim about total RAM, speed, model quality, or training huge models. Subsecond timings are especially sensitive to startup, caching, and measurement noise.

Training-only timing excludes validation and checkpoint I/O. The JSON includes a second throughput measurement that includes both. Equal update and token budgets give fewer parameter updates per layer under rotation. The synthetic data shares templates across train and validation; no independent cybersecurity task accuracy has been measured.

## Pretrained 4-bit Qwen3 0.6B

[pretrained-benchmark.json](pretrained-benchmark.json) records a second paired run with pinned upstream model provenance and verified weight hashes. Each strategy used 8 updates, 16 microbatches and 525 target tokens at learning rate 0.0001. Rotation selected four of 28 layers at each update. The validation NLL before either run was 4.236217.

| Measurement | All adapters | Rotating four layers |
|---|---:|---:|
| Peak resident Adam state, bytes | 4,587,532 | 655,372 |
| Peak MLX allocation, bytes | 525,940,297 | 509,352,976 |
| Process RSS high-water, bytes | 634,454,016 | 707,084,288 |
| Validation NLL after | 2.787757 | 4.178025 |
| Updates received by each layer | 8 | 1–2 |
| End-to-end elapsed seconds | 6.574926 | 4.415746 |

The optimizer-state reduction was 85.71%, compared with only 3.15% for MLX peak allocation; RSS increased 11.45%. The rotation schedule improved validation loss much less under this very short budget. The baseline ran first, so shared kernel caches and system load may favor later timings. No controlled speedup, independent task accuracy or production usefulness follows from these runs.

Raw checkpoints and full local reports are excluded from Git because they contain local paths and model artifacts. The public summaries retain dataset hashes, model configuration, package versions and the numeric measurements. Reproduce the commands in the project README in separate processes before drawing conclusions about another machine.

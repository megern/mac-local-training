# Single-pass grouped Adam: bounded local result

Original Qwen3 0.6B BF16, seed 17, eight updates, rank 4, four groups.
Acceptance thresholds were frozen before execution: adapter max absolute error
and validation NLL difference at most 1e-5, identical targets and layer coverage.
Observed array error and NLL difference were both **0.0**. No reserved test was scored.

Resident Adam state: 4,587,532 to 1,146,892 bytes (about 75% lower).
Full gradients retained: 2,293,760 bytes. Total MLX peak: 1,874,697,435
to 1,866,358,476 bytes (about **0.45% lower**). This is an allocation measurement,
not a claim about total system RAM or fitting a larger base model.

The candidate recorded 20.18 seconds of updates and 71.19 seconds end to end
(after model/data loading). Historical reference: 5.56 and 21.27 seconds.
These were not contemporaneous controlled runs, and update timing boundaries
differ regarding state I/O. No speed advantage is demonstrated. Avoid interpreting
the numerical ratio as a controlled slowdown either.

The method avoids repeated backwards but adds full-gradient residency and moment
I/O. It is an optimizer-offload implementation, not a new optimizer or discovery.
A useful follow-up would measure larger ranks under counterbalanced runs with
matched timing boundaries. No 70B uncompressed fit, quality improvement or general
large-model training claim follows from this experiment.

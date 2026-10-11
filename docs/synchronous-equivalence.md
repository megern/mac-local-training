# Why deferred block Adam can match all-adapter Adam

This is an implementation argument, not a new optimizer or priority claim.
Block-coordinate optimization and layer selection have prior art, including
[BAdam](https://arxiv.org/abs/2404.02827) and
[LISA](https://arxiv.org/abs/2403.17919). Our synchronous schedule differs from
ordinary rotating updates by using the same pre-update parameters for every
block in one logical step.

## Conditions

Partition adapter parameters into disjoint blocks. Keep base weights frozen.
Use identical initialization, examples, supervised target masks, hyperparameters
and optimizer state. Every block must receive one Adam update per logical step.
For stochastic forwards, replay the same RNG state for each block; the current
LoRA path uses dropout zero. Do not change any adapter until all block gradients
for this step have been computed and stored.

For block b at step t, compute the token-weighted gradient

$$g_{t,b}=\frac{1}{N}\sum_i\nabla_{\theta_b}\ell_i(\theta_t),$$

where N counts supervised targets, including completion masks and excluding
padding. This is exactly the corresponding slice of the all-adapter gradient
in exact arithmetic. Computing gradients after updating an earlier block would
evaluate a different parameter vector and invalidate the argument.

Adam's first and second moments update independently for every parameter:

$$m_{t,b}=\beta_1m_{t-1,b}+(1-\beta_1)g_{t,b},$$
$$v_{t,b}=\beta_2v_{t-1,b}+(1-\beta_2)g_{t,b}^2.$$

With the same per-block time counter t, bias correction and update therefore
match the all-adapter optimizer. By induction from equal initial weights and
moments, each completed logical step matches. Disk storage changes residency,
not these equations. Floating-point implementation differences remain possible;
this is why the software checks compare actual arrays and losses.

## What is tested

Tiny-model tests exercise unequal sequence lengths, unequal completion masks,
gradient accumulation, frozen base parameters and checkpoint/resume across
multiple logical steps. They compare final adapters and validation NLL against
all-adapter Adam. Checkpoint identity includes the synchronous schedule, so
an ordinary rotating run cannot silently resume a synchronous checkpoint.

The local original-BF16 check fixes acceptance before execution: maximum
adapter absolute error at most 1e-5, validation NLL absolute difference at most
1e-5, equal supervised targets and equal per-layer update coverage. It never
loads the reserved classification test. Source hashes and raw measured results
are retained locally for review.

## Limits

The base model and every adapter still remain resident. The method stores
temporary gradients and inactive Adam state on disk and recomputes backward
passes for each block. Reduced optimizer-state residency need not substantially
reduce total allocation. Runtime depends on additional computation, storage,
kernel caching and other machine activity. This is not evidence that an
uncompressed 70B model fits or trains practically on 16 GiB.

A failure during block application must not be treated as a completed update:
discard the current engine object and restore the last committed checkpoint.
The current implementation has no mid-update resume.

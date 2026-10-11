# Related work and claim boundary

The literature review changed the attribution of the endpoint experiment.
It is a replication and MLX port of an existing scheduling idea, not a new idea.

| Primary source | Relevant result | Relationship to this study |
|---|---|---|
| [ZeRO-Offload](https://arxiv.org/abs/2101.06840), 2021 | Places training data structures and optimizer work across host/device resources. | Optimizer offloading predates this project. |
| [ZeRO-Infinity](https://arxiv.org/abs/2104.07857), 2021 | Extends training-state placement to CPU and NVMe, with tiling and overlap. | Disk-backed optimizer state is not an original contribution here. |
| [BAdam](https://arxiv.org/abs/2404.02827), 2024 | Block-coordinate Adam changes which blocks are optimized at each step. | Our synchronous paths update every adapter per logical step; no claim of inventing block training. |
| [Deep Optimizer States](https://arxiv.org/html/2410.21316v2), Middleware 2024 | Schedules optimizer subgroups across CPU/GPU and overlaps transfers. | Subgroup scheduling and performance models have direct precedents. |
| [MLP-Offload](https://arxiv.org/html/2509.02480v1), 2025, section 3.2 | Alternates ascending and descending subgroup update order to exploit cached states. Also delays gradient precision conversion. | **Direct prior art for endpoint retention.** The same alternating-order idea cannot be attributed to us. |
| [MemAscend](https://arxiv.org/html/2505.23254v1), 2025 | Studies system-memory overhead, overflow checks and filesystem costs in SSD-offloaded training. | Memory-lifetime and storage-overhead optimization are established research directions. |
| [RATrain](https://arxiv.org/abs/2606.10415), 2026 | Schedules training-state lifecycles on a constrained heterogeneous platform. | Resource-aware scheduling itself is not a priority claim. |
| [MLX unified memory documentation](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) | CPU and GPU arrays share an Apple Silicon memory pool. | CPU placement alone must not be reported as a reduction in physical unified-memory usage. |

Our original artifact contribution can be the local implementation, numerical
state audit and replicated M1 measurements with explicit negative-result gates.
The group-local token-normalization extension avoids a second complete gradient
tree while retaining accumulation and the same scalar division. This is an
implementation hypothesis, not established algorithmic novelty. MLP-Offload's
delayed conversion is a close conceptual precedent; memory-efficient in-place
and grouped update methods also require comparison before any priority claim.

Searches included optimizer offload, subgroup reordering, reverse-order caching,
MLX training, gradient accumulation and delayed normalization. This was a
focused primary-source review, not an exhaustive systematic review or a
certification that a method is first. Dates follow the original submission or
conference where available, rather than search-engine crawl dates.

The protocol's original `novelty_status` was already 'not established'. The
subsequent discovery of direct prior art is preserved in `prior-art.json` as an
addendum; the frozen protocol and its acceptance criteria were not rewritten.

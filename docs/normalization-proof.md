# Group-local token normalization

This is a memory-lifetime implementation change, not a different optimizer.
Scalar loss normalization, gradient accumulation, grouped updates and buffer
reuse are established techniques. No first-of-kind claim is made.

Let S_b be the sum of gradients for adapter group b across all microbatches in a
logical update, and let N>0 be the total number of supervised targets, excluding
padding and masked prompt tokens. The reference computes a complete tree
`g_b = S_b / N`, retaining both S and g until all divisions are evaluated.

The alternative evaluates the complete raw tree S and checks every value for
finiteness **before any optimizer update**. For each group it then performs the
same division `S_b / N`, evaluates this group, releases its raw values, and
applies Adam with that group's original moments and time counter. All S_b were
computed from pre-update parameters. No new backward is performed after any
parameter changes. Group application uses the prior-art endpoint-retention order.

Because N is a positive count, dividing a finite raw gradient by N cannot
increase its magnitude or create overflow. All groups receive the same normalized
gradients as the reference in exact arithmetic. This does not replace global-norm
clipping or loss scaling; those would need their own complete-tree reductions and
would change the assumptions. Raw gradients must still fit in memory.

For a raw gradient tree of P bytes and largest group of p bytes, the fully
evaluated normalization path temporarily retains two complete trees: 2P bytes.
The local path needs at most a raw tree plus one group's output, bounded by P+p
for these gradient tensors, and releases consumed raw entries. With four equal
size groups, the tensor-lifetime bound changes from 2P to 1.25P. This is a bound
on this phase's gradient tensors, **not** on total MLX allocations or physical RAM.
Backward activations, accumulation, model weights, Adam states and allocator
caches can set a higher overall peak and eliminate the practical benefit.

The frozen comparison uses all-adapter Adam as a numerical oracle and the
endpoint-retained offload implementation as the performance baseline. It checks
final adapter arrays, every Adam moment and step scalar, completion NLL, targets
and all-layer coverage. The performance gate is fixed before execution: at least
3% lower geometric mean measured MLX peak at rank 256, its paired bootstrap
upper bound below one, and elapsed-time upper bound no greater than 1.10.
Rank 4 is a control. Every run is retained regardless of direction.

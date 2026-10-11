# Endpoint retention under a one-group state budget

This is a bounded scheduling result within existing optimizer offloading. It is
not a claim of a new optimizer, algorithm priority, or faster large-model training.

## Model and assumptions

There are G disjoint adapter groups. Every logical update computes and evaluates
all gradients at the same pre-update parameters, before changing any parameter.
Adam has no cross-group operation: no global gradient clipping performed after
partial updates, no changing learning rate within a logical update, and the same
time counter and hyperparameters in every group. Full gradients and the frozen
base remain resident. Only one group's Adam moments may remain resident.

Each logical update visits every group exactly once. Loading an absent group
requires evicting the resident group. This model concerns **optimizer-group
switches**, not physical SSD traffic or all memory allocations. It permits
retaining a group across update boundaries. Checkpoints copy state without
forcing eviction. Kernel and OS page caches remain active.

## Update equivalence

Once gradients are fixed, the update for group b reads only its own parameters,
gradient, moments and time counter. Its write set is disjoint from all other
groups. Therefore the group update functions commute. A permutation of their
application has the same final state in exact arithmetic. By induction, the same
initial state and examples yield the same states after every logical update.
Floating-point equality must still be checked on the actual implementation; this
argument does not cover stochastic or coupled optimizer implementations.

## Minimum switch count

Within each update, G distinct groups must be visited. At least G-1 switches
are required, regardless of its first group. Over T updates, the lower bound is
T(G-1). A sequence that alternates `0,...,G-1` and `G-1,...,0` starts each
subsequent update with the group retained from the previous update. It uses
exactly G-1 switches inside every update and zero at update boundaries, attaining
the lower bound. The first empty-cache activation is not an eviction.

For G>1, repeatedly applying the fixed order incurs TG-1 switches from cold:
T(G-1) inside updates and T-1 at boundaries. Thus endpoint retention saves T-1
switches. For G=1, both use zero switches.

After warmup initializes every group's Adam state, the fixed-order schedule
switches G times per measured update and endpoint retention switches G-1 times.
With G=4 this reduces **state-read and state-write operation counts by 25%**.
For equal-size states it also reduces logical payload movement by 25%; file
headers can introduce small byte differences. Neither statement guarantees a
25% runtime reduction. When backward computation, GC, synchronization, page
faults or other applications dominate, speed may change little or worsen.

## Validation

`tests/test_endpoint.py` compares the bound with exhaustive order enumeration
for G=1..4 and T=1..3. It also compares masked accumulation, final adapter arrays,
Adam moments and frozen base weights against all-adapter Adam. Resume is tested
at both traversal endpoints. An injected partial-apply failure must leave the
last checkpoint unchanged and allow correct recovery in a fresh engine.

The controlled local study fixes 36 fresh-process runs before execution, checks
adapter and moment arrays against the full Adam reference, and separately
measures synchronized elapsed time, MLX allocations, state operation counts and
logical bytes. No reserved task-quality test is evaluated.

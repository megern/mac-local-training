# Post-hoc gradient-root diagnostic

Two warmups followed by one instrumented rank-256 update, seed 17. This is not a confirmation study or speed benchmark. The endpoint trace preceded the root-release trace. Both use the same original model and pinned numerical-validation rows.

Releasing the redundant Python gradient root reduced active allocation after the second accumulation by 146,800,640 bytes, but the cumulative whole-loop peak stayed exactly 1,894,636,028 bytes in both traces. Root release therefore failed the diagnostic global-peak hypothesis; no positive peak-saving claim or success-directed confirmation follows. This limited trace does not locate every allocation or establish generality. MLX active allocation excludes its cache and is not total physical RAM.

The endpoint trace observes only the accumulated tree, not the additional `grads` root. Its array-buffer field is therefore not a census of every live gradient root. The active-allocation observations provide the separate comparison. The instrumented sources are included for inspection.

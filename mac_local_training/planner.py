"""Conservative local weight-footprint planning; this is not a fit guarantee."""
import math


def estimate(parameters_billions: float, bits: int, budget_gib: float) -> dict:
    if not math.isfinite(parameters_billions) or parameters_billions <= 0 or bits not in {2, 4, 8, 16, 32} or not math.isfinite(budget_gib) or budget_gib <= 0:
        raise ValueError("positive finite parameter count/budget and a supported bit width are required")
    raw = parameters_billions * 1e9 * bits / 8 / 2**30
    return {"parameters_billions": parameters_billions, "bits": bits, "budget_gib": budget_gib,
        "raw_weight_lower_bound_gib": raw,
        "decision": "reject_weights_exceed_budget" if raw >= budget_gib else "candidate_requires_measurement",
        "notes": ["Raw weights only: quantization metadata and unquantized tensors are excluded.",
            "Training also needs adapters, gradients, optimizer state, activations and OS/app headroom.",
            "Adapter rotation does not remove the base model's resident-weight requirement."]}


def estimate_adapter_state(trainable_parameters: int, active_fraction: float = 1.0) -> dict:
    """Raw FP32 adapter/gradient/Adam components, not measured process memory."""
    if isinstance(trainable_parameters, bool) or not isinstance(trainable_parameters, int) or trainable_parameters <= 0:
        raise ValueError("trainable_parameters must be a positive integer")
    if isinstance(active_fraction, bool) or not math.isfinite(active_fraction) or not 0 < active_fraction <= 1:
        raise ValueError("active_fraction must be finite in (0, 1]")
    active = math.ceil(trainable_parameters * active_fraction)
    components = {"resident_adapter_bytes": trainable_parameters * 4,
                  "active_gradient_bytes": active * 4, "active_adam_moments_bytes": active * 8}
    return {"trainable_parameters": trainable_parameters, "active_parameters": active,
            "components": components, "modeled_total_bytes": sum(components.values()),
            "scope": "FP32 components assuming inactive gradients and Adam moments are released. Excludes frozen base weights, activations, runtime overhead and OS headroom; no fit or measured savings guarantee."}

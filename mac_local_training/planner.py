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

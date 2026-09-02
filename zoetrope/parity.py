"""Parity in the units the controller receives.

One overall cosine on a 132-wide GR00T chunk is close to meaningless:
the LIBERO embodiment uses 7 of those columns and the other 125 are
padding the model is free to fill with anything. A treated arm can be
bit-perfect on every dimension the robot acts on and still score badly
if the padding is included — and, worse, the reverse. So parity is
measured after the host's own decode, per dimension, on the chunk the
controller is actually handed.
"""

from __future__ import annotations

import numpy as np


def per_dimension_parity(treated, reference) -> dict:
    treated = np.asarray(treated, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if treated.shape != reference.shape:
        return {"error": f"shape {treated.shape} vs {reference.shape}"}
    out = {"overall_cosine": _cos(treated.ravel(), reference.ravel()),
           "max_abs": float(np.max(np.abs(treated - reference))),
           "per_dim_cosine": [], "per_dim_max_abs": []}
    for dim in range(treated.shape[-1]):
        out["per_dim_cosine"].append(_cos(treated[..., dim],
                                          reference[..., dim]))
        out["per_dim_max_abs"].append(
            float(np.max(np.abs(treated[..., dim] - reference[..., dim]))))
    return out


def _cos(a, b) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 1.0

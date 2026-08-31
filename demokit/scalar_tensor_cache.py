"""Cache `torch.tensor(<scalar>, device="cuda")` so a capture can record it.

`pi052_denoise.py`'s `scalar_setitem_fill` pin already caches the scalar
tensors the denoise loop builds — but only the sequence form,
`torch.tensor([x], device=...)`. This host's `sample_actions` builds the
timestep with the bare-scalar form:

    time_tensor = torch.tensor(time, dtype=torch.float32, device=device)

which falls through to the real constructor and does a live host-to-device
copy, and a CUDA graph capture refuses that:

    RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA
    graph capture unless the CPU tensor is pinned.

Installing this *before* the lowering runs makes it the lowering's own
`real_tensor`, so the vendor pin's fall-through lands here instead. The
proper fix is one line in `pi052_denoise.py` — widen the guard from
`isinstance(data, (list, tuple))` to accept bare scalars — and then this
shim is redundant.

Only float/int/bool scalars with an explicit `device=` are cached, the
same discipline the vendor pin applies to the sequence form. The values
are constants of the denoise schedule and the tensors are read-only
downstream, so one tensor per value is what the graph should bake.
"""

from __future__ import annotations

import torch

_cache: dict[tuple, torch.Tensor] = {}
_installed = False


def install() -> str:
    global _installed
    if _installed:
        return "already installed"
    real_tensor = torch.tensor

    def caching_tensor(data, *args, **kwargs):
        device = kwargs.get("device")
        if device is not None and isinstance(data, (float, int, bool)) \
                and not isinstance(data, torch.Tensor):
            key = (data, str(device), str(kwargs.get("dtype")))
            hit = _cache.get(key)
            if hit is None:
                hit = real_tensor(data, *args, **kwargs)
                _cache[key] = hit
            return hit
        return real_tensor(data, *args, **kwargs)

    torch.tensor = caching_tensor
    _installed = True
    return "installed (bare-scalar cache under the vendor pin)"


def size() -> int:
    return len(_cache)

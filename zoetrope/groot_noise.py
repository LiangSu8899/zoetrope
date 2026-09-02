"""Pin the flow-matching noise, identically for every arm of the film.

The action head draws its starting noise with `torch.randn` inside the
timed region. Two things go wrong if that is left alone.

Within an arm: a captured graph replays without advancing the device
RNG in the same way the eager path does, so the fiftieth replay
denoises a different draw than the reference did, and the §5.1
repeatability test — the same observation twice must give bitwise
identical actions — can never pass.

Across arms: even seeding the process is not enough, because each host
consumes a different amount of randomness while loading, so "the first
draw" is a different tensor in every process. Two arms then integrate
from two different starting points and their trajectories diverge for a
reason that has nothing to do with what is being compared.

So the pin does not merely freeze the first draw — it *replaces* it with
a deterministic function of shape, dtype, device and seed. Every arm, in
every process, then starts from the same sample, which is what the
film's footer claims.
"""

from __future__ import annotations

import torch

SEED = 0


def pin_action_noise(seed: int = SEED):
    """Freeze the first CUDA `torch.randn` to a seeded draw.

    Returns `(undo, box)`; `box["value"]` is the tensor every arm
    integrates from, for the receipt and for the native arm's bundle.
    """
    original = torch.randn
    box: dict = {}

    def fixed(*size, **kwargs):
        try:
            shape = tuple(kwargs.get("size", ())) or (
                tuple(size[0])
                if len(size) == 1 and not isinstance(size[0], int)
                else tuple(size))
        except TypeError:
            shape = None
        dtype = kwargs.get("dtype")
        if shape is not None and box.get("shape") == shape:
            sample = box["value"]
            return sample if dtype in (None, sample.dtype) else sample.to(dtype)
        value = original(*size, **kwargs)
        if value.is_cuda and "value" not in box:
            generator = torch.Generator(device=value.device)
            generator.manual_seed(seed)
            # drawn in fp32 and cast on the way out, never drawn at the
            # caller's dtype: one host runs the head in fp32 and the
            # other under a bf16 autocast, and a generator consumed at
            # two different dtypes is not guaranteed to yield the same
            # sample. The caller still gets its own dtype.
            sample = original(*value.shape, dtype=torch.float32,
                              device=value.device, generator=generator)
            box["shape"] = tuple(sample.shape)
            box["value"] = sample
            box["seed"] = seed
            return sample.to(value.dtype)
        return value

    torch.randn = fixed
    return lambda: setattr(torch, "randn", original), box

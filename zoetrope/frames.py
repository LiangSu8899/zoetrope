"""How a run stores the pixels it recorded.

A rollout is a stack of `uint8 [N, H, W, 3]`, and stored raw that is enormous:
92 control steps of a 256px LIBERO camera is 18 MB, which is why the examples
in this repository used to ship truncated and shrunk. As an animated WebP the
same 92 frames are 0.27 MB with a mean error of about one part in 255 — so the
examples ship the whole episode at full size instead.

`frames.npy` is still read wherever it exists, because runs recorded before
this are still perfectly good runs.
"""

from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image

NPY, WEBP = "frames.npy", "frames.webp"


def frames_path(run_dir) -> pathlib.Path | None:
    """Whichever of the two a run happens to carry, raw first."""
    d = pathlib.Path(run_dir)
    for name in (NPY, WEBP):
        if (d / name).exists():
            return d / name
    return None


def has_frames(run_dir) -> bool:
    return frames_path(run_dir) is not None


def load_frames(run_dir) -> np.ndarray:
    p = frames_path(run_dir)
    if p is None:
        raise FileNotFoundError(f"{run_dir}: no {NPY} and no {WEBP}")
    if p.name == NPY:
        return np.load(p)
    im = Image.open(p)
    out = []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        out.append(np.asarray(im.convert("RGB")))
    return np.stack(out)


def save_frames(run_dir, arr, *, quality: int = 90) -> pathlib.Path:
    """Write the stack as an animated WebP, and drop any raw copy beside it."""
    d = pathlib.Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    ims = [Image.fromarray(np.asarray(f)) for f in arr]
    out = d / WEBP
    ims[0].save(out, format="WEBP", save_all=True, append_images=ims[1:],
                quality=quality, method=4, lossless=False)
    return out

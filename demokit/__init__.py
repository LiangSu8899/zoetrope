"""Record what an optimization did, then draw it.

A recorder writes down *when* each thing happened. A compositor replays those
timestamps and draws every frame. Nothing is screen-captured.

    from demokit import hook

    rec = hook.Recorder("stream", label="+ FlashRT structures",
                        sub="auto_swaps + capture", color="ours")
    with hook.on_tokens(rec, tokenizer):
        model.generate(**inputs, streamer=rec.streamer)
    rec.write("runs/myfilm/attach")

See docs/PROTOCOL.md for the run-directory contract.
"""

import pathlib as _pathlib
import sys as _sys

# The host adapters and pipeline recorders were written as scripts and import
# each other flatly (`from host_pi05_lerobot import ...`). They are validated
# as they stand, so rather than rewrite those imports and re-validate every
# one, the package puts their directories on the path. Anything new should use
# a normal package import.
_HERE = _pathlib.Path(__file__).resolve().parent
for _sub in ("", "hosts", "pipelines", "compose"):
    _p = str(_HERE / _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from . import hook                                       # noqa: E402
from .hook import Recorder                               # noqa: E402

__all__ = ["hook", "Recorder"]
__version__ = "0.1.0"

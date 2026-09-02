"""Pi0.5 on the OpenPI host (official PyTorch path).

Arms:
    eager      the host's own code, nothing added
    compiled   the same host under torch.compile + whole-graph capture
    attach     structures.auto_swaps -> swap.attach, captured the same way

The timed region is the model call alone: `model.sample_actions`. The
input transforms (image resize/pad, normalisation, tokenisation) and the
output transform (un-normalisation, action decode) sit outside it, and
they are the host's own code, unchanged, for every arm.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import torch

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # keep jax off the Thor GPU


def _tree_to_torch(tree, device):
    import jax

    return jax.tree.map(
        lambda x: torch.from_numpy(np.asarray(x)).to(device)[None, ...], tree)


def per_dimension_parity(treated, reference) -> dict:
    """Cosine per action dimension, in the units the controller receives.

    One overall cosine on a chunk whose gripper dimension dominates the
    norm hides a systematic bias on the motion dimensions, so the numbers
    that matter are per dimension after un-normalisation.
    """
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


def _tree_copy_(dst, src):
    """In-place refresh of a fixed observation window."""
    import jax

    jax.tree.map(lambda d, s: d.copy_(s), dst, src)


def _refresh_observation(window, fresh) -> None:
    """Write a fresh observation into the tensors the graph points at.

    Shape/dtype/device are the graph's contract, so a mismatch raises
    rather than letting `copy_` broadcast or cast underneath a replay.
    """
    import dataclasses

    for field in dataclasses.fields(window):
        old, new = getattr(window, field.name), getattr(fresh, field.name)
        if torch.is_tensor(old) and torch.is_tensor(new):
            _copy_exact(old, new, field.name)
        elif isinstance(old, dict) and isinstance(new, dict):
            for key in old:
                _copy_exact(old[key], new[key], f"{field.name}.{key}")


def _copy_exact(dst, src, name: str) -> None:
    if (dst.shape != src.shape or dst.dtype != src.dtype
            or dst.device != src.device):
        raise RuntimeError(
            f"observation window {name!r} changed form: "
            f"{tuple(dst.shape)}/{dst.dtype} -> "
            f"{tuple(src.shape)}/{src.dtype} — a replay over a coerced "
            "write is silently wrong")
    dst.copy_(src)


class OpenPiPi05Host:
    name = "openpi"

    def __init__(self, *, checkpoint: str, config_name: str = "pi05_libero",
                 arm: str = "eager", num_steps: int | None = None,
                 fixed_noise: bool = True, seed: int = 0,
                 compile_mode: str = "max-autotune",
                 host_src: str | None = None):
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        # transformers imports flash_attn at module scope
        # (modeling_flash_attention_utils), and the board's extension does
        # not load under the torch the kernel artifacts need
        from flash_attn_stub import install_if_broken
        self.flash_attn = install_if_broken()
        if host_src:
            sys.path.insert(0, str(host_src))
            for name in [m for m in sys.modules
                         if m == "openpi" or m.startswith("openpi.")]:
                del sys.modules[name]
        self.host_src = host_src
        self.checkpoint = checkpoint
        self.config_name = config_name
        self.arm = arm
        self.num_steps = num_steps
        self.fixed_noise = fixed_noise
        self.seed = seed
        self.compile_mode = compile_mode
        self.report: dict = {}
        self._obs = None
        self._flat = None
        self._rebuilds = 0
        self._prompt = None
        self._armed = False

    # -- build ---------------------------------------------------------
    @staticmethod
    def _patch_load_pytorch():
        """The vendor's own Thor workaround (sample_code/pi05_inference.py).

        The converted checkpoint carries the expert's unused pre/post
        layernorm weights — pi0.5 replaces them with AdaRMS — and strict
        `load_model` refuses them. Load non-strict, exactly as the shipped
        Thor sample does.
        """
        import safetensors.torch as _st

        import openpi.models.model as _model_mod
        from openpi.models_pytorch import pi0_pytorch as _pi0pt

        def _load(self, train_config, weight_path: str):
            model = _pi0pt.PI0Pytorch(config=train_config.model)
            model.load_state_dict(_st.load_file(weight_path), strict=False)
            return model

        for cls in vars(_model_mod).values():
            if isinstance(cls, type) and hasattr(cls, "load_pytorch"):
                cls.load_pytorch = _load

    def build(self):
        from openpi.policies import policy_config
        from openpi.training import config as _config

        self._patch_load_pytorch()
        train_config = _config.get_config(self.config_name)
        self.policy = policy_config.create_trained_policy(
            train_config, self.checkpoint, pytorch_device="cuda")
        self.model = self.policy._model
        self.device = self.policy._pytorch_device
        self.horizon = int(train_config.model.action_horizon)
        self.action_dim = int(train_config.model.action_dim)

        if self.fixed_noise:
            g = torch.Generator(device="cuda").manual_seed(self.seed)
            self._noise = torch.randn(1, self.horizon, self.action_dim,
                                      device="cuda", dtype=torch.float32,
                                      generator=g)
        else:
            self._noise = None

        self.report.update({
            "host": "openpi (official PyTorch)",
            "host_src": str(self.host_src),
            "flash_attn": self.flash_attn,
            "config": self.config_name,
            "checkpoint": str(self.checkpoint),
            "action_horizon": self.horizon,
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(0),
            "arm": self.arm,
            "fixed_noise": bool(self.fixed_noise),
            "compile_mode": self.compile_mode,
        })
        self._bind_sample_fn()

    # -- the model call ------------------------------------------------
    @staticmethod
    def _uncompiled(fn):
        """openpi wraps `sample_actions` in torch.compile inside
        `PI0Pytorch.__init__`, so this host has no eager form unless the
        wrapper is peeled off. The eager pane peels it; every other pane
        keeps the host's own compile. Recorded in the receipts."""
        seen = fn
        for _ in range(4):
            inner = getattr(seen, "_torchdynamo_orig_callable", None)
            if inner is None:
                break
            seen = inner
        return seen

    def _sample(self):
        kwargs = dict(self.policy._sample_kwargs)
        if self._noise is not None:
            kwargs["noise"] = self._noise
        if self.num_steps is not None:
            kwargs["num_steps"] = self.num_steps
        # resolved per call: the structures lowering records one invocation
        # through `model.sample_actions`, so it must not be bound early
        fn = self._sample_fn or self.model.sample_actions
        return fn(self.device, self._obs, **kwargs)

    def _arm_once(self):
        """Applied after the first real observation exists."""
        if self._armed:
            return
        self._armed = True
        if self.arm == "eager":
            self._call = self._sample
            return
        # Both captured panes take the host's own code *uncompiled* into
        # the capture door, so the only difference between them is
        # whether the swaps are in. Leaving the host's
        # torch.compile(max-autotune) wrapper on would put inductor's
        # cudagraph trees between the swaps and the graph, and those call
        # the bound kernels with storage-less tensors.
        self.model.sample_actions = self._uncompiled(self.model.sample_actions)
        self.report["host_compile_peeled"] = True
        if self.arm == "attach":
            self._attach()
        # `compiled` and `attach` take the same capture door — the
        # library's own: normalise the host's denoise schedule, compile,
        # record one CUDA graph, gate it. The only difference between
        # the two panes is whether the swaps are in.
        from flash_rt import structures

        stage = structures.capture(self._sample, model=self.model,
                                   gate_cos=0.0, min_speedup=0.0,
                                   verbose=True)
        self._stage = stage
        self.report["capture"] = stage.certification
        self._call = stage.replay

    def _bind_sample_fn(self):
        raw = self.model.sample_actions
        if self.arm == "eager":
            self._sample_fn = self._uncompiled(raw)
            self.report["host_compile_peeled"] = (self._sample_fn is not raw)
        else:
            self._sample_fn = None      # resolve `sample_actions` per call
            self.report["host_compile_peeled"] = False

    def _attach(self):
        from flash_rt import structures
        from flash_rt.structures import swap
        from flash_rt.structures.impls import unavailable_report

        def run_once():
            with torch.no_grad():
                return self._sample()

        plan = structures.auto_swaps(self.model, run_once, verbose=True)
        self._handle = swap.attach(self.model, plan.swaps,
                                   observe=plan.observed, revert=plan.revert)
        self.report.update({
            "swaps": len(plan.swaps),
            "observed": len(plan.observed),
            "refused": len(plan.notes.get("refused", [])),
            "kernel_unavailable": unavailable_report(),
        })

    # -- host preprocessing (untimed) ----------------------------------
    def set_task(self, text: str):
        self._prompt = text

    def observe(self, img, wrist, state):
        from openpi.models import model as _model

        raw = {
            "observation/image": img,
            "observation/wrist_image": wrist,
            "observation/state": np.asarray(state, dtype=np.float32),
            "prompt": self._prompt,
        }
        inputs = self.policy._input_transform(raw)
        self._raw_state = np.asarray(inputs["state"])
        fresh = _model.Observation.from_dict(_tree_to_torch(inputs,
                                                            self.device))
        if self._obs is None:
            # the fixed observation window: this object's tensors are the
            # ones a captured graph will hold pointers to, for the rest of
            # the episode
            self._obs = fresh
            return
        # `Observation.from_dict` converts uint8 HWC images into fresh
        # float NCHW tensors, so refreshing the dict it was built from
        # would leave the captured graph reading frame 0 forever. Write
        # into the observation's own tensors instead.
        _refresh_observation(self._obs, fresh)

    def sync(self):
        torch.cuda.synchronize()

    # -- the timed region ----------------------------------------------
    def infer(self):
        with torch.no_grad():
            return self._call()

    # -- host postprocessing (untimed) ---------------------------------
    def decode(self, chunk):
        actions = np.asarray(chunk[0].detach().float().cpu())
        out = self.policy._output_transform(
            {"state": self._raw_state, "actions": actions})
        return np.asarray(out["actions"])

    def warmup(self, img, wrist, state, rounds: int = 20):
        """Compile / autotune / settle — outside the recorded episode."""
        self.observe(img, wrist, state)

        # the host as loaded, before anything is applied to it: the only
        # honest parity reference for the treated arms
        with torch.no_grad():
            reference = self.decode(self._sample())
        self._arm_once()
        for _ in range(rounds):
            self.infer()
            torch.cuda.synchronize()
        treated = self.decode(self.infer())
        self.report["parity"] = per_dimension_parity(treated, reference)

    def finish(self):
        if hasattr(self, "_handle"):
            self.report["ledger"] = self._handle.summary()
        self.report["observation_rebuilds"] = self._rebuilds
        return self.report

"""GR00T N1.7 on the LeRobot port.

The second host of the race: the same NVIDIA LIBERO checkpoint, loaded
and driven by an independent reimplementation. LeRobot's GR00T needs
transformers 5.x while the Isaac package pins 4.57, so the two hosts
cannot share a process — which the protocol requires anyway.

Timed region: the model's own hot path —

    _groot_model.backbone(backbone_inputs)
    _groot_model.action_head.get_action(backbone_outputs, action_inputs)

which is what `predict_action_chunk` does after it has filtered the batch
and prepared the inputs. Preparation and the un-normalising postprocessor
sit outside, so all four arms of the film are timed at the same boundary.

Gripper convention: this port's postprocessor already applies the LIBERO
action transform (`action_decode_transform='auto'` resolves to 'libero'
for the `libero_sim` embodiment), which is the same operation the Isaac
sim wrapper applies in `step()`. Our loop drives the raw LIBERO env and
applies nothing, so the transform lands exactly once — here.
"""

from __future__ import annotations

import collections.abc
import pathlib
import sys

import numpy as np
import torch


class LeRobotGrootHost:
    name = "lerobot_groot"

    def __init__(self, *, checkpoint: str, lerobot_src: str,
                 arm: str = "eager", embodiment_tag: str | None = None,
                 num_views: int = 2,
                 compile_mode: str = "max-autotune"):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        sys.path.insert(0, str(pathlib.Path(lerobot_src)))
        for name in [m for m in sys.modules
                     if m == "lerobot" or m.startswith("lerobot.")]:
            del sys.modules[name]
        self.lerobot_src = lerobot_src
        self.checkpoint = str(checkpoint)
        self.arm = arm
        self.embodiment_tag = embodiment_tag
        self.num_views = int(num_views)
        self.compile_mode = compile_mode
        self.report: dict = {}
        self._armed = False
        self._windows = None
        self._rebuilds = 0
        self._prompt = None

    # -- build ---------------------------------------------------------
    def _import_port(self):
        """Get through lerobot's import chain, then get out of the way.

        `lerobot.policies.groot` pulls in `lerobot.policies.__init__` ->
        the policy factory -> xVLA -> Florence-2, which imports
        `flash_attn` at module scope; the board's extension does not load
        under the torch the kernel artifacts need. The stub carries that
        import — and is then removed, because this port does its own
        `try: import flash_attn / except ImportError: sdpa`, and leaving
        a stub in place would turn that clean fallback into a refusal at
        the first attention call.
        """
        from flash_attn_stub import install_if_broken, probe, remove

        carried = install_if_broken()
        try:
            from lerobot.policies.groot.configuration_groot import GrootConfig
            from lerobot.policies.groot.modeling_groot import GrootPolicy
            from lerobot.policies.groot.processor_groot import (
                make_groot_pre_post_processors_from_pretrained)
        finally:
            if carried != "real":
                remove()
        self.report["flash_attn"] = (
            f"{probe()} (stub carried the import chain: {carried})")
        return (GrootConfig, GrootPolicy,
                make_groot_pre_post_processors_from_pretrained)

    def build(self):
        GrootConfig, GrootPolicy, make_processors = self._import_port()

        # An NVIDIA checkpoint carries no LeRobot feature spec — it was
        # never written by LeRobot — so the port cannot infer one and
        # refuses to build without it. The spec below is the checkpoint's
        # own: its two video modality keys (which the port's packer
        # matches by name), LIBERO's 8-wide state and 7-wide action.
        from lerobot.configs.types import FeatureType, PolicyFeature

        # the embodiment tag has to go in through the constructor: the
        # config resolves `action_decode_transform='auto'` in
        # `__post_init__`, and that resolution is what turns the LIBERO
        # gripper transform on. Set the tag afterwards and the transform
        # stays off — the decoded gripper then comes out raw in [0, 1]
        # instead of the environment's ±1, the fingers never close, and
        # the arm hovers over the object for the whole episode.
        config = GrootConfig(pretrained_path=self.checkpoint,
                             **({"embodiment_tag": self.embodiment_tag}
                                if self.embodiment_tag else {}))
        config.input_features = {
            "observation.images.image": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE, shape=(8,)),
        }
        if self.num_views >= 2:
            config.input_features["observation.images.wrist_image"] = \
                PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
        config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
        config.device = "cuda"
        self.policy = GrootPolicy.from_pretrained(self.checkpoint,
                                                  config=config)
        self.policy.to("cuda").eval()
        self.config = self.policy.config
        self.model = self.policy._groot_model
        self.pre, self.post = make_processors(
            self.config, self.checkpoint,
            preprocessor_overrides={"device_processor": {"device": "cuda"}})

        from groot_noise import pin_action_noise
        self._unpin, self._noise_box = pin_action_noise()

        self.image_keys = [k for k in ("observation.images.image",
                                       "observation.images.wrist_image")
                           if k in self.config.input_features]
        import transformers
        self.report.update({
            "host": "LeRobot GR00T port",
            "lerobot_src": str(self.lerobot_src),
            "checkpoint": self.checkpoint,
            "embodiment_tag": str(getattr(self.config, "embodiment_tag", None)),
            "action_decode_transform": str(
                getattr(self.config, "action_decode_transform", None)),
            "image_keys": self.image_keys,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device_name": torch.cuda.get_device_name(0),
            "arm": self.arm,
            "compile_mode": self.compile_mode,
            "timed_region": "backbone + action_head.get_action "
                            "(input preparation and postprocessor excluded)",
        })

    # -- the model call ------------------------------------------------
    def _hot(self):
        backbone_inputs, action_inputs = self._windows
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=bool(getattr(self.config, "use_bf16",
                                                 False))):
            outputs = self.model.backbone(backbone_inputs)
            return self.model.action_head.get_action(
                outputs, action_inputs)["action_pred"]

    def _arm_once(self):
        if self._armed:
            return
        self._armed = True
        with torch.no_grad():
            reference = self._hot().detach().float().cpu()
        if self.arm == "eager":
            self._call = self._hot
            return
        if self.arm == "attach":
            self._attach()
            with torch.no_grad():
                treated = self._hot().detach().float().cpu()
            self.report["eager_parity_cosine"] = _cos(treated, reference)
        from flash_rt.structures import capture as capture_stage

        torch._dynamo.reset()
        stage = capture_stage(
            torch.compile(self._hot, mode="max-autotune-no-cudagraphs",
                          fullgraph=False),
            model=self.model, warmup=8, gate_cos=0, min_speedup=0)
        self._stage = stage
        self.report["capture"] = stage.certification
        self._call = stage.replay
        with torch.no_grad():
            captured = self._call().detach().float().cpu()
        self.report["captured_parity_cosine"] = _cos(captured, reference)

    def _attach(self):
        from flash_rt import structures
        from flash_rt.structures import swap
        from flash_rt.structures.impls import unavailable_report

        def run_once():
            with torch.no_grad():
                return self._hot()

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
        views = [img, wrist][:self.num_views]
        raw = {"observation.state": torch.tensor(np.asarray(state),
                                                 dtype=torch.float32),
               "task": self._prompt}
        for index, key in enumerate(self.image_keys):
            frame = views[min(index, len(views) - 1)]
            raw[key] = torch.from_numpy(
                np.ascontiguousarray(frame.astype(np.float32) / 255.0)
            ).permute(2, 0, 1).contiguous()
        batch = self.pre(raw)
        inputs = self.policy._filter_groot_inputs(batch, include_action=False)
        fresh = self.model.prepare_input(dict(inputs))
        if self._windows is None:
            self._windows = fresh
            return
        _refresh(self._windows, fresh)

    def sync(self):
        torch.cuda.synchronize()

    def infer(self):
        with torch.no_grad():
            return self._call()

    # -- host postprocessing (untimed) ---------------------------------
    def decode(self, pred):
        horizon = self.policy._resolve_prediction_horizon(pred)
        width = self.config.output_features["action"].shape[0]
        # the whole chunk, batch dimension intact: this embodiment's
        # actions are relative, and the decode step rebuilds absolute
        # poses by composing along the chunk against the cached state —
        # it refuses to do that one step at a time
        actions = pred[:, :horizon, :width].detach().float().cpu()
        out = self.post(actions)
        out = out["action"] if isinstance(out, dict) else out
        out = np.asarray(out)
        return out[0] if out.ndim == 3 else out

    def warmup(self, img, wrist, state, rounds: int = 20):
        self.observe(img, wrist, state)
        self._arm_once()
        for _ in range(rounds):
            self.infer()
            torch.cuda.synchronize()

    def finish(self):
        if hasattr(self, "_handle"):
            self.report["ledger"] = self._handle.summary()
        self.report["observation_rebuilds"] = self._rebuilds
        self.report["noise_pinned"] = {
            "shape": list(self._noise_box.get("shape", ())) or None,
            "seed": self._noise_box.get("seed")}
        return self.report


def _cos(a, b) -> float:
    return float(torch.nn.functional.cosine_similarity(
        torch.as_tensor(a).float().flatten(),
        torch.as_tensor(b).float().flatten(), dim=0))


def _refresh(dst, src, path: str = "") -> None:
    """Copy a fresh batch into the tensors a captured graph points at."""
    if torch.is_tensor(dst):
        if (dst.shape != src.shape or dst.dtype != src.dtype
                or dst.device != src.device):
            raise RuntimeError(
                f"observation window {path!r} changed form: "
                f"{tuple(dst.shape)}/{dst.dtype} -> "
                f"{tuple(src.shape)}/{src.dtype} — a replay over a coerced "
                "write is silently wrong")
        dst.copy_(src)
        return
    if isinstance(dst, collections.abc.Mapping):
        for key, value in dst.items():
            _refresh(value, src[key], f"{path}.{key}")
        return
    if isinstance(dst, (list, tuple)):
        for index, (value, other) in enumerate(zip(dst, src)):
            _refresh(value, other, f"{path}[{index}]")

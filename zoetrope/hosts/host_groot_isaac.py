"""GR00T N1.7 on the official Isaac-GR00T host.

Arms:
    eager      the host's own code, nothing added
    captured   the same host under torch.compile + whole-graph capture
    attach     structures.auto_swaps -> swap.attach, captured the same way

Timed region: the model's own hot path —

    model.backbone(backbone_inputs)
    model.action_head.get_action(backbone_outputs, action_inputs)

which is `Gr00tN1d7.get_action` minus its leading `prepare_input`. Input
preparation (the processor, the collator, the host-to-device move) and
`decode_action` sit outside, identically for every arm of the film,
because the FlashRT native arm replaces exactly this span and nothing
else. It is also the span the repository's own `full_graph.py` measures,
so the film and the library's receipts are quoting the same boundary.
"""

from __future__ import annotations

import collections.abc
import pathlib
import sys

import numpy as np
import torch


class IsaacGrootHost:
    name = "isaac"

    def __init__(self, *, checkpoint: str, host_src: str, arm: str = "eager",
                 embodiment_tag: str = "libero_sim",
                 denoising_steps: int | None = None,
                 num_views: int = 2,
                 compile_mode: str = "max-autotune"):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        sys.path.insert(0, str(pathlib.Path(host_src)))
        for name in [m for m in sys.modules
                     if m == "gr00t" or m.startswith("gr00t.")]:
            del sys.modules[name]
        # The board's flash_attn extension does not load under the torch the
        # kernel artifacts need. This host handles that itself — its backbone
        # does `try: import flash_attn / except ImportError: sdpa` — so the
        # right move here is to leave it broken and record which attention
        # the host therefore chose. Stubbing it would be actively harmful:
        # the stub imports, `is_flash_attn_2_available()` turns True, and
        # transformers routes attention into a function that refuses.
        from flash_attn_stub import probe
        self.flash_attn = probe()
        self.host_src = host_src
        self.checkpoint = checkpoint
        self.arm = arm
        self.embodiment_tag = embodiment_tag
        self.denoising_steps = denoising_steps
        self.num_views = int(num_views)
        self.compile_mode = compile_mode
        self.report: dict = {}
        self._armed = False
        self._windows = None
        self._rebuilds = 0
        self._prompt = None

    # -- build ---------------------------------------------------------
    def build(self):
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.policy = Gr00tPolicy(embodiment_tag=self.embodiment_tag,
                                  model_path=str(self.checkpoint),
                                  device="cuda")
        self.model = self.policy.model
        if self.denoising_steps is not None:
            # the checkpoint's own schedule unless the film asks otherwise
            self.model.action_head.num_inference_timesteps = int(
                self.denoising_steps)

        # after construction, so the pin catches the action head's own
        # flow-matching draw and not a tensor allocated during loading
        from groot_noise import pin_action_noise
        self._unpin, self._noise_box = pin_action_noise()
        import transformers
        self.report.update({
            "host": "Isaac-GR00T (official)",
            "host_src": str(self.host_src),
            "flash_attn": self.flash_attn,
            "checkpoint": str(self.checkpoint),
            "embodiment_tag": self.embodiment_tag,
            "video_keys": list(
                self.policy.modality_configs["video"].modality_keys),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device_name": torch.cuda.get_device_name(0),
            "arm": self.arm,
            "compile_mode": self.compile_mode,
            "denoising_steps": int(
                self.model.action_head.num_inference_timesteps),
            "timed_region": "model.backbone + action_head.get_action "
                            "(prepare_input and decode_action excluded)",
        })

    # -- the model call ------------------------------------------------
    def _hot(self):
        backbone_inputs, action_inputs = self._windows
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
        from parity import per_dimension_parity
        self.report["parity"] = per_dimension_parity(
            self.decode(self._call()), self.decode(reference))

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

    def raw_observation(self, img, wrist, state):
        """The host's own observation dict, in the host's own shapes.

        Flat keys, one timestep each; `parse_observation_gr00t` adds the
        batch dimension exactly as the host's evaluation path does. The
        state layout is LIBERO's: end-effector position, axis-angle
        orientation, two gripper joints — the same assembly this host's
        own sim wrapper builds in `_process_observation`.
        """
        from gr00t.eval.open_loop_eval import parse_observation_gr00t

        scalars = {"x": state[0], "y": state[1], "z": state[2],
                   "roll": state[3], "pitch": state[4], "yaw": state[5]}
        flat: dict = {}
        views = [img, wrist][:self.num_views]
        for index, key in enumerate(
                self.policy.modality_configs["video"].modality_keys):
            frame = views[min(index, len(views) - 1)]
            flat[f"video.{key}"] = np.asarray(frame)[None]
        for key in self.policy.modality_configs["state"].modality_keys:
            if key == "gripper":
                flat["state.gripper"] = np.asarray(state[6:8],
                                                   dtype=np.float32)[None]
            else:
                flat[f"state.{key}"] = np.asarray(
                    [scalars[key]], dtype=np.float32)[None]
        for key in self.policy.modality_configs["language"].modality_keys:
            flat[key] = self._prompt
        return parse_observation_gr00t(flat, self.policy.modality_configs)

    def collate(self, img, wrist, state):
        """The host's processor + collator, verbatim from `_get_action`."""
        from gr00t.data.types import MessageType
        from gr00t.policy.gr00t_policy import _rec_to_dtype

        observation = self.raw_observation(img, wrist, state)
        unbatched = self.policy._unbatch_observation(observation)
        processed, states = [], []
        for obs in unbatched:
            step = self.policy._to_vla_step_data(obs)
            states.append(step.states)
            processed.append(self.policy.processor(
                [{"type": MessageType.EPISODE_STEP.value, "content": step}]))
        collated = self.policy.collate_fn(processed)
        collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
        batched_states = {
            k: np.stack([s[k] for s in states], axis=0)
            for k in self.policy.modality_configs["state"].modality_keys}
        return collated, batched_states

    def observe(self, img, wrist, state):
        collated, batched_states = self.collate(img, wrist, state)
        self._states = batched_states
        fresh = self.model.prepare_input(dict(collated["inputs"]))
        if self._windows is None:
            # the fixed observation windows: these tensors are the ones a
            # captured graph holds pointers to for the rest of the episode
            self._windows = fresh
            return
        _refresh(self._windows, fresh)

    def sync(self):
        torch.cuda.synchronize()

    # -- the timed region ----------------------------------------------
    def infer(self):
        with torch.no_grad():
            return self._call()

    # -- host postprocessing (untimed) ---------------------------------
    def decode(self, pred):
        normalized = np.asarray(pred.detach().float().cpu())
        unnorm = self.policy.processor.decode_action(
            normalized, self.policy.embodiment_tag, self._states)
        keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        available = set(self.policy.modality_configs["action"].modality_keys)
        missing = [k for k in keys if k not in available]
        if missing:
            raise RuntimeError(f"action keys {missing} absent from this "
                               f"checkpoint (has {sorted(available)})")
        chunk = np.concatenate(
            [np.asarray(unnorm[k], dtype=np.float32)[0].reshape(-1, 1)
             for k in keys], axis=1)
        return apply_gripper_convention(chunk)

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


def apply_gripper_convention(chunk: np.ndarray) -> np.ndarray:
    """normalise [0,1] -> [-1,1], binarise, invert — once, in one place.

    This host's own sim wrapper (`gr00t/eval/sim/LIBERO/libero_env.py`
    `step`) applies exactly this before handing the action to LIBERO. Our
    loop drives the raw environment and applies nothing, so the transform
    lands here instead. Do it in both places and the gripper is exactly
    inverted: the arm aligns, descends, the fingers move, and it never
    picks anything up.
    """
    chunk = np.array(chunk, dtype=np.float32, copy=True)
    chunk[..., -1] = -np.sign(2.0 * chunk[..., -1] - 1.0)
    return chunk


def _cos(a, b) -> float:
    return float(torch.nn.functional.cosine_similarity(
        torch.as_tensor(a).float().flatten(),
        torch.as_tensor(b).float().flatten(), dim=0))


def _refresh(dst, src, path: str = "") -> None:
    """Copy a fresh batch into the tensors a captured graph points at.

    Shape/dtype/device are the graph's contract: a mismatch raises rather
    than letting `copy_` broadcast or cast underneath a replay.
    """
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

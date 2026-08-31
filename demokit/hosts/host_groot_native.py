"""GR00T N1.7, FlashRT native — the hand-written pipeline replaces the model.

The packaged Thor frontend bakes the prompt-constant half of the graph
inside `set_prompt` and exposes `run_backbone_graph(aux)` + `infer(state)`
as the per-observation hot path, so a closed loop is: refresh the ViT
input, replay the backbone graph, replay the DiT. Nothing of the host's
model runs inside the timed region.

The setup bundle (`aux`) is captured once, at the episode's first
observation, from the host's own forward — the same tensors
`tests/_helpers/groot_n17/capture_llm_aux.py` records. It carries raw
`pixel_values` and `input_ids`, so the patch embedding and the token
embedding lookup run in-kernel as part of the captured graph and the
per-frame path needs nothing from the host but the image tensor.

Timed region: `run_backbone_graph` + `infer` — the same span as every
other arm (`model.get_action` minus `prepare_input`). The host processor
and `decode_action` sit outside, as they do on every other arm.
"""

from __future__ import annotations

import functools
import pathlib
import sys

import numpy as np
import torch


class NativeGrootHost:
    name = "native"

    def __init__(self, *, checkpoint: str, host_src: str,
                 embodiment_tag: str = "libero_sim", use_fp4: bool = False,
                 num_views: int = 2, denoising_steps: int = 4,
                 action_horizon: int | None = None):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        self.host_src = str(host_src)
        self.checkpoint = str(checkpoint)
        self.embodiment_tag = embodiment_tag
        self.use_fp4 = use_fp4
        self.num_views = int(num_views)
        self.denoising_steps = int(denoising_steps)
        self.action_horizon = action_horizon
        self.report: dict = {}
        self._prompt = None
        self._pipe = None
        self._aux = None

    # -- build ---------------------------------------------------------
    def build(self):
        from host_groot_isaac import IsaacGrootHost

        # the observation pipeline: the host's own processor, nothing else
        self.obs_host = IsaacGrootHost(checkpoint=self.checkpoint,
                                       host_src=self.host_src, arm="eager",
                                       embodiment_tag=self.embodiment_tag,
                                       num_views=self.num_views,
                                       denoising_steps=self.denoising_steps)
        self.obs_host.build()
        self.report.update({
            "host": "FlashRT native (hand-written pipeline)",
            "observation_pipeline": "Isaac-GR00T processor",
            "flash_attn": self.obs_host.flash_attn,
            "checkpoint": self.checkpoint,
            "precision": "NVFP4 DiT + FP8 backbone + FA4" if self.use_fp4
                         else "FP8",
            "embodiment_tag": self.embodiment_tag,
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(0),
            "arm": "native",
            "denoising_steps": self.denoising_steps,
            "timed_region": "run_backbone_graph + infer "
                            "(prepare_input and decode_action excluded)",
        })

    # -- the aux bundle (once, from the host's own forward) -------------
    def _capture_aux(self):
        """One hooked host pass records the frontend's setup bundle.

        The same tensors, from the same hook points, as the repository's
        own `capture_llm_aux.py`; taken here from this episode's first
        real observation rather than from a saved fixture.
        """
        model = self.obs_host.model
        lm = model.backbone.model.model.language_model
        visual = model.backbone.model.model.visual
        block0 = visual.blocks[0]
        rot = lm.rotary_emb
        captured: dict = {}
        originals = {"lm": lm.forward, "rot": rot.forward,
                     "visual": visual.forward}

        def lm_hook(self_, *a, **kw):
            if kw.get("inputs_embeds") is not None:
                captured["llm_input_embeds"] = kw["inputs_embeds"].detach(
                ).to(torch.float32).cpu()
            if kw.get("input_ids") is not None:
                captured["input_ids"] = kw["input_ids"].detach().cpu()
            if kw.get("visual_pos_masks") is not None:
                captured["visual_pos_masks"] = kw[
                    "visual_pos_masks"].detach().cpu()
            if kw.get("position_ids") is not None:
                captured["position_ids"] = kw["position_ids"].detach().cpu()
            return originals["lm"](*a, **kw)

        def rot_hook(self_, x, position_ids):
            cos, sin = originals["rot"](x, position_ids)
            captured["rope_cos"] = cos.detach().cpu()
            captured["rope_sin"] = sin.detach().cpu()
            return cos, sin

        def visual_hook(self_, hidden_states, grid_thw, **kw):
            captured["grid_thw"] = grid_thw.detach().cpu()
            return originals["visual"](hidden_states, grid_thw, **kw)

        def block0_pre(module, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            captured["pixel_features"] = h.detach().to(torch.float32).cpu()
            return None

        lm.forward = functools.partial(lm_hook, lm)
        rot.forward = functools.partial(rot_hook, rot)
        visual.forward = functools.partial(visual_hook, visual)
        handle = block0.register_forward_pre_hook(block0_pre, with_kwargs=True)
        try:
            with torch.no_grad():
                self.obs_host._hot()
        finally:
            lm.forward = originals["lm"]
            rot.forward = originals["rot"]
            visual.forward = originals["visual"]
            handle.remove()

        # `input_ids` reaches the language model only as embeddings on this
        # transformers generation, so it comes from the prepared inputs
        # instead — the same tensor, one step earlier
        backbone_inputs = self.obs_host._windows[0]
        captured.setdefault("input_ids",
                            backbone_inputs["input_ids"].detach().cpu())
        missing = [k for k in ("llm_input_embeds", "input_ids", "rope_cos",
                               "rope_sin", "pixel_features", "grid_thw",
                               "visual_pos_masks") if k not in captured]
        if missing:
            raise RuntimeError(f"aux capture missed {missing}")
        captured["pixel_values"] = self._pixel_window.detach().float().cpu()
        # the noise the host's own action head drew on this pass, pinned by
        # the harness: every arm of the film integrates from this tensor
        noise = self.obs_host._noise_box.get("value")
        if noise is None:
            raise RuntimeError("the host's flow-matching noise was never "
                               "pinned — the arms would not be comparable")
        captured["initial_noise"] = noise.detach().float().cpu()
        return captured

    def _build_pipe(self):
        """Construct the frontend the way the repository's own benchmark
        does — by naming the tier's class.

        Not through `flash_rt.load_model(..., use_fp4=True)`. That router
        decides which keyword arguments a frontend accepts with
        `inspect.signature(pipe_cls)`, and the NVFP4 tier's `__init__` is
        `(*args, **kwargs)` (it probes for its kernel bindings before
        delegating), so the signature reports no `embodiment_tag` and the
        router drops it. The tier then loads under its default embodiment
        instead of this checkpoint's, and the actions come out wrong
        while everything about the run looks healthy: it loads, it is
        fast, its outputs are stable and repeatable, and its 132-wide
        chunk still correlates at 0.95 with the host's because 125 of
        those columns are padding. The FP8 tier's `__init__` names its
        parameters, so the same call works there — which makes the
        failure look exactly like an NVFP4 precision problem. It is not:
        bound by name, the two tiers agree to 0.99996.
        """
        self._aux = self._capture_aux()
        if self.use_fp4:
            from flash_rt.frontends.torch.groot_n17_thor_fp4 import (
                GrootN17TorchFrontendThorFP4 as Frontend)
        else:
            from flash_rt.frontends.torch.groot_n17_thor_fp8 import (
                GrootN17TorchFrontendThorFP8 as Frontend)
        self._pipe = Frontend(self.checkpoint, num_views=self.num_views,
                              embodiment_tag=self.embodiment_tag)
        if self._pipe.embodiment_tag != self.embodiment_tag:
            raise RuntimeError(
                f"the frontend loaded embodiment {self._pipe.embodiment_tag!r}, "
                f"not {self.embodiment_tag!r} — its state and action encoders "
                "would be reading the wrong embodiment embedding")
        self._pipe.set_prompt(aux=self._aux, prompt=self._prompt)
        self.report.update({
            "frontend": type(self._pipe).__name__,
            "frontend_embodiment_tag": self._pipe.embodiment_tag,
            "frontend_embodiment_id": int(self._pipe._embodiment_id),
            "prompt_tokens": int(self._aux["llm_input_embeds"].shape[1]),
            "vision_tokens": int(self._aux["pixel_features"].shape[0]),
        })
        self._noise = self._aux["initial_noise"].to("cuda").bfloat16(
        ).contiguous()

    # -- host preprocessing (untimed) ----------------------------------
    def set_task(self, text: str):
        self._prompt = text
        self.obs_host.set_task(text)

    def observe(self, img, wrist, state):
        self.obs_host.observe(img, wrist, state)
        backbone_inputs, action_inputs = self.obs_host._windows
        # the host's own prepared tensors, bit for bit: the patch input the
        # backbone graph consumes and the normalised state the DiT consumes
        self._pixel_window = backbone_inputs["pixel_values"]
        self._state = action_inputs["state"].reshape(1, 1, -1).float()
        if self._pipe is None:
            self._build_pipe()

    def sync(self):
        torch.cuda.synchronize()

    # -- the timed region ----------------------------------------------
    def infer(self):
        self._pipe._backbone_features = self._pipe.run_backbone_graph(
            {"pixel_values": self._pixel_window})
        kwargs = {"num_inference_timesteps": self.denoising_steps,
                  "initial_noise": self._noise}
        if self.action_horizon is not None:
            kwargs["action_horizon"] = int(self.action_horizon)
        return self._pipe.infer(self._state, **kwargs)

    # -- host postprocessing (untimed) ---------------------------------
    def decode(self, pred):
        # the host's own decode, so the four arms differ inside the timed
        # region and nowhere else
        return self.obs_host.decode(pred)

    def warmup(self, img, wrist, state, rounds: int = 20):
        self.observe(img, wrist, state)
        from parity import per_dimension_parity

        with torch.no_grad():
            reference = self.obs_host._hot()
        decoded_reference = self.obs_host.decode(reference)
        decoded_treated = self.decode(self.infer())
        self.report["parity"] = per_dimension_parity(decoded_treated,
                                                     decoded_reference)
        # the raw chunk too, for the record: 7 of its 132 columns are the
        # embodiment's, the rest is padding the model may fill freely
        self.report["parity_cosine_raw_chunk"] = float(
            torch.nn.functional.cosine_similarity(
                self.infer().detach().float().cpu().flatten(),
                reference.detach().float().cpu().flatten(), dim=0))
        for _ in range(rounds):
            self.infer()
            torch.cuda.synchronize()

    def finish(self):
        stats = getattr(self._pipe, "get_latency_stats", None)
        if callable(stats):
            self.report["frontend_latency"] = stats()
        return self.report

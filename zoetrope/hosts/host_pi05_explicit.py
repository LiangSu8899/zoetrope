"""Pi0.5 on the LeRobot host — the repository's own explicit pipeline.

The structures pane of a film should be the assembly the library
publishes, not a re-implementation of it, so this adapter imports
`examples/structure_pipeline/pi05.py` and calls its `load_host()`,
`build_inputs()` and `build()` unmodified. What it adds is the only
thing a film needs and a latency harness does not: the observation
windows are refreshed from live LIBERO frames every control step.

Arms:
    eager      the host as loaded, nothing applied
    captured   the same host through `structures.capture`, no swaps
    attach     the explicit seat book (region band ladder + the pi052
               denoise lowering + residual seats), then the same capture

`build_inputs` already allocates the inputs as persistent buffers and
hands back views into them, which is exactly the fixed-observation-window
discipline a captured graph needs; `observe()` writes through those
views. The stale-window gate in the loop is what proves it.
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import torch


class ExplicitPi05Host:
    name = "lerobot-explicit"

    def __init__(self, *, checkpoint: str, lerobot_src: str,
                 examples_src: str, arm: str = "attach",
                 tokenizer: str | None = None, num_views: int = 2,
                 seed: int = 7):
        self.checkpoint = str(checkpoint)
        self.lerobot_src = str(lerobot_src)
        self.examples_src = str(examples_src)
        self.arm = arm
        self.tokenizer = tokenizer
        self.num_views = int(num_views)
        self.seed = seed
        self.report: dict = {}
        self._prompt = None
        self._armed = False
        self._rebuilds = 0

    # -- build ---------------------------------------------------------
    def _import_vendor(self, bundle: str):
        os.environ["PI05_LEROBOT_SRC"] = self.lerobot_src
        os.environ["PI05_CKPT"] = self.checkpoint
        os.environ["PI05_TOKENIZER"] = "google/paligemma-3b-pt-224"
        os.environ["PI05_OBS_BUNDLE"] = bundle
        os.environ["PI05_VIEWS"] = str(self.num_views)
        sys.path.insert(0, self.lerobot_src)

        # the board's flash_attn extension does not load under the torch
        # the kernel artifacts need, and lerobot imports it transitively
        # through policies no arm here instantiates
        from flash_attn_stub import install_if_broken
        self.report["flash_attn"] = install_if_broken()

        # the hub copy of the tokenizer is gated on this board; the
        # sentencepiece model it is built from is on disk
        if self.tokenizer:
            from host_pi05_lerobot import _patch_tokenizer
            _patch_tokenizer(pathlib.Path(self.tokenizer))

        # must be installed before the denoise lowering captures its own
        # `real_tensor`, so the vendor pin's fall-through lands in the
        # cache instead of doing a host-to-device copy inside the capture
        import scalar_tensor_cache
        self.report["scalar_tensor_cache"] = scalar_tensor_cache.install()

        sys.path.insert(0, self.examples_src)
        import pi05 as vendor
        return vendor

    def _write_bundle(self, path: pathlib.Path, img, wrist, state):
        """The vendor example's own bundle format, from a live frame."""
        views = [img, wrist][:self.num_views]
        images = torch.from_numpy(
            np.stack(views).astype(np.float32) / 127.5 - 1.0)
        # the checkpoint's normaliser carries statistics at the dataset's
        # own state width (8 for LIBERO); the example's hard-coded 32-wide
        # state feature is a droid-shaped default, and padding to it makes
        # the normaliser raise 32-vs-8
        raw = torch.as_tensor(np.asarray(state, dtype=np.float32))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"input_images": images, "state": raw,
                    "prompt": self._prompt}, path)

    def build(self):
        # deferred: the bundle needs the episode's first observation, so
        # the real build happens on the first `observe`
        self.report.update({
            "host": "lerobot PI052Policy (explicit structure pipeline)",
            "lerobot_src": self.lerobot_src,
            "examples_src": self.examples_src,
            "checkpoint": self.checkpoint,
            "num_views": self.num_views,
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(0),
            "arm": self.arm,
            "assembly": "examples/structure_pipeline/pi05.py "
                        "(load_host + build_inputs + build), unmodified",
            "timed_region": "the captured stage replay "
                            "(host preprocessing and action decode outside)",
        })

    def _late_build(self, img, wrist, state):
        import json

        bundle = pathlib.Path(
            os.environ.get("PI05_BUNDLE_OUT", "_explicit_bundle.pt"))
        self._write_bundle(bundle, img, wrist, state)
        vendor = self._import_vendor(str(bundle))
        self.vendor = vendor

        self.policy = vendor.load_host()
        (self.views, self.img_masks, self.tokens,
         self.masks, self.noise) = vendor.build_inputs(self.policy)
        self.model = self.policy.model

        # the same processors the vendor example built, kept so the loop
        # can refresh the windows and decode the actions
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config

        raw = json.load(open(f"{self.checkpoint}/config.json"))
        cfg = PI05Config(
            action_expert_variant=raw.get("action_expert_variant",
                                          "gemma_300m"),
            paligemma_variant=raw.get("paligemma_variant", "gemma_2b"),
            dtype=raw.get("dtype", "float32"))
        cfg.input_features = self.policy.config.input_features
        cfg.output_features = self.policy.config.output_features
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=self.checkpoint,
            preprocessor_overrides={"tokenizer_processor": {
                "tokenizer_name": "google/paligemma-3b-pt-224"}})
        # `load_host()` overrides the policy's action feature to the
        # model's padded width (32); the checkpoint's un-normaliser carries
        # statistics at the dataset's own width (7 for LIBERO), so the
        # slice has to come from the checkpoint, not from that override
        self.action_dim = int(
            raw.get("output_features", {}).get("action", {})
               .get("shape", [7])[0])

    # -- the model call ------------------------------------------------
    def _hot(self):
        return self.model.sample_actions(self.views, self.img_masks,
                                         self.tokens, self.masks,
                                         noise=self.noise.clone())

    def _arm_once(self):
        if self._armed:
            return
        self._armed = True
        from flash_rt import structures

        with torch.no_grad():
            reference = self._hot().detach().float().clone()

        if self.arm == "attach":
            extras, notes, undo = self.vendor.build(self.policy,
                                                    lambda: self._hot())
            self._undo = undo
            self.report["book"] = [(r["family"], r["band"])
                                   for r in notes["regions_bound"]]
            self.report["regions_refused"] = notes["regions_refused"][:12]
            self.report["seats"] = notes.get("seats")
            self.report["seats_dropped_under_regions"] = notes.get(
                "seats_dropped_under_regions")
            with torch.no_grad():
                treated = self._hot().detach().float().clone()
            self.report["eager_parity_cosine"] = float(
                torch.nn.functional.cosine_similarity(
                    treated.flatten(), reference.flatten(), dim=0))

        if self.arm == "eager":
            self._call = self._hot
            return

        # the production window form the example measures: the noise
        # buffer is a true in-out window, consumed as the seed and
        # rewritten with the actions, so a replay is one graph launch
        torch._dynamo.reset()
        self._seed_noise = self.noise.clone()
        self._noise_win = torch.zeros_like(self.noise)

        def hot_body():
            acts = self.model.sample_actions(self.views, self.img_masks,
                                             self.tokens, self.masks,
                                             noise=self._noise_win)
            self._noise_win.copy_(acts)
            return self._noise_win

        self._noise_win.copy_(self._seed_noise)
        stage = structures.capture(torch.compile(hot_body),
                                   model=self.policy,
                                   windows={"noise": self._noise_win},
                                   reference=None, gate_cos=0,
                                   min_speedup=0)
        self._stage = stage
        self.report["capture"] = stage.certification

        def replay():
            self._noise_win.copy_(self._seed_noise)
            stage.replay()
            return self._noise_win

        self._call = replay
        with torch.no_grad():
            captured = self._call().detach().float().clone()
        self.report["captured_parity_cosine"] = float(
            torch.nn.functional.cosine_similarity(
                captured.flatten(), reference.flatten(), dim=0))

    # -- host preprocessing (untimed) ----------------------------------
    def set_task(self, text: str):
        self._prompt = text

    @staticmethod
    def _chw(img):
        return torch.from_numpy(
            np.ascontiguousarray(img.astype(np.float32) / 255.0)
        ).permute(2, 0, 1).unsqueeze(0).contiguous()

    def observe(self, img, wrist, state):
        if not hasattr(self, "policy"):
            self._late_build(img, wrist, state)

        obs = {"observation.state": torch.as_tensor(
                   np.asarray(state, dtype=np.float32)).unsqueeze(0),
               "task": self._prompt,
               "observation.images.image": self._chw(img)}
        if self.num_views >= 2:
            obs["observation.images.image2"] = self._chw(wrist)
        batch = self.pre(obs)
        batch = {k: (v.to("cuda") if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        self._batch = batch

        from lerobot.utils.constants import (OBS_LANGUAGE_ATTENTION_MASK,
                                             OBS_LANGUAGE_TOKENS)
        with torch.no_grad():
            images, img_masks = self.policy._preprocess_images(dict(batch))
        try:
            for index, view in enumerate(self.views):
                view.copy_(images[index])
            for index, mask in enumerate(self.img_masks):
                mask.copy_(img_masks[index])
            self.tokens.copy_(batch[OBS_LANGUAGE_TOKENS])
            self.masks.copy_(batch[OBS_LANGUAGE_ATTENTION_MASK])
        except RuntimeError as exc:
            raise RuntimeError(
                "the observation changed form mid-episode — a replay over "
                f"a coerced write is silently wrong: {exc}") from exc

    def sync(self):
        torch.cuda.synchronize()

    # -- the timed region ----------------------------------------------
    def infer(self):
        with torch.no_grad():
            return self._call()

    # -- host postprocessing (untimed) ---------------------------------
    def decode(self, chunk):
        actions = chunk[0, :, :self.action_dim].detach().float().cpu()
        out = self.post(actions)
        return np.asarray(out["action"] if isinstance(out, dict) else out)

    def warmup(self, img, wrist, state, rounds: int = 20):
        self.observe(img, wrist, state)
        self._arm_once()
        for _ in range(rounds):
            self.infer()
            torch.cuda.synchronize()

    def finish(self):
        self.report["observation_rebuilds"] = self._rebuilds
        return self.report

"""Pi0.5, FlashRT native — the hand-written pipeline replaces the model.

There is no host model here: `flash_rt.load_model` builds the whole tick
itself and the loop feeds it the same observations the host arms see.

Boundary note, stated rather than hidden: the packaged frontend's
`predict()` carries the image upload and the action un-normalisation
inside the call, where the host arms pay those outside their timed
region. The native number is therefore the pessimistic one — the wall
cost of a decision, not a graph replay.
"""

from __future__ import annotations

import numpy as np
import torch


class NativePi05Host:
    name = "native"

    def __init__(self, *, checkpoint: str, num_views: int = 2,
                 use_fp4: bool = False, action_horizon: int | None = None,
                 autotune: int = 3, fixed_noise: bool = True, seed: int = 0):
        self.fixed_noise = fixed_noise
        self.seed = seed
        self.checkpoint = checkpoint
        self.num_views = num_views
        self.use_fp4 = use_fp4
        self.action_horizon = action_horizon
        self.autotune = autotune
        self.report: dict = {}
        self._prompt = None
        self._images = None

    def build(self):
        import flash_rt

        # the repository's own published Thor configuration, knob for knob
        # (tests/bench_pi05_decoder_fp4_e2e.py :: public_api_kwargs) — the
        # FP8 column carries FA4 too, so the only difference between the
        # two tiers is the NVFP4 encoder/decoder path
        kwargs = dict(framework="torch", config="pi05", hardware="thor",
                      num_views=self.num_views, autotune=self.autotune,
                      use_fa4=True)
        if self.action_horizon is not None:
            kwargs["action_horizon"] = int(self.action_horizon)
        if self.use_fp4:
            kwargs.update(use_fp4=True, use_fp4_decoder=True)
        self.load_kwargs = dict(kwargs)
        self.model = flash_rt.load_model(self.checkpoint, **kwargs)
        self.report.update({
            "host": "FlashRT native (hand-written pipeline)",
            "checkpoint": str(self.checkpoint),
            "precision": "NVFP4 + FA4" if self.use_fp4 else "FP8 + FA4",
            "load_model_kwargs": kwargs,
            "num_views": self.num_views,
            "action_horizon": self.action_horizon,
            "torch": torch.__version__,
            "device_name": torch.cuda.get_device_name(0),
            "arm": "native",
            "timed_region": "flash_rt predict(): image upload + graph replay "
                            "+ download + un-normalisation",
        })

    def set_task(self, text: str):
        self._prompt = text

    def observe(self, img, wrist, state):
        # pi0.5-LIBERO takes no proprioceptive input: openpi's config has
        # discrete_state_input=False and the checkpoint carries no
        # state projection, so the state is not part of any arm's input.
        self._images = [np.ascontiguousarray(img)]
        if self.num_views >= 2:
            self._images.append(np.ascontiguousarray(wrist))
        while len(self._images) < self.num_views:
            self._images.append(self._images[-1])
        if self.fixed_noise:
            # this frontend draws its flow-matching sample from numpy's
            # global RNG inside the call; re-seeding here — outside the
            # timed region — holds the sample constant across steps, the
            # same discipline the other arms get from a fixed tensor
            np.random.seed(self.seed)

    def sync(self):
        torch.cuda.synchronize()

    def infer(self):
        return self.model.predict(self._images, prompt=self._prompt)

    def decode(self, chunk):
        return np.asarray(chunk)

    def warmup(self, img, wrist, state, rounds: int = 20):
        self.set_task(self._prompt)
        self.observe(img, wrist, state)
        for _ in range(rounds):
            self.infer()
            torch.cuda.synchronize()

    def finish(self):
        spec = getattr(self.model, "precision_spec", None)
        if spec is not None:
            try:
                self.report["precision_spec"] = str(spec)[:400]
            except Exception:  # noqa: BLE001
                pass
        # The frontend times its own `infer()` from the top of the call to
        # the last synchronise. Reporting it next to the loop's wall median
        # says how much of a decision is the pipeline and how much is the
        # public wrapper around it — image upload, download, the
        # un-normalisation, the prompt-equality check.
        pipe = getattr(self.model, "pipeline", None)
        stats = getattr(pipe, "get_latency_stats", None)
        if callable(stats):
            try:
                self.report["frontend_infer_latency"] = stats()
            except Exception:  # noqa: BLE001
                pass
        return self.report

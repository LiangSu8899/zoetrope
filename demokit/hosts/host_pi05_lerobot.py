"""Pi0.5 on the LeRobot host.

Arms:
    eager      the host's own code, nothing added
    compiled   the same host under torch.compile + whole-graph capture
    attach     structures.auto_swaps -> swap.attach, captured the same way

The timed region is the host's own model-side entry point,
`PI05Policy.predict_action_chunk`. The processor pipeline — image decode,
state normalisation, prompt tokenisation — and the un-normalising
postprocessor sit outside it, identically for every arm.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
from typing import Any

import numpy as np
import torch


class _LocalPaliGemmaTokenizer:
    """The checkpoint's tokenizer from a local sentencepiece model.

    The hub copy is not reachable from the board; the bytes are the same.
    """

    def __init__(self, model_path: pathlib.Path):
        import sentencepiece

        self._sp = sentencepiece.SentencePieceProcessor(
            model_proto=model_path.read_bytes())

    def __call__(self, text, *, max_length: int, truncation: bool = True,
                 padding: str = "max_length", padding_side: str = "right",
                 return_tensors: str = "pt", **_: Any):
        if return_tensors != "pt":
            raise ValueError("only return_tensors='pt' is supported")
        texts = [text] if isinstance(text, str) else list(text)
        ids_rows, mask_rows = [], []
        for item in texts:
            ids = list(self._sp.encode(item, add_bos=True))
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            mask = [1] * len(ids)
            if padding == "max_length" and len(ids) < max_length:
                pad = [0] * (max_length - len(ids))
                ids = (pad + ids) if padding_side == "left" else (ids + pad)
                mask = (pad + mask) if padding_side == "left" else (mask + pad)
            ids_rows.append(ids)
            mask_rows.append(mask)
        return {"input_ids": torch.tensor(ids_rows, dtype=torch.long),
                "attention_mask": torch.tensor(mask_rows, dtype=torch.long)}


def _patch_tokenizer(model_path: pathlib.Path) -> None:
    import transformers

    import lerobot.processor.tokenizer_processor as tokenizer_processor

    local = _LocalPaliGemmaTokenizer(model_path)
    original = transformers.AutoTokenizer.from_pretrained

    def from_pretrained(name, *args, **kwargs):
        if str(name) == "google/paligemma-3b-pt-224":
            return local
        return original(name, *args, **kwargs)

    transformers.AutoTokenizer.from_pretrained = from_pretrained
    tokenizer_processor.AutoTokenizer.from_pretrained = from_pretrained


class LeRobotPi05Host:
    name = "lerobot"

    def __init__(self, *, checkpoint: str, lerobot_src: str, arm: str = "eager",
                 tokenizer: str | None = None, num_steps: int | None = None,
                 fixed_noise: bool = True, seed: int = 0,
                 compile_mode: str = "max-autotune-no-cudagraphs"):
        sys.path.insert(0, str(pathlib.Path(lerobot_src)))
        for name in [m for m in sys.modules
                     if m == "lerobot" or m.startswith("lerobot.")]:
            del sys.modules[name]
        # lerobot's policy factory imports xVLA -> Florence-2 -> flash_attn,
        # whose extension does not load under the torch the kernel
        # artifacts need. No arm here instantiates those policies.
        from flash_attn_stub import install_if_broken
        self._flash_attn = install_if_broken()
        self.lerobot_src = lerobot_src
        self.checkpoint = pathlib.Path(checkpoint)
        self.arm = arm
        self.tokenizer = pathlib.Path(tokenizer) if tokenizer else None
        self.num_steps = num_steps
        self.fixed_noise = fixed_noise
        self.seed = seed
        self.compile_mode = compile_mode
        self.report: dict = {}
        self._armed = False
        self._batch = None
        self._rebuilds = 0
        self._prompt = None

    # -- build ---------------------------------------------------------
    def _config(self):
        import draccus

        from lerobot.policies.pi05.configuration_pi05 import PI05Config

        cfg = json.loads((self.checkpoint / "config.json").read_text())
        cfg.pop("type", None)
        cfg["device"] = "cuda"
        cfg["compile_model"] = False   # the arm decides, not the config
        cfg["gradient_checkpointing"] = False
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as handle:
            json.dump(cfg, handle)
            tmp = handle.name
        try:
            with draccus.config_type("json"):
                return draccus.parse(PI05Config, tmp, args=[])
        finally:
            pathlib.Path(tmp).unlink(missing_ok=True)

    def build(self):
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (batch_to_transition,
                                                  transition_to_batch)

        config = self._config()
        self.policy = PI05Policy.from_pretrained(str(self.checkpoint),
                                                 config=config, strict=False)
        self.policy.eval()
        self.config = config

        if self.tokenizer is not None:
            _patch_tokenizer(self.tokenizer)
        import lerobot.policies.pi05.processor_pi05  # noqa: F401

        self.pre = PolicyProcessorPipeline.from_pretrained(
            str(self.checkpoint), config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)
        self.post = PolicyProcessorPipeline.from_pretrained(
            str(self.checkpoint), config_filename="policy_postprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)

        if self.fixed_noise:
            g = torch.Generator(device="cuda").manual_seed(self.seed)
            self._noise = torch.randn(1, int(config.chunk_size),
                                      int(config.max_action_dim),
                                      device="cuda", dtype=torch.float32,
                                      generator=g)
        else:
            self._noise = None

        import transformers
        self.report.update({
            "host": "lerobot PI05Policy",
            "lerobot_src": str(self.lerobot_src),
            "checkpoint": str(self.checkpoint),
            "chunk_size": int(config.chunk_size),
            "num_inference_steps": int(config.num_inference_steps),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device_name": torch.cuda.get_device_name(0),
            "arm": self.arm,
            "fixed_noise": bool(self.fixed_noise),
            "compile_mode": self.compile_mode,
            "timed_region": "PI05Policy.predict_action_chunk",
            "flash_attn": self._flash_attn,
        })

    # -- the model call ------------------------------------------------
    def _predict(self):
        kwargs = {}
        if self._noise is not None:
            kwargs["noise"] = self._noise
        if self.num_steps is not None:
            kwargs["num_steps"] = self.num_steps
        return self.policy.predict_action_chunk(self._batch, **kwargs)

    def _arm_once(self):
        if self._armed:
            return
        self._armed = True
        if self.arm == "eager":
            self._call = self._predict
            return
        if self.arm == "attach":
            self._attach()
        torch._dynamo.reset()
        self._call = torch.compile(self._predict, mode=self.compile_mode,
                                   fullgraph=False, dynamic=False)

    def _attach(self):
        from flash_rt import structures
        from flash_rt.structures import swap
        from flash_rt.structures.impls import unavailable_report

        def run_once():
            with torch.no_grad():
                return self._predict()

        plan = structures.auto_swaps(self.policy.model, run_once, verbose=True)
        self._handle = swap.attach(self.policy.model, plan.swaps,
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

    @staticmethod
    def _chw(img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            np.ascontiguousarray(img.astype(np.float32) / 255.0)
        ).permute(2, 0, 1).contiguous()

    def observe(self, img, wrist, state):
        raw = {
            "observation.images.image": self._chw(img),
            "observation.images.image2": self._chw(wrist),
            "observation.state": torch.tensor(np.asarray(state),
                                              dtype=torch.float32),
            "task": self._prompt,
        }
        batch = self.pre(raw)
        if self._batch is None:
            self._batch = batch
            return
        try:
            for key, value in batch.items():
                if torch.is_tensor(value):
                    self._batch[key].copy_(value)
                else:
                    self._batch[key] = value
        except (KeyError, RuntimeError):
            self._rebuilds += 1
            self._batch = batch

    def sync(self):
        torch.cuda.synchronize()

    # -- the timed region ----------------------------------------------
    def infer(self):
        with torch.no_grad():
            return self._call()

    # -- host postprocessing (untimed) ---------------------------------
    def decode(self, chunk):
        out = self.post({"action": chunk[0].detach().float().cpu()})
        actions = out["action"] if "action" in out else out
        return np.asarray(actions)

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
        return self.report

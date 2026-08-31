"""GR00T N1.7 — the repository's own explicit structure pipeline.

The structures pane of a film should be the assembly the library
publishes, not a re-implementation of it, so this adapter imports
`examples/structure_pipeline/groot_n17.py` and calls its `build()`
unmodified — every seat path, every calibration hook, every binder call
as written there. It then closes the same door
`examples/structure_pipeline/full_graph.py` closes: wire the cadence
statics to their producer, compile, capture the whole hot path as one
CUDA graph, replay it.

What this adapter adds is the only thing a film needs and a latency
harness does not: the observation windows are refreshed from live LIBERO
frames every control step, and the §5.1 stale-window gate in the loop is
what proves the replay still looks at the camera.

The host itself is the official Isaac-GR00T policy, loaded by the same
adapter the eager pane uses, so the two panes differ in exactly one
thing: whether the seat book is bound.

Arms:
    attach     the explicit seat book, then the capture
    captured   the same capture with no seats bound
"""

from __future__ import annotations

import pathlib
import sys

import torch


class ExplicitGrootHost:
    name = "isaac-explicit"

    def __init__(self, *, checkpoint: str, host_src: str, examples_src: str,
                 arm: str = "attach", embodiment_tag: str = "libero_sim",
                 num_views: int = 2, denoising_steps: int | None = None):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        self.checkpoint = str(checkpoint)
        self.host_src = str(host_src)
        self.examples_src = str(examples_src)
        self.arm = arm
        self.embodiment_tag = embodiment_tag
        self.num_views = int(num_views)
        self.denoising_steps = denoising_steps
        self.report: dict = {}
        self._armed = False

    # -- build ---------------------------------------------------------
    def build(self):
        from host_groot_isaac import IsaacGrootHost

        self.obs_host = IsaacGrootHost(
            checkpoint=self.checkpoint, host_src=self.host_src, arm="eager",
            embodiment_tag=self.embodiment_tag, num_views=self.num_views,
            denoising_steps=self.denoising_steps)
        self.obs_host.build()
        self.model = self.obs_host.model

        sys.path.insert(0, self.examples_src)
        import groot_n17 as vendor
        self.vendor = vendor

        self.report.update(self.obs_host.report)
        self.report.update({
            "host": "Isaac-GR00T (official) + explicit structure pipeline",
            "arm": self.arm,
            "assembly": "examples/structure_pipeline/groot_n17.py build(), "
                        "unmodified; captured as in full_graph.py",
            "timed_region": "the captured stage replay of "
                            "backbone + action_head.get_action "
                            "(prepare_input and decode_action excluded)",
        })

    # -- the model call ------------------------------------------------
    def _hot(self):
        return self.obs_host._hot()

    def _arm_once(self):
        if self._armed:
            return
        self._armed = True
        from flash_rt.structures import capture as capture_stage
        from flash_rt.structures import swap
        from flash_rt.structures.impls import unavailable_report

        with torch.no_grad():
            reference = self._hot().detach().float().cpu()

        def run_once():
            with torch.no_grad():
                self._hot()

        if self.arm == "attach":
            asm, extras = self.vendor.build(self.model, run_once)
            self._handle = swap.attach(
                self.model, asm.swaps, consume=False,
                observe=extras["observed"], revert=extras["revert"],
                on_guard_fail="raise")
            self.report.update({
                "seats_bound": dict(asm.families),
                "swaps": len(asm.swaps),
                "observed": len(extras["observed"]),
                "refused": len(asm.refused),
                "refusals": [f"{p}: {why}" for p, why in asm.refused[:16]],
                "attention_variants": extras.get("variants"),
                "kernel_unavailable": unavailable_report(),
            })
            statics = extras["cadence_statics"]
            if statics:
                # the refresh rides the producer's forward, so the captured
                # graph writes the cross-attention banks from this call's
                # own observation rather than from the calibration frame
                from flash_rt.structures.impls.cadence_static.cross_attention \
                    import wire_refresh_to_producer
                wire_refresh_to_producer(self.model, statics, run_once)
                self.report["cadence_statics"] = len(statics)
            with torch.no_grad():
                treated = self._hot().detach().float().cpu()
            self.report["eager_parity_cosine"] = _cos(treated, reference)

        torch._dynamo.reset()
        stage = capture_stage(
            torch.compile(self._hot, mode="max-autotune-no-cudagraphs",
                          fullgraph=False),
            model=self.model, warmup=8, gate_cos=0, min_speedup=0)
        self._stage = stage
        self.report["capture"] = stage.certification
        self.report["graph_lowering"] = stage.certification.get(
            "graph_lowering")
        self._call = stage.replay
        with torch.no_grad():
            captured = self._call().detach().float().cpu()
        self.report["captured_parity_cosine"] = _cos(captured, reference)
        # and in the units the controller receives, which is the only
        # parity a film can be read against
        from parity import per_dimension_parity
        self.report["parity"] = per_dimension_parity(
            self.decode(self._call()), self.obs_host.decode(reference))

    # -- host preprocessing / postprocessing (untimed) ------------------
    def set_task(self, text: str):
        self.obs_host.set_task(text)

    def observe(self, img, wrist, state):
        self.obs_host.observe(img, wrist, state)

    def sync(self):
        torch.cuda.synchronize()

    # -- the timed region ----------------------------------------------
    def infer(self):
        with torch.no_grad():
            return self._call()

    def decode(self, pred):
        return self.obs_host.decode(pred)

    def warmup(self, img, wrist, state, rounds: int = 20):
        self.observe(img, wrist, state)
        self._arm_once()
        for _ in range(rounds):
            self.infer()
            torch.cuda.synchronize()

    def finish(self):
        if hasattr(self, "_handle"):
            self.report["ledger"] = self._handle.summary()
        self.report["noise_pinned"] = {
            "shape": list(self.obs_host._noise_box.get("shape", ())) or None,
            "seed": self.obs_host._noise_box.get("seed")}
        return self.report


def _cos(a, b) -> float:
    return float(torch.nn.functional.cosine_similarity(
        torch.as_tensor(a).float().flatten(),
        torch.as_tensor(b).float().flatten(), dim=0))

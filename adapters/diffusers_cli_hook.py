"""Attach FlashRT structures inside the diffusers CLI, without editing it.

The CLI loads a pipeline, applies its own optimizations, then calls it. This
wraps that optimization step: the denoiser gets a shim that lets the first few
real calls through untouched, calibrates on exactly those, attaches, and hands
the rest of the schedule to the bound structures.

FLASHRT_TIME=1   time every denoiser call (works with or without attach)
FLASHRT_STRUCTURES=1  attach after the warmup calls
"""
import os
import sys
import time

_TIME = os.environ.get("FLASHRT_TIME") == "1"
_ATTACH = os.environ.get("FLASHRT_STRUCTURES") == "1"

if _TIME or _ATTACH:
    import torch

    WARMUP = int(os.environ.get("FLASHRT_WARMUP_CALLS", "4"))
    SCHEME = os.environ.get("FLASHRT_SCHEME", "nvfp4_balance")
    WANTED = tuple(os.environ.get(
        "FLASHRT_STRUCTURE_SET",
        "vision_ffn,qkv_pack,linear_proj").split(","))

    class _Shim(torch.nn.Module):
        def __init__(self, host):
            super().__init__()
            self.host = host
            self.seen, self.ms = [], []
            self.handle, self.done = None, not _ATTACH
            self.attach_s = 0.0

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self._modules["host"], name)

        def _attach(self):
            from flash_rt import structures
            from flash_rt.structures.swap import attach as swap_attach
            host, seen = self.host, self.seen

            def fwd():
                with torch.no_grad():
                    for a, k in seen:
                        host(*a, **k)

            a0, k0 = seen[0]
            print(f"[flashrt] denoiser called with args={len(a0)} "
                  f"kwargs={sorted(k0)}", file=sys.stderr, flush=True)
            t0 = time.perf_counter()
            plan = structures.auto_swaps(host, fwd, structures=WANTED,
                                         scheme=SCHEME, verbose=True)
            try:
                print(structures.explain(plan), file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[flashrt] explain failed: {e!r}", file=sys.stderr)
            print(f"[flashrt] plan.swaps={len(plan.swaps)}",
                  file=sys.stderr, flush=True)
            self.handle = swap_attach(host, plan.swaps,
                                      observe=plan.observed,
                                      revert=plan.revert)
            self.attach_s = time.perf_counter() - t0
            print(f"[flashrt] {len(plan.swaps)} seats bound in "
                  f"{self.attach_s:.1f} s (scheme {SCHEME}, calibrated on "
                  f"{len(seen)} real calls)", file=sys.stderr, flush=True)
            if os.environ.get("FLASHRT_AOT") == "1":
                self._export()
            self.seen = []
            self.done = True

        def _export(self):
            """Take the swapped denoiser whole: export, AOT-compile, and
            run the package instead of the module tree."""
            import gc
            from flash_rt.structures import aot_load, aot_package
            from flash_rt.structures.aot import AotModule
            host = self.host
            a0, k0 = self.seen[0]

            rep = self.handle.report()
            fb = sum(r.get("fallbacks", 0) for r in rep.values())
            n = sum(r.get("calls", 0) for r in rep.values())
            print(f"[flashrt] ledger before export: {n} structure calls, "
                  f"{fb} fallbacks", file=sys.stderr, flush=True)

            t0 = time.perf_counter()
            path = os.environ.get("FLASHRT_AOT_PATH",
                                  f"/tmp/frt_aot/cli_{SCHEME}.pt2")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pkg = aot_package(host, args=a0, kwargs=k0, package_path=path)
            export_s = time.perf_counter() - t0

            self.handle.detach()
            self.handle = None
            host_cpu = host.to("cpu")
            gc.collect(); torch.cuda.empty_cache()
            t1 = time.perf_counter()
            loaded = aot_load(pkg)
            load_s = time.perf_counter() - t1
            sig = dict(k0)

            def call(*a, **kw):
                for kk, v in sig.items():
                    if kk not in kw and not torch.is_tensor(v):
                        kw[kk] = v
                return loaded(*a, **{kk: kw[kk] for kk in sig if kk in kw})

            self._modules["host"] = AotModule(call, host_cpu)
            self.attach_s += export_s + load_s
            print(f"[flashrt] whole-graph export {export_s:.1f} s, "
                  f"aot_load {load_s:.1f} s", file=sys.stderr, flush=True)

        def forward(self, *a, **k):
            if not self.done:
                if len(self.seen) < WARMUP:
                    self.seen.append((
                        tuple(x.detach().clone() if torch.is_tensor(x) else x
                              for x in a),
                        {kk: (v.detach().clone() if torch.is_tensor(v) else v)
                         for kk, v in k.items()}))
                else:
                    self._attach()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = self.host(*a, **k)
            torch.cuda.synchronize()
            self.ms.append((time.perf_counter() - t0) * 1e3)
            return out

        def report(self):
            import statistics as st
            n = len(self.ms)
            if not n:
                return
            warm = self.ms[WARMUP + 1:] if _ATTACH else self.ms[2:]
            tag = "attached" if _ATTACH else "as shipped"
            print(f"[flashrt] {tag}: {n} denoiser calls, "
                  f"median {st.median(warm):.1f} ms warm, "
                  f"sum {sum(self.ms)/1e3:.2f} s"
                  + (f", attach {self.attach_s:.1f} s" if _ATTACH else ""),
                  file=sys.stderr, flush=True)

    def _patch():
        from diffusers.commands import run as _run
        original_opt = _run._apply_optimizations
        shims = []

        def patched_opt(pipeline, args):
            original_opt(pipeline, args)
            for attr in dir(pipeline):
                if not attr.startswith(_run._DENOISER_COMPONENT_KEYS):
                    continue
                mod = getattr(pipeline, attr, None)
                if isinstance(mod, torch.nn.Module):
                    sh = _Shim(mod)
                    setattr(pipeline, attr, sh)
                    shims.append(sh)
                    print(f"[flashrt] armed on pipeline.{attr}"
                          + ("" if _ATTACH else " (timing only)"),
                          file=sys.stderr, flush=True)

        _run._apply_optimizations = patched_opt
        import atexit
        atexit.register(lambda: [s.report() for s in shims])

    class _Hook:
        """Patch diffusers.commands.run the moment it finishes importing."""
        def find_module(self, name, path=None):
            return None

        def find_spec(self, name, path=None, target=None):
            if name == "diffusers.commands.run":
                sys.meta_path.remove(self)
                import importlib
                spec = importlib.util.find_spec(name)
                if spec is not None:
                    orig_exec = spec.loader.exec_module

                    def exec_module(module):
                        orig_exec(module)
                        _patch()

                    spec.loader.exec_module = exec_module
                return spec
            return None

    sys.meta_path.insert(0, _Hook())

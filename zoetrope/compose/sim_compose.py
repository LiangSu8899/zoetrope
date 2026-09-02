"""Compose two rollouts into one side-by-side video on a shared wall clock.

Each arm's rollout was recorded with the wall latency of every policy call.
A robot closes its loop at a fixed control period and cannot move on until
the next action arrives, so an arm's timeline is:

    step time = control period, plus the policy latency whenever the arm
                had to think before that step

Both panes then play against the same clock: identical actions (the arms
are parity-identical), different rates. Nothing is sped up or slowed down.
"""

import argparse
import json
import pathlib
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..frames import load_frames
from .race_compose import wrap

CONTROL_HZ = 20.0                     # LIBERO's control rate
FONT_DIRS = ["/usr/share/fonts/truetype/dejavu",
             "/usr/share/fonts/truetype/liberation"]

INK = (232, 236, 233)
MUTED = (147, 160, 154)
BG = (18, 25, 23)
CARD = (24, 33, 32)
LINE = (38, 50, 48)
ACCENT = (52, 194, 154)
STOCK = (171, 175, 164)
COMPILED = (140, 165, 200)
NATIVE = (226, 178, 96)


def font(size, bold=False):
    names = (["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] if bold
             else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf"])
    for d in FONT_DIRS:
        for n in names:
            p = pathlib.Path(d) / n
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def timeline(events, control_dt):
    """Wall-clock time at which each recorded frame is reached."""
    t, out = 0.0, []
    for e in events:
        if e["infer_ms"] is not None:
            t += e["infer_ms"] / 1000.0      # the robot waits for the policy
        t += control_dt
        out.append(t)
    return out


def _ffmpeg() -> str:
    """ffmpeg on PATH, or the one imageio-ffmpeg ships, or $FFMPEG."""
    import os
    import shutil

    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "no ffmpeg: install it, pip install imageio-ffmpeg, or set $FFMPEG"
        ) from exc


def hz_trace(events, times):
    """Policy decision rate: 1 / the latency of the decision in force."""
    out, cur = [], None
    for e in events:
        if e["infer_ms"] is not None:
            cur = 1000.0 / e["infer_ms"]
        out.append(cur if cur is not None else 0.0)
    return out


class Arm:
    def __init__(self, run_dir, label, sub, color, steps_cap=None):
        d = pathlib.Path(run_dir)
        self.frames = load_frames(d)
        blob = json.loads((d / "events.json").read_text())
        self.meta, self.events = blob["meta"], blob["events"]
        if steps_cap is not None:            # same robot work in every pane
            self.frames = self.frames[:steps_cap]
            self.events = self.events[:steps_cap]
        self.steps_cap = steps_cap
        self.times = timeline(self.events, 1.0 / CONTROL_HZ)
        self.hz = hz_trace(self.events, self.times)
        # a recording that named itself wins: the presets below are defaults
        # for runs made before `label`/`sub` were written into the meta, and
        # a stale default can describe an arm the recording is not
        self.label = self.meta.get("label") or label
        self.sub = self.meta.get("sub") or sub
        self.color = color
        self.median_ms = self.meta["median_infer_ms"]
        self.final_hz = float(np.median(self.hz[len(self.hz) // 3:]))

    def at(self, t):
        i = int(np.searchsorted(self.times, t, side="right")) - 1
        return max(0, min(i, len(self.frames) - 1))

    def finished(self, t):
        return t > self.times[-1]


#: One canvas width for the whole kit; the panes fill it.
CANVAS_W = 1280


def render(arms, out_path, fps=30, pane=440, seconds=None, speed=1.0,
           tail=1.6,
           title=None, note=None, footer=None):
    # hold the finished page: a film that ends on the frame its slowest arm
    # lands never shows that arm landing
    span = seconds or max(a.times[-1] for a in arms) + tail
    gap, pad, head, foot = 26, 34, 168, 178
    # every film in this kit is the same width, so a page of them lines up.
    # The panes divide that width rather than setting it.
    W = CANVAS_W
    pane = (W - pad * 2 - gap * (len(arms) - 1)) // len(arms)
    H = head + pane + foot
    f_title, f_lab, f_sub = font(27, True), font(19, True), font(14)
    f_hz, f_unit, f_small = font(52, True), font(15, True), font(13)

    frames_dir = pathlib.Path(out_path).parent / "_compose_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()

    n = int(span / speed * fps)
    for k in range(n):
        t = k / fps * speed
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((pad, 26), title or "pi0.5 · LIBERO closed loop", INK,
               font=f_title)
        d.text((pad, 62), arms[0].meta["task"], MUTED, font=f_sub)
        # the note is long and the canvas is only as wide as the panes:
        # wrap it rather than letting it run off the right edge
        ny = 84
        for line in wrap(d, note or (
                "same checkpoint · same task · same initial state · parity "
                f"0.99995 · control {CONTROL_HZ:.0f} Hz · re-plans every "
                f"{arms[0].meta['replan']} step(s) — each arm waits for its "
                "own policy"), f_small, W - 2 * pad)[:2]:
            d.text((pad, ny), line, MUTED, font=f_small)
            ny += 16

        for idx, a in enumerate(arms):
            x = pad + idx * (pane + gap)
            i = a.at(t)
            img = Image.fromarray(a.frames[i]).resize((pane, pane),
                                                      Image.BILINEAR)
            im.paste(img, (x, head))
            d.rectangle([x, head, x + pane - 1, head + pane - 1],
                        outline=LINE)
            d.text((x, head - 46), a.label, a.color, font=f_lab)
            d.text((x, head - 24), a.sub, MUTED, font=f_sub)

            y = head + pane + 12
            hz = a.hz[i] if not a.finished(t) else a.final_hz
            hz_txt = f"{hz:.1f}"
            d.text((x, y), hz_txt, a.color, font=f_hz)
            d.text((x + d.textlength(hz_txt, font=f_hz) + 8, y + 30), "Hz",
                   MUTED, font=f_unit)
            d.text((x + d.textlength(hz_txt, font=f_hz) + 8, y + 6),
                   f"{a.median_ms:.1f} ms", INK, font=f_sub)
            done = sum(1 for e, tt in zip(a.events, a.times)
                       if tt <= t and e["infer_ms"] is not None)
            d.text((x, y + 62), f"{done} decisions · "
                   f"{min(a.at(t) + 1, len(a.frames))} control steps",
                   MUTED, font=f_small)
            if a.finished(t):
                # A rollout only "completes" if the recording says the episode
                # succeeded.  Running out of recorded steps — because the arms
                # were capped, or because the run was truncated to ship — is a
                # different fact, and the pane says which one it is.
                ok = a.meta.get("success")
                if a.steps_cap or ok is None:
                    done_txt = (f"{len(a.events)} steps recorded, "
                                f"to {a.times[-1]:.1f} s")
                elif ok:
                    done_txt = f"task complete at {a.times[-1]:.1f} s"
                else:
                    done_txt = f"episode ended at {a.times[-1]:.1f} s"
                d.text((x, y + 80), done_txt, a.color, font=f_small)

            bar_y = y + 102
            width = int(pane * min(t / a.times[-1], 1.0))
            d.rectangle([x, bar_y, x + pane, bar_y + 5], fill=LINE)
            d.rectangle([x, bar_y, x + width, bar_y + 5], fill=a.color)

        d.text((W - pad - 96, 30), f"{t:5.1f} s", INK, font=f_lab)
        summary = " · ".join(f"{a.median_ms:.1f} ms" for a in arms)
        loops = " · ".join(f"{1000 / (a.median_ms + 1000 / CONTROL_HZ):.1f}"
                           for a in arms)
        d.text((pad, H - 42),
               f"policy latency {summary}   —   " + (
                   footer or "the same checkpoint, the same task, "
                             "different ways of executing it"), MUTED,
               font=f_small)
        d.text((pad, H - 24),
               f"closed loop incl. the robot's own {CONTROL_HZ:.0f} Hz "
               f"control step: {loops} Hz", MUTED, font=f_small)
        im.save(frames_dir / f"{k:05d}.jpg", quality=88)

    subprocess.run([
        _ffmpeg(), "-y", "-v", "error",
        "-framerate", str(fps), "-i", str(frames_dir / "%05d.jpg"),
        "-c:v", "libvpx-vp9", "-b:v", "1400k", "-pix_fmt", "yuv420p",
        str(out_path)], check=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()
    frames_dir.rmdir()
    print(f"wrote {out_path} ({n} frames, {n / fps:.1f}s at {speed}x)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="directory holding one sub-directory per arm, "
                         "each with events.json + frames.webp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--pane", type=int, default=400)
    ap.add_argument("--preset", default="ladder",
                    choices=["ladder", "cross-host", "groot", "groot4",
                             "groot4x", "groot-hosts",
                             "thor-pi05-race", "thor-pi05-ladder",
                             "thor-pi05-hosts",
                             "thor-groot-race", "thor-groot-hosts"])
    ap.add_argument("--title", default=None)
    ap.add_argument("--note", default=None)
    ap.add_argument("--footer", default=None,
                    help="the line under the panes; the default claims one "
                         "checkpoint across all panes, which is not true of "
                         "every cut")
    ap.add_argument("--steps-cap", type=int, default=None,
                    help="race a fixed number of control steps instead of "
                         "running each arm to its own completion")
    args = ap.parse_args()

    runs = pathlib.Path(args.runs)
    if args.preset == "thor-pi05-race":
        # two hosts in the form their own authors ship, then the same
        # model accelerated two ways. No torch.compile pane: the panes
        # are what you would actually deploy.
        specs = [
            ("openpi_eager", "OpenPI host, as shipped", "eager PyTorch",
             STOCK),
            ("lerobot_eager", "LeRobot host, as shipped", "eager PyTorch",
             STOCK),
            ("attach", "+ FlashRT structures",
             "explicit pipeline: attach + capture", ACCENT),
            ("native", "FlashRT native",
             "hand-written pipeline, NVFP4", NATIVE)]
    elif args.preset == "thor-pi05-ladder":
        # one checkpoint (openpi pi05_libero), four ways to execute it,
        # all four in the OpenPI host's own observation pipeline
        specs = [
            ("eager", "OpenPI host, eager", "the host's own code, uncompiled",
             STOCK),
            ("compiled", "the same host, captured",
             "compile + whole-graph capture", COMPILED),
            ("attach", "+ FlashRT structures", "auto_swaps + attach", ACCENT),
            ("native", "FlashRT native", "hand-written pipeline, FP8",
             NATIVE)]
    elif args.preset == "thor-pi05-hosts":
        specs = [
            ("eager", "OpenPI host, as shipped", "eager PyTorch", STOCK),
            ("attach", "OpenPI + structures", "auto_swaps + attach", ACCENT),
            ("lerobot_eager", "LeRobot host, as shipped", "eager PyTorch",
             STOCK),
            ("lerobot_attach", "LeRobot + structures", "auto_swaps + attach",
             ACCENT)]
    elif args.preset == "thor-groot-race":
        # two hosts in the form their own authors ship, then the same
        # checkpoint accelerated two ways
        specs = [
            ("isaac_eager", "Isaac-GR00T, as shipped", "eager PyTorch",
             STOCK),
            ("lerobot_eager", "LeRobot GR00T port, as shipped",
             "eager PyTorch", STOCK),
            ("attach", "+ FlashRT structures",
             "explicit pipeline: attach + capture", ACCENT),
            ("native", "FlashRT native",
             "hand-written pipeline, NVFP4", NATIVE)]
    elif args.preset == "thor-groot-hosts":
        specs = [
            ("eager", "Isaac-GR00T, as shipped", "eager PyTorch", STOCK),
            ("attach", "Isaac-GR00T + structures", "auto_swaps + attach",
             ACCENT),
            ("lerobot_eager", "LeRobot GR00T port, as shipped",
             "eager PyTorch", STOCK),
            ("lerobot_attach", "LeRobot + structures", "auto_swaps + attach",
             ACCENT)]
    elif args.preset == "groot-hosts":
        # each host is shown against the form its own authors ship, so the
        # panes read as "what this host does" -> "what it does attached"
        specs = [
            ("eager", "Isaac-GR00T, as shipped", "eager PyTorch", STOCK),
            ("attach", "Isaac-GR00T + structures", "auto_swaps + capture",
             ACCENT),
            ("lerobot_eager", "LeRobot GR00T port, as shipped",
             "eager PyTorch", STOCK),
            ("lerobot_attach", "LeRobot + structures", "auto_swaps + capture",
             ACCENT)]
    elif args.preset == "groot":
        specs = [
            ("eager", "Isaac-GR00T, as shipped", "eager PyTorch", STOCK),
            ("captured", "the same host, captured",
             "compile + whole-graph capture", COMPILED),
            ("attach", "+ FlashRT structures", "auto_swaps + capture",
             ACCENT)]
    elif args.preset == "groot4x":
        # the three host arms come from the official Isaac-GR00T host; the
        # native arm replaces the model outright and takes its observations
        # from the LeRobot port, which is the only host that can sit in the
        # same process as the native kernels
        specs = [
            ("eager", "Isaac-GR00T, as shipped", "eager PyTorch", STOCK),
            ("captured", "the same host, captured",
             "compile + whole-graph capture", COMPILED),
            ("attach", "+ FlashRT structures", "auto_swaps + capture",
             ACCENT),
            ("native", "FlashRT native", "hand-written pipeline, FP8",
             NATIVE)]
    elif args.preset == "groot4":
        # all four in one host process, so the native pane can be fed by the
        # same observation pipeline the other three run on
        specs = [
            ("lerobot_eager", "LeRobot GR00T host, as shipped", "eager PyTorch",
             STOCK),
            ("lerobot_captured", "the same host, captured",
             "compile + whole-graph capture", COMPILED),
            ("lerobot_attach", "+ FlashRT structures", "auto_swaps + capture",
             ACCENT),
            ("lerobot_native", "FlashRT native",
             "hand-written pipeline, FP8", NATIVE)]
    elif args.preset == "cross-host":
        # each host against the form its own authors ship, so the pair of
        # panes reads as "what this host does" -> "what it does attached"
        specs = [
            ("eager", "LeRobot host, as shipped", "eager PyTorch", STOCK),
            ("attach", "LeRobot + structures", "auto_swaps + capture", ACCENT),
            ("openpi_eager", "OpenPI host, as shipped", "eager PyTorch",
             STOCK),
            ("openpi_attach", "OpenPI + structures", "auto_swaps + capture",
             ACCENT)]
    else:
        specs = [
            ("eager", "LeRobot host, as shipped", "eager PyTorch", STOCK),
            ("compiled", "the same host, compiled", "torch.compile", COMPILED),
            ("attach", "+ FlashRT structures", "auto_swaps + capture", ACCENT),
            ("native", "FlashRT native", "hand-written pipeline, FP8",
             NATIVE)]
    arms = [Arm(runs / name, label, sub, color, steps_cap=args.steps_cap)
            for name, label, sub, color in specs
            if (runs / name / "events.json").exists()]
    render(arms, args.out, fps=args.fps, seconds=args.seconds,
           speed=args.speed, pane=args.pane, title=args.title,
           note=args.note, footer=args.footer)


if __name__ == "__main__":
    main()

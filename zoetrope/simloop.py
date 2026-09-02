"""One closed-loop LIBERO rollout, recorded.

Protocol (see RECORDING.md in the demo repository):

* one arm per process — this module is imported by exactly one arm;
* the robot waits for its own policy: every control step that re-plans
  records the wall latency of that decision and nothing else;
* host preprocessing (image transform, tokenizer, normalisation) and
  action decode sit **outside** the timed region, identically for every
  arm;
* the episode runs to its own completion.

Output per run directory:

    events.json   {"meta": {...}, "events": [{"step", "infer_ms", "action"}]}
    frames.webp   the rollout, one frame per control step
"""

from __future__ import annotations

import collections
import json
import math
import pathlib
import time

import numpy as np

from .frames import save_frames

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
CONTROL_HZ = 20.0

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def quat2axisangle(quat):
    """Copied from robosuite (the convention LIBERO training data used)."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def make_env(suite: str, task_id: int, seed: int, resolution: int):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    bddl = (pathlib.Path(get_libero_path("bddl_files"))
            / task.problem_folder / task.bddl_file)
    env = OffScreenRenderEnv(**{"bddl_file_name": str(bddl),
                                "camera_heights": resolution,
                                "camera_widths": resolution})
    env.seed(seed)  # object positions move with the seed even at a fixed state
    return env, task_suite, task


def resize_with_pad(image: np.ndarray, size: int) -> np.ndarray:
    """`tf.image.resize_with_pad`, the way the LIBERO policies were trained.

    Same arithmetic as openpi's `image_tools.resize_with_pad`, written out
    here so the loop has one image path on every host — the GR00T
    containers do not carry openpi.
    """
    from PIL import Image

    if image.shape[:2] == (size, size):
        return np.ascontiguousarray(image)
    pil = Image.fromarray(image)
    cur_w, cur_h = pil.size
    ratio = max(cur_w / size, cur_h / size)
    resized = pil.resize((int(cur_w / ratio), int(cur_h / ratio)),
                         resample=Image.BILINEAR)
    canvas = Image.new(resized.mode, (size, size), 0)
    canvas.paste(resized, (max(0, (size - resized.size[0]) // 2),
                           max(0, (size - resized.size[1]) // 2)))
    return np.asarray(canvas, dtype=np.uint8)


def observation_views(obs, resize_to: int):
    """The two camera views the LIBERO policies were trained on.

    Rotated 180 degrees to match the training preprocessing, then padded
    to the policy's square input. Returns uint8 HWC.
    """
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return resize_with_pad(img, resize_to), resize_with_pad(wrist, resize_to)


def observation_state(obs):
    return np.concatenate((obs["robot0_eef_pos"],
                           quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"]))


def settle(host, *, min_rounds: int = 60, min_seconds: float = 20.0,
           block: int = 10, tol: float = 0.01, max_seconds: float = 240.0):
    """Run the arm until it is in the board's hot regime, and prove it.

    A count of warm-up calls is the wrong unit: 20 calls is 16 s for the
    eager arm and 0.9 s for the native one, and this board needs wall
    time under load, not iterations, to leave the cold regime. So the
    settle loop has three floors — a call count, a wall-clock minimum,
    and a stability test on consecutive blocks — and it returns the
    per-block medians so the receipt can show the arm had stopped
    moving before a single frame was recorded.
    """
    medians: list[float] = []
    calls = 0
    t0 = time.perf_counter()
    while True:
        samples = []
        for _ in range(block):
            host.sync()
            call_t0 = time.perf_counter()
            host.infer()
            host.sync()
            samples.append((time.perf_counter() - call_t0) * 1000.0)
        medians.append(float(np.median(samples)))
        calls += block
        elapsed = time.perf_counter() - t0
        if elapsed > max_seconds:
            break
        if calls < min_rounds or elapsed < min_seconds:
            continue
        if len(medians) >= 3:
            recent = medians[-3:]
            if (max(recent) - min(recent)) / max(recent) <= tol:
                break
    return {"calls": calls, "seconds": round(time.perf_counter() - t0, 2),
            "block": block, "block_medians_ms": [round(m, 3) for m in medians],
            "settled": len(medians) >= 3
                       and (max(medians[-3:]) - min(medians[-3:]))
                       / max(medians[-3:]) <= tol,
            "drift_first_to_last_pct": round(
                100.0 * (medians[-1] - medians[0]) / medians[0], 2)
            if medians else None}


def input_sensitivity(host, env, obs, resize_to: int, lookahead: int = 25):
    """A(x), A(y), A(x) — the only test that catches a stale window.

    "The output looks reasonable" is not a test; "the output changes when
    the input changes" is. A captured graph that kept a pointer to the
    first frame passes every cosine check ever written and fails this one.

    The second observation has to be a genuinely different one, which the
    dummy action does not guarantee: on a scene that has already settled,
    holding still for 25 steps can leave the cameras byte-identical, and
    then "the output did not change" is the correct answer rather than a
    stale window. So this drives the arm with a real motion and reports
    how far the input actually moved; the verdict below is only read when
    the input moved at all.
    """
    img_a, wrist_a = observation_views(obs, resize_to)
    state_a = observation_state(obs)

    probe = np.array([0.0, 0.0, -0.35, 0.0, 0.0, 0.0, -1.0])
    forward = obs
    for _ in range(lookahead):
        forward, _, _, _ = env.step(probe.tolist())
    img_b, wrist_b = observation_views(forward, resize_to)
    state_b = observation_state(forward)
    input_delta = max(
        float(np.max(np.abs(img_a.astype(np.int32) - img_b.astype(np.int32)))),
        float(np.max(np.abs(wrist_a.astype(np.int32)
                            - wrist_b.astype(np.int32)))),
        float(np.max(np.abs(state_a - state_b))))

    def once(img, wrist, state):
        host.observe(img, wrist, state)
        host.sync()
        return np.asarray(host.decode(host.infer()), dtype=np.float64)

    first = once(img_a, wrist_a, state_a)
    other = once(img_b, wrist_b, state_b)
    again = once(img_a, wrist_a, state_a)
    return {
        "input_moved": input_delta > 0.0,
        "max_abs_input_A_vs_B": input_delta,
        "changes_with_input": bool(np.max(np.abs(first - other)) > 0.0),
        "max_abs_A_vs_B": float(np.max(np.abs(first - other))),
        "repeatable": bool(np.array_equal(first, again)),
        "max_abs_A_vs_A": float(np.max(np.abs(first - again))),
        "frames_apart": lookahead,
        "probe_action": probe.tolist(),
    }


def rollout(host, *, suite: str, task_id: int, trial: int, seed: int,
            replan: int, out_dir: pathlib.Path, resize_to: int = 224,
            num_steps_wait: int = 10, max_steps: int | None = None,
            extra_meta: dict | None = None, progress_every: int = 25):
    """Drive `host` around a closed LIBERO loop and record the film data.

    `host` must expose:
        set_task(text)                 once, before the loop
        observe(img, wrist, state)     host preprocessing (untimed)
        infer() -> (H, action_dim)     the only thing being compared
        decode(chunk) -> (H, 7)        action decode (untimed)
    """
    env, task_suite, task = make_env(suite, task_id, seed, LIBERO_ENV_RESOLUTION)
    cap = max_steps or MAX_STEPS[suite]
    initial_states = task_suite.get_task_init_states(task_id)

    host.set_task(str(task.language))

    env.reset()
    obs = env.set_init_state(initial_states[trial])

    # the simulator drops the objects; wait for them to settle
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    # compile / autotune / settle on the episode's own first observation,
    # before anything is recorded
    warm_img, warm_wrist = observation_views(obs, resize_to)
    warm_t0 = time.perf_counter()
    host.warmup(warm_img, warm_wrist, observation_state(obs),
                rounds=int(getattr(host, "warmup_rounds", 20)))
    settle_report = settle(host,
                           min_rounds=int(getattr(host, "settle_rounds", 60)),
                           min_seconds=float(getattr(host, "settle_seconds",
                                                     20.0)))
    warmup_s = time.perf_counter() - warm_t0
    print(f"[rollout] settled: {settle_report['calls']} calls / "
          f"{settle_report['seconds']}s, drift "
          f"{settle_report['drift_first_to_last_pct']}%, "
          f"steady={settle_report['settled']}", flush=True)

    # §5.1: a path that has only ever been used once. Prove this arm
    # still looks at the camera after the graph is captured, before a
    # single frame of film is recorded.
    sensitivity = input_sensitivity(host, env, obs, resize_to)
    print(f"[rollout] input sensitivity: {sensitivity}", flush=True)
    if not sensitivity["input_moved"]:
        raise SystemExit("the probe did not move the observation, so the "
                         "stale-window test proves nothing — refusing to "
                         "record this arm on an untested path")
    if not sensitivity["changes_with_input"]:
        raise SystemExit("this arm's output does not change when the "
                         "observation changes — refusing to record it")

    # back to the episode's initial state for the recorded run
    env.reset()
    obs = env.set_init_state(initial_states[trial])
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    frames: list[np.ndarray] = []
    events: list[dict] = []
    plan: collections.deque = collections.deque()
    success = False
    t = 0
    t0_wall = time.perf_counter()

    while t < cap:
        img, wrist = observation_views(obs, resize_to)
        state = observation_state(obs)
        frames.append(img)

        infer_ms = None
        if not plan:
            host.observe(img, wrist, state)          # untimed, every arm alike
            host.sync()
            call_t0 = time.perf_counter()
            chunk = host.infer()                     # the timed region
            host.sync()
            infer_ms = (time.perf_counter() - call_t0) * 1000.0
            actions = host.decode(chunk)             # untimed, every arm alike
            plan.extend(actions[:replan])

        action = plan.popleft()
        events.append({"step": len(events),
                       "infer_ms": infer_ms,
                       "action": [float(a) for a in action]})
        obs, _, done, _ = env.step(np.asarray(action, dtype=np.float64).tolist())
        if done:
            success = True
            break
        t += 1
        if progress_every and len(events) % progress_every == 0:
            lat = [e["infer_ms"] for e in events if e["infer_ms"] is not None]
            print(f"[rollout] step {len(events)} "
                  f"median {np.median(lat):.2f} ms", flush=True)

    env.close()

    # after the episode, never before: an arm's closing report carries
    # things that only exist once it has run — the swap ledger's run-time
    # fallback counts, the frontend's latency records, how many times the
    # observation window had to be rebuilt. Collected as a call argument
    # it would be collected at build time and every one of them would
    # read empty.
    host_report = host.finish() if hasattr(host, "finish") else None

    latencies = [e["infer_ms"] for e in events if e["infer_ms"] is not None]
    median_ms = float(np.median(latencies)) if latencies else float("nan")
    meta = {
        "suite": suite,
        "task_id": task_id,
        "trial": trial,
        "seed": seed,
        "task": str(task.language),
        "replan": replan,
        "control_hz": CONTROL_HZ,
        "num_steps_wait": num_steps_wait,
        "resize_to": resize_to,
        "control_steps": len(events),
        "policy_calls": len(latencies),
        "median_infer_ms": median_ms,
        "mean_infer_ms": float(np.mean(latencies)) if latencies else None,
        "p10_infer_ms": float(np.percentile(latencies, 10)) if latencies else None,
        "p90_infer_ms": float(np.percentile(latencies, 90)) if latencies else None,
        "policy_hz": (1000.0 / median_ms) if latencies else None,
        "success": success,
        "wall_s": time.perf_counter() - t0_wall,
        "warmup_s": warmup_s,
        "settle": settle_report,
        "input_sensitivity": sensitivity,
        "timed_region": "model call only; preprocessing and action decode "
                        "outside, identically for every arm",
        "host_report": host_report,
    }
    meta.update(extra_meta or {})

    out_dir.mkdir(parents=True, exist_ok=True)
    save_frames(out_dir, np.asarray(frames, dtype=np.uint8))
    (out_dir / "events.json").write_text(
        json.dumps({"meta": meta, "events": events}, indent=2))
    print(json.dumps(meta, indent=2, default=str), flush=True)
    return meta

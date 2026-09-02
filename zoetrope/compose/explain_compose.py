"""Explainers: one published mechanism, drawn.

The films elsewhere in this kit argue from a recording. These panels argue
from a paper. Each one takes a single idea a framework's own authors put at
the centre of their design, and draws it — the mechanism, not a benchmark,
and never one framework against another.

The rules that keep it honest are the ones the rest of the kit already
follows, pointed at a different source:

* **Every number on a panel is the paper's, and says so.** The spec carries
  a `source` line and it is printed on every page.
* **Nothing here is measured.** A panel that would need a measurement to be
  true is a panel that should have been a recording.
* **An illustration says it is one.** Where a picture needs a concrete
  scenario — three requests, a tree of prompts — it is labelled as an
  example, so it is never read as data.

The panels live in `zoetrope/explainers/*.json`, so pointing this at another
framework is a spec, not a patch. Six painters, and each is a template for
a shape that recurs:

    paged        a resource reserved for a worst case, versus handed out
                 in fixed pieces
    blocktable   an indirection table: contiguous to the reader, scattered
                 underneath
    share        one copy read by many, and what happens when one writes
    radix        work kept in a tree, so a shared prefix is done once
    schedule     the same queue in two orders, and what the cache thinks
    fsm          steps that had to be taken, versus steps already decided

    zoetrope explain vllm_paged --menu menu.png
    zoetrope explain sglang_radix --panel 0 --palette blueprint --frame a.png
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import tempfile

from PIL import Image, ImageDraw

from .canvas import (R_LG, R_MD, R_SM, Canvas, appear, arrow, ease, ease_out,
                     font, link, _truetype)
from .palettes import PALETTES, palette
from .race_compose import _ffmpeg, wrap

W, H = 1280, 720
PAD = 96
BODY_TOP = 164

def reveal(n, t):
    return int(round(n * min(max(t, 0.0), 1.0)))


def stage(t, a, b):
    """Progress within one beat of a panel, so beats can follow each other.

    A counter that states its conclusion while the row it counts is still
    being drawn is saying something that is not yet true on the page.
    Gate on `stage(...) >= 1`.
    """
    return ease(min(max((t - a) / (b - a), 0.0), 1.0))


def _text(d, xy, s, f, fill, center=None):
    x, y = xy
    if center is not None:
        x = center - d.textlength(s, font=f) / 2
    d.text((x, y), s, font=f, fill=fill)


def sweep(im, d, xy, s, f, pal, ink, q):
    """A highlighter pulled across a number as it lands, and then off again.

    `q` runs 0 to 1 over the number's own beat: the plate enters from the
    left, covers the line, and leaves the same way, so the page settles on
    the number itself rather than on a permanent block of colour.

    The line is drawn once in ink; the highlighted version is drawn into a
    tile and pasted back in a window, so the plate and the text under it can
    never disagree about where the glyphs are.
    """
    x, y = xy
    d.text((x, y), s, font=f, fill=ink)
    if q <= 0 or q >= 1:
        return
    lo, hi = (0.0, q * 2) if q < 0.5 else (q * 2 - 1, 1.0)
    pad, w, h = 10, d.textlength(s, font=f), f.size * 1.42
    span = w + 2 * pad
    d.plate_text((x - pad + span * lo, y, x - pad + span * hi, y + h),
                 (x, y), s, f, ink, pal.bg)


def _chip(d, box, pal, ink, *, filled=True, r=6):
    if filled:
        d.rounded_rectangle(box, r, fill=ink)
    else:
        d.rounded_rectangle(box, r, fill=pal.card, outline=pal.line)


# ------------------------------------------------------- derived from spec
# Every count a panel prints is computed from the spec here, so a changed
# scenario cannot leave a stale number typed into a caption.

def paged_counts(p):
    """(tokens held, slots reserved ahead, slots a block allocator hands out)."""
    used = sum(r["used"] for r in p["requests"])
    reserved = len(p["requests"]) * p["reserve"]
    blocks = sum(math.ceil(r["used"] / p["block"]) for r in p["requests"])
    return used, reserved, blocks * p["block"]


def fsm_steps(p):
    """(tokens, decode steps once the forced spans are jumped)."""
    return len(p["text"]), sum(1 for f in p["forced"] if not f)


def radix_tokens(tree):
    """(tokens computed if every request pays for its own prefix, tokens
    computed if the tree is kept) — read off the illustration itself."""
    total_nodes, per_request = 0, []

    def walk(node, carried):
        nonlocal total_nodes
        total_nodes += node["tokens"]
        run = carried + node["tokens"]
        kids = node.get("kids", [])
        if not kids:
            per_request.append(run)
        for k in kids:
            walk(k, run)

    walk(tree, 0)
    return sum(per_request), total_nodes


def prefix_computes(order):
    """How often a prefix has to be computed if the queue runs in this order."""
    seen, n = None, 0
    for g in order:
        n += 0 if g == seen else 1
        seen = g
    return n


DERIVE = {
    "radix_tokens": lambda p: radix_tokens(p["tree"]),
    "prefix_computes": lambda p: (prefix_computes(p["groups"]),
                                  prefix_computes(sorted(p["groups"]))),
    "fsm_steps": lambda p: fsm_steps(p),
    "paged_held": lambda p: tuple(
        round(100 * paged_counts(p)[0] / v, 1) for v in paged_counts(p)[1:]),
}


def chart_bars(p):
    """The bars a chart draws: given outright, or derived from the panel it
    sits beside, so a diagram and the number under it cannot drift apart."""
    c = p["chart"]
    if "bars" in c:
        return c["bars"]
    values = DERIVE[c["derive"]](p)
    kinds = c.get("kinds", ["base", "ours"])
    return [{"label": lab, "value": v, "kind": k}
            for lab, v, k in zip(c["labels"], values, kinds)]


def chart(im, d, pal, p, t, ink, box):
    """A bar chart: the payoff, beside the mechanism rather than a page away.

    Tall boxes get vertical bars (a side column), wide boxes horizontal ones
    (a results page). A bar may carry a range, which is drawn to its upper
    end with the lower end marked, because a published range is not a point.
    """
    c, bars = p["chart"], chart_bars(p)
    x0, y0, x1, y1 = box
    unit = c.get("unit", "")
    hi = max(b.get("high", b["value"]) for b in bars)
    top = c.get("cap", hi * 1.18)
    fmt = (lambda v: f"{v:,.0f}") if hi >= 100 else (lambda v: f"{v:g}")

    def ink_for(b):
        k = b.get("kind", "ours")
        return (ink if k == "ours" else
                pal.dim(ink, 0.68) if k == "mid" else pal.dim(ink, 0.42))

    if (y1 - y0) > (x1 - x0):                       # a column beside a diagram
        ay = y0
        for line in wrap(d, c["axis"], font(15), x1 - x0)[:3]:
            _text(d, (x0, ay), line, font(15), pal.muted)
            ay += 19
        base, headroom = y1 - 46, ay + 44
        w = min(84, (x1 - x0 - 30 * (len(bars) - 1)) / len(bars))
        for i, b in enumerate(bars):
            x = x0 + i * (w + 30)
            v = b["value"] * ease_out(stage(t, i * 0.12, 0.75 + i * 0.12))
            h = (base - headroom) * v / top
            d.rounded_rectangle((x, base - h, x + w, base), R_SM,
                                fill=ink_for(b))
            if t >= 1:
                _text(d, (0, base - h - 30), b.get("text") or
                      f"{fmt(b['value'])}{unit}", font(21, True), ink_for(b),
                      center=x + w / 2)
                yy = base + 10
                for line in wrap(d, b["label"], font(14), w + 26)[:3]:
                    _text(d, (0, yy), line, font(14), pal.muted,
                          center=x + w / 2)
                    yy += 17
        if t >= 1 and len(bars) == 2 and c.get("ratio", True):
            a, bb = bars[0]["value"], bars[1]["value"]
            r = max(a, bb) / max(min(a, bb), 1e-9)
            _text(d, (0, ay + 4), f"{r:.1f}x {c.get('ratio_word', 'fewer')}",
                  font(22, True), ink, center=(x0 + x1) / 2)
        return

    rows = min((y1 - y0 - 26) / len(bars), 78)      # a results page
    y1 = y0 + rows * len(bars) + 26
    _text(d, (x0, y1 - 16), c["axis"], font(15), pal.muted)
    lf = font(17)
    lw = min(max(d.textlength(b["label"], font=lf) for b in bars) + 24,
             (x1 - x0) * 0.42)
    vf = font(int(min(28, rows * 0.62)), True)
    bx0, bx1 = x0 + lw, x1 - 170
    if c.get("baseline"):
        rx = bx0 + (bx1 - bx0) * c["baseline"] / top
        d.line((rx, y0, rx, y0 + len(bars) * rows), fill=pal.line)
    for i, b in enumerate(bars):
        y = y0 + i * rows
        lines = wrap(d, b["label"], lf, lw - 24)[:2]
        ly = y + rows / 2 - 10 * len(lines)
        for line in lines:
            _text(d, (x0, ly), line, lf, pal.ink)
            ly += 21
        w = (bx1 - bx0) * b["value"] * ease_out(
            stage(t, i * 0.1, 0.7 + i * 0.1)) / top
        bh = min(rows - 18, 52)
        d.rounded_rectangle((bx0, y + (rows - bh) / 2, bx0 + max(w, 2),
                             y + (rows + bh) / 2), 5, fill=ink_for(b))
        if b.get("low") is not None and t >= 1:
            lx = bx0 + (bx1 - bx0) * b["low"] / top
            d.line((lx, y + (rows - bh) / 2 - 4, lx, y + (rows + bh) / 2 + 4),
                   fill=pal.bg, width=3)
        if t >= 1:
            _text(d, (bx1 + 20, y + rows / 2 - vf.size * 0.7),
                  b.get("text") or f"{fmt(b['value'])}{unit}", vf, ink_for(b))


# ---------------------------------------------------------------- painters

def paged(im, d, pal, p, t, ink):
    right = p.get("_right", W - PAD)
    gap, hh = 2, 34
    reserve, block = p["reserve"], p["block"]
    slot = (right - PAD) / (len(p["requests"]) * reserve) - gap
    reqs = p["requests"]
    used_total, reserved_total, paged_total = paged_counts(p)

    # as reserved: one contiguous run per request, most of it never taken
    x0, y = PAD, BODY_TOP + 40
    _text(d, (PAD, y - 30), p["top"], font(18, True), pal.ink)
    ta, tb = stage(t, 0.0, 0.5), stage(t, 0.5, 1.0)
    shown = reveal(len(reqs) * reserve, ta)
    for j, r in enumerate(reqs):
        base = j * reserve
        _text(d, (x0 + base * (slot + gap), y - 6), r["label"], font(13),
              pal.muted)
    for i in range(len(reqs) * reserve):
        if i >= shown:
            break
        j, off = divmod(i, reserve)
        x = x0 + i * (slot + gap)
        _chip(d, (x, y + 14, x + slot, y + 14 + hh), pal, ink,
              filled=off < reqs[j]["used"], r=3)
    if ta >= 1:
        held = f"{used_total} of {reserved_total} slots hold a token"
        _text(d, (PAD, y + 58), held, font(16), pal.muted)

    # paged: blocks handed out only when a token needs one
    y = BODY_TOP + 216
    _text(d, (PAD, y - 30), p["bottom"], font(18, True), pal.ink)
    x = PAD
    nblocks = sum(math.ceil(r["used"] / block) for r in reqs)
    seen = 0
    for j, r in enumerate(reqs):
        for b in range(math.ceil(r["used"] / block)):
            if seen >= reveal(nblocks, tb):
                break
            for k in range(block):
                filled = b * block + k < r["used"]
                cx = x + k * (slot + gap)
                _chip(d, (cx, y + 14, cx + slot, y + 14 + hh), pal, ink,
                      filled=filled, r=3)
            x += block * (slot + gap) + 12
            seen += 1
    if tb >= 1:
        end = right
        d.line((x + 6, y + 31, end, y + 31), fill=pal.line)
        _text(d, (x + 18, y + 40), "free for other requests", font(15),
              pal.muted)
    if tb >= 1:
        _text(d, (PAD, y + 58),
              f"{used_total} of {paged_total} slots hold a token", font(16),
              pal.muted)
        _text(d, (PAD, BODY_TOP + 350), p["tail"], font(20, True), ink)
        _text(d, (PAD, BODY_TOP + 384), p["scale"], font(14), pal.muted)


def blocktable(im, d, pal, p, t, ink):
    lw, lh, lgap = 250, 54, 16
    _text(d, (PAD, BODY_TOP - 4), "one sequence, in order", font(18, True),
          pal.ink)
    cols, rows = p["grid"]
    cw, ch, cg = 56, 46, 8
    gx = W - PAD - cols * (cw + cg) + cg
    n_log = len(p["logical"])
    gy = BODY_TOP + 46 + max(0, (n_log * (lh + lgap) - rows * (ch + cg)) / 2)
    ly = BODY_TOP + 46 + max(0, (rows * (ch + cg) - n_log * (lh + lgap)) / 2)
    _text(d, (gx, BODY_TOP - 4), "GPU memory, block by block",
          font(18, True), pal.ink)
    for i in range(cols * rows):
        r, c = divmod(i, cols)
        x, y = gx + c * (cw + cg), gy + r * (ch + cg)
        lit = i in p["physical"][:reveal(len(p["physical"]), t)]
        _chip(d, (x, y, x + cw, y + ch), pal, ink, filled=lit, r=5)

    for i, lab in enumerate(p["logical"]):
        if i >= reveal(len(p["logical"]), t):
            break
        y = ly + i * (lh + lgap)
        d.rounded_rectangle((PAD, y, PAD + lw, y + lh), 6, fill=pal.card,
                            outline=pal.line)
        _text(d, (PAD + 16, y + 17), lab, font(17), pal.ink)
        j = p["physical"][i]
        r, c = divmod(j, cols)
        link(d, PAD + lw + 10, y + lh / 2,
             gx + c * (cw + cg) - 4, gy + r * (ch + cg) + ch / 2, pal.muted,
             width=2, head=13, half=5)
    _text(d, (gx, gy + rows * (ch + cg) + 14), p["free_note"], font(15),
          pal.muted)


def share(im, d, pal, p, t, ink):
    bw, bh = 190, 62
    n = len(p["prompt"])
    x0 = (W - (n * (bw + 14) - 14)) / 2
    top = BODY_TOP + 30
    _text(d, (0, BODY_TOP - 6), p["prompt_label"], font(19, True), pal.ink,
          center=W / 2)
    for i in range(n):
        if i >= reveal(n, stage(t, 0.0, 0.35)):
            break
        x = x0 + i * (bw + 14)
        d.rounded_rectangle((x, top, x + bw, top + bh), 8, fill=ink)
        _text(d, (0, top + 20), f"block {i + 1}", font(18, True), pal.bg,
              center=x + bw / 2)

    br = p["branches"]
    cw = 150
    bx0 = (W - (len(br) * (cw + 20) - 20)) / 2
    by = BODY_TOP + 210
    for i, lab in enumerate(br):
        if i >= reveal(len(br), stage(t, 0.3, 0.75)):
            break
        x = bx0 + i * (cw + 20)
        d.rounded_rectangle((x, by, x + cw, by + 54), 8, fill=pal.card,
                            outline=pal.line)
        _text(d, (0, by + 16), lab, font(17), pal.ink, center=x + cw / 2)
        # every branch points at the same blocks: that is the whole idea.
        # They arrive fanned across one block rather than stacked on a
        # point, which would pile four heads into a blot.
        link(d, x + cw / 2, by - 8,
             W / 2 + (i - (len(br) - 1) / 2) * 26, top + bh + 11, pal.muted,
             axis="y", width=2, head=10, half=4)
        if i == len(br) - 1 and t > 0.85:
            link(d, x + cw / 2, by + 58, x + 18 + cw / 2, by + 72, ink,
                 axis="y", width=2, head=0)
            d.rounded_rectangle((x + 18, by + 74, x + cw + 18, by + 128),
                                R_MD, fill=pal.card, outline=ink)
            _text(d, (0, by + 92), "its own block", font(16), ink,
                  center=x + 18 + cw / 2)
    if t > 0.85:
        _text(d, (PAD, by + 154), p["cow"], font(19, True), ink)
    _text(d, (PAD, by + 186), p["tail"], font(16), pal.muted)


def _tree_rows(node, depth=0, out=None):
    out = [] if out is None else out
    kids = node.get("kids", [])
    if not kids:
        node["_row"] = len([n for n in out if not n.get("kids")])
    out.append(node)
    for k in kids:
        _tree_rows(k, depth + 1, out)
    return out


def radix(im, d, pal, p, t, ink):
    root = json.loads(json.dumps(p["tree"]))          # do not mutate the spec
    levels, leaves = [], []

    def walk(node, depth):
        while len(levels) <= depth:
            levels.append([])
        levels[depth].append(node)
        kids = node.get("kids", [])
        if kids:
            for k in kids:
                walk(k, depth + 1)
            node["y"] = sum(k["y"] for k in kids) / len(kids)
        else:
            node["y"] = BODY_TOP + 26 + len(leaves) * 86
            leaves.append(node)

    walk(root, 0)
    avail = p.get("_right", W - PAD) - PAD
    gapx = 88 if avail > 900 else 52
    nw, nh = min(208, (avail - (len(levels) - 1) * gapx) / len(levels)), 54
    order = [n for lvl in levels for n in lvl]
    live = {}
    for depth, lvl in enumerate(levels):
        for node in lvl:
            a = appear(order.index(node), len(order), t)
            live[id(node)] = a
            if a <= 0:
                continue
            x = PAD + depth * (nw + gapx)
            y = node["y"] + (1 - a) * 12
            new = node.get("new")
            d.rounded_rectangle((x, y, x + nw, y + nh), R_MD,
                                fill=pal.dim(pal.card if new
                                             else pal.dim(ink, 0.55), a),
                                outline=pal.dim(ink, a) if new else None,
                                width=3)
            _text(d, (x + 14, y + 8), node["label"], font(17, True),
                  pal.dim(ink if new else pal.ink, a))
            _text(d, (x + 14, y + 30), f"{node['tokens']} tokens", font(14),
                  pal.dim(pal.muted if new else pal.ink, a))
            for k in node.get("kids", []):
                ka = live.get(id(k), appear(order.index(k), len(order), t))
                if ka > 0:
                    link(d, x + nw + 4, y + nh / 2,
                         x + nw + gapx - 4, k["y"] + nh / 2,
                         pal.dim(pal.muted, ka * 0.55), width=2)

    y = BODY_TOP + 340
    _chip(d, (PAD, y, PAD + 26, y + 18), pal, pal.dim(ink, 0.55))
    _text(d, (PAD + 38, y - 1), p["tail"], font(17), pal.ink)
    x = PAD + 60 + d.textlength(p["tail"], font=font(17))
    d.rounded_rectangle((x, y, x + 26, y + 18), 5, fill=pal.card,
                        outline=ink, width=3)
    _text(d, (x + 38, y - 1), p["new_tail"], font(17), ink)
    _text(d, (PAD, y + 34), p["scale"], font(14), pal.muted)


def schedule(im, d, pal, p, t, ink):
    groups = p["groups"]
    keys = sorted(set(groups))
    inks = {k: pal.role(r) for k, r in
            zip(keys, ("ours", "compiled", "native", "stock"))}
    cg, chh = 14, 58
    cw = (p.get("_right", W - PAD) - PAD - (len(groups) - 1) * cg) / len(groups)

    def row(order, y, head, tt):
        _text(d, (PAD, y - 52), head, font(18, True), pal.ink)
        seen = None
        for i, g in enumerate(order):
            hit = g == seen
            seen = g
            a = appear(i, len(order), tt)
            if a <= 0:
                continue
            x = PAD + i * (cw + cg)
            yy = y + (1 - a) * 10
            d.rounded_rectangle((x, yy, x + cw, yy + chh), R_MD,
                                fill=pal.dim(inks[g], a))
            _text(d, (0, yy + 16), f"prefix {g}", font(17, True),
                  pal.dim(pal.bg, a * 0.15 + 0.85) if a >= 1 else
                  pal.dim(pal.bg, a), center=x + cw / 2)
            r = 7
            cx, cy = x + cw / 2, y - 14
            if a < 1:
                continue
            if hit:
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=inks[g])
            else:
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=pal.card,
                          outline=pal.muted, width=2)

    a = prefix_computes(groups)
    b = prefix_computes(sorted(groups))
    ta, tb = stage(t, 0.0, 0.5), stage(t, 0.5, 1.0)
    row(groups, BODY_TOP + 50, p["top"], ta)
    row(sorted(groups), BODY_TOP + 210, p["bottom"], tb)
    if ta >= 1:
        _text(d, (PAD, BODY_TOP + 122),
              f"the prefix is computed {a} times", font(17), pal.muted)
    if tb >= 1:
        _text(d, (PAD, BODY_TOP + 282),
              f"the prefix is computed {b} times, and read "
              f"{len(groups) - b} times from the tree", font(17), ink)
        _text(d, (PAD, BODY_TOP + 326), p["tail"], font(19, True), ink)


def fsm(im, d, pal, p, t, ink):
    text, forced = p["text"], p["forced"]
    n, decode_steps = fsm_steps(p)
    tw = (p.get("_right", W - PAD) - PAD) / n
    ta, tb = stage(t, 0.0, 0.5), stage(t, 0.5, 1.0)
    y = BODY_TOP + 30
    _text(d, (PAD, y - 30), p["top"], font(18, True), pal.ink)
    for i, s in enumerate(text):
        if i >= reveal(n, ta):
            break
        x = PAD + i * tw
        d.rounded_rectangle((x + 3, y, x + tw - 3, y + 62), 7, fill=pal.card,
                            outline=pal.line)
        _text(d, (0, y + 10), s, font(20, True), pal.ink, center=x + tw / 2)
        _text(d, (0, y + 38), f"step {i + 1}", font(13), pal.muted,
              center=x + tw / 2)

    y = BODY_TOP + 216
    _text(d, (PAD, y - 30), p["bottom"], font(18, True), pal.ink)
    i, step = 0, 0
    while i < reveal(n, tb):
        j = i
        while j < n and forced[j] == forced[i]:
            j += 1
        x0, x1 = PAD + i * tw + 3, PAD + j * tw - 3
        if forced[i]:
            d.rounded_rectangle((x0, y, x1, y + 62), 7, fill=pal.dim(ink, .4))
            # a short span cannot hold the long label; it keeps the short one
            lab, lf = "forced by the schema", font(16)
            if d.textlength(lab, font=lf) > x1 - x0 - 16:
                lab, lf = "forced", font(15)
            _text(d, (0, y + 12), lab, lf, pal.ink, center=(x0 + x1) / 2)
            _text(d, (0, y + 36), "jumped", font(14), pal.muted,
                  center=(x0 + x1) / 2)
        else:
            step += 1
            d.rounded_rectangle((x0, y, x1, y + 62), 7, fill=ink)
            _text(d, (0, y + 12), "".join(text[i:j]), font(19, True), pal.bg,
                  center=(x0 + x1) / 2)
            _text(d, (0, y + 38), f"step {step}", font(13), pal.bg,
                  center=(x0 + x1) / 2)
        i = j
    if tb >= 1:
        _text(d, (PAD, y + 100), f"{n} decode steps becomes {decode_steps}",
              font(20, True), ink)
        _text(d, (PAD, y + 134), p["tail"], font(16), pal.muted)


def tiling(im, d, pal, p, t, ink):
    """The loops the paper draws: K and V outside, Q inside, SRAM in the
    middle, and the N x N matrix that is formed a tile at a time and thrown
    away before the next one."""
    n, dm, cell = p.get("blocks", 8), p.get("dblocks", 3), 32
    sy = 292
    qx = PAD
    sx = qx + dm * cell + 26
    vx = sx + n * cell + 24
    ox = vx + dm * cell + 26
    bx, bx1 = ox + dm * cell + 44, W - PAD - 84
    ky = sy - dm * cell - 26

    step = min(int(t * n * n), n * n - 1)
    j, i = divmod(step, n)                     # outer over K/V, inner over Q

    def block(x, y, w, h, fill, outline=None):
        d.rectangle((x, y, x + w - 2, y + h - 2), fill=fill, outline=outline)

    _text(d, (qx, sy - 26), p["q"], font(17, True), pal.ink)
    for r in range(n):
        for c in range(dm):
            hot = r == i
            block(qx + c * cell, sy + r * cell, cell, cell,
                  ink if hot else pal.dim(ink, .3))
    _text(d, (qx, sy + n * cell + 10), p["inner"], font(14), pal.muted)

    _text(d, (sx, ky - 26), p["k"], font(17, True), pal.ink)
    for r in range(dm):
        for c in range(n):
            hot = c == j
            block(sx + c * cell, ky + r * cell, cell, cell,
                  ink if hot else pal.dim(ink, .3))
    oy = ky + 20
    for line in p["outer"].split("\n"):
        _text(d, (vx, oy), line, font(14), pal.muted)
        oy += 18

    _text(d, (vx, sy - 26), p["v"], font(17, True), pal.ink)
    for r in range(n):
        for c in range(dm):
            hot = r == j
            block(vx + c * cell, sy + r * cell, cell, cell,
                  ink if hot else pal.dim(ink, .3))

    for r in range(n):                          # S, formed a tile at a time
        for c in range(n):
            done = c < j or (c == j and r < i)
            if c == j and r == i:
                block(sx + c * cell, sy + r * cell, cell, cell, ink)
            else:
                block(sx + c * cell, sy + r * cell, cell, cell,
                      pal.card if done else pal.bg, pal.line)
    _text(d, (sx, sy + n * cell + 10), p["s"], font(15), pal.muted)

    _text(d, (ox, sy - 26), p["o"], font(17, True), pal.ink)
    for r in range(n):
        for c in range(dm):
            if r == i:
                fill = ink                       # the block being added to
            elif r < i or j:
                fill = pal.dim(ink, .5)          # carried from earlier passes
            else:
                fill = pal.card
            block(ox + c * cell, sy + r * cell, cell, cell, fill, pal.line)

    boxh = 92 + len(p["carried"]) * 27 + 62
    d.rounded_rectangle((bx, ky, bx1, ky + boxh), 10, fill=pal.card,
                        outline=ink, width=2)
    _text(d, (bx + 20, ky + 16), p["sram"], font(18, True), ink)
    yy = ky + 52
    for line in p["carried"]:
        _text(d, (bx + 20, yy), line, font(16), pal.ink)
        yy += 27
    d.line((bx + 20, yy + 6, bx1 - 20, yy + 6), fill=pal.line)
    for line in wrap(d, p["discard"], font(15), bx1 - bx - 40)[:3]:
        _text(d, (bx + 20, yy + 18), line, font(15), ink)
        yy += 20


def memory(im, d, pal, p, t, ink):
    """The ladder the whole argument stands on: what is near, and what is far."""
    levels = p["levels"]
    h, gap = 74, 26
    n = len(levels)
    y = BODY_TOP + 20 + max(0, (400 - (n * h + (n - 1) * gap + 46)) / 2)
    widest = W - 2 * PAD - 300
    for i, lv in enumerate(levels):
        a = appear(i, len(levels), t)
        if a <= 0:
            break
        w = widest * lv["w"] * (0.94 + 0.06 * a)
        near = lv.get("near")
        d.rounded_rectangle((PAD, y, PAD + w, y + h), R_MD,
                            fill=pal.dim(ink if near else pal.card, a),
                            outline=None if near else pal.dim(pal.line, a))
        # a level too narrow to hold its own name is labelled beside itself
        nf, sf = font(21, True), font(16)
        wide = max(d.textlength(lv["name"], font=nf),
                   d.textlength(lv["size"], font=sf)) + 36 <= w
        tx = PAD + 18 if wide else PAD + w + 20
        _text(d, (tx, y + 14), lv["name"], nf,
              (pal.bg if near else pal.ink) if wide else pal.ink)
        _text(d, (tx, y + 42), lv["size"], sf,
              (pal.bg if near else pal.muted) if wide else pal.muted)
        if lv.get("bw"):
            f = font(28, True)
            _text(d, (PAD + widest + 40, y + 22), lv["bw"], f, ink)
        y += h + gap
    if appear(len(levels) - 1, len(levels), t) >= 1:
        _text(d, (PAD, y + 12), p["tail"], font(20, True), ink)


def pieces(im, d, pal, p, t, ink):
    """The same work, cut into a different number of pieces."""
    rows, per = p["rows"], p.get("per_block", 100)
    x0, x1 = PAD, W - PAD - 250
    y = BODY_TOP + 30
    for i, r in enumerate(rows):
        tt = stage(t, i * 0.5, i * 0.5 + 0.5) if len(rows) == 2 else t
        n = max(1, round(r["pieces"] / per))
        _text(d, (x0, y - 26), r["label"], font(18, True), pal.ink)
        w = (x1 - x0) / n
        for k in range(reveal(n, tt)):
            x = x0 + k * w
            d.rectangle((x + 1, y, x + w - 1, y + 72),
                        fill=ink if i else pal.dim(ink, .5))
        if tt >= 1:
            f = font(30, True)
            sweep(im, d, (x1 + 40, y + 8), f"{r['pieces']:,}", f, pal, ink,
                  1.0 if i else 0.0)
            _text(d, (x1 + 40, y + 48), r["unit"], font(15), pal.muted)
        y += 150
    if t >= 1:
        _text(d, (PAD, y + 4), p["tail"], font(20, True), ink)
        _text(d, (PAD, y + 36), p["scale"], font(14), pal.muted)


def result(im, d, pal, p, t, ink):
    """The page the mechanism was for: what it came to, and whose figure it is."""
    ta, tb = stage(t, 0.0, 0.45), stage(t, 0.45, 1.0)
    y = BODY_TOP + 6
    for m in p.get("meters", []):
        cap = m.get("cap", 100)
        x0, x1 = PAD, W - PAD - 300
        _text(d, (x0, y), m["label"], font(18, True), pal.ink)
        ty = y + 30
        d.rounded_rectangle((x0, ty, x1, ty + 26), 6, fill=pal.card,
                            outline=pal.line)
        b = x0 + (x1 - x0) * m["before"] / cap
        a = x0 + (x1 - x0) * (m["before"] + (m["after"] - m["before"]) * ta) / cap
        d.rounded_rectangle((x0, ty, a, ty + 26), 6, fill=ink)
        d.line((b, ty - 8, b, ty + 34), fill=pal.muted, width=2)
        _text(d, (0, ty + 34), m.get("before_label",
                                     f"{m['before']}{m['unit']}"),
              font(15), pal.muted, center=b)
        if ta >= 1:
            _text(d, (x1 + 24, ty - 2),
                  m.get("after_label", f"{m['after']}{m['unit']}"),
                  font(26, True), ink)
            ny_ = ty + 34
            room = 3 if len(p["meters"]) == 1 else 2
            for line in wrap(d, m.get("note", ""), font(14),
                             W - PAD - (x1 + 24))[:room]:
                _text(d, (x1 + 24, ny_), line, font(14), pal.muted)
                ny_ += 18
        y += 92

    if p.get("chart"):
        top_y = y + 24
        used = min(566 - top_y, 78 * len(chart_bars(p)) + 26)
        chart(im, d, pal, p, tb, ink, (PAD, top_y, W - PAD, top_y + used))
        by = top_y + used + 18
        if tb >= 1:
            for line in p.get("bullets", []):
                if by + 30 > 570:
                    break
                _text(d, (PAD + 4, by), "-", font(18, True), ink)
                _text(d, (PAD + 24, by), line, font(18), pal.ink)
                by += 30
        return

    nums = p.get("numbers", [])
    if not nums:
        return
    size = 60 if len(p.get("meters", [])) < 2 else 48
    block = size * 1.4 + 3 * 23 + 24
    ny = y + 20 + max(0, (586 - (y + 20) - block) / 2)
    colw = (W - 2 * PAD) / len(nums)
    for i, num in enumerate(nums):
        k = stage(tb, i / len(nums), (i + 0.7) / len(nums))
        if k <= 0:
            continue
        x = PAD + i * colw
        sweep(im, d, (x, ny), num["value"], font(size, True), pal, ink, k)
        if k > 0.5:
            yy = ny + size * 1.4
            for line in wrap(d, num["caption"], font(17), colw - 40)[:3]:
                _text(d, (x, yy), line, font(17), pal.ink)
                yy += 23
            _text(d, (x, yy + 6), num["cite"], font(13), pal.muted)
    if tb >= 1:
        by = ny + block + 10
        for line in p.get("bullets", []):
            _text(d, (PAD + 4, by), "-", font(18, True), ink)
            _text(d, (PAD + 24, by), line, font(18), pal.ink)
            by += 30


def chart_page(im, d, pal, p, t, ink):
    chart(im, d, pal, p, t, ink, (PAD, BODY_TOP + 10, W - PAD, 560))


STYLES = {"paged": paged, "chart": chart_page, "blocktable": blocktable, "share": share,
          "radix": radix, "schedule": schedule, "fsm": fsm,
          "tiling": tiling, "memory": memory, "pieces": pieces,
          "result": result}


# ---------------------------------------------------------------- page

def _scrub(d, pal, ink, progress):
    """A hairline along the foot of the page: where the film has got to.

    It costs three pixels and it is the difference between watching a stack
    of slides and watching something with a length.
    """
    if progress is None:
        return
    d.rectangle((0, H - 3, W, H), fill=pal.line)
    d.rectangle((0, H - 3, W * min(max(progress, 0.0), 1.0), H), fill=ink)


def frame(spec, panel, pal, t=1.0, progress=None):
    p = spec["panels"][panel] if isinstance(panel, int) else panel
    d = Canvas(pal.bg)
    im = d
    ink = pal.role(spec.get("accent", "ours"))

    _text(d, (PAD, 40), p["title"], font(31, True), pal.ink)
    y = 84
    for line in wrap(d, p["sub"], font(17), W - 2 * PAD - 220)[:2]:
        _text(d, (PAD, y), line, font(17), pal.muted)
        y += 23
    tag = spec["framework"]
    f = font(17, True)
    _text(d, (W - PAD - d.textlength(tag, font=f), 44), tag, f, ink)
    cite = p.get("cite", spec["paper"])
    _text(d, (W - PAD - d.textlength(cite, font=font(14)), 70), cite,
          font(14), pal.muted)
    d.line((PAD, 136, W - PAD, 136), fill=pal.line)

    if p.get("chart", {}).get("side"):
        col = 330
        p = {**p, "_right": W - PAD - col}
        STYLES[p["style"]](im, d, pal, p, t, ink)
        d.line((W - PAD - col + 40, BODY_TOP - 10, W - PAD - col + 40, 560),
               fill=pal.line)
        chart(im, d, pal, p, stage(t, 0.5, 1.0), ink,
              (W - PAD - col + 80, BODY_TOP + 10, W - PAD, 540))
    else:
        STYLES[p["style"]](im, d, pal, p, t, ink)

    d.line((PAD, 596, W - PAD, 596), fill=pal.line)
    yy = 612
    for line in wrap(d, p["note"], font(21, True), W - 2 * PAD)[:2]:
        _text(d, (PAD, yy), line, font(21, True), ink)
        yy += 26
    _text(d, (PAD, 678), spec["source"], font(13), pal.muted)
    _scrub(d, pal, ink, progress)
    return d.image()


def card(spec, pal, t=1.0, progress=None):
    """The opening card: whose idea this is, and where it was published."""
    d = Canvas(pal.bg)
    ink = pal.role(spec.get("accent", "ours"))
    k = ease_out(min(t * 1.6, 1.0))
    _text(d, (PAD, 232), spec["framework"], font(76, True), ink)
    y = 336
    for line in wrap(d, spec["tagline"], font(28), W - 2 * PAD - 120)[:2]:
        _text(d, (PAD, y), line, font(28), pal.ink)
        y += 38
    d.line((PAD, y + 26, PAD + 260 * k, y + 26), fill=pal.line)
    if k > 0.4:
        _text(d, (PAD, y + 48), spec["paper"], font(19), pal.muted)
    _text(d, (PAD, 678), spec["source"], font(13), pal.muted)
    _scrub(d, pal, ink, progress)
    return d.image()


def load(name):
    p = pathlib.Path(name)
    if not p.exists():
        p = pathlib.Path(__file__).parent.parent / "explainers" / f"{name}.json"
    if not p.exists():
        have = sorted(q.stem for q in p.parent.glob("*.json"))
        raise SystemExit(f"no explainer {name!r}; have: {', '.join(have)}")
    return json.loads(p.read_text())


def render(spec, panel, out, pal, fps=30, build=3.0, hold=2.5):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="explain_"))
    total = int((build + hold) * fps)
    for i in range(total):
        t = min(i / fps / build, 1.0)
        frame(spec, panel, pal, t).save(tmp / f"{i:05d}.png")
    subprocess.run([_ffmpeg(), "-y", "-v", "error", "-framerate", str(fps),
                    "-i", str(tmp / "%05d.png"), "-c:v", "libvpx-vp9",
                    "-b:v", "0", "-crf", "24", "-row-mt", "1", "-pix_fmt", "yuv420p",
                    str(out)], check=True)
    return out


#: A panel is drawn in two beats, so a build of 3.6 s gives each about 1.8,
#: and then the finished page is held: landing a result and cutting away from
#: it immediately is the single commonest way one of these films reads badly.
def film(spec, out, pal, fps=30, card_s=2.8, build=3.6, hold=3.2, fade=0.45):
    """Every panel of one spec, in order: the framework's own film."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="explain_film_"))
    n = len(spec["panels"])
    seg = build + hold
    total = card_s + n * seg
    nf = int(fade * fps)
    i, elapsed, prev = 0, 0.0, None

    def write(im, j, first):
        """Cross-fade into a segment: a hard cut re-starts the eye."""
        nonlocal i, prev
        if prev is not None and not first and j < nf:
            im = Image.blend(prev, im, ease((j + 1) / nf))
        im.save(tmp / f"{i:05d}.png")
        i += 1

    for j in range(int(card_s * fps)):
        write(card(spec, pal, j / fps / card_s,
                   progress=(elapsed + j / fps) / total), j, True)
    prev = card(spec, pal, 1.0, progress=card_s / total)
    elapsed += card_s

    for panel in range(n):
        for j in range(int(seg * fps)):
            im = frame(spec, panel, pal, min(j / fps / build, 1.0),
                       progress=(elapsed + j / fps) / total)
            write(im, j, False)
        prev = frame(spec, panel, pal, 1.0,
                     progress=(elapsed + seg) / total)
        elapsed += seg

    subprocess.run([_ffmpeg(), "-y", "-v", "error", "-framerate", str(fps),
                    "-i", str(tmp / "%05d.png"), "-c:v", "libvpx-vp9",
                    "-b:v", "0", "-crf", "24", "-row-mt", "1",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


#: the sheet is a print, not a film: neutral so no palette owns the page
SHEET_BG, SHEET_INK = (245, 245, 245), (28, 28, 28)


def _clip(d, s, f, width):
    """A sheet label that outgrows its tile reads as two labels colliding."""
    if d.textlength(s, font=f) <= width:
        return s
    while s and d.textlength(s + "...", font=f) > width:
        s = s[:-1]
    return s.rstrip(" ·") + "..."


def _tile(panels, cols, head, scale, labels):
    tw, th = round(W * scale), round(H * scale)
    rows = math.ceil(len(panels) / cols)
    sheet = Image.new("RGB", (tw * cols, (th + head) * rows), SHEET_BG)
    d = ImageDraw.Draw(sheet)
    for i, tile in enumerate(panels):
        r, c = divmod(i, cols)
        x, y = c * tw, r * (th + head)
        sheet.paste(tile.resize((tw, th), Image.LANCZOS), (x, y + head))
        f = _truetype(17, True)
        d.text((x + 10, y + 9), _clip(d, labels[i], f, tw - 24), font=f,
               fill=SHEET_INK)
    return sheet


def menu(spec, out, pal, scale=0.42, cols=3):
    """Every panel in one spec: the template menu for a framework."""
    tiles = [frame(spec, i, pal) for i in range(len(spec["panels"]))]
    labels = [f"{p['style']} · {p['title']}" for p in spec["panels"]]
    _tile(tiles, cols, 34, scale, labels).save(out)
    return out


def sheet(spec, panel, out, scale=0.34, cols=3):
    """One panel in every colour system."""
    names = list(PALETTES)
    tiles = [frame(spec, panel, palette(n)) for n in names]
    _tile(tiles, cols, 34, scale,
          [f"{spec['panels'][panel]['style']} · {n}" for n in names]).save(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="an explainer name or a path to one")
    ap.add_argument("--panel", type=int, default=0)
    ap.add_argument("--palette", default="midnight", choices=sorted(PALETTES))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frame"); ap.add_argument("--out")
    ap.add_argument("--menu"); ap.add_argument("--sheet")
    ap.add_argument("--film", help="every panel in order, one film")
    a = ap.parse_args(argv)
    spec = load(a.spec)
    pal = palette(a.palette)
    if a.menu:
        print(menu(spec, a.menu, pal))
    if a.sheet:
        print(sheet(spec, a.panel, a.sheet))
    if a.frame:
        frame(spec, a.panel, pal).save(a.frame)
        print(a.frame)
    if a.out:
        print(render(spec, a.panel, a.out, pal, fps=a.fps))
    if a.film:
        print(film(spec, a.film, pal, fps=a.fps))
    if not (a.menu or a.sheet or a.frame or a.out or a.film):
        ap.error("nothing to write: pass --frame, --out, --menu or --sheet")


if __name__ == "__main__":
    main()

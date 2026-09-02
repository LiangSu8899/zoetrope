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

The panels live in `demokit/explainers/*.json`, so pointing this at another
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

    demokit explain vllm_paged --menu menu.png
    demokit explain sglang_radix --panel 0 --palette blueprint --frame a.png
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import tempfile

from PIL import Image, ImageDraw

from .palettes import PALETTES, palette
from .race_compose import _arrow, _ffmpeg, font, wrap

W, H = 1280, 720
PAD = 96
BODY_TOP = 164


def reveal(n, t):
    return int(round(n * min(max(t, 0.0), 1.0)))


def _text(d, xy, s, f, fill, center=None):
    x, y = xy
    if center is not None:
        x = center - d.textlength(s, font=f) / 2
    d.text((x, y), s, font=f, fill=fill)


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


def prefix_computes(order):
    """How often a prefix has to be computed if the queue runs in this order."""
    seen, n = None, 0
    for g in order:
        n += 0 if g == seen else 1
        seen = g
    return n


# ---------------------------------------------------------------- painters

def paged(d, pal, p, t, ink):
    slot, gap, hh = 20, 2, 34
    reserve, block = p["reserve"], p["block"]
    reqs = p["requests"]
    used_total, reserved_total, paged_total = paged_counts(p)

    # as reserved: one contiguous run per request, most of it never taken
    x0, y = PAD, BODY_TOP + 40
    _text(d, (PAD, y - 30), p["top"], font(18, True), pal.ink)
    shown = reveal(len(reqs) * reserve, t)
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
            if seen >= reveal(nblocks, t):
                break
            for k in range(block):
                filled = b * block + k < r["used"]
                cx = x + k * (slot + gap)
                _chip(d, (cx, y + 14, cx + slot, y + 14 + hh), pal, ink,
                      filled=filled, r=3)
            x += block * (slot + gap) + 12
            seen += 1
    if t > 0.9:
        end = PAD + (len(reqs) * reserve) * (slot + gap)
        d.line((x + 6, y + 31, end, y + 31), fill=pal.line)
        _text(d, (x + 18, y + 40), "free for other requests", font(15),
              pal.muted)
    _text(d, (PAD, y + 58),
          f"{used_total} of {paged_total} slots hold a token", font(16),
          pal.muted)

    _text(d, (PAD, BODY_TOP + 350), p["tail"], font(20, True), ink)
    _text(d, (PAD, BODY_TOP + 384), p["scale"], font(14), pal.muted)


def blocktable(d, pal, p, t, ink):
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
        _arrow(d, PAD + lw + 10, y + lh / 2,
               gx + c * (cw + cg) - 6, gy + r * (ch + cg) + ch / 2, pal.muted)
    _text(d, (gx, gy + rows * (ch + cg) + 14), p["free_note"], font(15),
          pal.muted)


def share(d, pal, p, t, ink):
    bw, bh = 190, 62
    n = len(p["prompt"])
    x0 = (W - (n * (bw + 14) - 14)) / 2
    top = BODY_TOP + 30
    _text(d, (0, BODY_TOP - 6), p["prompt_label"], font(19, True), pal.ink,
          center=W / 2)
    for i in range(n):
        if i >= reveal(n, t):
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
        if i >= reveal(len(br), t + 0.2):
            break
        x = bx0 + i * (cw + 20)
        d.rounded_rectangle((x, by, x + cw, by + 54), 8, fill=pal.card,
                            outline=pal.line)
        _text(d, (0, by + 16), lab, font(17), pal.ink, center=x + cw / 2)
        # every branch points at the same blocks: that is the whole idea
        _arrow(d, x + cw / 2, by - 8, W / 2, top + bh + 10, pal.muted)
        if i == len(br) - 1 and t > 0.85:
            d.rounded_rectangle((x + 18, by + 74, x + cw + 18, by + 128), 8,
                                fill=pal.card, outline=ink)
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


def radix(d, pal, p, t, ink):
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
    nw, nh = 208, 54
    order = [n for lvl in levels for n in lvl]
    shown = reveal(len(order), t)
    for depth, lvl in enumerate(levels):
        for node in lvl:
            if order.index(node) >= shown:
                continue
            x = PAD + depth * (nw + 88)
            y = node["y"]
            new = node.get("new")
            d.rounded_rectangle((x, y, x + nw, y + nh), 8,
                                fill=pal.card if new else pal.dim(ink, 0.55),
                                outline=ink if new else None, width=3)
            _text(d, (x + 14, y + 8), node["label"], font(17, True),
                  ink if new else pal.ink)
            _text(d, (x + 14, y + 30), f"{node['tokens']} tokens", font(14),
                  pal.muted if new else pal.ink)
            for k in node.get("kids", []):
                if order.index(k) < shown:
                    _arrow(d, x + nw + 6, y + nh / 2,
                           x + nw + 82, k["y"] + nh / 2, pal.line)

    y = BODY_TOP + 340
    _chip(d, (PAD, y, PAD + 26, y + 18), pal, pal.dim(ink, 0.55))
    _text(d, (PAD + 38, y - 1), p["tail"], font(17), pal.ink)
    x = PAD + 60 + d.textlength(p["tail"], font=font(17))
    d.rounded_rectangle((x, y, x + 26, y + 18), 5, fill=pal.card,
                        outline=ink, width=3)
    _text(d, (x + 38, y - 1), p["new_tail"], font(17), ink)
    _text(d, (PAD, y + 34), p["scale"], font(14), pal.muted)


def schedule(d, pal, p, t, ink):
    groups = p["groups"]
    keys = sorted(set(groups))
    inks = {k: pal.role(r) for k, r in
            zip(keys, ("ours", "compiled", "native", "stock"))}
    cw, cg, chh = 106, 14, 58

    def row(order, y, head, ):
        _text(d, (PAD, y - 52), head, font(18, True), pal.ink)
        seen = None
        for i, g in enumerate(order):
            hit = g == seen
            seen = g
            if i >= reveal(len(order), t):
                continue
            x = PAD + i * (cw + cg)
            d.rounded_rectangle((x, y, x + cw, y + chh), 8, fill=inks[g])
            _text(d, (0, y + 16), f"prefix {g}", font(17, True), pal.bg,
                  center=x + cw / 2)
            r = 7
            cx, cy = x + cw / 2, y - 14
            if hit:
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=inks[g])
            else:
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=pal.card,
                          outline=pal.muted, width=2)

    a = prefix_computes(groups)
    b = prefix_computes(sorted(groups))
    row(groups, BODY_TOP + 50, p["top"])
    row(sorted(groups), BODY_TOP + 210, p["bottom"])
    _text(d, (PAD, BODY_TOP + 122),
          f"the prefix is computed {a} times", font(17), pal.muted)
    _text(d, (PAD, BODY_TOP + 282),
          f"the prefix is computed {b} times, and read {len(groups) - b} "
          f"times from the tree", font(17), ink)
    _text(d, (PAD, BODY_TOP + 326), p["tail"], font(19, True), ink)


def fsm(d, pal, p, t, ink):
    text, forced = p["text"], p["forced"]
    n, decode_steps = fsm_steps(p)
    tw = (W - 2 * PAD) / n
    y = BODY_TOP + 30
    _text(d, (PAD, y - 30), p["top"], font(18, True), pal.ink)
    for i, s in enumerate(text):
        if i >= reveal(n, t):
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
    while i < n:
        j = i
        while j < n and forced[j] == forced[i]:
            j += 1
        x0, x1 = PAD + i * tw + 3, PAD + j * tw - 3
        if forced[i]:
            d.rounded_rectangle((x0, y, x1, y + 62), 7, fill=pal.dim(ink, .4))
            _text(d, (0, y + 12), "forced by the schema", font(16), pal.ink,
                  center=(x0 + x1) / 2)
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
    _text(d, (PAD, y + 100), f"{n} decode steps becomes {decode_steps}",
          font(20, True), ink)
    _text(d, (PAD, y + 134), p["tail"], font(16), pal.muted)


STYLES = {"paged": paged, "blocktable": blocktable, "share": share,
          "radix": radix, "schedule": schedule, "fsm": fsm}


# ---------------------------------------------------------------- page

def frame(spec, panel, pal, t=1.0):
    p = spec["panels"][panel] if isinstance(panel, int) else panel
    im = Image.new("RGB", (W, H), pal.bg)
    d = ImageDraw.Draw(im)
    ink = pal.role(spec.get("accent", "ours"))

    _text(d, (PAD, 40), p["title"], font(31, True), pal.ink)
    y = 84
    for line in wrap(d, p["sub"], font(17), W - 2 * PAD - 220)[:2]:
        _text(d, (PAD, y), line, font(17), pal.muted)
        y += 23
    tag = spec["framework"]
    f = font(17, True)
    _text(d, (W - PAD - d.textlength(tag, font=f), 44), tag, f, ink)
    _text(d, (W - PAD - d.textlength(spec["paper"], font=font(14)), 70),
          spec["paper"], font(14), pal.muted)
    d.line((PAD, 136, W - PAD, 136), fill=pal.line)

    STYLES[p["style"]](d, pal, p, t, ink)

    d.line((PAD, 596, W - PAD, 596), fill=pal.line)
    yy = 612
    for line in wrap(d, p["note"], font(21, True), W - 2 * PAD)[:2]:
        _text(d, (PAD, yy), line, font(21, True), ink)
        yy += 26
    _text(d, (PAD, 678), spec["source"], font(13), pal.muted)
    return im


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
                    "-b:v", "0", "-crf", "30", "-pix_fmt", "yuv420p",
                    str(out)], check=True)
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
        f = font(17, True)
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
    if not (a.menu or a.sheet or a.frame or a.out):
        ap.error("nothing to write: pass --frame, --out, --menu or --sheet")


if __name__ == "__main__":
    main()

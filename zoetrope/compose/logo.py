"""The mark: a drum of frames, read as motion.

A zoetrope is a slitted drum with a strip of still drawings inside it. Spin
it, look through the slits, and the stills become one moving thing. That is
this whole project in an object, so the mark is the drum seen end-on: a ring
of slits, and a crest of light travelling round it.

Nothing in the ring moves. Each slit only grows and shrinks in place, and the
rotation is entirely in the order they do it — which is the same trick the
drum plays, and the same one every film here plays with a list of timestamps.

    zoetrope logo docs/logo_dark.gif  --palette midnight
    zoetrope logo docs/logo_light.gif --palette paper --word
"""

from __future__ import annotations

import argparse
import math
import pathlib

from PIL import Image

from .canvas import Canvas, ease_out, font
from .palettes import PALETTES, palette

SLITS = 22
FRAMES = 44


def _draw(pal, t, *, word, size, k=3):
    """One turn of the drum, at `t` in 0..1 of a full revolution."""
    w, h = size
    im = Image.new("RGB", (w * k, h * k), pal.bg)
    cv = Canvas.__new__(Canvas)          # a canvas of our own size
    cv.k, cv.im = k, im
    from PIL import ImageDraw
    cv.d = ImageDraw.Draw(im)

    ink = pal.role("ours")
    cx, cy = (h / 2 + 4, h / 2)
    r_in, r_out = h * 0.19, h * 0.43
    phase = t * 2 * math.pi

    for i in range(SLITS):
        a = 2 * math.pi * i / SLITS
        # one crest travelling round: the slit does not move, its turn does
        lift = max(0.0, math.cos(a - phase)) ** 3
        r1 = r_in + (r_out - r_in) * (0.34 + 0.66 * lift)
        col = pal.dim(ink, 0.30 + 0.70 * lift)
        ca, sa = math.cos(a), math.sin(a)
        cv.line((cx + r_in * ca, cy + r_in * sa, cx + r1 * ca, cy + r1 * sa),
                fill=col, width=h * 0.036)

    # the hub: the strip of stills the drum is spinning
    cv.ellipse((cx - r_in * 0.42, cy - r_in * 0.42,
                cx + r_in * 0.42, cy + r_in * 0.42), fill=pal.dim(ink, 0.22))
    ha = phase
    cv.ellipse((cx + r_in * 0.62 * math.cos(ha) - h * 0.022,
                cy + r_in * 0.62 * math.sin(ha) - h * 0.022,
                cx + r_in * 0.62 * math.cos(ha) + h * 0.022,
                cy + r_in * 0.62 * math.sin(ha) + h * 0.022), fill=ink)

    if word:
        cv.text((h + 22, cy - h * 0.23), "zoetrope",
                font=font(h * 0.30, True), fill=pal.ink)
        cv.text((h + 24, cy + h * 0.13), "frames, and the motion in them",
                font=font(h * 0.108), fill=pal.muted)
    return im.resize((w, h), Image.LANCZOS)


def _width(height, word):
    """Fit the box to the mark and the words, not to a guessed ratio."""
    if not word:
        return int(height * 1.16)
    probe = Canvas(( 0, 0, 0), k=1)
    a = probe.textlength("zoetrope", font=font(height * 0.30, True))
    b = probe.textlength("frames, and the motion in them",
                         font=font(height * 0.108))
    return int(height + 24 + max(a, b) + 26)


def render(out, pal, *, word=True, height=104, fps=22):
    size = (_width(height, word), height)
    frames = [_draw(pal, i / FRAMES, word=word, size=size)
              for i in range(FRAMES)]
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--palette", default="midnight", choices=sorted(PALETTES))
    ap.add_argument("--word", action="store_true", help="with the wordmark")
    ap.add_argument("--height", type=int, default=104)
    ap.add_argument("--fps", type=int, default=22)
    a = ap.parse_args(argv)
    print(render(a.out, palette(a.palette), word=a.word, height=a.height,
                 fps=a.fps))


if __name__ == "__main__":
    main()

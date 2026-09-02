"""Colour systems.

A film is a piece of design, and the same recording reads differently in
different ink.  A painter should ask a palette for a *role* — the ground,
the rule, the reading text, the arm that ships with the framework — and
never name a colour itself.  Then one flag changes the whole look and
nothing in the drawing code moves.

    pal = palette("paper")
    d.rectangle(box, fill=pal.card, outline=pal.line)
    d.line(pts, fill=pal.role("ours"), width=3)

Roles, and what they mean in every palette:

    stock       the framework as it ships
    compiled    the same framework, compiled
    ours        the arm the film is about
    native      a third arm, when there is one

The two greys are `ink` (reading text) and `muted` (labels, units, axis
numbers).  `grid` is fainter than `line`: rules you should be able to
ignore.  Every palette here has been rendered and looked at; the light
ones carry darker arm colours, which is not a detail you can skip.
"""

from __future__ import annotations


class Palette:
    def __init__(self, name, *, bg, card, line, grid, ink, muted, roles,
                 note=""):
        self.name = name
        self.bg, self.card, self.line, self.grid = bg, card, line, grid
        self.ink, self.muted = ink, muted
        self.roles = dict(roles)
        self.note = note
        self.light = sum(bg) > 380

    def role(self, name):
        """`ours` is the protocol's name; `accent` is the older spelling."""
        if name == "accent":
            name = "ours"
        return self.roles.get(name, self.roles["stock"])

    def dim(self, color, k=0.35):
        """A colour laid over the ground — for trails and spent things."""
        return tuple(round(c * k + b * (1 - k))
                     for c, b in zip(color, self.bg))


PALETTES = {
    "midnight": Palette(
        "midnight", note="the kit's own look: dark, green accent",
        bg=(18, 25, 23), card=(24, 33, 32), line=(38, 50, 48),
        grid=(30, 41, 39), ink=(232, 236, 233), muted=(147, 160, 154),
        roles={"stock": (171, 175, 164), "compiled": (140, 165, 200),
               "ours": (52, 194, 154), "native": (226, 178, 96)}),

    "paper": Palette(
        "paper", note="light, for slides and print",
        bg=(247, 245, 240), card=(255, 255, 255), line=(216, 211, 202),
        grid=(233, 230, 223), ink=(30, 32, 34), muted=(126, 128, 130),
        roles={"stock": (158, 156, 150), "compiled": (66, 104, 160),
               "ours": (196, 66, 54), "native": (176, 124, 40)}),

    "phosphor": Palette(
        "phosphor", note="terminal green, for a runtime that feels close "
                         "to the metal",
        bg=(8, 13, 10), card=(12, 20, 15), line=(28, 48, 35),
        grid=(19, 33, 24), ink=(198, 240, 206), muted=(94, 138, 108),
        roles={"stock": (72, 108, 82), "compiled": (198, 168, 74),
               "ours": (104, 240, 138), "native": (94, 204, 204)}),

    "blueprint": Palette(
        "blueprint", note="drafting blue, for anything with a diagram in it",
        bg=(12, 22, 44), card=(17, 30, 58), line=(36, 58, 98),
        grid=(24, 41, 72), ink=(222, 232, 248), muted=(126, 152, 192),
        roles={"stock": (118, 144, 182), "compiled": (240, 196, 96),
               "ours": (86, 196, 255), "native": (255, 138, 118)}),

    "ember": Palette(
        "ember", note="warm dark, for heat and throughput",
        bg=(26, 19, 24), card=(36, 27, 34), line=(60, 45, 56),
        grid=(43, 32, 40), ink=(250, 238, 232), muted=(174, 148, 150),
        roles={"stock": (142, 124, 120), "compiled": (216, 152, 90),
               "ours": (250, 88, 108), "native": (250, 198, 112)}),

    "mono": Palette(
        "mono", note="grey, one accent: nothing competes with the point",
        bg=(15, 15, 15), card=(23, 23, 23), line=(46, 46, 46),
        grid=(32, 32, 32), ink=(238, 238, 238), muted=(138, 138, 138),
        roles={"stock": (104, 104, 104), "compiled": (172, 172, 172),
               "ours": (236, 82, 58), "native": (208, 208, 208)}),
}


def palette(name):
    try:
        return PALETTES[name]
    except KeyError:
        raise SystemExit(
            f"unknown palette {name!r}; have: "
            + ", ".join(sorted(PALETTES))) from None

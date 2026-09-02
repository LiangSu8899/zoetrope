"""One drawing surface, shared by the compositors.

PIL draws hard edges: a rounded corner, a diagonal arrow and a circle all
come out stepped, and at 1280x720 that is the whole difference between a page
that looks designed and one that looks like a screenshot of a program. So
every page is drawn several times larger and brought back down, and painters
never see the factor — they pass page coordinates and page font sizes, and
`textlength` answers in page units too.

The easing lives here for the same reason. Nothing physical starts at full
speed; a page whose elements switch on reads as a slideshow, and one whose
elements arrive reads as motion. `ease` for a thing travelling, `ease_out`
for a thing landing, `appear` for a list that should arrive as a list.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

from .race_compose import font as _truetype

W, H = 1280, 720

#: Every page is drawn this many times larger and brought back down.  PIL has
#: no anti-aliasing: a rounded corner, a diagonal arrow and a circle are all
#: drawn hard, and at 1x they are the difference between a page that looks
#: printed and one that looks like a screenshot of a program.
SS = 2

R_SM, R_MD, R_LG = 5, 8, 12          # one set of corner radii, used by all


class Face:
    """A font asked for in page units; the canvas decides the real size."""

    __slots__ = ("size", "bold")

    def __init__(self, size, bold=False):
        self.size, self.bold = size, bold


def font(size, bold=False):
    return Face(size, bold)


class Canvas:
    """Draw in 1280x720 page units; render at SS times that and come down.

    Painters never see the supersampling: they pass page coordinates and page
    font sizes, and `textlength` answers in page units too, so wrapping and
    centring are unaffected by the factor.
    """

    def __init__(self, bg, k=SS):
        self.k = k
        self.im = Image.new("RGB", (W * k, H * k), bg)
        self.d = ImageDraw.Draw(self.im)

    def _b(self, box):
        """Page units to device units, for a box or a run of points."""
        if box and isinstance(box[0], (tuple, list)):
            return [(x * self.k, y * self.k) for x, y in box]
        return tuple(v * self.k for v in box)

    def _f(self, face):
        return _truetype(max(1, round(face.size * self.k)), face.bold)

    def rectangle(self, box, fill=None, outline=None, width=1):
        self.d.rectangle(self._b(box), fill=fill, outline=outline,
                         width=width * self.k)

    def rounded_rectangle(self, box, r, fill=None, outline=None, width=1):
        self.d.rounded_rectangle(self._b(box), r * self.k, fill=fill,
                                 outline=outline, width=width * self.k)

    def ellipse(self, box, fill=None, outline=None, width=1):
        self.d.ellipse(self._b(box), fill=fill, outline=outline,
                       width=width * self.k)

    def line(self, box, fill=None, width=1, joint=None):
        self.d.line(self._b(box), fill=fill,
                    width=max(1, round(width * self.k)), joint=joint)

    def polygon(self, points, fill=None):
        self.d.polygon([(x * self.k, y * self.k) for x, y in points], fill=fill)

    def text(self, xy, s, font=None, fill=None):
        self.d.text(self._b(xy), s, font=self._f(font), fill=fill)

    def textlength(self, s, font=None):
        return self.d.textlength(s, font=self._f(font)) / self.k

    def plate_text(self, window, xy, s, face, plate, glyph):
        """Text cut out of a filled plate, inside `window` only.

        A highlighter is a plate with the glyphs knocked out of it; drawing
        the plate and then the text cannot be clipped in PIL, so the window
        is composed on its own and pasted back.
        """
        k = self.k
        x0, y0, x1, y1 = (int(round(v * k)) for v in window)
        if x1 <= x0 or y1 <= y0:
            return
        tile = Image.new("RGB", (x1 - x0, y1 - y0), plate)
        ImageDraw.Draw(tile).text((xy[0] * k - x0, xy[1] * k - y0), s,
                                  font=self._f(face), fill=glyph)
        self.im.paste(tile, (x0, y0))

    def image_at(self, img, xy, size):
        """Paste a picture in page units; it is resized at device scale."""
        k = self.k
        w, h = int(size[0] * k), int(size[1] * k)
        self.im.paste(img.resize((w, h), Image.LANCZOS),
                      (int(xy[0] * k), int(xy[1] * k)))

    def image(self):
        return self.im.resize((W, H), Image.LANCZOS)


def ease(t):
    """Cubic in and out.  Nothing in the physical world starts at full speed,
    and a page that does reads as a slideshow rather than as motion."""
    t = min(max(t, 0.0), 1.0)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out(t):
    """Fast, then settling — for a thing arriving rather than travelling."""
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def _head(d, x, y, ux, uy, ink, length, half):
    """A head lying along the line it ends.

    An axis-aligned triangle on a diagonal is the single cheapest-looking
    thing a diagram can do: the head points one way and the shaft arrives
    from another, and the join reads as a mistake.
    """
    px, py = -uy, ux
    d.polygon([(x, y),
               (x - ux * length + px * half, y - uy * length + py * half),
               (x - ux * length - px * half, y - uy * length - py * half)],
              fill=ink)


def arrow(d, x0, y0, x1, y1, ink, *, width=1.6, head=9, half=3.4):
    """A straight arrow: aligned head, and a shaft that stops where it begins.

    Running the shaft under the head thickens the tip and blunts it, which
    is the other half of why a drawn arrow looks worse than a designed one.
    """
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist
    if head:
        d.line([x0, y0, x1 - ux * head * 0.9, y1 - uy * head * 0.9],
               fill=ink, width=width)
        _head(d, x1, y1, ux, uy, ink, head, half)
    else:
        d.line([x0, y0, x1, y1], fill=ink, width=width)


def link(d, x0, y0, x1, y1, ink, *, axis="x", width=1.6, head=0, half=3.4,
         bend=0.55, steps=26):
    """A connector that leaves and arrives along one axis, curving between.

    This is the shape a node graph or an org chart uses, and the reason is
    not decoration: a straight diagonal between two boxes crosses whatever
    lies between them at an arbitrary angle, while a curve that leaves
    horizontally and arrives horizontally reads as a route.
    """
    if axis == "x":
        c0, c1 = (x0 + (x1 - x0) * bend, y0), (x1 - (x1 - x0) * bend, y1)
    else:
        c0, c1 = (x0, y0 + (y1 - y0) * bend), (x1, y1 - (y1 - y0) * bend)
    pts = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        pts.append((u ** 3 * x0 + 3 * u * u * t * c0[0] + 3 * u * t * t * c1[0]
                    + t ** 3 * x1,
                    u ** 3 * y0 + 3 * u * u * t * c0[1] + 3 * u * t * t * c1[1]
                    + t ** 3 * y1))
    if head:
        pts = pts[:-1]
        ex, ey = pts[-1]
        dx, dy = x1 - ex, y1 - ey
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        d.line(pts, fill=ink, width=width, joint="curve")
        _head(d, x1, y1, ux, uy, ink, head, half)
    else:
        d.line(pts, fill=ink, width=width, joint="curve")


#: kept for callers that predate `arrow`
def _arrow(d, x0, y0, x1, y1, ink, head=5):
    arrow(d, x0, y0, x1, y1, ink, head=head + 4)




def appear(i, n, t, span=0.55):
    """One item's own entrance inside a group's beat.

    Staggered and eased, so a list arrives as a list rather than as a single
    switch, and a little of it is always still moving while the eye lands.
    """
    lead = (1 - span) / max(n - 1, 1)
    return ease_out(min(max((t - i * lead) / span, 0.0), 1.0))

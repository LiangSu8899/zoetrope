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


def _arrow(d, x0, y0, x1, y1, ink, head=5):
    d.line([x0, y0, x1, y1], fill=ink, width=2)
    if y1 > y0:
        d.polygon([(x1, y1), (x1 - head, y1 - head), (x1 + head, y1 - head)],
                  fill=ink)
    elif x1 > x0:
        d.polygon([(x1, y1), (x1 - head, y1 - head), (x1 - head, y1 + head)],
                  fill=ink)




def appear(i, n, t, span=0.55):
    """One item's own entrance inside a group's beat.

    Staggered and eased, so a list arrives as a list rather than as a single
    switch, and a little of it is always still moving while the eye lands.
    """
    lead = (1 - span) / max(n - 1, 1)
    return ease_out(min(max((t - i * lead) / span, 0.0), 1.0))

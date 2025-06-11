# fmt: off

import time

from ase.io.utils import PlottingVariables, make_patch_list
from ase.utils import writer


class EPS(PlottingVariables):
    def __init__(self, atoms,
                 rotation='', radii=None,
                 bbox=None, colors=None, scale=20, maxwidth=500,
                 **kwargs):
        """Encapsulated PostScript writer.

        show_unit_cell: int
            0: Don't show unit cell (default).  1: Show unit cell.
            2: Show unit cell and make sure all of it is visible.
        """
        PlottingVariables.__init__(
            self, atoms, rotation=rotation,
            radii=radii, bbox=bbox, colors=colors, scale=scale,
            maxwidth=maxwidth,
            **kwargs)

    def write(self, fd):
        renderer = self._renderer(fd)
        self.write_header(fd)
        self.write_body(fd, renderer)
        self.write_trailer(fd, renderer)

    def write_header(self, fd):
        fd.write('%!PS-Adobe-3.0 EPSF-3.0\n')
        fd.write('%%Creator: G2\n')
        fd.write('%%CreationDate: %s\n' % time.ctime(time.time()))
        fd.write('%%Orientation: portrait\n')
        bbox = (0, 0, self.w, self.h)
        fd.write('%%%%BoundingBox: %d %d %d %d\n' % bbox)
        fd.write('%%EndComments\n')

        Ndict = len(psDefs)
        fd.write('%%BeginProlog\n')
        fd.write('/mpldict %d dict def\n' % Ndict)
        fd.write('mpldict begin\n')
        for d in psDefs:
            d = d.strip()
            for line in d.split('\n'):
                fd.write(line.strip() + '\n')
        fd.write('%%EndProlog\n')

        fd.write('mpldict begin\n')
        fd.write('%d %d 0 0 clipbox\n' % (self.w, self.h))

    def _renderer(self, fd):
        # Subclass can override
        from matplotlib.backends.backend_ps import RendererPS
        return RendererPS(self.w, self.h, fd)

    def write_body(self, fd, renderer):
        patch_list = make_patch_list(self)
        for patch in patch_list:
            patch.draw(renderer)

    def write_trailer(self, fd, renderer):
        fd.write('end\n')
        fd.write('showpage\n')


@writer
def write_eps(fd, atoms, **parameters):
    EPS(atoms, **parameters).write(fd)


# Adapted from Matplotlib 3.7.3

# The following Python dictionary psDefs contains the entries for the
# PostScript dictionary mpldict.  This dictionary implements most of
# the matplotlib primitives and some abbreviations.
#
# References:
# https://www.adobe.com/content/dam/acom/en/devnet/actionscript/articles/PLRM.pdf
# http://preserve.mactech.com/articles/mactech/Vol.09/09.04/PostscriptTutorial
# http://www.math.ubc.ca/people/faculty/cass/graphics/text/www/
#

# The usage comments use the notation of the operator summary
# in the PostScript Language reference manual.
psDefs = [
    # name proc  *_d*  -
    # Note that this cannot be bound to /d, because when embedding a Type3 font
    # we may want to define a "d" glyph using "/d{...} d" which would locally
    # overwrite the definition.
    "/_d { bind def } bind def",
    # x y  *m*  -
    "/m { moveto } _d",
    # x y  *l*  -
    "/l { lineto } _d",
    # x y  *r*  -
    "/r { rlineto } _d",
    # x1 y1 x2 y2 x y *c*  -
    "/c { curveto } _d",
    # *cl*  -
    "/cl { closepath } _d",
    # *ce*  -
    "/ce { closepath eofill } _d",
    # w h x y  *box*  -
    """/box {
      m
      1 index 0 r
      0 exch r
      neg 0 r
      cl
    } _d""",
    # w h x y  *clipbox*  -
    """/clipbox {
      box
      clip
      newpath
    } _d""",
    # wx wy llx lly urx ury  *setcachedevice*  -
    "/sc { setcachedevice } _d",
]

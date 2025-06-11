# fmt: off
from itertools import islice
from typing import IO

import numpy as np

from ase.data import atomic_numbers, covalent_radii
from ase.data.colors import jmol_colors as default_colors
from ase.io.formats import string2index
from ase.utils import irotate, rotate


def normalize(a):
    return np.array(a) / np.linalg.norm(a)


def complete_camera_vectors(look=None, up=None, right=None):
    """Creates the camera (or look) basis vectors from user input and
     will autocomplete missing vector or non-orthogonal vectors using dot
     products. The look direction will be maintained, up direction has higher
     priority than right direction"""

    # ensure good input
    if look is not None:
        assert len(look) == 3
        l = np.array(look)

    if up is not None:
        assert len(up) == 3
        u = np.array(up)

    if right is not None:
        assert len(right) == 3
        r = np.array(right)

    if look is not None and up is not None:
        r = normalize(np.cross(l, u))
        u = normalize(np.cross(r, l))  # ensures complete perpendicularity
        l = normalize(np.cross(u, r))
    elif look is not None and right is not None:
        u = normalize(np.cross(r, l))
        r = normalize(np.cross(l, u))  # ensures complete perpendicularity
        l = normalize(np.cross(u, r))
    elif up is not None and right is not None:
        l = normalize(np.cross(u, r))
        r = normalize(np.cross(l, u))  # ensures complete perpendicularity
        u = normalize(np.cross(r, u))
    else:
        raise ValueError('''At least two camera vectors of <look>, <up>,
 or <right> must be specified''')
    return l, u, r


def get_cell_vertex_points(cell, disp=(0.0, 0.0, 0.0)):
    """Returns 8x3 list of the cell vertex coordinates"""
    cell_vertices = np.empty((2, 2, 2, 3))
    displacement = np.array(disp)
    for c1 in range(2):
        for c2 in range(2):
            for c3 in range(2):
                cell_vertices[c1, c2, c3] = [c1, c2, c3] @ cell + displacement
    cell_vertices.shape = (8, 3)
    return cell_vertices


def update_line_order_for_atoms(L, T, D, atoms, radii):
    # why/how does this happen before the camera rotation???
    R = atoms.get_positions()
    r2 = radii**2
    for n in range(len(L)):
        d = D[T[n]]
        if ((((R - L[n] - d)**2).sum(1) < r2) &
                (((R - L[n] + d)**2).sum(1) < r2)).any():
            T[n] = -1
    return T


def combine_bboxes(bbox_a, bbox_b):
    """Combines bboxes using their extrema"""
    bbox_low = np.minimum(bbox_a[0], bbox_b[0])
    bbox_high = np.maximum(bbox_a[1], bbox_b[1])
    return np.array([bbox_low, bbox_high])


def has_cell(atoms):
    return atoms.cell.rank > 0


HIDE = 0
SHOW_CELL = 1
SHOW_CELL_AND_FIT_TO_ALL = 2
SHOW_CELL_AND_FIT_TO_CELL = 3


class PlottingVariables:
    # removed writer - self
    def __init__(self, atoms, rotation='', show_unit_cell=2,
                 radii=None, bbox=None, colors=None, scale=20,
                 maxwidth=500, extra_offset=(0., 0.),
                 auto_bbox_size=1.05,
                 auto_image_plane_z='front_all',
                 ):

        assert show_unit_cell in (0, 1, 2, 3)
        """Handles camera/paper space transformations used for rendering, 2D
        plots, ...and a few legacy features. after camera rotations, the image
        plane is set to the front of structure.

        atoms: Atoms object
            The Atoms object to render/plot.

        rotation: string or 3x3 matrix
            Controls camera rotation. Can be a string with euler angles in
            degrees like '45x, 90y, 0z' or a rotation matrix.
            (defaults to '0x, 0y, 0z')

        show_unit_cell: int 0, 1, 2, or 3
            0 cell is not shown, 1 cell is shown, 2 cell is shown and bounding
            box is computed to fit atoms and cell, 3 bounding box is fixed to
            cell only. (default 2)

        radii: list of floats
            a list of atomic radii for the atoms. (default None)

        bbox: list of four floats
            Allows explicit control of the image plane bounding box in the form
            (xlo, ylo, xhi, yhi) where x and y are the horizontal and vertical
            axes of the image plane. The units are in atomic coordinates without
            the paperspace scale factor. (defaults to None the automatic
            bounding box is used)

        colors : a list of RGB color triples
            a list of the RGB color triples for each atom. (default None, uses
            Jmol colors)

        scale: float
            The ratio between the image plane units and atomic units, e.g.
            Angstroms per cm. (default 20.0)

        maxwidth: float
            Limits the width of the image plane. (why?) Uses paperspace units.
            (default 500)

        extra_offset: (float, float)
            Translates the image center in the image plane by (x,y) where x and
            y are the horizontal and vertical shift distances, respectively.
            (default (0.0, 0.0)) should only be used for small tweaks to the
            automatically fit image plane

        auto_bbox_size: float
            Controls the padding given to the bounding box in the image plane.
            With auto_bbox_size=1.0 the structure touches the edges of the
            image. auto_bbox_size>1.0 gives whitespace padding. (default 1.05)

        auto_image_plane_z: string ('front_all', 'front_auto', 'legacy')
            After a camera rotation, controls where to put camera image plane
            relative to the atoms and cell. 'front_all' puts everything in front
            of the camera. 'front_auto' sets the image plane location to
            respect the show_unit_cell option so that the atoms or cell can be
            ignored when setting the image plane. 'legacy' leaves the image
            plane passing through the origin for backwards compatibility.
            (default: 'front_all')
        """

        self.show_unit_cell = show_unit_cell
        self.numbers = atoms.get_atomic_numbers()
        self.maxwidth = maxwidth
        self.atoms = atoms
        # not used in PlottingVariables, keeping for legacy
        self.natoms = len(atoms)

        self.auto_bbox_size = auto_bbox_size
        self.auto_image_plane_z = auto_image_plane_z
        self.offset = np.zeros(3)
        self.extra_offset = np.array(extra_offset)

        self.constraints = atoms.constraints
        # extension for partial occupancies
        self.frac_occ = False
        self.tags = None
        self.occs = None

        if 'occupancy' in atoms.info:
            self.occs = atoms.info['occupancy']
            self.tags = atoms.get_tags()
            self.frac_occ = True

        # colors
        self.colors = colors
        if colors is None:
            ncolors = len(default_colors)
            self.colors = default_colors[self.numbers.clip(max=ncolors - 1)]

        # radius
        if radii is None:
            radii = covalent_radii[self.numbers]
        elif isinstance(radii, float):
            radii = covalent_radii[self.numbers] * radii
        else:
            radii = np.array(radii)

        self.radii = radii  # radius in Angstroms
        self.scale = scale  # Angstroms per cm

        self.set_rotation(rotation)
        self.update_image_plane_offset_and_size_from_structure(bbox=bbox)

    def to_dict(self):
        out = {
            'bbox': self.get_bbox(),
            'rotation': self.rotation,
            'scale':    self.scale,
            'colors': self.colors}
        return out

    @property
    def d(self):
        # XXX hopefully this can be deprecated someday.
        """Returns paperspace diameters for scale and radii lists"""
        return 2 * self.scale * self.radii

    def set_rotation(self, rotation):
        if rotation is not None:
            if isinstance(rotation, str):
                rotation = rotate(rotation)
            self.rotation = rotation
        self.update_patch_and_line_vars()

    def update_image_plane_offset_and_size_from_structure(self, bbox=None):
        """Updates image size to fit structure according to show_unit_cell
        if bbox=None. Otherwise, sets the image size from bbox. bbox is in the
        image plane. Note that bbox format is (xlo, ylo, xhi, yhi) for
        compatibility reasons the internal functions use (2,3)"""

        # zero out the offset so it's not involved in the
        # to_image_plane_positions() calculations which are used to calcucate
        #  the offset
        self.offset = np.zeros(3)

        # computing the bboxes in self.atoms here makes it easier to follow the
        # various options selection/choices later
        bbox_atoms = self.get_bbox_from_atoms(self.atoms, self.d / 2)
        if has_cell(self.atoms):
            cell = self.atoms.get_cell()
            disp = self.atoms.get_celldisp().flatten()
            bbox_cell = self.get_bbox_from_cell(cell, disp)
            bbox_combined = combine_bboxes(bbox_atoms, bbox_cell)
        else:
            bbox_combined = bbox_atoms

        # bbox_auto is the bbox that matches the show_unit_cell option
        if has_cell(self.atoms) and self.show_unit_cell in (
            SHOW_CELL_AND_FIT_TO_ALL, SHOW_CELL_AND_FIT_TO_CELL):

            if self.show_unit_cell == SHOW_CELL_AND_FIT_TO_ALL:
                bbox_auto = bbox_combined
            else:
                bbox_auto = bbox_cell
        else:
            bbox_auto = bbox_atoms

        #
        if bbox is None:
            middle = (bbox_auto[0] + bbox_auto[1]) / 2
            im_size = self.auto_bbox_size * (bbox_auto[1] - bbox_auto[0])
            # should auto_bbox_size pad the z_heght via offset?

            if im_size[0] > self.maxwidth:
                rescale_factor = self.maxwidth / im_size[0]
                im_size *= rescale_factor
                self.scale *= rescale_factor
            offset = middle - im_size / 2
        else:
            width = (bbox[2] - bbox[0]) * self.scale
            height = (bbox[3] - bbox[1]) * self.scale

            im_size = np.array([width, height, 0])
            offset = np.array([bbox[0], bbox[1], 0]) * self.scale

        # this section shifts the image plane up and down parallel to the look
        # direction to match the legacy option, or to force it allways touch the
        # front most objects regardless of the show_unit_cell setting
        if self.auto_image_plane_z == 'front_all':
            offset[2] = bbox_combined[1, 2]  # highest z in image orientation
        elif self.auto_image_plane_z == 'legacy':
            offset[2] = 0
        elif self.auto_image_plane_z == 'front_auto':
            offset[2] = bbox_auto[1, 2]
        else:
            raise ValueError(
                f'bad image plane setting {self.auto_image_plane_z!r}')

        # since we are moving the origin in the image plane (camera coordinates)
        self.offset += offset

        # Previously, the picture size changed with extra_offset, This is very
        # counter intuitive and seems like a bug. Leaving it commented out in
        # case someone relying on this likely bug needs to revert it.
        self.w = im_size[0]  # + self.extra_offset[0]
        self.h = im_size[1]  # + self.extra_offset[1]

        # allows extra_offset to be 2D or 3D
        for i in range(len(self.extra_offset)):
            self.offset[i] -= self.extra_offset[i]

        # we have to update the arcane stuff after every camera update.
        self.update_patch_and_line_vars()

    def center_camera_on_position(self, pos, scaled_position=False):
        if scaled_position:
            pos = pos @ self.atoms.cell
        im_pos = self.to_image_plane_positions(pos)
        cam_pos = self.to_image_plane_positions(self.get_image_plane_center())
        in_plane_shift = im_pos - cam_pos
        self.offset[0:2] += in_plane_shift[0:2]
        self.update_patch_and_line_vars()

    def get_bbox(self):
        xlo = self.offset[0]
        ylo = self.offset[1]
        xhi = xlo + self.w
        yhi = ylo + self.h
        return np.array([xlo, ylo, xhi, yhi]) / self.scale

    def set_rotation_from_camera_directions(self,
                                            look=None, up=None, right=None,
                                            scaled_position=False):

        if scaled_position:
            if look is not None:
                look = look @ self.atoms.cell
            if right is not None:
                right = right @ self.atoms.cell
            if up is not None:
                up = up @ self.atoms.cell

        look, up, right = complete_camera_vectors(look, up, right)

        rotation = np.zeros((3, 3))
        rotation[:, 0] = right
        rotation[:, 1] = up
        rotation[:, 2] = -look
        self.rotation = rotation
        self.update_patch_and_line_vars()

    def get_rotation_angles(self):
        """Gets the rotation angles from the rotation matrix in the current
        PlottingVariables object"""
        return irotate(self.rotation)

    def get_rotation_angles_string(self, digits=5):
        fmt = '%.{:d}f'.format(digits)
        angles = self.get_rotation_angles()
        outstring = (fmt + 'x, ' + fmt + 'y, ' + fmt + 'z') % (angles)
        return outstring

    def update_patch_and_line_vars(self):
        """Updates all the line and path stuff that is still inobvious, this
        function should be deprecated if nobody can understand why it's features
        exist."""
        cell = self.atoms.get_cell()
        disp = self.atoms.get_celldisp().flatten()
        positions = self.atoms.get_positions()

        if self.show_unit_cell in (
            SHOW_CELL, SHOW_CELL_AND_FIT_TO_ALL, SHOW_CELL_AND_FIT_TO_CELL):

            L, T, D = cell_to_lines(self, cell)
            cell_verts_in_atom_coords = get_cell_vertex_points(cell, disp)
            cell_vertices = self.to_image_plane_positions(
                cell_verts_in_atom_coords)
            T = update_line_order_for_atoms(L, T, D, self.atoms, self.radii)
            # D are a positions in the image plane,
            # not sure why it's setup like this
            D = (self.to_image_plane_positions(D) + self.offset)[:, :2]
            positions = np.concatenate((positions, L), axis=0)
        else:
            L = np.empty((0, 3))
            T = None
            D = None
            cell_vertices = None
        # just a rotations and scaling since offset is currently [0,0,0]
        positions = self.to_image_plane_positions(positions)
        self.positions = positions
        # list of 2D cell points in the imageplane without the offset
        self.D = D
        # integers, probably z-order for lines?
        self.T = T
        self.cell_vertices = cell_vertices

        # no displacement since it's a vector
        cell_vec_im = self.scale * self.atoms.get_cell() @ self.rotation
        self.cell = cell_vec_im

    def to_image_plane_positions(self, positions):
        """Converts atomic coordinates to image plane positions. The third
        coordinate is distance above/below the image plane"""
        im_positions = (positions @ self.rotation) * self.scale - self.offset
        return im_positions

    def to_atom_positions(self, im_positions):
        """Converts image plane positions to atomic coordinates."""
        positions = ((im_positions + self.offset) /
                     self.scale) @ self.rotation.T
        return positions

    def get_bbox_from_atoms(self, atoms, im_radii):
        """Uses supplied atoms and radii to compute the bounding box of the
        atoms in the image plane"""
        im_positions = self.to_image_plane_positions(atoms.get_positions())
        im_low = (im_positions - im_radii[:, None]).min(0)
        im_high = (im_positions + im_radii[:, None]).max(0)
        return np.array([im_low, im_high])

    def get_bbox_from_cell(self, cell, disp=(0.0, 0.0, 0.0)):
        """Uses supplied cell to compute the bounding box of the cell in the
        image plane"""
        displacement = np.array(disp)
        cell_verts_in_atom_coords = get_cell_vertex_points(cell, displacement)
        cell_vertices = self.to_image_plane_positions(cell_verts_in_atom_coords)
        im_low = cell_vertices.min(0)
        im_high = cell_vertices.max(0)
        return np.array([im_low, im_high])

    def get_image_plane_center(self):
        return self.to_atom_positions(np.array([self.w / 2, self.h / 2, 0]))

    def get_atom_direction(self, direction):
        c0 = self.to_atom_positions([0, 0, 0])  # self.get_image_plane_center()
        c1 = self.to_atom_positions(direction)
        atom_direction = c1 - c0
        return atom_direction / np.linalg.norm(atom_direction)

    def get_camera_direction(self):
        """Returns vector pointing away from camera toward atoms/cell in atomic
        coordinates"""
        return self.get_atom_direction([0, 0, -1])

    def get_camera_up(self):
        """Returns the image plane up direction in atomic coordinates"""
        return self.get_atom_direction([0, 1, 0])

    def get_camera_right(self):
        """Returns the image plane right direction in atomic coordinates"""
        return self.get_atom_direction([1, 0, 0])


def cell_to_lines(writer, cell):
    # XXX this needs to be updated for cell vectors that are zero.
    # Cannot read the code though!  (What are T and D? nn?)
    nlines = 0
    nsegments = []
    for c in range(3):
        d = np.sqrt((cell[c]**2).sum())
        n = max(2, int(d / 0.3))
        nsegments.append(n)
        nlines += 4 * n

    positions = np.empty((nlines, 3))
    T = np.empty(nlines, int)
    D = np.zeros((3, 3))

    n1 = 0
    for c in range(3):
        n = nsegments[c]
        dd = cell[c] / (4 * n - 2)
        D[c] = dd
        P = np.arange(1, 4 * n + 1, 4)[:, None] * dd
        T[n1:] = c
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            n2 = n1 + n
            positions[n1:n2] = P + i * cell[c - 2] + j * cell[c - 1]
            n1 = n2

    return positions, T, D


def make_patch_list(writer):
    from matplotlib.patches import Circle, PathPatch, Wedge
    from matplotlib.path import Path

    indices = writer.positions[:, 2].argsort()
    patch_list = []
    for a in indices:
        xy = writer.positions[a, :2]
        if a < writer.natoms:
            r = writer.d[a] / 2
            if writer.frac_occ:
                site_occ = writer.occs[str(writer.tags[a])]
                # first an empty circle if a site is not fully occupied
                if (np.sum([v for v in site_occ.values()])) < 1.0:
                    # fill with white
                    fill = '#ffffff'
                    patch = Circle(xy, r, facecolor=fill,
                                   edgecolor='black')
                    patch_list.append(patch)

                start = 0
                # start with the dominant species
                for sym, occ in sorted(site_occ.items(),
                                       key=lambda x: x[1],
                                       reverse=True):
                    if np.round(occ, decimals=4) == 1.0:
                        patch = Circle(xy, r, facecolor=writer.colors[a],
                                       edgecolor='black')
                        patch_list.append(patch)
                    else:
                        # jmol colors for the moment
                        extent = 360. * occ
                        patch = Wedge(
                            xy, r, start, start + extent,
                            facecolor=default_colors[atomic_numbers[sym]],
                            edgecolor='black')
                        patch_list.append(patch)
                        start += extent

            else:
                if ((xy[1] + r > 0) and (xy[1] - r < writer.h) and
                        (xy[0] + r > 0) and (xy[0] - r < writer.w)):
                    patch = Circle(xy, r, facecolor=writer.colors[a],
                                   edgecolor='black')
                    patch_list.append(patch)
        else:
            a -= writer.natoms
            c = writer.T[a]
            if c != -1:
                hxy = writer.D[c]
                patch = PathPatch(Path((xy + hxy, xy - hxy)))
                patch_list.append(patch)
    return patch_list


class ImageChunk:
    """Base Class for a file chunk which contains enough information to
    reconstruct an atoms object."""

    def build(self, **kwargs):
        """Construct the atoms object from the stored information,
        and return it"""


class ImageIterator:
    """Iterate over chunks, to return the corresponding Atoms objects.
    Will only build the atoms objects which corresponds to the requested
    indices when called.
    Assumes ``ichunks`` is in iterator, which returns ``ImageChunk``
    type objects. See extxyz.py:iread_xyz as an example.
    """

    def __init__(self, ichunks):
        self.ichunks = ichunks

    def __call__(self, fd: IO, index=None, **kwargs):
        if isinstance(index, str):
            index = string2index(index)

        if index is None or index == ':':
            index = slice(None, None, None)

        if not isinstance(index, (slice, str)):
            index = slice(index, (index + 1) or None)

        for chunk in self._getslice(fd, index):
            yield chunk.build(**kwargs)

    def _getslice(self, fd: IO, indices: slice):
        try:
            iterator = islice(self.ichunks(fd),
                              indices.start, indices.stop,
                              indices.step)
        except ValueError:
            # Negative indices.  Go through the whole thing to get the length,
            # which allows us to evaluate the slice, and then read it again
            if not hasattr(fd, 'seekable') or not fd.seekable():
                raise ValueError('Negative indices only supported for '
                                 'seekable streams')

            startpos = fd.tell()
            nchunks = 0
            for _ in self.ichunks(fd):
                nchunks += 1
            fd.seek(startpos)
            indices_tuple = indices.indices(nchunks)
            iterator = islice(self.ichunks(fd), *indices_tuple)
        return iterator


def verify_cell_for_export(cell, check_orthorhombric=True):
    """Function to verify if the cell size is defined and if the cell is

    Parameters:

    cell: cell object
        cell to be checked.

    check_orthorhombric: bool
        If True, check if the cell is orthorhombric, raise an ``ValueError`` if
        the cell is orthorhombric. If False, doesn't check if the cell is
        orthorhombric.

    Raise a ``ValueError`` if the cell if not suitable for export to mustem xtl
    file or prismatic/computem xyz format:
        - if cell is not orthorhombic (only when check_orthorhombric=True)
        - if cell size is not defined
    """

    if check_orthorhombric and not cell.orthorhombic:
        raise ValueError('To export to this format, the cell needs to be '
                         'orthorhombic.')
    if cell.rank < 3:
        raise ValueError('To export to this format, the cell size needs '
                         'to be set: current cell is {}.'.format(cell))


def verify_dictionary(atoms, dictionary, dictionary_name):
    """
    Verify a dictionary have a key for each symbol present in the atoms object.

    Parameters:

    dictionary: dict
        Dictionary to be checked.


    dictionary_name: dict
        Name of the dictionary to be displayed in the error message.

    cell: cell object
        cell to be checked.


    Raise a ``ValueError`` if the key doesn't match the atoms present in the
    cell.
    """
    # Check if we have enough key
    for key in set(atoms.symbols):
        if key not in dictionary:
            raise ValueError('Missing the {} key in the `{}` dictionary.'
                             ''.format(key, dictionary_name))


def segment_list(data, segment_size):
    """Segments a list into sublists of a specified size."""
    return [data[i:i + segment_size] for i in range(0, len(data), segment_size)]

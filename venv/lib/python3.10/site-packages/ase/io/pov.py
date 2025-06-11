# fmt: off

"""
Module for povray file format support.

See http://www.povray.org/ for details on the format.
"""
from collections.abc import Sequence
from os import unlink
from pathlib import Path
from subprocess import DEVNULL, check_call

import numpy as np

from ase import Atoms
from ase.constraints import FixAtoms
from ase.io.utils import PlottingVariables


def pa(array):
    """Povray array syntax"""
    return '<' + ', '.join(f"{x:>6.2f}" for x in tuple(array)) + '>'


def pc(array):
    """Povray color syntax"""
    if isinstance(array, str):
        return 'color ' + array
    if isinstance(array, float):
        return f'rgb <{array:.2f}>*3'.format(array)
    L = len(array)
    if L > 2 and L < 6:
        return f"rgb{'' if L == 3 else 't' if L == 4 else 'ft'} <" +\
            ', '.join(f"{x:.2f}" for x in tuple(array)) + '>'


def get_bondpairs(atoms, radius=1.1):
    """Get all pairs of bonding atoms

    Return all pairs of atoms which are closer than radius times the
    sum of their respective covalent radii.  The pairs are returned as
    tuples::

      (a, b, (i1, i2, i3))

    so that atoms a bonds to atom b displaced by the vector::

        _     _     _
      i c + i c + i c ,
       1 1   2 2   3 3

    where c1, c2 and c3 are the unit cell vectors and i1, i2, i3 are
    integers."""

    from ase.data import covalent_radii
    from ase.neighborlist import NeighborList
    cutoffs = radius * covalent_radii[atoms.numbers]
    nl = NeighborList(cutoffs=cutoffs, self_interaction=False)
    nl.update(atoms)
    bondpairs = []
    for a in range(len(atoms)):
        indices, offsets = nl.get_neighbors(a)
        bondpairs.extend([(a, a2, offset)
                          for a2, offset in zip(indices, offsets)])
    return bondpairs


def set_high_bondorder_pairs(bondpairs, high_bondorder_pairs=None):
    """Set high bondorder pairs

    Modify bondpairs list (from get_bondpairs((atoms)) to include high
    bondorder pairs.

    Parameters:
    -----------
    bondpairs: List of pairs, generated from get_bondpairs(atoms)
    high_bondorder_pairs: Dictionary of pairs with high bond orders
                          using the following format:
                          { ( a1, b1 ): ( offset1, bond_order1, bond_offset1),
                            ( a2, b2 ): ( offset2, bond_order2, bond_offset2),
                            ...
                          }
                          offset, bond_order, bond_offset are optional.
                          However, if they are provided, the 1st value is
                          offset, 2nd value is bond_order,
                          3rd value is bond_offset """

    if high_bondorder_pairs is None:
        high_bondorder_pairs = {}
    bondpairs_ = []
    for pair in bondpairs:
        (a, b) = (pair[0], pair[1])
        if (a, b) in high_bondorder_pairs.keys():
            bondpair = [a, b] + [item for item in high_bondorder_pairs[(a, b)]]
            bondpairs_.append(bondpair)
        elif (b, a) in high_bondorder_pairs.keys():
            bondpair = [a, b] + [item for item in high_bondorder_pairs[(b, a)]]
            bondpairs_.append(bondpair)
        else:
            bondpairs_.append(pair)
    return bondpairs_


class POVRAY:
    # These new styles were an attempt to port the old styles o the correct
    # gamma, many had or still have unphysical light properties inorder to
    # acheive a certain look.
    material_styles_dict = dict(
        simple='finish {phong 0.7 ambient 0.4 diffuse 0.55}',
        # In general, 'pale' doesn't conserve energy and can look
        # strange in many cases.
        pale=('finish {ambient 0.9 diffuse 0.30 roughness 0.001 '
              'specular 0.2 }'),
        intermediate=('finish {ambient 0.4 diffuse 0.6 specular 0.1 '
                      'roughness 0.04}'),
        vmd=(
            'finish {ambient 0.2 diffuse 0.80 phong 0.25 phong_size 10.0 '
            'specular 0.2 roughness 0.1}'),
        jmol=('finish {ambient 0.4 diffuse 0.6 specular 1 roughness 0.001 '
              'metallic}'),
        ase2=('finish {ambient 0.2 brilliance 3 diffuse 0.6 metallic '
              'specular 0.7 roughness 0.04 reflection 0.15}'),
        ase3=('finish {ambient 0.4 brilliance 2 diffuse 0.6 metallic '
              'specular 1.0 roughness 0.001 reflection 0.0}'),
        glass=('finish {ambient 0.4 diffuse 0.35 specular 1.0 '
               'roughness 0.001}'),
        glass2=('finish {ambient 0.3 diffuse 0.3 specular 1.0 '
                'reflection 0.25 roughness 0.001}'),
    )

    # These styles were made when assumed_gamma was 1.0 which gives poor color
    # reproduction, the correct gamma is 2.2 for the sRGB standard.
    material_styles_dict_old = dict(
        simple='finish {phong 0.7}',
        pale=('finish {ambient 0.5 diffuse 0.85 roughness 0.001 '
              'specular 0.200 }'),
        intermediate=('finish {ambient 0.3 diffuse 0.6 specular 0.1 '
                      'roughness 0.04}'),
        vmd=('finish {ambient 0.0 diffuse 0.65 phong 0.1 phong_size 40.0 '
             'specular 0.5 }'),
        jmol=('finish {ambient 0.2 diffuse 0.6 specular 1 roughness 0.001 '
              'metallic}'),
        ase2=('finish {ambient 0.05 brilliance 3 diffuse 0.6 metallic '
              'specular 0.7 roughness 0.04 reflection 0.15}'),
        ase3=('finish {ambient 0.15 brilliance 2 diffuse 0.6 metallic '
              'specular 1.0 roughness 0.001 reflection 0.0}'),
        glass=('finish {ambient 0.05 diffuse 0.3 specular 1.0 '
               'roughness 0.001}'),
        glass2=('finish {ambient 0.01 diffuse 0.3 specular 1.0 '
                'reflection 0.25 roughness 0.001}'),
    )

    def __init__(self, cell, cell_vertices, positions, diameters, colors,
                 image_width, image_height, constraints=(), isosurfaces=[],
                 display=False, pause=True, transparent=True, canvas_width=None,
                 canvas_height=None, camera_dist=50., image_plane=None,
                 camera_type='orthographic', point_lights=[],
                 area_light=[(2., 3., 40.), 'White', .7, .7, 3, 3],
                 background='White', textures=None, transmittances=None,
                 depth_cueing=False, cue_density=5e-3,
                 celllinewidth=0.05, bondlinewidth=0.10, bondatoms=[],
                 exportconstraints=False):
        """
        # x, y is the image plane, z is *out* of the screen
        cell: ase.cell
            cell object
        cell_vertices: 2-d numpy array
            contains the 8 vertices of the cell, each with three coordinates
        positions: 2-d numpy array
            number of atoms length array with three coordinates for positions
        diameters: 1-d numpy array
            diameter of atoms (in order with positions)
        colors: list of str
            colors of atoms (in order with positions)
        image_width: float
            image width in pixels
        image_height: float
            image height in pixels
        constraints: Atoms.constraints
            constraints to be visualized
        isosurfaces: list of POVRAYIsosurface
            composite object to write/render POVRAY isosurfaces
        display: bool
            display while rendering
        pause: bool
            pause when done rendering (only if display)
        transparent: bool
            make background transparent
        canvas_width: int
            width of canvas in pixels
        canvas_height: int
            height of canvas in pixels
        camera_dist: float
            distance from camera to front atom
        image_plane: float
            distance from front atom to image plane
        camera_type: str
            if 'orthographic' perspective, ultra_wide_angle
        point_lights: list of 2-element sequences
            like [[loc1, color1], [loc2, color2],...]
        area_light: 3-element sequence of location (3-tuple), color (str),
                   width (float), height (float),
                   Nlamps_x (int), Nlamps_y (int)
            example [(2., 3., 40.), 'White', .7, .7, 3, 3]
        background: str
            color specification, e.g., 'White'
        textures: list of str
            length of atoms list of texture names
        transmittances: list of floats
            length of atoms list of transmittances of the atoms
        depth_cueing: bool
            whether or not to use depth cueing a.k.a. fog
            use with care - adjust the camera_distance to be closer
        cue_density: float
            if there is depth_cueing, how dense is it (how dense is the fog)
        celllinewidth: float
            radius of the cylinders representing the cell (Ang.)
        bondlinewidth: float
            radius of the cylinders representing bonds (Ang.)
        bondatoms: list of lists (polymorphic)
            [[atom1, atom2], ... ] pairs of bonding atoms
             For bond order > 1 = [[atom1, atom2, offset,
                                    bond_order, bond_offset],
                                   ... ]
             bond_order: 1, 2, 3 for single, double,
                          and triple bond
             bond_offset: vector for shifting bonds from
                           original position. Coordinates are
                           in Angstrom unit.
        exportconstraints: bool
            honour FixAtoms and mark?"""

        # attributes from initialization
        self.area_light = area_light
        self.background = background
        self.bondatoms = bondatoms
        self.bondlinewidth = bondlinewidth
        self.camera_dist = camera_dist
        self.camera_type = camera_type
        self.celllinewidth = celllinewidth
        self.cue_density = cue_density
        self.depth_cueing = depth_cueing
        self.display = display
        self.exportconstraints = exportconstraints
        self.isosurfaces = isosurfaces
        self.pause = pause
        self.point_lights = point_lights
        self.textures = textures
        self.transmittances = transmittances
        self.transparent = transparent

        self.image_width = image_width
        self.image_height = image_height
        self.colors = colors
        self.cell = cell
        self.diameters = diameters

        # calculations based on passed inputs

        z0 = positions[:, 2].max()
        self.offset = (image_width / 2, image_height / 2, z0)
        self.positions = positions - self.offset

        if cell_vertices is not None:
            self.cell_vertices = cell_vertices - self.offset
            self.cell_vertices.shape = (2, 2, 2, 3)
        else:
            self.cell_vertices = None

        ratio = float(self.image_width) / self.image_height
        if canvas_width is None:
            if canvas_height is None:
                self.canvas_width = min(self.image_width * 15, 640)
                self.canvas_height = min(self.image_height * 15, 640)
            else:
                self.canvas_width = canvas_height * ratio
                self.canvas_height = canvas_height
        elif canvas_height is None:
            self.canvas_width = canvas_width
            self.canvas_height = self.canvas_width / ratio
        else:
            raise RuntimeError("Can't set *both* width and height!")

        # Distance to image plane from camera
        if image_plane is None:
            if self.camera_type == 'orthographic':
                self.image_plane = 1 - self.camera_dist
            else:
                self.image_plane = 0
        self.image_plane += self.camera_dist

        self.constrainatoms = []
        for c in constraints:
            if isinstance(c, FixAtoms):
                # self.constrainatoms.extend(c.index) # is this list-like?
                for n, i in enumerate(c.index):
                    self.constrainatoms += [i]

    @classmethod
    def from_PlottingVariables(cls, pvars, **kwargs):
        cell = pvars.cell
        cell_vertices = pvars.cell_vertices
        if 'colors' in kwargs:
            colors = kwargs.pop('colors')
        else:
            colors = pvars.colors
        diameters = pvars.d
        image_height = pvars.h
        image_width = pvars.w
        positions = pvars.positions
        constraints = pvars.constraints
        return cls(cell=cell, cell_vertices=cell_vertices, colors=colors,
                   constraints=constraints, diameters=diameters,
                   image_height=image_height, image_width=image_width,
                   positions=positions, **kwargs)

    @classmethod
    def from_atoms(cls, atoms, **kwargs):
        return cls.from_plotting_variables(
            PlottingVariables(atoms, scale=1.0), **kwargs)

    def write_ini(self, path):
        """Write ini file."""

        ini_str = f"""\
Input_File_Name={path.with_suffix('.pov').name}
Output_to_File=True
Output_File_Type=N
Output_Alpha={'on' if self.transparent else 'off'}
; if you adjust Height, and width, you must preserve the ratio
; Width / Height = {self.canvas_width / self.canvas_height:f}
Width={self.canvas_width}
Height={self.canvas_height}
Antialias=True
Antialias_Threshold=0.1
Display={self.display}
Display_Gamma=2.2
Pause_When_Done={self.pause}
Verbose=False
"""
        with open(path, 'w') as fd:
            fd.write(ini_str)
        return path

    def write_pov(self, path):
        """Write pov file."""

        point_lights = '\n'.join(f"light_source {{{pa(loc)} {pc(rgb)}}}"
                                 for loc, rgb in self.point_lights)

        area_light = ''
        if self.area_light is not None:
            loc, color, width, height, nx, ny = self.area_light
            area_light += f"""\nlight_source {{{pa(loc)} {pc(color)}
  area_light <{width:.2f}, 0, 0>, <0, {height:.2f}, 0>, {nx:n}, {ny:n}
  adaptive 1 jitter}}"""

        fog = ''
        if self.depth_cueing and (self.cue_density >= 1e-4):
            # same way vmd does it
            if self.cue_density > 1e4:
                # larger does not make any sense
                dist = 1e-4
            else:
                dist = 1. / self.cue_density
            fog += f'fog {{fog_type 1 distance {dist:.4f} '\
                   f'color {pc(self.background)}}}'

        mat_style_keys = (f'#declare {k} = {v}'
                          for k, v in self.material_styles_dict.items())
        mat_style_keys = '\n'.join(mat_style_keys)

        # Draw unit cell
        cell_vertices = ''
        if self.cell_vertices is not None:
            for c in range(3):
                for j in ([0, 0], [1, 0], [1, 1], [0, 1]):
                    p1 = self.cell_vertices[tuple(j[:c]) + (0,) + tuple(j[c:])]
                    p2 = self.cell_vertices[tuple(j[:c]) + (1,) + tuple(j[c:])]

                    distance = np.linalg.norm(p2 - p1)
                    if distance < 1e-12:
                        continue

                    cell_vertices += f'cylinder {{{pa(p1)}, {pa(p2)}, '\
                                     'Rcell pigment {Black}}\n'
                    # all strings are f-strings for consistency
            cell_vertices = cell_vertices.strip('\n')

        # Draw atoms
        a = 0
        atoms = ''
        for loc, dia, col in zip(self.positions, self.diameters, self.colors):
            tex = 'ase3'
            trans = 0.
            if self.textures is not None:
                tex = self.textures[a]
            if self.transmittances is not None:
                trans = self.transmittances[a]
            atoms += f'atom({pa(loc)}, {dia / 2.:.2f}, {pc(col)}, '\
                     f'{trans}, {tex}) // #{a:n}\n'
            a += 1
        atoms = atoms.strip('\n')

        # Draw atom bonds
        bondatoms = ''
        for pair in self.bondatoms:
            # Make sure that each pair has 4 componets: a, b, offset,
            #                                           bond_order, bond_offset
            # a, b: atom index to draw bond
            # offset: original meaning to make offset for mid-point.
            # bond_oder: if not supplied, set it to 1 (single bond).
            #            It can be  1, 2, 3, corresponding to single,
            #            double, triple bond
            # bond_offset: displacement from original bond position.
            #              Default is (bondlinewidth, bondlinewidth, 0)
            #              for bond_order > 1.
            if len(pair) == 2:
                a, b = pair
                offset = (0, 0, 0)
                bond_order = 1
                bond_offset = (0, 0, 0)
            elif len(pair) == 3:
                a, b, offset = pair
                bond_order = 1
                bond_offset = (0, 0, 0)
            elif len(pair) == 4:
                a, b, offset, bond_order = pair
                bond_offset = (self.bondlinewidth, self.bondlinewidth, 0)
            elif len(pair) > 4:
                a, b, offset, bond_order, bond_offset = pair
            else:
                raise RuntimeError('Each list in bondatom must have at least '
                                   '2 entries. Error at %s' % pair)

            if len(offset) != 3:
                raise ValueError('offset must have 3 elements. '
                                 'Error at %s' % pair)
            if len(bond_offset) != 3:
                raise ValueError('bond_offset must have 3 elements. '
                                 'Error at %s' % pair)
            if bond_order not in [0, 1, 2, 3]:
                raise ValueError('bond_order must be either 0, 1, 2, or 3. '
                                 'Error at %s' % pair)

            # Up to here, we should have all a, b, offset, bond_order,
            # bond_offset for all bonds.

            # Rotate bond_offset so that its direction is 90 deg. off the bond
            # Utilize Atoms object to rotate
            if bond_order > 1 and np.linalg.norm(bond_offset) > 1.e-9:
                tmp_atoms = Atoms('H3')
                tmp_atoms.set_cell(self.cell)
                tmp_atoms.set_positions([
                    self.positions[a],
                    self.positions[b],
                    self.positions[b] + np.array(bond_offset),
                ])
                tmp_atoms.center()
                tmp_atoms.set_angle(0, 1, 2, 90)
                bond_offset = tmp_atoms[2].position - tmp_atoms[1].position

            R = np.dot(offset, self.cell)
            mida = 0.5 * (self.positions[a] + self.positions[b] + R)
            midb = 0.5 * (self.positions[a] + self.positions[b] - R)
            if self.textures is not None:
                texa = self.textures[a]
                texb = self.textures[b]
            else:
                texa = texb = 'ase3'

            if self.transmittances is not None:
                transa = self.transmittances[a]
                transb = self.transmittances[b]
            else:
                transa = transb = 0.

            # draw bond, according to its bond_order.
            # bond_order == 0: No bond is plotted
            # bond_order == 1: use original code
            # bond_order == 2: draw two bonds, one is shifted by bond_offset/2,
            #                  and another is shifted by -bond_offset/2.
            # bond_order == 3: draw two bonds, one is shifted by bond_offset,
            #                  and one is shifted by -bond_offset, and the
            #                  other has no shift.
            # To shift the bond, add the shift to the first two coordinate in
            # write statement.

            posa = self.positions[a]
            posb = self.positions[b]
            cola = self.colors[a]
            colb = self.colors[b]

            if bond_order == 1:
                draw_tuples = (
                    (posa, mida, cola, transa, texa),
                    (posb, midb, colb, transb, texb))

            elif bond_order == 2:
                bs = [x / 2 for x in bond_offset]
                draw_tuples = (
                    (posa - bs, mida - bs, cola, transa, texa),
                    (posb - bs, midb - bs, colb, transb, texb),
                    (posa + bs, mida + bs, cola, transa, texa),
                    (posb + bs, midb + bs, colb, transb, texb))

            elif bond_order == 3:
                bs = bond_offset
                draw_tuples = (
                    (posa, mida, cola, transa, texa),
                    (posb, midb, colb, transb, texb),
                    (posa + bs, mida + bs, cola, transa, texa),
                    (posb + bs, midb + bs, colb, transb, texb),
                    (posa - bs, mida - bs, cola, transa, texa),
                    (posb - bs, midb - bs, colb, transb, texb))

            bondatoms += ''.join(f'cylinder {{{pa(p)}, '
                                 f'{pa(m)}, Rbond texture{{pigment '
                                 f'{{color {pc(c)} '
                                 f'transmit {tr}}} finish{{{tx}}}}}}}\n'
                                 for p, m, c, tr, tx in
                                 draw_tuples)

        bondatoms = bondatoms.strip('\n')

        # Draw constraints if requested
        constraints = ''
        if self.exportconstraints:
            for a in self.constrainatoms:
                dia = self.diameters[a]
                loc = self.positions[a]
                trans = 0.0
                if self.transmittances is not None:
                    trans = self.transmittances[a]
                constraints += f'constrain({pa(loc)}, {dia / 2.:.2f}, Black, '\
                    f'{trans}, {tex}) // #{a:n} \n'
        constraints = constraints.strip('\n')

        pov = f"""#version 3.6;
#include "colors.inc"
#include "finish.inc"

global_settings {{assumed_gamma 2.2 max_trace_level 6}}
background {{{pc(self.background)}{' transmit 1.0' if self.transparent else ''}}}
camera {{{self.camera_type}
  right -{self.image_width:.2f}*x up {self.image_height:.2f}*y
  direction {self.image_plane:.2f}*z
  location <0,0,{self.camera_dist:.2f}> look_at <0,0,0>}}
{point_lights}
{area_light if area_light != '' else '// no area light'}
{fog if fog != '' else '// no fog'}
{mat_style_keys}
#declare Rcell = {self.celllinewidth:.3f};
#declare Rbond = {self.bondlinewidth:.3f};

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{{LOC, R texture{{pigment{{color COL transmit TRANS}} finish{{FIN}}}}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{{torus{{R, Rcell rotate 45*z texture{{pigment{{color COL transmit TRANS}} finish{{FIN}}}}}}
     torus{{R, Rcell rotate -45*z texture{{pigment{{color COL transmit TRANS}} finish{{FIN}}}}}}
     translate LOC}}
#end

{cell_vertices if cell_vertices != '' else '// no cell vertices'}
{atoms}
{bondatoms}
{constraints if constraints != '' else '// no constraints'}
"""  # noqa: E501

        with open(path, 'w') as fd:
            fd.write(pov)

        return path

    def write(self, pov_path):
        pov_path = require_pov(pov_path)
        ini_path = pov_path.with_suffix('.ini')
        self.write_ini(ini_path)
        self.write_pov(pov_path)
        if self.isosurfaces is not None:
            with open(pov_path, 'a') as fd:
                for iso in self.isosurfaces:
                    fd.write(iso.format_mesh())
        return POVRAYInputs(ini_path)


def require_pov(path):
    path = Path(path)
    if path.suffix != '.pov':
        raise ValueError(f'Expected .pov path, got {path}')
    return path


class POVRAYInputs:
    def __init__(self, path):
        self.path = path

    def render(self, povray_executable='povray', stderr=DEVNULL,
               clean_up=False):
        cmd = [povray_executable, str(self.path)]

        check_call(cmd, stderr=stderr)
        png_path = self.path.with_suffix('.png').absolute()
        if not png_path.is_file():
            raise RuntimeError(f'Povray left no output PNG file "{png_path}"')

        if clean_up:
            unlink(self.path)
            unlink(self.path.with_suffix('.pov'))

        return png_path


class POVRAYIsosurface:
    def __init__(self, density_grid, cut_off, cell, cell_origin,
                 closed_edges=False, gradient_ascending=False,
                 color=(0.85, 0.80, 0.25, 0.2), material='ase3'):
        """
        density_grid: 3D float ndarray
            A regular grid on that spans the cell. The first dimension
            corresponds to the first cell vector and so on.
        cut_off: float
            The density value of the isosurface.
        cell: 2D float ndarray or ASE cell object
            The 3 vectors which give the cell's repetition
        cell_origin: 4 float tuple
            The cell origin as used by POVRAY object
        closed_edges: bool
            Setting this will fill in isosurface edges at the cell boundaries.
            Filling in the edges can help with visualizing
            highly porous structures.
        gradient_ascending: bool
            Lets you pick the area you want to enclose, i.e., should the denser
            or less dense area be filled in.
        color: povray color string, float, or float tuple
            1 float is interpreted as grey scale, a 3 float tuple is rgb,
            4 float tuple is rgbt, and 5 float tuple is rgbft, where
            t is transmission fraction and f is filter fraction.
            Named Povray colors are set in colors.inc
            (http://wiki.povray.org/content/Reference:Colors.inc)
        material: string
            Can be a finish macro defined by POVRAY.material_styles
            or a full Povray material {...} specification. Using a
            full material specification willoverride the color parameter.
        """

        self.gradient_direction = 'ascent' if gradient_ascending else 'descent'
        self.color = color
        self.material = material
        self.closed_edges = closed_edges
        self._cut_off = cut_off

        if self.gradient_direction == 'ascent':
            cv = 2 * cut_off
        else:
            cv = 0

        if closed_edges:
            shape_old = density_grid.shape
            # since well be padding, we need to keep the data at origin
            cell_origin += -(1.0 / np.array(shape_old)) @ cell
            density_grid = np.pad(
                density_grid, pad_width=(
                    1,), mode='constant', constant_values=cv)
            shape_new = density_grid.shape
            s = np.array(shape_new) / np.array(shape_old)
            cell = cell @ np.diag(s)

        self.cell = cell
        self.cell_origin = cell_origin
        self.density_grid = density_grid
        self.spacing = tuple(1.0 / np.array(self.density_grid.shape))

        scaled_verts, faces, _normals, _values = self.compute_mesh(
            self.density_grid,
            self.cut_off,
            self.spacing,
            self.gradient_direction)

        # The verts are scaled by default, this is the super easy way of
        # distributing them in real space but it's easier to do affine
        # transformations/rotations on a unit cube so I leave it like that
        # verts = scaled_verts.dot(self.cell)
        self.verts = scaled_verts
        self.faces = faces

    @property
    def cut_off(self):
        return self._cut_off

    @cut_off.setter
    def cut_off(self, value):
        raise Exception("Use the set_cut_off method")

    def set_cut_off(self, value):
        self._cut_off = value

        if self.gradient_direction == 'ascent':
            cv = 2 * self.cut_off
        else:
            cv = 0

        if self.closed_edges:
            shape_old = self.density_grid.shape
            # since well be padding, we need to keep the data at origin
            self.cell_origin += -(1.0 / np.array(shape_old)) @ self.cell
            self.density_grid = np.pad(
                self.density_grid, pad_width=(
                    1,), mode='constant', constant_values=cv)
            shape_new = self.density_grid.shape
            s = np.array(shape_new) / np.array(shape_old)
            self.cell = self.cell @ np.diag(s)

        self.spacing = tuple(1.0 / np.array(self.density_grid.shape))

        scaled_verts, faces, _, _ = self.compute_mesh(
            self.density_grid,
            self.cut_off,
            self.spacing,
            self.gradient_direction)

        self.verts = scaled_verts
        self.faces = faces

    @classmethod
    def from_POVRAY(cls, povray, density_grid, cut_off, **kwargs):
        return cls(cell=povray.cell,
                   cell_origin=povray.cell_vertices[0, 0, 0],
                   density_grid=density_grid,
                   cut_off=cut_off, **kwargs)

    @staticmethod
    def wrapped_triples_section(triple_list,
                                triple_format="<{:f}, {:f}, {:f}>".format,
                                triples_per_line=4):

        triples = [triple_format(*x) for x in triple_list]
        n = len(triples)
        s = ''
        tpl = triples_per_line
        c = 0

        while c < n - tpl:
            c += tpl
            s += '\n     '
            s += ', '.join(triples[c - tpl:c])
        s += '\n    '
        s += ', '.join(triples[c:])
        return s

    @staticmethod
    def compute_mesh(density_grid, cut_off, spacing, gradient_direction):
        """

        Import statement is in this method and not file header
        since few users will use isosurface rendering.

        Returns scaled_verts, faces, normals, values. See skimage docs.

        """

        # marching_cubes name was changed in skimage v0.19
        try:
            # New skimage
            from skimage.measure import marching_cubes
        except ImportError:
            # Old skimage (remove at some point)
            from skimage.measure import marching_cubes_lewiner as marching_cubes

        return marching_cubes(
            density_grid,
            level=cut_off,
            spacing=spacing,
            gradient_direction=gradient_direction,
            allow_degenerate=False)

    def format_mesh(self):
        """Returns a formatted data output for POVRAY files

        Example:
        material = '''
          material { // This material looks like pink jelly
            texture {
              pigment { rgbt <0.8, 0.25, 0.25, 0.5> }
              finish{ diffuse 0.85 ambient 0.99 brilliance 3 specular 0.5 roughness 0.001
                reflection { 0.05, 0.98 fresnel on exponent 1.5 }
                conserve_energy
              }
            }
            interior { ior 1.3 }
          }
          photons {
              target
              refraction on
              reflection on
              collect on
          }'''
        """  # noqa: E501

        if self.material in POVRAY.material_styles_dict:
            material = f"""material {{
        texture {{
          pigment {{ {pc(self.color)} }}
          finish {{ {self.material} }}
        }}
      }}"""
        else:
            material = self.material

        # Start writing the mesh2
        vertex_vectors = self.wrapped_triples_section(
            triple_list=self.verts,
            triple_format="<{:f}, {:f}, {:f}>".format,
            triples_per_line=4)

        face_indices = self.wrapped_triples_section(
            triple_list=self.faces,
            triple_format="<{:n}, {:n}, {:n}>".format,
            triples_per_line=5)

        cell = self.cell
        cell_or = self.cell_origin
        mesh2 = f"""\n\nmesh2 {{
    vertex_vectors {{  {len(self.verts):n},
    {vertex_vectors}
    }}
    face_indices {{ {len(self.faces):n},
    {face_indices}
    }}
{material if material != '' else '// no material'}
  matrix < {cell[0][0]:f}, {cell[0][1]:f}, {cell[0][2]:f},
           {cell[1][0]:f}, {cell[1][1]:f}, {cell[1][2]:f},
           {cell[2][0]:f}, {cell[2][1]:f}, {cell[2][2]:f},
           {cell_or[0]:f}, {cell_or[1]:f}, {cell_or[2]:f}>
    }}
    """
        return mesh2


def pop_deprecated(dct, name):
    import warnings
    if name in dct:
        del dct[name]
        warnings.warn(f'The "{name}" keyword of write_pov() is deprecated '
                      'and has no effect; this will raise an error in the '
                      'future.', FutureWarning)


def write_pov(filename, atoms, *,
              povray_settings=None, isosurface_data=None,
              **generic_projection_settings):

    for name in ['run_povray', 'povray_path', 'stderr', 'extras']:
        pop_deprecated(generic_projection_settings, name)

    if povray_settings is None:
        povray_settings = {}

    pvars = PlottingVariables(atoms, scale=1.0, **generic_projection_settings)
    pov_obj = POVRAY.from_PlottingVariables(pvars, **povray_settings)

    if isosurface_data is None:
        isosurface_data = []
    elif not isinstance(isosurface_data, Sequence):
        isosurface_data = [isosurface_data]

    isosurfaces = []
    for isodata in isosurface_data:
        if isinstance(isodata, POVRAYIsosurface):
            iso = isodata
        else:
            iso = POVRAYIsosurface.from_POVRAY(pov_obj, **isodata)
        isosurfaces.append(iso)
    pov_obj.isosurfaces = isosurfaces

    return pov_obj.write(filename)

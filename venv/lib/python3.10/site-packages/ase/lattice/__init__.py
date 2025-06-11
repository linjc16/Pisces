# fmt: off

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from ase.cell import Cell
from ase.dft.kpoints import BandPath, parse_path_string, sc_special_points
from ase.utils import pbc2pbc

_degrees = np.pi / 180


class BravaisLattice(ABC):
    """Represent Bravais lattices and data related to the Brillouin zone.

    There are 14 3D Bravais classes: CUB, FCC, BCC, ..., and TRI, and
    five 2D classes.

    Each class stores basic static information:

    >>> from ase.lattice import FCC, MCL
    >>> FCC.name
    'FCC'
    >>> FCC.longname
    'face-centred cubic'
    >>> FCC.pearson_symbol
    'cF'
    >>> MCL.parameters
    ('a', 'b', 'c', 'alpha')

    Each class can be instantiated with the specific lattice parameters
    that apply to that lattice:

    >>> MCL(3, 4, 5, 80)
    MCL(a=3, b=4, c=5, alpha=80)

    """
    # These parameters can be set by the @bravais decorator for a subclass.
    # (We could also use metaclasses to do this, but that's more abstract)
    name = None  # e.g. 'CUB', 'BCT', 'ORCF', ...
    longname = None  # e.g. 'cubic', 'body-centred tetragonal', ...
    parameters = None  # e.g. ('a', 'c')
    variants = None  # e.g. {'BCT1': <variant object>,
    #                        'BCT2': <variant object>}
    ndim = None

    def __init__(self, **kwargs):
        p = {}
        eps = kwargs.pop('eps', 2e-4)
        for k, v in kwargs.items():
            p[k] = float(v)
        assert set(p) == set(self.parameters)
        self._parameters = p
        self._eps = eps

        if len(self.variants) == 1:
            # If there's only one it has the same name as the lattice type
            self._variant = self.variants[self.name]
        else:
            name = self._variant_name(**self._parameters)
            self._variant = self.variants[name]

    @property
    def variant(self) -> str:
        """Return name of lattice variant.

        >>> from ase.lattice import BCT

        >>> BCT(3, 5).variant
        'BCT2'
        """
        return self._variant.name

    def __getattr__(self, name: str):
        if name in self._parameters:
            return self._parameters[name]
        return self.__getattribute__(name)  # Raises error

    def vars(self) -> Dict[str, float]:
        """Get parameter names and values of this lattice as a dictionary."""
        return dict(self._parameters)

    def conventional(self) -> 'BravaisLattice':
        """Get the conventional cell corresponding to this lattice."""
        cls = bravais_lattices[self.conventional_cls]
        return cls(**self._parameters)

    def tocell(self) -> Cell:
        """Return this lattice as a :class:`~ase.cell.Cell` object."""
        cell = self._cell(**self._parameters)
        return Cell(cell)

    def cellpar(self) -> np.ndarray:
        """Get cell lengths and angles as array of length 6.

        See :func:`ase.geometry.Cell.cellpar`."""
        # (Just a brute-force implementation)
        cell = self.tocell()
        return cell.cellpar()

    @property
    def special_path(self) -> str:
        """Get default special k-point path for this lattice as a string.

        >>> BCT(3, 5).special_path
        'GXYSGZS1NPY1Z,XP'
        """
        return self._variant.special_path

    @property
    def special_point_names(self) -> List[str]:
        """Return all special point names as a list of strings.

        >>> from ase.lattice import BCT

        >>> BCT(3, 5).special_point_names
        ['G', 'N', 'P', 'S', 'S1', 'X', 'Y', 'Y1', 'Z']
        """
        labels = parse_path_string(self._variant.special_point_names)
        assert len(labels) == 1  # list of lists
        return labels[0]

    def get_special_points_array(self) -> np.ndarray:
        """Return all special points for this lattice as an array.

        Ordering is consistent with special_point_names."""
        if self._variant.special_points is not None:
            # Fixed dictionary of special points
            d = self.get_special_points()
            labels = self.special_point_names
            assert len(d) == len(labels)
            points = np.empty((len(d), 3))
            for i, label in enumerate(labels):
                points[i] = d[label]
            return points

        # Special points depend on lattice parameters:
        points = self._special_points(variant=self._variant,
                                      **self._parameters)
        assert len(points) == len(self.special_point_names)
        return np.array(points)

    def get_special_points(self) -> Dict[str, np.ndarray]:
        """Return a dictionary of named special k-points for this lattice."""
        if self._variant.special_points is not None:
            return self._variant.special_points

        labels = self.special_point_names
        points = self.get_special_points_array()

        return dict(zip(labels, points))

    def plot_bz(self, path=None, special_points=None, **plotkwargs):
        """Plot the reciprocal cell and default bandpath."""
        # Create a generic bandpath (no interpolated kpoints):
        bandpath = self.bandpath(path=path, special_points=special_points,
                                 npoints=0)
        return bandpath.plot(dimension=self.ndim, **plotkwargs)

    def bandpath(self, path=None, npoints=None, special_points=None,
                 density=None) -> BandPath:
        """Return a :class:`~ase.dft.kpoints.BandPath` for this lattice.

        See :meth:`ase.cell.Cell.bandpath` for description of parameters.

        >>> from ase.lattice import BCT

        >>> BCT(3, 5).bandpath()
        BandPath(path='GXYSGZS1NPY1Z,XP', cell=[3x3], \
special_points={GNPSS1XYY1Z}, kpts=[51x3])

        .. note:: This produces the standard band path following AFlow
           conventions.  If your cell does not follow this convention,
           you will need :meth:`ase.cell.Cell.bandpath` instead or
           the kpoints may not correspond to your particular cell.

        """
        if special_points is None:
            special_points = self.get_special_points()

        if path is None:
            path = self._variant.special_path
        elif not isinstance(path, str):
            from ase.dft.kpoints import resolve_custom_points
            path, special_points = resolve_custom_points(path,
                                                         special_points,
                                                         self._eps)

        cell = self.tocell()
        bandpath = BandPath(cell=cell, path=path,
                            special_points=special_points)
        return bandpath.interpolate(npoints=npoints, density=density)

    @abstractmethod
    def _cell(self, **kwargs):
        """Return a Cell object from this Bravais lattice.

        Arguments are the dictionary of Bravais parameters."""

    def _special_points(self, **kwargs):
        """Return the special point coordinates as an npoints x 3 sequence.

        Subclasses typically return a nested list.

        Ordering must be same as kpoint labels.

        Arguments are the dictionary of Bravais parameters and the variant."""
        raise NotImplementedError

    def _variant_name(self, **kwargs):
        """Return the name (e.g. 'ORCF3') of variant.

        Arguments will be the dictionary of Bravais parameters."""
        raise NotImplementedError

    def __format__(self, spec):
        tokens = []
        if not spec:
            spec = '.6g'
        template = f'{{}}={{:{spec}}}'

        for name in self.parameters:
            value = self._parameters[name]
            tokens.append(template.format(name, value))
        return '{}({})'.format(self.name, ', '.join(tokens))

    def __str__(self) -> str:
        return self.__format__('')

    def __repr__(self) -> str:
        return self.__format__('.20g')

    def description(self) -> str:
        """Return complete description of lattice and Brillouin zone."""
        points = self.get_special_points()
        labels = self.special_point_names

        coordstring = '\n'.join(['    {:2s} {:7.4f} {:7.4f} {:7.4f}'
                                 .format(label, *points[label])
                                 for label in labels])

        string = """\
{repr}
  {variant}
  Special point coordinates:
{special_points}
""".format(repr=str(self),
           variant=self._variant,
           special_points=coordstring)
        return string

    @classmethod
    def type_description(cls):
        """Return complete description of this Bravais lattice type."""
        desc = """\
Lattice name: {name}
  Long name: {longname}
  Parameters: {parameters}
""".format(**vars(cls))

        chunks = [desc]
        for name in cls.variant_names:
            var = cls.variants[name]
            txt = str(var)
            lines = ['  ' + L for L in txt.splitlines()]
            lines.append('')
            chunks.extend(lines)

        return '\n'.join(chunks)


class Variant:
    variant_desc = """\
Variant name: {name}
  Special point names: {special_point_names}
  Default path: {special_path}
"""

    def __init__(self, name, special_point_names, special_path,
                 special_points=None):
        self.name = name
        self.special_point_names = special_point_names
        self.special_path = special_path
        if special_points is not None:
            special_points = dict(special_points)
            for key, value in special_points.items():
                special_points[key] = np.array(value)
        self.special_points = special_points
        # XXX Should make special_points available as a single array as well
        # (easier to transform that way)

    def __str__(self) -> str:
        return self.variant_desc.format(**vars(self))


bravais_names = []
bravais_lattices = {}
bravais_classes = {}


def bravaisclass(longname, crystal_family, lattice_system, pearson_symbol,
                 parameters, variants, ndim=3):
    """Decorator for Bravais lattice classes.

    This sets a number of class variables and processes the information
    about different variants into a list of Variant objects."""

    def decorate(cls):
        btype = cls.__name__
        cls.name = btype
        cls.longname = longname
        cls.crystal_family = crystal_family
        cls.lattice_system = lattice_system
        cls.pearson_symbol = pearson_symbol
        cls.parameters = tuple(parameters)
        cls.variant_names = []
        cls.variants = {}
        cls.ndim = ndim

        for [name, special_point_names, special_path,
             special_points] in variants:
            cls.variant_names.append(name)
            cls.variants[name] = Variant(name, special_point_names,
                                         special_path, special_points)

        # Register in global list and dictionary
        bravais_names.append(btype)
        bravais_lattices[btype] = cls
        bravais_classes[pearson_symbol] = cls
        return cls

    return decorate


# Common mappings of primitive to conventional cells:
_identity = np.identity(3, int)
_fcc_map = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
_bcc_map = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])


class UnconventionalLattice(ValueError):
    pass


class Cubic(BravaisLattice):
    """Abstract class for cubic lattices."""
    conventional_cls = 'CUB'

    def __init__(self, a):
        super().__init__(a=a)


@bravaisclass('primitive cubic', 'cubic', 'cubic', 'cP', 'a',
              [['CUB', 'GXRM', 'GXMGRX,MR', sc_special_points['cubic']]])
class CUB(Cubic):
    conventional_cellmap = _identity

    def _cell(self, a):
        return a * np.eye(3)


@bravaisclass('face-centred cubic', 'cubic', 'cubic', 'cF', 'a',
              [['FCC', 'GKLUWX', 'GXWKGLUWLK,UX', sc_special_points['fcc']]])
class FCC(Cubic):
    conventional_cellmap = _bcc_map

    def _cell(self, a):
        return 0.5 * np.array([[0., a, a], [a, 0, a], [a, a, 0]])


@bravaisclass('body-centred cubic', 'cubic', 'cubic', 'cI', 'a',
              [['BCC', 'GHPN', 'GHNGPH,PN', sc_special_points['bcc']]])
class BCC(Cubic):
    conventional_cellmap = _fcc_map

    def _cell(self, a):
        return 0.5 * np.array([[-a, a, a], [a, -a, a], [a, a, -a]])


@bravaisclass('primitive tetragonal', 'tetragonal', 'tetragonal', 'tP', 'ac',
              [['TET', 'GAMRXZ', 'GXMGZRAZ,XR,MA',
                sc_special_points['tetragonal']]])
class TET(BravaisLattice):
    conventional_cls = 'TET'
    conventional_cellmap = _identity

    def __init__(self, a, c):
        super().__init__(a=a, c=c)

    def _cell(self, a, c):
        return np.diag(np.array([a, a, c]))


# XXX in BCT2 we use S for Sigma.
# Also in other places I think
@bravaisclass('body-centred tetragonal', 'tetragonal', 'tetragonal', 'tI',
              'ac',
              [['BCT1', 'GMNPXZZ1', 'GXMGZPNZ1M,XP', None],
               ['BCT2', 'GNPSS1XYY1Z', 'GXYSGZS1NPY1Z,XP', None]])
class BCT(BravaisLattice):
    conventional_cls = 'TET'
    conventional_cellmap = _fcc_map

    def __init__(self, a, c):
        super().__init__(a=a, c=c)

    def _cell(self, a, c):
        return 0.5 * np.array([[-a, a, c], [a, -a, c], [a, a, -c]])

    def _variant_name(self, a, c):
        return 'BCT1' if c < a else 'BCT2'

    def _special_points(self, a, c, variant):
        a2 = a * a
        c2 = c * c

        assert variant.name in self.variants

        if variant.name == 'BCT1':
            eta = .25 * (1 + c2 / a2)
            points = [[0, 0, 0],
                      [-.5, .5, .5],
                      [0., .5, 0.],
                      [.25, .25, .25],
                      [0., 0., .5],
                      [eta, eta, -eta],
                      [-eta, 1 - eta, eta]]
        else:
            eta = .25 * (1 + a2 / c2)  # Not same eta as BCT1!
            zeta = 0.5 * a2 / c2
            points = [[0., .0, 0.],
                      [0., .5, 0.],
                      [.25, .25, .25],
                      [-eta, eta, eta],
                      [eta, 1 - eta, -eta],
                      [0., 0., .5],
                      [-zeta, zeta, .5],
                      [.5, .5, -zeta],
                      [.5, .5, -.5]]
        return points


def check_orc(a, b, c):
    if not a < b < c:
        raise UnconventionalLattice('Expected a < b < c, got {}, {}, {}'
                                    .format(a, b, c))


class Orthorhombic(BravaisLattice):
    """Abstract class for orthorhombic types."""

    def __init__(self, a, b, c):
        check_orc(a, b, c)
        super().__init__(a=a, b=b, c=c)


@bravaisclass('primitive orthorhombic', 'orthorhombic', 'orthorhombic', 'oP',
              'abc',
              [['ORC', 'GRSTUXYZ', 'GXSYGZURTZ,YT,UX,SR',
                sc_special_points['orthorhombic']]])
class ORC(Orthorhombic):
    conventional_cls = 'ORC'
    conventional_cellmap = _identity

    def _cell(self, a, b, c):
        return np.diag([a, b, c]).astype(float)


@bravaisclass('face-centred orthorhombic', 'orthorhombic', 'orthorhombic',
              'oF', 'abc',
              [['ORCF1', 'GAA1LTXX1YZ', 'GYTZGXA1Y,TX1,XAZ,LG', None],
               ['ORCF2', 'GCC1DD1LHH1XYZ', 'GYCDXGZD1HC,C1Z,XH1,HY,LG', None],
               ['ORCF3', 'GAA1LTXX1YZ', 'GYTZGXA1Y,XAZ,LG', None]])
class ORCF(Orthorhombic):
    conventional_cls = 'ORC'
    conventional_cellmap = _bcc_map

    def _cell(self, a, b, c):
        return 0.5 * np.array([[0, b, c], [a, 0, c], [a, b, 0]])

    def _special_points(self, a, b, c, variant):
        a2 = a * a
        b2 = b * b
        c2 = c * c
        xminus = 0.25 * (1 + a2 / b2 - a2 / c2)
        xplus = 0.25 * (1 + a2 / b2 + a2 / c2)

        if variant.name == 'ORCF1' or variant.name == 'ORCF3':
            zeta = xminus
            eta = xplus

            points = [[0, 0, 0],
                      [.5, .5 + zeta, zeta],
                      [.5, .5 - zeta, 1 - zeta],
                      [.5, .5, .5],
                      [1., .5, .5],
                      [0., eta, eta],
                      [1., 1 - eta, 1 - eta],
                      [.5, 0, .5],
                      [.5, .5, 0]]
        else:
            assert variant.name == 'ORCF2'
            phi = 0.25 * (1 + c2 / b2 - c2 / a2)
            delta = 0.25 * (1 + b2 / a2 - b2 / c2)
            eta = xminus

            points = [[0, 0, 0],
                      [.5, .5 - eta, 1 - eta],
                      [.5, .5 + eta, eta],
                      [.5 - delta, .5, 1 - delta],
                      [.5 + delta, .5, delta],
                      [.5, .5, .5],
                      [1 - phi, .5 - phi, .5],
                      [phi, .5 + phi, .5],
                      [0., .5, .5],
                      [.5, 0., .5],
                      [.5, .5, 0.]]

        return points

    def _variant_name(self, a, b, c):
        diff = 1.0 / (a * a) - 1.0 / (b * b) - 1.0 / (c * c)
        if abs(diff) < self._eps:
            return 'ORCF3'
        return 'ORCF1' if diff > 0 else 'ORCF2'


@bravaisclass('body-centred orthorhombic', 'orthorhombic', 'orthorhombic',
              'oI', 'abc',
              [['ORCI', 'GLL1L2RSTWXX1YY1Z', 'GXLTWRX1ZGYSW,L1Y,Y1Z', None]])
class ORCI(Orthorhombic):
    conventional_cls = 'ORC'
    conventional_cellmap = _fcc_map

    def _cell(self, a, b, c):
        return 0.5 * np.array([[-a, b, c], [a, -b, c], [a, b, -c]])

    def _special_points(self, a, b, c, variant):
        a2 = a**2
        b2 = b**2
        c2 = c**2

        zeta = .25 * (1 + a2 / c2)
        eta = .25 * (1 + b2 / c2)
        delta = .25 * (b2 - a2) / c2
        mu = .25 * (a2 + b2) / c2

        points = [[0., 0., 0.],
                  [-mu, mu, .5 - delta],
                  [mu, -mu, .5 + delta],
                  [.5 - delta, .5 + delta, -mu],
                  [0, .5, 0],
                  [.5, 0, 0],
                  [0., 0., .5],
                  [.25, .25, .25],
                  [-zeta, zeta, zeta],
                  [zeta, 1 - zeta, -zeta],
                  [eta, -eta, eta],
                  [1 - eta, eta, -eta],
                  [.5, .5, -.5]]
        return points


@bravaisclass('base-centred orthorhombic', 'orthorhombic', 'orthorhombic',
              'oC', 'abc',
              [['ORCC', 'GAA1RSTXX1YZ', 'GXSRAZGYX1A1TY,ZT', None]])
class ORCC(BravaisLattice):
    conventional_cls = 'ORC'
    conventional_cellmap = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]])

    def __init__(self, a, b, c):
        # ORCC is the only ORCx lattice with a<b and not a<b<c
        if a >= b:
            raise UnconventionalLattice(f'Expected a < b, got a={a}, b={b}')
        super().__init__(a=a, b=b, c=c)

    def _cell(self, a, b, c):
        return np.array([[0.5 * a, -0.5 * b, 0], [0.5 * a, 0.5 * b, 0],
                         [0, 0, c]])

    def _special_points(self, a, b, c, variant):
        zeta = .25 * (1 + a * a / (b * b))
        points = [[0, 0, 0],
                  [zeta, zeta, .5],
                  [-zeta, 1 - zeta, .5],
                  [0, .5, .5],
                  [0, .5, 0],
                  [-.5, .5, .5],
                  [zeta, zeta, 0],
                  [-zeta, 1 - zeta, 0],
                  [-.5, .5, 0],
                  [0, 0, .5]]
        return points


@bravaisclass('primitive hexagonal', 'hexagonal', 'hexagonal', 'hP',
              'ac',
              [['HEX', 'GMKALH', 'GMKGALHA,LM,KH',
                sc_special_points['hexagonal']]])
class HEX(BravaisLattice):
    conventional_cls = 'HEX'
    conventional_cellmap = _identity

    def __init__(self, a, c):
        super().__init__(a=a, c=c)

    def _cell(self, a, c):
        x = 0.5 * np.sqrt(3)
        return np.array([[0.5 * a, -x * a, 0], [0.5 * a, x * a, 0],
                         [0., 0., c]])


@bravaisclass('primitive rhombohedral', 'hexagonal', 'rhombohedral', 'hR',
              ('a', 'alpha'),
              [['RHL1', 'GBB1FLL1PP1P2QXZ', 'GLB1,BZGX,QFP1Z,LP', None],
               ['RHL2', 'GFLPP1QQ1Z', 'GPZQGFP1Q1LZ', None]])
class RHL(BravaisLattice):
    conventional_cls = 'RHL'
    conventional_cellmap = _identity

    def __init__(self, a, alpha):
        if alpha >= 120:
            raise UnconventionalLattice('Need alpha < 120 degrees, got {}'
                                        .format(alpha))
        super().__init__(a=a, alpha=alpha)

    def _cell(self, a, alpha):
        alpha *= np.pi / 180
        acosa = a * np.cos(alpha)
        acosa2 = a * np.cos(0.5 * alpha)
        asina2 = a * np.sin(0.5 * alpha)
        acosfrac = acosa / acosa2
        xx = (1 - acosfrac**2)
        assert xx > 0.0
        return np.array([[acosa2, -asina2, 0], [acosa2, asina2, 0],
                         [a * acosfrac, 0, a * xx**0.5]])

    def _variant_name(self, a, alpha):
        return 'RHL1' if alpha < 90 else 'RHL2'

    def _special_points(self, a, alpha, variant):
        if variant.name == 'RHL1':
            cosa = np.cos(alpha * _degrees)
            eta = (1 + 4 * cosa) / (2 + 4 * cosa)
            nu = .75 - 0.5 * eta
            points = [[0, 0, 0],
                      [eta, .5, 1 - eta],
                      [.5, 1 - eta, eta - 1],
                      [.5, .5, 0],
                      [.5, 0, 0],
                      [0, 0, -.5],
                      [eta, nu, nu],
                      [1 - nu, 1 - nu, 1 - eta],
                      [nu, nu, eta - 1],
                      [1 - nu, nu, 0],
                      [nu, 0, -nu],
                      [.5, .5, .5]]
        else:
            eta = 1 / (2 * np.tan(alpha * _degrees / 2)**2)
            nu = .75 - 0.5 * eta
            points = [[0, 0, 0],
                      [.5, -.5, 0],
                      [.5, 0, 0],
                      [1 - nu, -nu, 1 - nu],
                      [nu, nu - 1, nu - 1],
                      [eta, eta, eta],
                      [1 - eta, -eta, -eta],
                      [.5, -.5, .5]]
        return points


def check_mcl(a, b, c, alpha):
    if b > c or alpha >= 90:
        raise UnconventionalLattice('Expected b <= c, alpha < 90; '
                                    'got a={}, b={}, c={}, alpha={}'
                                    .format(a, b, c, alpha))


@bravaisclass('primitive monoclinic', 'monoclinic', 'monoclinic', 'mP',
              ('a', 'b', 'c', 'alpha'),
              [['MCL', 'GACDD1EHH1H2MM1M2XYY1Z', 'GYHCEM1AXH1,MDZ,YD', None]])
class MCL(BravaisLattice):
    conventional_cls = 'MCL'
    conventional_cellmap = _identity

    def __init__(self, a, b, c, alpha):
        check_mcl(a, b, c, alpha)
        super().__init__(a=a, b=b, c=c, alpha=alpha)

    def _cell(self, a, b, c, alpha):
        alpha *= _degrees
        return np.array([[a, 0, 0], [0, b, 0],
                         [0, c * np.cos(alpha), c * np.sin(alpha)]])

    def _special_points(self, a, b, c, alpha, variant):
        cosa = np.cos(alpha * _degrees)
        eta = (1 - b * cosa / c) / (2 * np.sin(alpha * _degrees)**2)
        nu = .5 - eta * c * cosa / b

        points = [[0, 0, 0],
                  [.5, .5, 0],
                  [0, .5, .5],
                  [.5, 0, .5],
                  [.5, 0, -.5],
                  [.5, .5, .5],
                  [0, eta, 1 - nu],
                  [0, 1 - eta, nu],
                  [0, eta, -nu],
                  [.5, eta, 1 - nu],
                  [.5, 1 - eta, nu],
                  [.5, eta, -nu],
                  [0, .5, 0],
                  [0, 0, .5],
                  [0, 0, -.5],
                  [.5, 0, 0]]
        return points

    def _variant_name(self, a, b, c, alpha):
        check_mcl(a, b, c, alpha)
        return 'MCL'


@bravaisclass('base-centred monoclinic', 'monoclinic', 'monoclinic', 'mC',
              ('a', 'b', 'c', 'alpha'),
              [['MCLC1', 'GNN1FF1F2F3II1LMXX1X2YY1Z',
                'GYFLI,I1ZF1,YX1,XGN,MG', None],
               ['MCLC2', 'GNN1FF1F2F3II1LMXX1X2YY1Z',
                'GYFLI,I1ZF1,NGM', None],
               ['MCLC3', 'GFF1F2HH1H2IMNN1XYY1Y2Y3Z',
                'GYFHZIF1,H1Y1XGN,MG', None],
               ['MCLC4', 'GFF1F2HH1H2IMNN1XYY1Y2Y3Z',
                'GYFHZI,H1Y1XGN,MG', None],
               ['MCLC5', 'GFF1F2HH1H2II1LMNN1XYY1Y2Y3Z',
                'GYFLI,I1ZHF1,H1Y1XGN,MG', None]])
class MCLC(BravaisLattice):
    conventional_cls = 'MCL'
    conventional_cellmap = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]])

    def __init__(self, a, b, c, alpha):
        check_mcl(a, b, c, alpha)
        super().__init__(a=a, b=b, c=c, alpha=alpha)

    def _cell(self, a, b, c, alpha):
        alpha *= np.pi / 180
        return np.array([[0.5 * a, 0.5 * b, 0], [-0.5 * a, 0.5 * b, 0],
                         [0, c * np.cos(alpha), c * np.sin(alpha)]])

    def _variant_name(self, a, b, c, alpha):
        # from ase.geometry.cell import mclc
        # okay, this is a bit hacky

        # We need the same parameters here as when determining the points.
        # Right now we just repeat the code:
        check_mcl(a, b, c, alpha)

        a2 = a * a
        b2 = b * b
        cosa = np.cos(alpha * _degrees)
        sina = np.sin(alpha * _degrees)
        sina2 = sina**2

        cell = self.tocell()
        lengths_angles = Cell(cell.reciprocal()).cellpar()

        kgamma = lengths_angles[-1]

        eps = self._eps
        # We should not compare angles in degrees versus lengths with
        # the same precision.
        if abs(kgamma - 90) < eps:
            variant = 2
        elif kgamma > 90:
            variant = 1
        elif kgamma < 90:
            num = b * cosa / c + b2 * sina2 / a2
            if abs(num - 1) < eps:
                variant = 4
            elif num < 1:
                variant = 3
            else:
                variant = 5
        variant = 'MCLC' + str(variant)
        return variant

    def _special_points(self, a, b, c, alpha, variant):
        variant = int(variant.name[-1])

        a2 = a * a
        b2 = b * b
        # c2 = c * c
        cosa = np.cos(alpha * _degrees)
        sina = np.sin(alpha * _degrees)
        sina2 = sina**2

        if variant == 1 or variant == 2:
            zeta = (2 - b * cosa / c) / (4 * sina2)
            eta = 0.5 + 2 * zeta * c * cosa / b
            psi = .75 - a2 / (4 * b2 * sina * sina)
            phi = psi + (.75 - psi) * b * cosa / c

            points = [[0, 0, 0],
                      [.5, 0, 0],
                      [0, -.5, 0],
                      [1 - zeta, 1 - zeta, 1 - eta],
                      [zeta, zeta, eta],
                      [-zeta, -zeta, 1 - eta],
                      [1 - zeta, -zeta, 1 - eta],
                      [phi, 1 - phi, .5],
                      [1 - phi, phi - 1, .5],
                      [.5, .5, .5],
                      [.5, 0, .5],
                      [1 - psi, psi - 1, 0],
                      [psi, 1 - psi, 0],
                      [psi - 1, -psi, 0],
                      [.5, .5, 0],
                      [-.5, -.5, 0],
                      [0, 0, .5]]
        elif variant == 3 or variant == 4:
            mu = .25 * (1 + b2 / a2)
            delta = b * c * cosa / (2 * a2)
            zeta = mu - 0.25 + (1 - b * cosa / c) / (4 * sina2)
            eta = 0.5 + 2 * zeta * c * cosa / b
            phi = 1 + zeta - 2 * mu
            psi = eta - 2 * delta

            points = [[0, 0, 0],
                      [1 - phi, 1 - phi, 1 - psi],
                      [phi, phi - 1, psi],
                      [1 - phi, -phi, 1 - psi],
                      [zeta, zeta, eta],
                      [1 - zeta, -zeta, 1 - eta],
                      [-zeta, -zeta, 1 - eta],
                      [.5, -.5, .5],
                      [.5, 0, .5],
                      [.5, 0, 0],
                      [0, -.5, 0],
                      [.5, -.5, 0],
                      [mu, mu, delta],
                      [1 - mu, -mu, -delta],
                      [-mu, -mu, -delta],
                      [mu, mu - 1, delta],
                      [0, 0, .5]]
        elif variant == 5:
            zeta = .25 * (b2 / a2 + (1 - b * cosa / c) / sina2)
            eta = 0.5 + 2 * zeta * c * cosa / b
            mu = .5 * eta + b2 / (4 * a2) - b * c * cosa / (2 * a2)
            nu = 2 * mu - zeta
            omega = (4 * nu - 1 - b2 * sina2 / a2) * c / (2 * b * cosa)
            delta = zeta * c * cosa / b + omega / 2 - .25
            rho = 1 - zeta * a2 / b2

            points = [[0, 0, 0],
                      [nu, nu, omega],
                      [1 - nu, 1 - nu, 1 - omega],
                      [nu, nu - 1, omega],
                      [zeta, zeta, eta],
                      [1 - zeta, -zeta, 1 - eta],
                      [-zeta, -zeta, 1 - eta],
                      [rho, 1 - rho, .5],
                      [1 - rho, rho - 1, .5],
                      [.5, .5, .5],
                      [.5, 0, .5],
                      [.5, 0, 0],
                      [0, -.5, 0],
                      [.5, -.5, 0],
                      [mu, mu, delta],
                      [1 - mu, -mu, -delta],
                      [-mu, -mu, -delta],
                      [mu, mu - 1, delta],
                      [0, 0, .5]]

        return points


tri_angles_explanation = """\
Angles kalpha, kbeta and kgamma of TRI lattice must be 1) all greater \
than 90 degrees with kgamma being the smallest, or 2) all smaller than \
90 with kgamma being the largest, or 3) kgamma=90 being the \
smallest of the three, or 4) kgamma=90 being the largest of the three.  \
Angles of reciprocal lattice are kalpha={}, kbeta={}, kgamma={}.  \
If you don't care, please use Cell.fromcellpar() instead."""

# XXX labels, paths, are all the same.


@bravaisclass('primitive triclinic', 'triclinic', 'triclinic', 'aP',
              ('a', 'b', 'c', 'alpha', 'beta', 'gamma'),
              [['TRI1a', 'GLMNRXYZ', 'XGY,LGZ,NGM,RG', None],
               ['TRI2a', 'GLMNRXYZ', 'XGY,LGZ,NGM,RG', None],
               ['TRI1b', 'GLMNRXYZ', 'XGY,LGZ,NGM,RG', None],
               ['TRI2b', 'GLMNRXYZ', 'XGY,LGZ,NGM,RG', None]])
class TRI(BravaisLattice):
    conventional_cls = 'TRI'
    conventional_cellmap = _identity

    def __init__(self, a, b, c, alpha, beta, gamma):
        super().__init__(a=a, b=b, c=c, alpha=alpha, beta=beta,
                         gamma=gamma)

    def _cell(self, a, b, c, alpha, beta, gamma):
        alpha, beta, gamma = np.array([alpha, beta, gamma])
        singamma = np.sin(gamma * _degrees)
        cosgamma = np.cos(gamma * _degrees)
        cosbeta = np.cos(beta * _degrees)
        cosalpha = np.cos(alpha * _degrees)
        a3x = c * cosbeta
        a3y = c / singamma * (cosalpha - cosbeta * cosgamma)
        a3z = c / singamma * np.sqrt(singamma**2 - cosalpha**2 - cosbeta**2
                                     + 2 * cosalpha * cosbeta * cosgamma)
        return np.array([[a, 0, 0], [b * cosgamma, b * singamma, 0],
                         [a3x, a3y, a3z]])

    def _variant_name(self, a, b, c, alpha, beta, gamma):
        cell = Cell.new([a, b, c, alpha, beta, gamma])
        icellpar = Cell(cell.reciprocal()).cellpar()
        kangles = kalpha, kbeta, kgamma = icellpar[3:]

        def raise_unconventional():
            raise UnconventionalLattice(tri_angles_explanation
                                        .format(*kangles))

        eps = self._eps
        if abs(kgamma - 90) < eps:
            if kalpha > 90 and kbeta > 90:
                var = '2a'
            elif kalpha < 90 and kbeta < 90:
                var = '2b'
            else:
                # Is this possible?  Maybe due to epsilon
                raise_unconventional()
        elif all(kangles > 90):
            if kgamma > min(kangles):
                raise_unconventional()
            var = '1a'
        elif all(kangles < 90):  # and kgamma > max(kalpha, kbeta):
            if kgamma < max(kangles):
                raise_unconventional()
            var = '1b'
        else:
            raise_unconventional()

        return 'TRI' + var

    def _special_points(self, a, b, c, alpha, beta, gamma, variant):
        # (None of the points actually depend on any parameters)
        # (We should store the points openly on the variant objects)
        if variant.name == 'TRI1a' or variant.name == 'TRI2a':
            points = [[0., 0., 0.],
                      [.5, .5, 0],
                      [0, .5, .5],
                      [.5, 0, .5],
                      [.5, .5, .5],
                      [.5, 0, 0],
                      [0, .5, 0],
                      [0, 0, .5]]
        else:
            points = [[0, 0, 0],
                      [.5, -.5, 0],
                      [0, 0, .5],
                      [-.5, -.5, .5],
                      [0, -.5, .5],
                      [0, -0.5, 0],
                      [.5, 0, 0],
                      [-.5, 0, .5]]

        return points


def get_subset_points(names, points):
    newpoints = {name: points[name] for name in names}
    return newpoints


@bravaisclass('primitive oblique', 'monoclinic', None, 'mp',
              ('a', 'b', 'alpha'), [['OBL', 'GYHCH1X', 'GYHCH1XG', None]],
              ndim=2)
class OBL(BravaisLattice):
    def __init__(self, a, b, alpha, **kwargs):
        check_rect(a, b)
        if alpha >= 90:
            raise UnconventionalLattice(
                f'Expected alpha < 90, got alpha={alpha}')
        super().__init__(a=a, b=b, alpha=alpha, **kwargs)

    def _cell(self, a, b, alpha):
        cosa = np.cos(alpha * _degrees)
        sina = np.sin(alpha * _degrees)

        return np.array([[a, 0, 0],
                         [b * cosa, b * sina, 0],
                         [0., 0., 0.]])

    def _special_points(self, a, b, alpha, variant):
        cosa = np.cos(alpha * _degrees)
        eta = (1 - a * cosa / b) / (2 * np.sin(alpha * _degrees)**2)
        nu = .5 - eta * b * cosa / a

        points = [[0, 0, 0],
                  [0, 0.5, 0],
                  [eta, 1 - nu, 0],
                  [.5, .5, 0],
                  [1 - eta, nu, 0],
                  [.5, 0, 0]]

        return points


@bravaisclass('primitive hexagonal', 'hexagonal', None, 'hp', 'a',
              [['HEX2D', 'GMK', 'GMKG',
                get_subset_points('GMK',
                                  sc_special_points['hexagonal'])]],
              ndim=2)
class HEX2D(BravaisLattice):
    def __init__(self, a, **kwargs):
        super().__init__(a=a, **kwargs)

    def _cell(self, a):
        x = 0.5 * np.sqrt(3)
        return np.array([[a, 0, 0],
                         [-0.5 * a, x * a, 0],
                         [0., 0., 0.]])


def check_rect(a, b):
    if a >= b:
        raise UnconventionalLattice(f'Expected a < b, got a={a}, b={b}')


@bravaisclass('primitive rectangular', 'orthorhombic', None, 'op', 'ab',
              [['RECT', 'GXSY', 'GXSYGS',
                get_subset_points('GXSY',
                                  sc_special_points['orthorhombic'])]],
              ndim=2)
class RECT(BravaisLattice):
    def __init__(self, a, b, **kwargs):
        check_rect(a, b)
        super().__init__(a=a, b=b, **kwargs)

    def _cell(self, a, b):
        return np.array([[a, 0, 0],
                         [0, b, 0],
                         [0, 0, 0.]])


@bravaisclass('centred rectangular', 'orthorhombic', None, 'oc',
              ('a', 'alpha'), [['CRECT', 'GXA1Y', 'GXA1YG', None]], ndim=2)
class CRECT(BravaisLattice):
    def __init__(self, a, alpha, **kwargs):
        # It would probably be better to define the CRECT cell
        # by (a, b) rather than (a, alpha).  Then we can require a < b
        # like in ordinary RECT.
        #
        # In 3D, all lattices in the same family generally take
        # identical parameters.
        if alpha >= 90:
            raise UnconventionalLattice(
                f'Expected alpha < 90.  Got alpha={alpha}')
        super().__init__(a=a, alpha=alpha, **kwargs)

    def _cell(self, a, alpha):
        x = np.cos(alpha * _degrees)
        y = np.sin(alpha * _degrees)
        return np.array([[a, 0, 0],
                         [a * x, a * y, 0],
                         [0, 0, 0.]])

    def _special_points(self, a, alpha, variant):
        sina2 = np.sin(alpha / 2 * _degrees)**2
        sina = np.sin(alpha * _degrees)**2
        eta = sina2 / sina
        cosa = np.cos(alpha * _degrees)
        xi = eta * cosa

        points = [[0, 0, 0],
                  [eta, - eta, 0],
                  [0.5 + xi, 0.5 - xi, 0],
                  [0.5, 0.5, 0]]

        return points


@bravaisclass('primitive square', 'tetragonal', None, 'tp', ('a',),
              [['SQR', 'GMX', 'MGXM',
                get_subset_points('GMX', sc_special_points['tetragonal'])]],
              ndim=2)
class SQR(BravaisLattice):
    def __init__(self, a, **kwargs):
        super().__init__(a=a, **kwargs)

    def _cell(self, a):
        return np.array([[a, 0, 0],
                         [0, a, 0],
                         [0, 0, 0.]])


@bravaisclass('primitive line', 'line', None, '?', ('a',),
              [['LINE', 'GX', 'GX', {'G': [0, 0, 0], 'X': [0.5, 0, 0]}]],
              ndim=1)
class LINE(BravaisLattice):
    def __init__(self, a, **kwargs):
        super().__init__(a=a, **kwargs)

    def _cell(self, a):
        return np.array([[a, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])


def celldiff(cell1, cell2):
    """Return a unitless measure of the difference between two cells."""
    cell1 = Cell.ascell(cell1).complete()
    cell2 = Cell.ascell(cell2).complete()
    v1v2 = cell1.volume * cell2.volume
    if v1v2 < 1e-10:
        # (Proposed cell may be linearly dependent)
        return np.inf

    scale = v1v2**(-1. / 3.)  # --> 1/Ang^2
    x1 = cell1 @ cell1.T
    x2 = cell2 @ cell2.T
    dev = scale * np.abs(x2 - x1).max()
    return dev


def get_lattice_from_canonical_cell(cell, eps=2e-4):
    """Return a Bravais lattice representing the given cell.

    This works only for cells that are derived from the standard form
    (as generated by lat.tocell()) or rotations thereof.

    If the given cell does not resemble the known form of a Bravais
    lattice, raise RuntimeError."""
    return LatticeChecker(cell, eps).match()


def identify_lattice(cell, eps=2e-4, *, pbc=True):
    """Find Bravais lattice representing this cell.

    Returns Bravais lattice object representing the cell along with
    an operation that, applied to the cell, yields the same lengths
    and angles as the Bravais lattice object."""
    from ase.geometry.bravais_type_engine import niggli_op_table

    pbc = cell.any(1) & pbc2pbc(pbc)
    npbc = sum(pbc)

    cell = cell.uncomplete(pbc)
    rcell, reduction_op = cell.niggli_reduce(eps=eps)

    # We tabulate the cell's Niggli-mapped versions so we don't need to
    # redo any work when the same Niggli-operation appears multiple times
    # in the table:
    memory = {}

    # We loop through the most symmetric kinds (CUB etc.) and return
    # the first one we find:
    for latname in LatticeChecker.check_orders[npbc]:
        # There may be multiple Niggli operations that produce valid
        # lattices, at least for MCL.  In that case we will pick the
        # one whose angle is closest to 90, but it means we cannot
        # just return the first one we find so we must remember then:
        matching_lattices = []

        for op_key in niggli_op_table[latname]:
            checker_and_op = memory.get(op_key)
            if checker_and_op is None:
                normalization_op = np.array(op_key).reshape(3, 3)
                candidate = Cell(np.linalg.inv(normalization_op.T) @ rcell)
                checker = LatticeChecker(candidate, eps=eps)
                memory[op_key] = (checker, normalization_op)
            else:
                checker, normalization_op = checker_and_op

            lat = checker.query(latname)
            if lat is not None:
                op = normalization_op @ np.linalg.inv(reduction_op)
                matching_lattices.append((lat, op))

        if not matching_lattices:
            continue  # Move to next Bravais lattice

        lat, op = pick_best_lattice(matching_lattices)

        if npbc == 2 and op[2, 2] < 0:
            op = flip_2d_handedness(op)

        return lat, op

    raise RuntimeError('Failed to recognize lattice')


def flip_2d_handedness(op):
    # The 3x3 operation may flip the z axis, but then the x/y
    # components are necessarily also left-handed which
    # means a defacto left-handed 2D bandpath.
    #
    # We repair this by applying an operation that unflips the
    # z axis and interchanges x/y:
    repair_op = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    return repair_op @ op


def pick_best_lattice(matching_lattices):
    """Return (lat, op) with lowest orthogonality defect."""
    best = None
    best_defect = np.inf
    for lat, op in matching_lattices:
        cell = lat.tocell().complete()
        orthogonality_defect = np.prod(cell.lengths()) / cell.volume
        if orthogonality_defect < best_defect:
            best = lat, op
            best_defect = orthogonality_defect
    return best


class LatticeChecker:
    # The check order is slightly different than elsewhere listed order
    # as we need to check HEX/RHL before the ORCx family.
    check_orders = {
        1: ['LINE'],
        2: ['SQR', 'RECT', 'HEX2D', 'CRECT', 'OBL'],
        3: ['CUB', 'FCC', 'BCC', 'TET', 'BCT', 'HEX', 'RHL',
            'ORC', 'ORCF', 'ORCI', 'ORCC', 'MCL', 'MCLC', 'TRI']}

    def __init__(self, cell, eps=2e-4):
        """Generate Bravais lattices that look (or not) like the given cell.

        The cell must be reduced to canonical form, i.e., it must
        be possible to produce a cell with the same lengths and angles
        by directly through one of the Bravais lattice classes.

        Generally for internal use (this module).

        For each of the 14 Bravais lattices, this object can produce
        a lattice object which represents the same cell, or None if
        the tolerance eps is not met."""
        self.cell = cell
        self.eps = eps

        self.cellpar = cell.cellpar()
        self.lengths = self.A, self.B, self.C = self.cellpar[:3]
        self.angles = self.cellpar[3:]

        # Use a 'neutral' length for checking cubic lattices
        self.A0 = self.lengths.mean()

        # Vector of the diagonal and then off-diagonal dot products:
        #   [a1 · a1, a2 · a2, a3 · a3, a2 · a3, a3 · a1, a1 · a2]
        self.prods = (cell @ cell.T).flat[[0, 4, 8, 5, 2, 1]]

    def _check(self, latcls, *args):
        if any(arg <= 0 for arg in args):
            return None
        try:
            lat = latcls(*args)
        except UnconventionalLattice:
            return None

        newcell = lat.tocell()
        err = celldiff(self.cell, newcell)
        if err < self.eps:
            return lat

    def match(self):
        """Match cell against all lattices, returning most symmetric match.

        Returns the lattice object.  Raises RuntimeError on failure."""
        for name in self.check_orders[self.cell.rank]:
            lat = self.query(name)
            if lat:
                return lat
        raise RuntimeError('Could not find lattice type for cell '
                           'with lengths and angles {}'
                           .format(self.cell.cellpar().tolist()))

    def query(self, latname):
        """Match cell against named Bravais lattice.

        Return lattice object on success, None on failure."""
        meth = getattr(self, latname)
        lat = meth()
        return lat

    def LINE(self):
        return self._check(LINE, self.lengths[0])

    def SQR(self):
        return self._check(SQR, self.lengths[0])

    def RECT(self):
        return self._check(RECT, *self.lengths[:2])

    def CRECT(self):
        return self._check(CRECT, self.lengths[0], self.angles[2])

    def HEX2D(self):
        return self._check(HEX2D, self.lengths[0])

    def OBL(self):
        return self._check(OBL, *self.lengths[:2], self.angles[2])

    def CUB(self):
        # These methods (CUB, FCC, ...) all return a lattice object if
        # it matches, else None.
        return self._check(CUB, self.A0)

    def FCC(self):
        return self._check(FCC, np.sqrt(2) * self.A0)

    def BCC(self):
        return self._check(BCC, 2.0 * self.A0 / np.sqrt(3))

    def TET(self):
        return self._check(TET, self.A, self.C)

    def _bct_orci_lengths(self):
        # Coordinate-system independent relation for BCT and ORCI
        # standard cells:
        #   a1 · a1 + a2 · a3 == a² / 2
        #   a2 · a2 + a3 · a1 == a² / 2 (BCT)
        #                     == b² / 2 (ORCI)
        #   a3 · a3 + a1 · a2 == c² / 2
        # We use these to get a, b, and c in those cases.
        prods = self.prods
        lengthsqr = 2.0 * (prods[:3] + prods[3:])
        if any(lengthsqr < 0):
            return None
        return np.sqrt(lengthsqr)

    def BCT(self):
        lengths = self._bct_orci_lengths()
        if lengths is None:
            return None
        return self._check(BCT, lengths[0], lengths[2])

    def HEX(self):
        return self._check(HEX, self.A, self.C)

    def RHL(self):
        return self._check(RHL, self.A, self.angles[0])

    def ORC(self):
        return self._check(ORC, *self.lengths)

    def ORCF(self):
        # ORCF standard cell:
        #   a2 · a3 = a²/4
        #   a3 · a1 = b²/4
        #   a1 · a2 = c²/4
        prods = self.prods
        if all(prods[3:] > 0):
            orcf_abc = 2 * np.sqrt(prods[3:])
            return self._check(ORCF, *orcf_abc)

    def ORCI(self):
        lengths = self._bct_orci_lengths()
        if lengths is None:
            return None
        return self._check(ORCI, *lengths)

    def _orcc_ab(self):
        # ORCC: a1 · a1 + a2 · a3 = a²/2
        #       a2 · a2 - a2 · a3 = b²/2
        prods = self.prods
        orcc_sqr_ab = np.empty(2)
        orcc_sqr_ab[0] = 2.0 * (prods[0] + prods[5])
        orcc_sqr_ab[1] = 2.0 * (prods[1] - prods[5])
        if all(orcc_sqr_ab > 0):
            return np.sqrt(orcc_sqr_ab)

    def ORCC(self):
        orcc_lengths_ab = self._orcc_ab()
        if orcc_lengths_ab is None:
            return None
        return self._check(ORCC, *orcc_lengths_ab, self.C)

    def MCL(self):
        return self._check(MCL, *self.lengths, self.angles[0])

    def MCLC(self):
        # MCLC is similar to ORCC:
        orcc_ab = self._orcc_ab()
        if orcc_ab is None:
            return None

        prods = self.prods
        C = self.C
        mclc_a, mclc_b = orcc_ab[::-1]  # a, b reversed wrt. ORCC
        mclc_cosa = 2.0 * prods[3] / (mclc_b * C)
        if -1 < mclc_cosa < 1:
            mclc_alpha = np.arccos(mclc_cosa) * 180 / np.pi
            if mclc_b > C:
                # XXX Temporary fix for certain otherwise
                # unrecognizable lattices.
                #
                # This error could happen if the input lattice maps to
                # something just outside the domain of conventional
                # lattices (less than the tolerance).  Our solution is to
                # propose a nearby conventional lattice instead, which
                # will then be accepted if it's close enough.
                mclc_b = 0.5 * (mclc_b + C)
                C = mclc_b
            return self._check(MCLC, mclc_a, mclc_b, C, mclc_alpha)

    def TRI(self):
        return self._check(TRI, *self.cellpar)


def all_variants():
    """For testing and examples; yield all variants of all lattices."""
    a, b, c = 3., 4., 5.
    alpha = 55.0
    yield CUB(a)
    yield FCC(a)
    yield BCC(a)
    yield TET(a, c)
    bct1 = BCT(2 * a, c)
    bct2 = BCT(a, c)
    assert bct1.variant == 'BCT1'
    assert bct2.variant == 'BCT2'

    yield bct1
    yield bct2

    yield ORC(a, b, c)

    a0 = np.sqrt(1.0 / (1 / b**2 + 1 / c**2))
    orcf1 = ORCF(0.5 * a0, b, c)
    orcf2 = ORCF(1.2 * a0, b, c)
    orcf3 = ORCF(a0, b, c)
    assert orcf1.variant == 'ORCF1'
    assert orcf2.variant == 'ORCF2'
    assert orcf3.variant == 'ORCF3'
    yield orcf1
    yield orcf2
    yield orcf3

    yield ORCI(a, b, c)
    yield ORCC(a, b, c)

    yield HEX(a, c)

    rhl1 = RHL(a, alpha=55.0)
    assert rhl1.variant == 'RHL1'
    yield rhl1

    rhl2 = RHL(a, alpha=105.0)
    assert rhl2.variant == 'RHL2'
    yield rhl2

    # With these lengths, alpha < 65 (or so) would result in a lattice that
    # could also be represented with alpha > 65, which is more conventional.
    yield MCL(a, b, c, alpha=70.0)

    mclc1 = MCLC(a, b, c, 80)
    assert mclc1.variant == 'MCLC1'
    yield mclc1
    # mclc2 has same special points as mclc1

    mclc3 = MCLC(1.8 * a, b, c * 2, 80)
    assert mclc3.variant == 'MCLC3'
    yield mclc3
    # mclc4 has same special points as mclc3

    # XXX We should add MCLC2 and MCLC4 as well.

    mclc5 = MCLC(b, b, 1.1 * b, 70)
    assert mclc5.variant == 'MCLC5'
    yield mclc5

    def get_tri(kcellpar):
        # We build the TRI lattices from cellpars of reciprocal cell
        icell = Cell.fromcellpar(kcellpar)
        cellpar = Cell(4 * icell.reciprocal()).cellpar()
        return TRI(*cellpar)

    tri1a = get_tri([1., 1.2, 1.4, 120., 110., 100.])
    assert tri1a.variant == 'TRI1a'
    yield tri1a

    tri1b = get_tri([1., 1.2, 1.4, 50., 60., 70.])
    assert tri1b.variant == 'TRI1b'
    yield tri1b

    tri2a = get_tri([1., 1.2, 1.4, 120., 110., 90.])
    assert tri2a.variant == 'TRI2a'
    yield tri2a
    tri2b = get_tri([1., 1.2, 1.4, 50., 60., 90.])
    assert tri2b.variant == 'TRI2b'
    yield tri2b

    # Choose an OBL lattice that round-trip-converts to itself.
    # The default a/b/alpha parameters result in another representation
    # of the same lattice.
    yield OBL(a=3.0, b=3.35, alpha=77.85)
    yield RECT(a, b)
    yield CRECT(a, alpha=alpha)
    yield HEX2D(a)
    yield SQR(a)
    yield LINE(a)

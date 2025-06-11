# fmt: off

"""Prism"""
import warnings
from typing import Sequence, Union

import numpy as np

from ase.geometry import wrap_positions


def calc_box_parameters(cell: np.ndarray) -> np.ndarray:
    """Calculate box parameters

    https://docs.lammps.org/Howto_triclinic.html
    """
    ax = np.sqrt(cell[0] @ cell[0])
    bx = cell[0] @ cell[1] / ax
    by = np.sqrt(cell[1] @ cell[1] - bx ** 2)
    cx = cell[0] @ cell[2] / ax
    cy = (cell[1] @ cell[2] - bx * cx) / by
    cz = np.sqrt(cell[2] @ cell[2] - cx ** 2 - cy ** 2)
    return np.array((ax, by, cz, bx, cx, cy))


def calc_rotated_cell(cell: np.ndarray) -> np.ndarray:
    """Calculate rotated cell in LAMMPS coordinates

    Parameters
    ----------
    cell : np.ndarray
        Cell to be rotated.

    Returns
    -------
    rotated_cell : np.ndarray
        Rotated cell represented by a lower triangular matrix.
    """
    ax, by, cz, bx, cx, cy = calc_box_parameters(cell)
    return np.array(((ax, 0.0, 0.0), (bx, by, 0.0), (cx, cy, cz)))


def calc_reduced_cell(
    cell: np.ndarray,
    pbc: Union[np.ndarray, Sequence[bool]],
) -> np.ndarray:
    """Calculate LAMMPS cell with short lattice basis vectors

    The lengths of the second and the third lattice basis vectors, b and c, are
    shortened with keeping the same periodicity of the system. b is modified by
    adding multiple a vectors, and c is modified by adding first multiple b
    vectors and then multiple a vectors.

    Parameters
    ----------
    cell : np.ndarray
        Cell to be reduced. This must be already a lower triangular matrix.
    pbc : Sequence[bool]
        True if the system is periodic along the corresponding direction.

    Returns
    -------
    reduced_cell : np.ndarray
        Reduced cell. `xx`, `yy`, `zz` are the same as the original cell,
        and `abs(xy) <= xx`, `abs(xz) <= xx`, `abs(yz) <= yy`.
    """
    # cell 1 (reduced) <- cell 2 (original)
    # o-----------------------------/==o-----------------------------/--o
    #  \                        /--/    \                        /--/
    #   \                   /--/         \                   /--/
    #    \         1    /--/              \   2          /--/
    #     \         /--/                   \         /--/
    #      \    /--/                        \    /--/
    #       o==/-----------------------------o--/
    reduced_cell = cell.copy()
    # Order in which off-diagonal elements are checked for strong tilt
    # yz is updated before xz so that the latter does not affect the former
    flip_order = ((1, 0), (2, 1), (2, 0))
    for i, j in flip_order:
        if not pbc[j]:
            continue
        ratio = reduced_cell[i, j] / reduced_cell[j, j]
        if abs(ratio) > 0.5:
            reduced_cell[i] -= reduced_cell[j] * np.round(ratio)
    return reduced_cell


class Prism:
    """The representation of the unit cell in LAMMPS

    The main purpose of the prism-object is to create suitable string
    representations of prism limits and atom positions within the prism.

    Parameters
    ----------
    cell : np.ndarray
        Cell in ASE coordinate system.
    pbc : one or three bool
        Periodic boundary conditions flags.
    reduce_cell : bool
        If True, the LAMMPS cell is reduced for short lattice basis vectors.
        The atomic positions are always wraped into the reduced cell,
        regardress of `wrap` in `vector_to_lammps` and `vector_to_ase`.
    tolerance : float
        Precision for skewness test.

    Methods
    -------
    vector_to_lammps
        Rotate vectors from ASE to LAMMPS coordinates.
        Positions can be further wrapped into the LAMMPS cell by `wrap=True`.

    vector_to_ase
        Rotate vectors from LAMMPS to ASE coordinates.
        Positions can be further wrapped into the LAMMPS cell by `wrap=True`.

    Notes
    -----
    LAMMPS prefers triangular matrixes without a strong tilt.
    Therefore the 'Prism'-object contains three coordinate systems:

    - ase_cell (the simulated system in the ASE coordination system)
    - lammps_tilt (ase-cell rotated to be an lower triangular matrix)
    - lammps_cell (same volume as tilted cell, but reduce edge length)

    The translation between 'ase_cell' and 'lammps_tilt' is done with a
    rotation matrix 'rot_mat'. (Mathematically this is a QR decomposition.)

    The transformation between 'lammps_tilt' and 'lammps_cell' is done by
    changing the off-diagonal elements.

    Depending on the option `reduce`, vectors in ASE coordinates are
    transformed either `lammps_tilt` or `lammps_cell`.

    The vector conversion can fail as depending on the simulation run LAMMPS
    might have changed the simulation box significantly. This is for example a
    problem with hexagonal cells. LAMMPS might also wrap atoms across periodic
    boundaries, which can lead to problems for example NEB calculations.
    """

    # !TODO: derive tolerance from cell-dimensions
    def __init__(
        self,
        cell: np.ndarray,
        pbc: Union[bool, np.ndarray] = True,
        reduce_cell: bool = False,
        tolerance: float = 1.0e-8,
    ):
        #    rot_mat * lammps_tilt^T = ase_cell^T
        # => lammps_tilt * rot_mat^T = ase_cell
        # => lammps_tilt             = ase_cell * rot_mat
        # LAMMPS requires positive diagonal elements of the triangular matrix.
        # The diagonals of `lammps_tilt` are always positive by construction.
        self.lammps_tilt = calc_rotated_cell(cell)
        self.rot_mat = np.linalg.solve(self.lammps_tilt, cell).T
        self.ase_cell = cell
        self.tolerance = tolerance
        self.pbc = tuple(np.zeros(3, bool) + pbc)
        self.lammps_cell = calc_reduced_cell(self.lammps_tilt, self.pbc)
        self.is_reduced = reduce_cell

    @property
    def cell(self) -> np.ndarray:
        return self.lammps_cell if self.is_reduced else self.lammps_tilt

    def get_lammps_prism(self) -> np.ndarray:
        """Return box parameters of the rotated cell in LAMMPS coordinates

        Returns
        -------
        np.ndarray
            xhi - xlo, yhi - ylo, zhi - zlo, xy, xz, yz
        """
        return self.cell[(0, 1, 2, 1, 2, 2), (0, 1, 2, 0, 0, 1)]

    def update_cell(self, lammps_cell: np.ndarray) -> np.ndarray:
        """Rotate new LAMMPS cell into ASE coordinate system

        Parameters
        ----------
        lammps_cell : np.ndarray
            New Cell in LAMMPS coordinates received after executing LAMMPS

        Returns
        -------
        np.ndarray
            New cell in ASE coordinates
        """
        # Transformation: integer matrix
        # lammps_cell * transformation = lammps_tilt
        transformation = np.linalg.solve(self.lammps_cell, self.lammps_tilt)

        if self.is_reduced:
            self.lammps_cell = lammps_cell
            self.lammps_tilt = lammps_cell @ transformation
        else:
            self.lammps_tilt = lammps_cell
            self.lammps_cell = calc_reduced_cell(self.lammps_tilt, self.pbc)

        # try to detect potential flips in lammps
        # (lammps minimizes the cell-vector lengths)
        new_ase_cell = self.lammps_tilt @ self.rot_mat.T
        # assuming the cell changes are mostly isotropic
        new_vol = np.linalg.det(new_ase_cell)
        old_vol = np.linalg.det(self.ase_cell)
        test_residual = self.ase_cell.copy()
        test_residual *= (new_vol / old_vol) ** (1.0 / 3.0)
        test_residual -= new_ase_cell
        if any(
                np.linalg.norm(test_residual, axis=1)
                > 0.5 * np.linalg.norm(self.ase_cell, axis=1)
        ):
            warnings.warn(
                "Significant simulation cell changes from LAMMPS detected. "
                "Backtransformation to ASE might fail!"
            )
        return new_ase_cell

    def vector_to_lammps(
        self,
        vec: np.ndarray,
        wrap: bool = False,
    ) -> np.ndarray:
        """Rotate vectors from ASE to LAMMPS coordinates

        Parameters
        ----------
        vec : np.ndarray
            Vectors in ASE coordinates to be rotated into LAMMPS coordinates
        wrap : bool
            If True, the vectors are wrapped into the cell

        Returns
        -------
        np.array
            Vectors in LAMMPS coordinates
        """
        # !TODO: right eps-limit
        # lammps might not like atoms outside the cell
        if wrap or self.is_reduced:
            return wrap_positions(
                vec @ self.rot_mat,
                cell=self.cell,
                pbc=self.pbc,
                eps=1e-18,
            )
        return vec @ self.rot_mat

    def vector_to_ase(
        self,
        vec: np.ndarray,
        wrap: bool = False,
    ) -> np.ndarray:
        """Rotate vectors from LAMMPS to ASE coordinates

        Parameters
        ----------
        vec : np.ndarray
            Vectors in LAMMPS coordinates to be rotated into ASE coordinates
        wrap : bool
            If True, the vectors are wrapped into the cell

        Returns
        -------
        np.ndarray
            Vectors in ASE coordinates
        """
        if wrap or self.is_reduced:
            # fractional in `lammps_tilt` (the same shape as ASE cell)
            fractional = np.linalg.solve(self.lammps_tilt.T, vec.T).T
            # wrap into 0 to 1 for periodic directions
            fractional -= np.floor(fractional) * self.pbc
            # Cartesian coordinates wrapped into `lammps_tilt`
            vec = fractional @ self.lammps_tilt
        # rotate back to the ASE cell
        return vec @ self.rot_mat.T

    def tensor2_to_ase(self, tensor: np.ndarray) -> np.ndarray:
        """Rotate a second order tensor from LAMMPS to ASE coordinates

        Parameters
        ----------
        tensor : np.ndarray
            Tensor in LAMMPS coordinates to be rotated into ASE coordinates

        Returns
        -------
        np.ndarray
            Tensor in ASE coordinates
        """
        return self.rot_mat @ tensor @ self.rot_mat.T

    def is_skewed(self) -> bool:
        """Test if the lammps cell is skewed, i.e., monoclinic or triclinic.

        Returns
        -------
        bool
            True if the lammps cell is skewed.
        """
        cell_sq = self.cell ** 2
        on_diag = np.sum(np.diag(cell_sq))
        off_diag = np.sum(np.tril(cell_sq, -1))
        return off_diag / on_diag > self.tolerance

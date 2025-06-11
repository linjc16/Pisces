# fmt: off

"""
Provides utility functions for FixSymmetry class
"""
from collections.abc import MutableMapping
from typing import Optional

import numpy as np

from ase.utils import atoms_to_spglib_cell

__all__ = ['refine_symmetry', 'check_symmetry']


def spglib_get_symmetry_dataset(*args, **kwargs):
    """Temporary compatibility adapter around spglib dataset.

    Return an object that allows attribute-based access
    in line with recent spglib.  This allows ASE code to not care about
    older spglib versions.
    """
    import spglib
    dataset = spglib.get_symmetry_dataset(*args, **kwargs)
    if dataset is None:
        return None
    if isinstance(dataset, dict):  # spglib < 2.5.0
        return SpglibDatasetWrapper(dataset)
    return dataset  # spglib >= 2.5.0


class SpglibDatasetWrapper(MutableMapping):
    # Spglib 2.5.0 returns SpglibDataset with deprecated __getitem__.
    # Spglib 2.4.0 and earlier return dict.
    #
    # We use this object to wrap dictionaries such that both types of access
    # work correctly.
    def __init__(self, spglib_dct):
        self._spglib_dct = spglib_dct

    def __getattr__(self, attr):
        return self[attr]

    def __getitem__(self, key):
        return self._spglib_dct[key]

    def __len__(self):
        return len(self._spglib_dct)

    def __iter__(self):
        return iter(self._spglib_dct)

    def __setitem__(self, key, value):
        self._spglib_dct[key] = value

    def __delitem__(self, item):
        del self._spglib_dct[item]


def print_symmetry(symprec, dataset):
    print("ase.spacegroup.symmetrize: prec", symprec,
          "got symmetry group number", dataset.number,
          ", international (Hermann-Mauguin)", dataset.international,
          ", Hall ", dataset.hall)


def refine_symmetry(atoms, symprec=0.01, verbose=False):
    """
    Refine symmetry of an Atoms object

    Parameters
    ----------
    atoms - input Atoms object
    symprec - symmetry precicion
    verbose - if True, print out symmetry information before and after

    Returns
    -------

    spglib dataset

    """
    _check_and_symmetrize_cell(atoms, symprec=symprec, verbose=verbose)
    _check_and_symmetrize_positions(atoms, symprec=symprec, verbose=verbose)
    return check_symmetry(atoms, symprec=1e-4, verbose=verbose)


class IntermediateDatasetError(Exception):
    """The symmetry dataset in `_check_and_symmetrize_positions` can be at odds
    with the original symmetry dataset in `_check_and_symmetrize_cell`.
    This implies a faulty partial symmetrization if not handled by exception."""


def get_symmetrized_atoms(atoms,
                          symprec: float = 0.01,
                          final_symprec: Optional[float] = None):
    """Get new Atoms object with refined symmetries.

    Checks internal consistency of the found symmetries.

    Parameters
    ----------
    atoms : Atoms
        Input atoms object.
    symprec : float
        Symmetry precision used to identify symmetries with spglib.
    final_symprec : float
        Symmetry precision used for testing the symmetrization.

    Returns
    -------
    symatoms : Atoms
        New atoms object symmetrized according to the input symprec.
    """
    atoms = atoms.copy()
    original_dataset = _check_and_symmetrize_cell(atoms, symprec=symprec)
    intermediate_dataset = _check_and_symmetrize_positions(
        atoms, symprec=symprec)
    if intermediate_dataset.number != original_dataset.number:
        raise IntermediateDatasetError()
    final_symprec = final_symprec or symprec
    final_dataset = check_symmetry(atoms, symprec=final_symprec)
    assert final_dataset.number == original_dataset.number
    return atoms, final_dataset


def _check_and_symmetrize_cell(atoms, **kwargs):
    dataset = check_symmetry(atoms, **kwargs)
    _symmetrize_cell(atoms, dataset)
    return dataset


def _symmetrize_cell(atoms, dataset):
    # set actual cell to symmetrized cell vectors by copying
    # transformed and rotated standard cell
    std_cell = dataset.std_lattice
    trans_std_cell = dataset.transformation_matrix.T @ std_cell
    rot_trans_std_cell = trans_std_cell @ dataset.std_rotation_matrix
    atoms.set_cell(rot_trans_std_cell, True)


def _check_and_symmetrize_positions(atoms, *, symprec, **kwargs):
    import spglib
    dataset = check_symmetry(atoms, symprec=symprec, **kwargs)
    # here we are assuming that primitive vectors returned by find_primitive
    #    are compatible with std_lattice returned by get_symmetry_dataset
    res = spglib.find_primitive(atoms_to_spglib_cell(atoms), symprec=symprec)
    _symmetrize_positions(atoms, dataset, res)
    return dataset


def _symmetrize_positions(atoms, dataset, primitive_spglib_cell):
    prim_cell, _prim_scaled_pos, _prim_types = primitive_spglib_cell

    # calculate offset between standard cell and actual cell
    std_cell = dataset.std_lattice
    rot_std_cell = std_cell @ dataset.std_rotation_matrix
    rot_std_pos = dataset.std_positions @ rot_std_cell
    pos = atoms.get_positions()
    dp0 = (pos[list(dataset.mapping_to_primitive).index(0)] - rot_std_pos[
        list(dataset.std_mapping_to_primitive).index(0)])

    # create aligned set of standard cell positions to figure out mapping
    rot_prim_cell = prim_cell @ dataset.std_rotation_matrix
    inv_rot_prim_cell = np.linalg.inv(rot_prim_cell)
    aligned_std_pos = rot_std_pos + dp0

    # find ideal positions from position of corresponding std cell atom +
    #    integer_vec . primitive cell vectors
    mapping_to_primitive = list(dataset.mapping_to_primitive)
    std_mapping_to_primitive = list(dataset.std_mapping_to_primitive)
    pos = atoms.get_positions()
    for i_at in range(len(atoms)):
        std_i_at = std_mapping_to_primitive.index(mapping_to_primitive[i_at])
        dp = aligned_std_pos[std_i_at] - pos[i_at]
        dp_s = dp @ inv_rot_prim_cell
        pos[i_at] = (aligned_std_pos[std_i_at] - np.round(dp_s) @ rot_prim_cell)
    atoms.set_positions(pos)


def check_symmetry(atoms, symprec=1.0e-6, verbose=False):
    """
    Check symmetry of `atoms` with precision `symprec` using `spglib`

    Prints a summary and returns result of `spglib.get_symmetry_dataset()`
    """
    dataset = spglib_get_symmetry_dataset(atoms_to_spglib_cell(atoms),
                                          symprec=symprec)
    if verbose:
        print_symmetry(symprec, dataset)
    return dataset


def is_subgroup(sup_data, sub_data, tol=1e-10):
    """
    Test if spglib dataset `sub_data` is a subgroup of dataset `sup_data`
    """
    for rot1, trns1 in zip(sub_data.rotations, sub_data.translations):
        for rot2, trns2 in zip(sup_data.rotations, sup_data.translations):
            if np.all(rot1 == rot2) and np.linalg.norm(trns1 - trns2) < tol:
                break
        else:
            return False
    return True


def prep_symmetry(atoms, symprec=1.0e-6, verbose=False):
    """
    Prepare `at` for symmetry-preserving minimisation at precision `symprec`

    Returns a tuple `(rotations, translations, symm_map)`
    """
    dataset = spglib_get_symmetry_dataset(atoms_to_spglib_cell(atoms),
                                          symprec=symprec)
    if verbose:
        print_symmetry(symprec, dataset)
    rotations = dataset.rotations.copy()
    translations = dataset.translations.copy()
    symm_map = []
    scaled_pos = atoms.get_scaled_positions()
    for (rot, trans) in zip(rotations, translations):
        this_op_map = [-1] * len(atoms)
        for i_at in range(len(atoms)):
            new_p = rot @ scaled_pos[i_at, :] + trans
            dp = scaled_pos - new_p
            dp -= np.round(dp)
            i_at_map = np.argmin(np.linalg.norm(dp, axis=1))
            this_op_map[i_at] = i_at_map
        symm_map.append(this_op_map)
    return (rotations, translations, symm_map)


def symmetrize_rank1(lattice, inv_lattice, forces, rot, trans, symm_map):
    """
    Return symmetrized forces

    lattice vectors expected as row vectors (same as ASE get_cell() convention),
    inv_lattice is its matrix inverse (reciprocal().T)
    """
    scaled_symmetrized_forces_T = np.zeros(forces.T.shape)

    scaled_forces_T = np.dot(inv_lattice.T, forces.T)
    for (r, t, this_op_map) in zip(rot, trans, symm_map):
        transformed_forces_T = np.dot(r, scaled_forces_T)
        scaled_symmetrized_forces_T[:, this_op_map] += transformed_forces_T
    scaled_symmetrized_forces_T /= len(rot)
    symmetrized_forces = (lattice.T @ scaled_symmetrized_forces_T).T

    return symmetrized_forces


def symmetrize_rank2(lattice, lattice_inv, stress_3_3, rot):
    """
    Return symmetrized stress

    lattice vectors expected as row vectors (same as ASE get_cell() convention),
    inv_lattice is its matrix inverse (reciprocal().T)
    """
    scaled_stress = np.dot(np.dot(lattice, stress_3_3), lattice.T)

    symmetrized_scaled_stress = np.zeros((3, 3))
    for r in rot:
        symmetrized_scaled_stress += np.dot(np.dot(r.T, scaled_stress), r)
    symmetrized_scaled_stress /= len(rot)

    sym = np.dot(np.dot(lattice_inv, symmetrized_scaled_stress), lattice_inv.T)
    return sym

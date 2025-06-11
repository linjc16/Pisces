# fmt: off

import numpy as np

from ase.build.general_surface import surface
from ase.geometry import get_layers
from ase.symbols import string2symbols


def surfaces_with_termination(lattice, indices, layers, vacuum=None, tol=1e-10,
                              termination=None, return_all=False,
                              verbose=False):
    """Create surface from a given lattice and Miller indices with a given
        termination

        Parameters
        ==========
        lattice: Atoms object or str
            Bulk lattice structure of alloy or pure metal.  Note that the
            unit-cell must be the conventional cell - not the primitive cell.
            One can also give the chemical symbol as a string, in which case the
            correct bulk lattice will be generated automatically.
        indices: sequence of three int
            Surface normal in Miller indices (h,k,l).
        layers: int
            Number of equivalent layers of the slab. (not the same as the layers
            you choose from for terminations)
        vacuum: float
            Amount of vacuum added on both sides of the slab.
        termination: str
            the atoms you wish to be in the top layer. There may be many such
            terminations, this function returns all terminations with the same
            atomic composition.
            e.g. 'O' will return oxygen terminated surfaces.
            e.g.'TiO' returns surfaces terminated with layers containing both
            O and Ti
        Returns:
        return_surfs: List
            a list of surfaces that match the specifications given

    """
    lats = translate_lattice(lattice, indices)
    return_surfs = []
    check = []
    check2 = []
    for item in lats:
        too_similar = False
        surf = surface(item, indices, layers, vacuum=vacuum, tol=tol)
        surf.wrap(pbc=[True] * 3)  # standardize slabs

        positions = surf.get_scaled_positions().flatten()
        for i, value in enumerate(positions):
            if value >= 1 - tol:  # move things closer to zero within tol
                positions[i] -= 1
        surf.set_scaled_positions(np.reshape(positions, (len(surf), 3)))
        # rep = find_z_layers(surf)
        z_layers, _hs = get_layers(surf, (0, 0, 1))  # just z layers matter
        # get the indicies of the atoms in the highest layer
        top_layer = [
            i for i, val in enumerate(
                z_layers == max(z_layers)) if val]

        if termination is not None:
            comp = [surf.get_chemical_symbols()[a] for a in top_layer]
            term = string2symbols(termination)
            # list atoms in top layer and not in requested termination
            check = [a for a in comp if a not in term]
            # list of atoms in requested termination and not in top layer
            check2 = [a for a in term if a not in comp]
        if len(return_surfs) > 0:
            pos_diff = [a.get_positions() - surf.get_positions()
                        for a in return_surfs]
            for i, su in enumerate(pos_diff):
                similarity_test = su.flatten() < tol * 1000
                if similarity_test.all():
                    # checks if surface is too similar to another surface
                    too_similar = True
        if too_similar:
            continue
        if return_all is True:
            pass
        elif check != [] or check2 != []:
            continue
        return_surfs.append(surf)
    return return_surfs


def translate_lattice(lattice, indices, tol=10**-3):
    """translates a bulk unit cell along a normal vector given by the a set of
    miller indices to the next symetric position. This is used to control the
    termination of the surface in the smart_surface command
    Parameters:
    ==========
        lattice: Atoms object
            atoms object of the bulk unit cell
        indices: 1x3 list,tuple, or numpy array
            the miller indices you wish to cut along.
    returns:
        lattice_list: list of Atoms objects
            a list of all the different translations of the unit cell that will
            yield different terminations of a surface cut along the miller
            indices provided.
    """
    lattice_list = []
    cell = lattice.get_cell()
    pt = [0, 0, 0]
    h, k, l = indices  # noqa (E741 ambiguous name 'l')
    millers = list(indices)
    for index, item in enumerate(millers):
        if item == 0:
            millers[index] = 10**9  # make zeros large numbers
        elif pt == [0, 0, 0]:  # for numerical stability
            pt = list(cell[index] / float(item) / np.linalg.norm(cell[index]))
    h1, k1, l1 = millers
    N = np.array(cell[0] / h1 + cell[1] / k1 + cell[2] / l1)
    n = N / np.linalg.norm(N)  # making a unit vector normal to cut plane
    # finding distance from cut plan vector
    d = [np.round(np.dot(n, (a - pt)) * n, 5) for
         a in lattice.get_scaled_positions()]
    duplicates = []
    for i, item in enumerate(d):
        g = [True for a in d[i + 1:] if np.linalg.norm(a - item) < tol]
        if g != []:
            duplicates.append(i)
    duplicates.reverse()
    for i in duplicates:
        del d[i]
    # put distance to the plane at the end of the array
    for i, item in enumerate(d):
        d[i] = np.append(item,
                         np.dot(n, (lattice.get_scaled_positions()[i] - pt)))
    d = np.array(d)
    d = d[d[:, 3].argsort()]  # sort by distance to the plane
    d = [a[:3] for a in d]  # remove distance
    d = list(d)  # make it a list again
    for i in d:
        """
        The above method gives you the boundries of between terminations that
        will allow you to build a complete set of terminations. However, it
        does not return all the boundries. Thus you must check both above and
        below the boundary, and not stray too far from the boundary. If you move
        too far away, you risk hitting another boundary you did not find.
        """
        lattice1 = lattice.copy()
        displacement = (h * cell[0] + k * cell[1] + l * cell[2]) \
            * (i + 10 ** -8)
        lattice1.positions -= displacement
        lattice_list.append(lattice1)
        lattice1 = lattice.copy()
        displacement = (h * cell[0] + k * cell[1] + l * cell[2]) \
            * (i - 10 ** -8)
        lattice1.positions -= displacement
        lattice_list.append(lattice1)
    return lattice_list

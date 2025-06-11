# fmt: off

from itertools import combinations_with_replacement
from math import erf

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from ase.neighborlist import NeighborList
from ase.utils import pbc2pbc


class OFPComparator:
    """Implementation of comparison using Oganov's fingerprint (OFP)
    functions, based on:

      * :doi:`Oganov, Valle, J. Chem. Phys. 130, 104504 (2009)
        <10.1063/1.3079326>`

      * :doi:`Lyakhov, Oganov, Valle, Comp. Phys. Comm. 181 (2010) 1623-1632
        <10.1016/j.cpc.2010.06.007>`

    Parameters:

    n_top: int or None
           The number of atoms to optimize (None = include all).

    dE: float
        Energy difference above which two structures are
        automatically considered to be different. (Default 1 eV)

    cos_dist_max: float
        Maximal cosine distance between two structures in
        order to be still considered the same structure. Default 5e-3

    rcut: float
        Cutoff radius in Angstrom for the fingerprints.
        (Default 20 Angstrom)

    binwidth: float
        Width in Angstrom of the bins over which the fingerprints
        are discretized. (Default 0.05 Angstrom)

    pbc: list of three booleans or None
         Specifies whether to apply periodic boundary conditions
         along each of the three unit cell vectors when calculating
         the fingerprint. The default (None) is to apply PBCs in all
         3 directions.

         Note: for isolated systems (pbc = [False, False, False]),
         the pair correlation function itself is always short-ranged
         (decays to zero beyond a certain radius), so unity is not
         subtracted for calculating the fingerprint. Also the
         volume normalization disappears.

    maxdims: list of three floats or None
             If PBCs in only 1 or 2 dimensions are specified, the
             maximal thicknesses along the non-periodic directions can
             be specified here (the values given for the periodic
             directions will not be used). If set to None (the
             default), the length of the cell vector along the
             non-periodic direction is used.

             Note: in this implementation, the cell vectors are
             assumed to be orthogonal.

    sigma: float
           Standard deviation of the gaussian smearing to be applied
           in the calculation of the fingerprints (in
           Angstrom). Default 0.02 Angstrom.

    nsigma: int
            Distance (as the number of standard deviations sigma) at
            which the gaussian smearing is cut off (i.e. no smearing
            beyond that distance). (Default 4)

    recalculate: boolean
                 If True, ignores the fingerprints stored in
                 atoms.info and recalculates them. (Default False)

    """

    def __init__(self, n_top=None, dE=1.0, cos_dist_max=5e-3, rcut=20.,
                 binwidth=0.05, sigma=0.02, nsigma=4, pbc=True,
                 maxdims=None, recalculate=False):
        self.n_top = n_top or 0
        self.dE = dE
        self.cos_dist_max = cos_dist_max
        self.rcut = rcut
        self.binwidth = binwidth
        self.pbc = pbc2pbc(pbc)

        if maxdims is None:
            self.maxdims = [None] * 3
        else:
            self.maxdims = maxdims

        self.sigma = sigma
        self.nsigma = nsigma
        self.recalculate = recalculate
        self.dimensions = self.pbc.sum()

        if self.dimensions == 1 or self.dimensions == 2:
            for direction in range(3):
                if not self.pbc[direction]:
                    if self.maxdims[direction] is not None:
                        if self.maxdims[direction] <= 0:
                            e = '''If a max thickness is specificed in maxdims
                                  for a non-periodic direction, it has to be
                                  strictly positive.'''
                            raise ValueError(e)

    def looks_like(self, a1, a2):
        """ Return if structure a1 or a2 are similar or not. """
        if len(a1) != len(a2):
            raise Exception('The two configurations are not the same size.')

        # first we check the energy criteria
        if a1.calc is not None and a2.calc is not None:
            dE = abs(a1.get_potential_energy() - a2.get_potential_energy())
            if dE >= self.dE:
                return False

        # then we check the structure
        cos_dist = self._compare_structure(a1, a2)
        verdict = cos_dist < self.cos_dist_max
        return verdict

    def _json_encode(self, fingerprints, typedic):
        """ json does not accept tuples nor integers as dict keys,
        so in order to write the fingerprints to atoms.info, we need
        to convert them to strings """
        fingerprints_encoded = {}
        for key, val in fingerprints.items():
            try:
                newkey = "_".join(map(str, list(key)))
            except TypeError:
                newkey = str(key)
            if isinstance(val, dict):
                fingerprints_encoded[newkey] = {
                    str(key2): val2 for key2, val2 in val.items()}
            else:
                fingerprints_encoded[newkey] = val
        typedic_encoded = {}
        for key, val in typedic.items():
            newkey = str(key)
            typedic_encoded[newkey] = val
        return [fingerprints_encoded, typedic_encoded]

    def _json_decode(self, fingerprints, typedic):
        """ This is the reverse operation of _json_encode """
        fingerprints_decoded = {}
        for key, val in fingerprints.items():
            newkey = list(map(int, key.split("_")))
            if len(newkey) > 1:
                newkey = tuple(newkey)
            else:
                newkey = newkey[0]

            if isinstance(val, dict):
                fingerprints_decoded[newkey] = {
                    int(key2): np.array(val2) for key2, val2 in val.items()
                }
            else:
                fingerprints_decoded[newkey] = np.array(val)
        typedic_decoded = {}
        for key, val in typedic.items():
            newkey = int(key)
            typedic_decoded[newkey] = val
        return [fingerprints_decoded, typedic_decoded]

    def _compare_structure(self, a1, a2):
        """ Returns the cosine distance between the two structures,
            using their fingerprints. """

        if len(a1) != len(a2):
            raise Exception('The two configurations are not the same size.')

        a1top = a1[-self.n_top:]
        a2top = a2[-self.n_top:]

        if 'fingerprints' in a1.info and not self.recalculate:
            fp1, typedic1 = a1.info['fingerprints']
            fp1, typedic1 = self._json_decode(fp1, typedic1)
        else:
            fp1, typedic1 = self._take_fingerprints(a1top)
            a1.info['fingerprints'] = self._json_encode(fp1, typedic1)

        if 'fingerprints' in a2.info and not self.recalculate:
            fp2, typedic2 = a2.info['fingerprints']
            fp2, typedic2 = self._json_decode(fp2, typedic2)
        else:
            fp2, typedic2 = self._take_fingerprints(a2top)
            a2.info['fingerprints'] = self._json_encode(fp2, typedic2)

        if sorted(fp1) != sorted(fp2):
            raise AssertionError('The two structures have fingerprints '
                                 'with different compounds.')
        for key in typedic1:
            if not np.array_equal(typedic1[key], typedic2[key]):
                raise AssertionError('The two structures have a different '
                                     'stoichiometry or ordering!')

        cos_dist = self._cosine_distance(fp1, fp2, typedic1)
        return cos_dist

    def _get_volume(self, a):
        ''' Calculates the normalizing value, and other parameters
        (pmin,pmax,qmin,qmax) that are used for surface area calculation
        in the case of 1 or 2-D periodicity.'''

        cell = a.get_cell()
        scalpos = a.get_scaled_positions()

        # defaults:
        volume = 1.
        pmin, pmax, qmin, qmax = [0.] * 4

        if self.dimensions == 1 or self.dimensions == 2:
            for direction in range(3):
                if not self.pbc[direction]:
                    if self.maxdims[direction] is None:
                        maxdim = np.linalg.norm(cell[direction, :])
                        self.maxdims[direction] = maxdim

        pbc_dirs = [i for i in range(3) if self.pbc[i]]
        non_pbc_dirs = [i for i in range(3) if not self.pbc[i]]

        if self.dimensions == 3:
            volume = abs(np.dot(np.cross(cell[0, :], cell[1, :]), cell[2, :]))

        elif self.dimensions == 2:
            non_pbc_dir = non_pbc_dirs[0]

            a = np.cross(cell[pbc_dirs[0], :], cell[pbc_dirs[1], :])
            b = self.maxdims[non_pbc_dir]
            b /= np.linalg.norm(cell[non_pbc_dir, :])

            volume = np.abs(np.dot(a, b * cell[non_pbc_dir, :]))

            maxpos = np.max(scalpos[:, non_pbc_dir])
            minpos = np.min(scalpos[:, non_pbc_dir])
            pwidth = maxpos - minpos
            pmargin = 0.5 * (b - pwidth)
            # note: here is a place where we assume that the
            # non-periodic direction is orthogonal to the periodic ones:
            pmin = np.min(scalpos[:, non_pbc_dir]) - pmargin
            pmin *= np.linalg.norm(cell[non_pbc_dir, :])
            pmax = np.max(scalpos[:, non_pbc_dir]) + pmargin
            pmax *= np.linalg.norm(cell[non_pbc_dir, :])

        elif self.dimensions == 1:
            pbc_dir = pbc_dirs[0]

            v0 = cell[non_pbc_dirs[0], :]
            b0 = self.maxdims[non_pbc_dirs[0]]
            b0 /= np.linalg.norm(cell[non_pbc_dirs[0], :])
            v1 = cell[non_pbc_dirs[1], :]
            b1 = self.maxdims[non_pbc_dirs[1]]
            b1 /= np.linalg.norm(cell[non_pbc_dirs[1], :])

            volume = np.abs(np.dot(np.cross(b0 * v0, b1 * v1),
                                   cell[pbc_dir, :]))

            # note: here is a place where we assume that the
            # non-periodic direction is orthogonal to the periodic ones:
            maxpos = np.max(scalpos[:, non_pbc_dirs[0]])
            minpos = np.min(scalpos[:, non_pbc_dirs[0]])
            pwidth = maxpos - minpos
            pmargin = 0.5 * (b0 - pwidth)

            pmin = np.min(scalpos[:, non_pbc_dirs[0]]) - pmargin
            pmin *= np.linalg.norm(cell[non_pbc_dirs[0], :])
            pmax = np.max(scalpos[:, non_pbc_dirs[0]]) + pmargin
            pmax *= np.linalg.norm(cell[non_pbc_dirs[0], :])

            maxpos = np.max(scalpos[:, non_pbc_dirs[1]])
            minpos = np.min(scalpos[:, non_pbc_dirs[1]])
            qwidth = maxpos - minpos
            qmargin = 0.5 * (b1 - qwidth)

            qmin = np.min(scalpos[:, non_pbc_dirs[1]]) - qmargin
            qmin *= np.linalg.norm(cell[non_pbc_dirs[1], :])
            qmax = np.max(scalpos[:, non_pbc_dirs[1]]) + qmargin
            qmax *= np.linalg.norm(cell[non_pbc_dirs[1], :])

        elif self.dimensions == 0:
            volume = 1.

        return [volume, pmin, pmax, qmin, qmax]

    def _take_fingerprints(self, atoms, individual=False):
        """ Returns a [fingerprints,typedic] list, where fingerprints
        is a dictionary with the fingerprints, and typedic is a
        dictionary with the list of atom indices for each element
        (or "type") in the atoms object.
        The keys in the fingerprints dictionary are the (A,B) tuples,
        which are the different element-element combinations in the
        atoms object (A and B are the atomic numbers).
        When A != B, the (A,B) tuple is sorted (A < B).

        If individual=True, a dict is returned, where each atom index
        has an {atomic_number:fingerprint} dict as value.
        If individual=False, the fingerprints from atoms of the same
        atomic number are added together."""

        pos = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.get_cell()

        unique_types = np.unique(num)
        posdic = {}
        typedic = {}
        for t in unique_types:
            tlist = [i for i, atom in enumerate(atoms) if atom.number == t]
            typedic[t] = tlist
            posdic[t] = pos[tlist]

        # determining the volume normalization and other parameters
        volume, pmin, pmax, qmin, qmax = self._get_volume(atoms)

        # functions for calculating the surface area
        non_pbc_dirs = [i for i in range(3) if not self.pbc[i]]

        def surface_area_0d(r):
            return 4 * np.pi * (r**2)

        def surface_area_1d(r, pos):
            q0 = pos[non_pbc_dirs[1]]
            phi1 = np.lib.scimath.arccos((qmax - q0) / r).real
            phi2 = np.pi - np.lib.scimath.arccos((qmin - q0) / r).real
            factor = 1 - (phi1 + phi2) / np.pi
            return surface_area_2d(r, pos) * factor

        def surface_area_2d(r, pos):
            p0 = pos[non_pbc_dirs[0]]
            area = np.minimum(pmax - p0, r) + np.minimum(p0 - pmin, r)
            area *= 2 * np.pi * r
            return area

        def surface_area_3d(r):
            return 4 * np.pi * (r**2)

        # build neighborlist
        # this is computationally the most intensive part
        a = atoms.copy()
        a.set_pbc(self.pbc)
        nl = NeighborList([self.rcut / 2.] * len(a), skin=0.,
                          self_interaction=False, bothways=True)
        nl.update(a)

        # parameters for the binning:
        m = int(np.ceil(self.nsigma * self.sigma / self.binwidth))
        x = 0.25 * np.sqrt(2) * self.binwidth * (2 * m + 1) * 1. / self.sigma
        smearing_norm = erf(x)
        nbins = int(np.ceil(self.rcut * 1. / self.binwidth))
        bindist = self.binwidth * np.arange(1, nbins + 1)

        def take_individual_rdf(index, unique_type):
            # Computes the radial distribution function of atoms
            # of type unique_type around the atom with index "index".
            rdf = np.zeros(nbins)

            if self.dimensions == 3:
                weights = 1. / surface_area_3d(bindist)
            elif self.dimensions == 2:
                weights = 1. / surface_area_2d(bindist, pos[index])
            elif self.dimensions == 1:
                weights = 1. / surface_area_1d(bindist, pos[index])
            elif self.dimensions == 0:
                weights = 1. / surface_area_0d(bindist)
            weights /= self.binwidth

            indices, offsets = nl.get_neighbors(index)
            valid = np.where(num[indices] == unique_type)
            p = pos[indices[valid]] + np.dot(offsets[valid], cell)
            r = cdist(p, [pos[index]])
            bins = np.floor(r / self.binwidth)

            for i in range(-m, m + 1):
                newbins = bins + i
                valid = np.where((newbins >= 0) & (newbins < nbins))
                valid_bins = newbins[valid].astype(int)
                values = weights[valid_bins]

                c = 0.25 * np.sqrt(2) * self.binwidth * 1. / self.sigma
                values *= 0.5 * erf(c * (2 * i + 1)) - \
                    0.5 * erf(c * (2 * i - 1))
                values /= smearing_norm

                for j, valid_bin in enumerate(valid_bins):
                    rdf[valid_bin] += values[j]

            rdf /= len(typedic[unique_type]) * 1. / volume
            return rdf

        fingerprints = {}
        if individual:
            for i in range(len(atoms)):
                fingerprints[i] = {}
                for unique_type in unique_types:
                    fingerprint = take_individual_rdf(i, unique_type)
                    if self.dimensions > 0:
                        fingerprint -= 1
                    fingerprints[i][unique_type] = fingerprint
        else:
            for t1, t2 in combinations_with_replacement(unique_types, r=2):
                key = (t1, t2)
                fingerprint = np.zeros(nbins)
                for i in typedic[t1]:
                    fingerprint += take_individual_rdf(i, t2)
                fingerprint /= len(typedic[t1])
                if self.dimensions > 0:
                    fingerprint -= 1
                fingerprints[key] = fingerprint

        return [fingerprints, typedic]

    def _calculate_local_orders(self, individual_fingerprints, typedic,
                                volume):
        """ Returns a list with the local order for every atom,
        using the definition of local order from
        Lyakhov, Oganov, Valle, Comp. Phys. Comm. 181 (2010) 1623-1632
        :doi:`10.1016/j.cpc.2010.06.007`"""

        # total number of atoms:
        n_tot = sum(len(typedic[key]) for key in typedic)
        inv_n_tot = 1. / n_tot

        local_orders = []
        for fingerprints in individual_fingerprints.values():
            local_order = 0
            for unique_type, fingerprint in fingerprints.items():
                term = np.linalg.norm(fingerprint)**2
                term *= self.binwidth
                term *= (volume * inv_n_tot)**(-1 / 3)
                term *= len(typedic[unique_type]) * inv_n_tot
                local_order += term
            local_orders.append(np.sqrt(local_order))

        return local_orders

    def get_local_orders(self, a):
        """ Returns the local orders of all the atoms."""

        a_top = a[-self.n_top:]
        key = 'individual_fingerprints'

        if key in a.info and not self.recalculate:
            fp, typedic = self._json_decode(*a.info[key])
        else:
            fp, typedic = self._take_fingerprints(a_top, individual=True)
            a.info[key] = self._json_encode(fp, typedic)

        volume, _pmin, _pmax, _qmin, _qmax = self._get_volume(a_top)
        return self._calculate_local_orders(fp, typedic, volume)

    def _cosine_distance(self, fp1, fp2, typedic):
        """ Returns the cosine distance from two fingerprints.
        It also needs information about the number of atoms from
        each element, which is included in "typedic"."""

        keys = sorted(fp1)

        # calculating the weights:
        w = {}
        wtot = 0
        for key in keys:
            weight = len(typedic[key[0]]) * len(typedic[key[1]])
            wtot += weight
            w[key] = weight
        for key in keys:
            w[key] *= 1. / wtot

        # calculating the fingerprint norms:
        norm1 = 0
        norm2 = 0
        for key in keys:
            norm1 += (np.linalg.norm(fp1[key])**2) * w[key]
            norm2 += (np.linalg.norm(fp2[key])**2) * w[key]
        norm1 = np.sqrt(norm1)
        norm2 = np.sqrt(norm2)

        # calculating the distance:
        distance = 0
        for key in keys:
            distance += np.sum(fp1[key] * fp2[key]) * w[key] / (norm1 * norm2)

        distance = 0.5 * (1 - distance)
        return distance

    def plot_fingerprints(self, a, prefix=''):
        """ Function for quickly plotting all the fingerprints.
        Prefix = a prefix you want to give to the resulting PNG file."""

        if 'fingerprints' in a.info and not self.recalculate:
            fp, typedic = a.info['fingerprints']
            fp, typedic = self._json_decode(fp, typedic)
        else:
            a_top = a[-self.n_top:]
            fp, typedic = self._take_fingerprints(a_top)
            a.info['fingerprints'] = self._json_encode(fp, typedic)

        npts = int(np.ceil(self.rcut * 1. / self.binwidth))
        x = np.linspace(0, self.rcut, npts, endpoint=False)

        for key, val in fp.items():
            plt.plot(x, val)
            suffix = f"_fp_{key[0]}_{key[1]}.png"
            plt.savefig(prefix + suffix)
            plt.clf()

    def plot_individual_fingerprints(self, a, prefix=''):
        """ Function for plotting all the individual fingerprints.
        Prefix = a prefix for the resulting PNG file."""
        if 'individual_fingerprints' in a.info and not self.recalculate:
            fp, typedic = a.info['individual_fingerprints']
        else:
            a_top = a[-self.n_top:]
            fp, typedic = self._take_fingerprints(a_top, individual=True)
            a.info['individual_fingerprints'] = [fp, typedic]

        npts = int(np.ceil(self.rcut * 1. / self.binwidth))
        x = np.linspace(0, self.rcut, npts, endpoint=False)

        for key, val in fp.items():
            for key2, val2 in val.items():
                plt.plot(x, val2)
                plt.ylim([-1, 10])
                suffix = f"_individual_fp_{key}_{key2}.png"
                plt.savefig(prefix + suffix)
                plt.clf()

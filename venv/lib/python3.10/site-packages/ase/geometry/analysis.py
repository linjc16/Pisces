# fmt: off

# flake8: noqa
"""Tools for analyzing instances of :class:`~ase.Atoms`
"""

from typing import List, Optional

import numpy as np

from ase import Atoms
from ase.geometry.rdf import get_containing_cell_length, get_rdf
from ase.neighborlist import (build_neighbor_list, get_distance_indices,
                              get_distance_matrix)

__all__ = ['Analysis']


def get_max_containing_cell_length(images: List[Atoms]):
    i2diff = np.zeros(3)
    for image in images:
        np.maximum(get_containing_cell_length(image), i2diff, out=i2diff)
    return i2diff


def get_max_volume_estimate(images: List[Atoms]) -> float:
    return np.prod(get_max_containing_cell_length(images))


class Analysis:
    """Analysis class

    Parameters for initialization:

    images: :class:`~ase.Atoms` object or list of such
        Images to analyze.
    nl: None, :class:`~ase.neighborlist.NeighborList` object or list of such
        Neighborlist(s) for the given images. One or nImages, depending if bonding
        pattern changes or is constant. Using one Neigborlist greatly improves speed.
    kwargs: options, dict
        Arguments for constructing :class:`~ase.neighborlist.NeighborList` object if :data:`nl` is None.

    The choice of ``bothways=True`` for the :class:`~ase.neighborlist.NeighborList` object
    will not influence the amount of bonds/angles/dihedrals you get, all are reported
    in both directions. Use the *unique*-labeled properties to get lists without
    duplicates.
    """

    def __init__(self, images, nl=None, **kwargs):
        self.images = images

        if isinstance(nl, list):
            assert len(nl) == self.nImages
            self._nl = nl
        elif nl is not None:
            self._nl = [nl]
        else:
            self._nl = [build_neighbor_list(self.images[0], **kwargs)]

        self._cache = {}

    def _get_slice(self, imageIdx):
        """Return a slice from user input.
        Using *imageIdx* (can be integer or slice) the analyzed frames can be specified.
        If *imageIdx* is None, all frames will be analyzed.
        """
        # get slice from imageIdx
        if isinstance(imageIdx, int):
            sl = slice(imageIdx, imageIdx + 1)
        elif isinstance(imageIdx, slice):
            sl = imageIdx
        elif imageIdx is None:
            sl = slice(0, None)
        else:
            raise ValueError(
                "Unsupported type for imageIdx in ase.geometry.analysis.Analysis._get_slice")
        return sl

    @property
    def images(self):
        """Images.

        Set during initialization but can also be set later.
        """
        return self._images

    @images.setter
    def images(self, images):
        """Set images"""
        if isinstance(images, list):
            self._images = images
        else:
            self._images = [images]

    @images.deleter
    def images(self):
        """Delete images"""
        self._images = None

    @property
    def nImages(self):
        """Number of Images in this instance.

        Cannot be set, is determined automatically.
        """
        return len(self.images)

    @property
    def nl(self):
        """Neighbor Lists in this instance.

        Set during initialization.

        **No setter or deleter, only getter**
        """
        return self._nl

    def _get_all_x(self, distance):
        """Helper function to get bonds, angles, dihedrals"""
        maxIter = self.nImages
        if len(self.nl) == 1:
            maxIter = 1

        xList = []
        for i in range(maxIter):
            xList.append(get_distance_indices(
                self.distance_matrix[i], distance))

        return xList

    @property
    def all_bonds(self):
        """All Bonds.

        A list with indices of bonded atoms for each neighborlist in *self*.
        Atom i is connected to all atoms inside result[i]. Duplicates from PBCs are
        removed. See also :data:`unique_bonds`.

        **No setter or deleter, only getter**
        """
        if 'allBonds' not in self._cache:
            self._cache['allBonds'] = self._get_all_x(1)

        return self._cache['allBonds']

    @property
    def all_angles(self):
        """All angles

        A list with indices of atoms in angles for each neighborlist in *self*.
        Atom i forms an angle to the atoms inside the tuples in result[i]:
        i -- result[i][x][0] -- result[i][x][1]
        where x is in range(number of angles from i). See also :data:`unique_angles`.

        **No setter or deleter, only getter**
        """
        if 'allAngles' not in self._cache:
            self._cache['allAngles'] = []
            distList = self._get_all_x(2)

            for imI in range(len(distList)):
                self._cache['allAngles'].append([])
                # iterate over second neighbors of all atoms
                for iAtom, secNeighs in enumerate(distList[imI]):
                    self._cache['allAngles'][-1].append([])
                    if len(secNeighs) == 0:
                        continue
                    firstNeighs = self.all_bonds[imI][iAtom]
                    # iterate over second neighbors of iAtom
                    for kAtom in secNeighs:
                        relevantFirstNeighs = [
                            idx for idx in firstNeighs if kAtom in self.all_bonds[imI][idx]]
                        # iterate over all atoms that are connected to iAtom and kAtom
                        for jAtom in relevantFirstNeighs:
                            self._cache['allAngles'][-1][-1].append(
                                (jAtom, kAtom))

        return self._cache['allAngles']

    @property
    def all_dihedrals(self):
        """All dihedrals

        Returns a list with indices of atoms in dihedrals for each neighborlist in this instance.
        Atom i forms a dihedral to the atoms inside the tuples in result[i]:
        i -- result[i][x][0] -- result[i][x][1] -- result[i][x][2]
        where x is in range(number of dihedrals from i). See also :data:`unique_dihedrals`.

        **No setter or deleter, only getter**
        """
        if 'allDihedrals' not in self._cache:
            self._cache['allDihedrals'] = []
            distList = self._get_all_x(3)

            for imI in range(len(distList)):
                self._cache['allDihedrals'].append([])
                for iAtom, thirdNeighs in enumerate(distList[imI]):
                    self._cache['allDihedrals'][-1].append([])
                    if len(thirdNeighs) == 0:
                        continue
                    anglesI = self.all_angles[imI][iAtom]
                    # iterate over third neighbors of iAtom
                    for lAtom in thirdNeighs:
                        secondNeighs = [angle[-1] for angle in anglesI]
                        firstNeighs = [angle[0] for angle in anglesI]
                        relevantSecondNeighs = [
                            idx for idx in secondNeighs if lAtom in self.all_bonds[imI][idx]]
                        relevantFirstNeighs = [
                            firstNeighs[secondNeighs.index(idx)] for idx in relevantSecondNeighs]
                        # iterate over all atoms that are connected to iAtom and lAtom
                        for jAtom, kAtom in zip(relevantFirstNeighs, relevantSecondNeighs):
                            # remove dihedrals in circles
                            tupl = (jAtom, kAtom, lAtom)
                            if len(set((iAtom, ) + tupl)) != 4:
                                continue
                            # avoid duplicates
                            elif tupl in self._cache['allDihedrals'][-1][-1]:
                                continue
                            elif iAtom in tupl:
                                raise RuntimeError(
                                    "Something is wrong in analysis.all_dihedrals!")
                            self._cache['allDihedrals'][-1][-1].append(
                                (jAtom, kAtom, lAtom))

        return self._cache['allDihedrals']

    @property
    def adjacency_matrix(self):
        """The adjacency/connectivity matrix.

        If not already done, build a list of adjacency matrices for all :data:`nl`.

        **No setter or deleter, only getter**
        """

        if 'adjacencyMatrix' not in self._cache:
            self._cache['adjacencyMatrix'] = []
            for i in range(len(self.nl)):
                self._cache['adjacencyMatrix'].append(
                    self.nl[i].get_connectivity_matrix())

        return self._cache['adjacencyMatrix']

    @property
    def distance_matrix(self):
        """The distance matrix.

        If not already done, build a list of distance matrices for all :data:`nl`. See
        :meth:`ase.neighborlist.get_distance_matrix`.

        **No setter or deleter, only getter**
        """

        if 'distanceMatrix' not in self._cache:
            self._cache['distanceMatrix'] = []
            for i in range(len(self.nl)):
                self._cache['distanceMatrix'].append(
                    get_distance_matrix(self.adjacency_matrix[i]))

        return self._cache['distanceMatrix']

    @property
    def unique_bonds(self):
        """Get Unique Bonds.

        :data:`all_bonds` i-j without j-i. This is the upper triangle of the
        connectivity matrix (i,j), `i < j`

        """
        bonds = []
        for imI in range(len(self.all_bonds)):
            bonds.append([])
            for iAtom, bonded in enumerate(self.all_bonds[imI]):
                bonds[-1].append([jAtom for jAtom in bonded if jAtom > iAtom])

        return bonds

    def _filter_unique(self, l):
        """Helper function to filter for unique lists in a list
        that also contains the reversed items.
        """
        r = []
        # iterate over images
        for imI in range(len(l)):
            r.append([])
            # iterate over atoms
            for i, tuples in enumerate(l[imI]):
                # add the ones where i is smaller than the last element
                r[-1].append([x for x in tuples if i < x[-1]])
        return r

    def clear_cache(self):
        """Delete all cached information."""
        self._cache = {}

    @property
    def unique_angles(self):
        """Get Unique Angles.

        :data:`all_angles` i-j-k without k-j-i.

        """
        return self._filter_unique(self.all_angles)

    @property
    def unique_dihedrals(self):
        """Get Unique Dihedrals.

        :data:`all_dihedrals` i-j-k-l without l-k-j-i.

        """
        return self._filter_unique(self.all_dihedrals)

    def _get_symbol_idxs(self, imI, sym):
        """Get list of indices of element *sym*"""
        if isinstance(imI, int):
            return [idx for idx in range(len(self.images[imI])) if self.images[imI][idx].symbol == sym]
        else:
            return [idx for idx in range(len(imI)) if imI[idx].symbol == sym]

    def _idxTuple2SymbolTuple(self, imI, tup):
        """Converts a tuple of indices to their symbols"""
        return (self.images[imI][idx].symbol for idx in tup)

    def get_bonds(self, A, B, unique=True):
        """Get bonds from element A to element B.

        Parameters:

        A, B: str
            Get Bonds between elements A and B
        unique: bool
            Return the bonds both ways or just one way (A-B and B-A or only A-B)

        Returns:

        return: list of lists of tuples
            return[imageIdx][atomIdx][bondI], each tuple starts with atomIdx.

        Use :func:`get_values` to convert the returned list to values.
        """
        r = []
        for imI in range(len(self.all_bonds)):
            r.append([])
            aIdxs = self._get_symbol_idxs(imI, A)
            if A != B:
                bIdxs = self._get_symbol_idxs(imI, B)
            for idx in aIdxs:
                bonded = self.all_bonds[imI][idx]
                if A == B:
                    r[-1].extend([(idx, x)
                                 for x in bonded if (x in aIdxs) and (x > idx)])
                else:
                    r[-1].extend([(idx, x) for x in bonded if x in bIdxs])

            if not unique:
                r[-1] += [x[::-1] for x in r[-1]]

        return r

    def get_angles(self, A, B, C, unique=True):
        """Get angles from given elements A-B-C.

        Parameters:

        A, B, C: str
            Get Angles between elements A, B and C. **B will be the central atom**.
        unique: bool
            Return the angles both ways or just one way (A-B-C and C-B-A or only A-B-C)

        Returns:

        return: list of lists of tuples
            return[imageIdx][atomIdx][angleI], each tuple starts with atomIdx.

        Use :func:`get_values` to convert the returned list to values.
        """
        from itertools import combinations, product
        r = []
        for imI in range(len(self.all_angles)):
            r.append([])
            # Middle Atom is fixed
            bIdxs = self._get_symbol_idxs(imI, B)
            for bIdx in bIdxs:
                bondedA = [idx for idx in self.all_bonds[imI]
                           [bIdx] if self.images[imI][idx].symbol == A]
                if len(bondedA) == 0:
                    continue

                if A != C:
                    bondedC = [idx for idx in self.all_bonds[imI]
                               [bIdx] if self.images[imI][idx].symbol == C]
                    if len(bondedC) == 0:
                        continue

                if A == C:
                    extend = [(x[0], bIdx, x[1])
                              for x in list(combinations(bondedA, 2))]
                else:
                    extend = list(product(bondedA, [bIdx], bondedC))

                if not unique:
                    extend += [x[::-1] for x in extend]

                r[-1].extend(extend)
        return r

    def get_dihedrals(self, A, B, C, D, unique=True):
        """Get dihedrals A-B-C-D.

        Parameters:

        A, B, C, D: str
            Get Dihedralss between elements A, B, C and D. **B-C will be the central axis**.
        unique: bool
            Return the dihedrals both ways or just one way (A-B-C-D and D-C-B-A or only A-B-C-D)

        Returns:

        return: list of lists of tuples
            return[imageIdx][atomIdx][dihedralI], each tuple starts with atomIdx.

        Use :func:`get_values` to convert the returned list to values.
        """
        r = []
        for imI in range(len(self.all_dihedrals)):
            r.append([])
            # get indices of elements
            aIdxs = self._get_symbol_idxs(imI, A)
            bIdxs = self._get_symbol_idxs(imI, B)
            cIdxs = self._get_symbol_idxs(imI, C)
            dIdxs = self._get_symbol_idxs(imI, D)
            for aIdx in aIdxs:
                dihedrals = [(aIdx, ) + d for d in self.all_dihedrals[imI][aIdx]
                             if (d[0] in bIdxs) and (d[1] in cIdxs) and (d[2] in dIdxs)]
                if not unique:
                    dihedrals += [d[::-1] for d in dihedrals]
                r[-1].extend(dihedrals)

        return r

    def get_bond_value(self, imIdx, idxs, mic=True, **kwargs):
        """Get bond length.

        Parameters:

        imIdx: int
            Index of Image to get value from.
        idxs: tuple or list of integers
            Get distance between atoms idxs[0]-idxs[1].
        mic: bool
            Passed on to :func:`ase.Atoms.get_distance` for retrieving the value, defaults to True.
            If the cell of the image is correctly set, there should be no reason to change this.
        kwargs: options or dict
            Passed on to :func:`ase.Atoms.get_distance`.

        Returns:

        return: float
            Value returned by image.get_distance.
        """
        return self.images[imIdx].get_distance(idxs[0], idxs[1], mic=mic, **kwargs)

    def get_angle_value(self, imIdx, idxs, mic=True, **kwargs):
        """Get angle.

        Parameters:

        imIdx: int
            Index of Image to get value from.
        idxs: tuple or list of integers
            Get angle between atoms idxs[0]-idxs[1]-idxs[2].
        mic: bool
            Passed on to :func:`ase.Atoms.get_angle` for retrieving the value, defaults to True.
            If the cell of the image is correctly set, there should be no reason to change this.
        kwargs: options or dict
            Passed on to :func:`ase.Atoms.get_angle`.

        Returns:

        return: float
            Value returned by image.get_angle.
        """
        return self.images[imIdx].get_angle(idxs[0], idxs[1], idxs[2], mic=True, **kwargs)

    def get_dihedral_value(self, imIdx, idxs, mic=True, **kwargs):
        """Get dihedral.

        Parameters:

        imIdx: int
            Index of Image to get value from.
        idxs: tuple or list of integers
            Get angle between atoms idxs[0]-idxs[1]-idxs[2]-idxs[3].
        mic: bool
            Passed on to :func:`ase.Atoms.get_dihedral` for retrieving the value, defaults to True.
            If the cell of the image is correctly set, there should be no reason to change this.
        kwargs: options or dict
            Passed on to :func:`ase.Atoms.get_dihedral`.

        Returns:

        return: float
            Value returned by image.get_dihedral.
        """
        return self.images[imIdx].get_dihedral(idxs[0], idxs[1], idxs[2], idxs[3], mic=mic, **kwargs)

    def get_values(self, inputList, imageIdx=None, mic=True, **kwargs):
        """Get Bond/Angle/Dihedral values.

        Parameters:

        inputList: list of lists of tuples
            Can be any list provided by :meth:`~ase.geometry.analysis.Analysis.get_bonds`,
            :meth:`~ase.geometry.analysis.Analysis.get_angles` or
            :meth:`~ase.geometry.analysis.Analysis.get_dihedrals`.
        imageIdx: integer or slice
            The images from :data:`images` to be analyzed. If None, all frames will be analyzed.
            See :func:`~ase.geometry.analysis.Analysis._get_slice` for details.
        mic: bool
            Passed on to :class:`~ase.Atoms` for retrieving the values, defaults to True.
            If the cells of the images are correctly set, there should be no reason to change this.
        kwargs: options or dict
            Passed on to the :class:`~ase.Atoms` classes functions for retrieving the values.

        Returns:

        return: list of lists of floats
            return[imageIdx][valueIdx]. Has the same shape as the *inputList*, instead of each
            tuple there is a float with the value this tuple yields.

        The type of value requested is determined from the length of the tuple inputList[0][0].
        The methods from the :class:`~ase.Atoms` class are used.
        """

        sl = self._get_slice(imageIdx)

        # get method to call from length of inputList
        if len(inputList[0][0]) == 2:
            get = self.get_bond_value
        elif len(inputList[0][0]) == 3:
            get = self.get_angle_value
        elif len(inputList[0][0]) == 4:
            get = self.get_dihedral_value
        else:
            raise ValueError(
                "inputList in ase.geometry.analysis.Analysis.get_values has a bad shape.")

        # check if length of slice and inputList match
        singleNL = False
        if len(inputList) != len(self.images[sl]):
            # only one nl for all images
            if len(inputList) == 1 and len(self.nl) == 1:
                singleNL = True
            else:
                raise RuntimeError("Length of inputList does not match length of \
                        images requested, but it also is not one item long.")

        r = []
        for inputIdx, image in enumerate(self.images[sl]):
            imageIdx = self.images.index(image)
            r.append([])
            # always use first list from input if only a single neighborlist was used
            if singleNL:
                inputIdx = 0
            for tupl in inputList[inputIdx]:
                r[-1].append(get(imageIdx, tupl, mic=mic, **kwargs))

        return r

    def get_max_volume_estimate(self):
        return get_max_volume_estimate(self.images)

    def get_rdf(self, rmax, nbins, imageIdx=None, elements=None, return_dists=False,
                volume: Optional[float] = None):
        """Get RDF.

        Wrapper for :meth:`ase.ga.utilities.get_rdf` with more selection possibilities.

        Parameters:

        rmax: float
            Maximum distance of RDF.
        nbins: int
            Number of bins to divide RDF.
        imageIdx: int/slice/None
            Images to analyze, see :func:`_get_slice` for details.
        elements: str/int/list/tuple
            Make partial RDFs.

        If elements is *None*, a full RDF is calculated. If elements is an *integer* or a *list/tuple
        of integers*, only those atoms will contribute to the RDF (like a mask). If elements
        is a *string* or a *list/tuple of strings*, only Atoms of those elements will contribute.

        Returns:

        return: list of lists / list of tuples of lists
            If return_dists is True, the returned tuples contain (rdf, distances). Otherwise
            only rdfs for each image are returned.
        """

        sl = self._get_slice(imageIdx)

        ls_rdf = []
        el = None

        for image in self.images[sl]:
            if elements is None:
                tmp_image = image
            # integers
            elif isinstance(elements, int):
                tmp_image = Atoms(cell=image.get_cell(), pbc=image.get_pbc())
                tmp_image.append(image[elements])
            # strings
            elif isinstance(elements, str):
                tmp_image = Atoms(cell=image.get_cell(), pbc=image.get_pbc())
                for idx in self._get_symbol_idxs(image, elements):
                    tmp_image.append(image[idx])
            # lists
            elif isinstance(elements, (list, tuple)):
                # list of ints
                if all(isinstance(x, int) for x in elements):
                    if len(elements) == 2:
                        # use builtin get_rdf mask
                        el = elements
                        tmp_image = image
                    else:
                        # create dummy image
                        tmp_image = Atoms(
                            cell=image.get_cell(), pbc=image.get_pbc())
                        for idx in elements:
                            tmp_image.append(image[idx])
                # list of strings
                elif all(isinstance(x, str) for x in elements):
                    tmp_image = Atoms(cell=image.get_cell(),
                                      pbc=image.get_pbc())
                    for element in elements:
                        for idx in self._get_symbol_idxs(image, element):
                            tmp_image.append(image[idx])
                else:
                    raise ValueError(
                        "Unsupported type of elements given in ase.geometry.analysis.Analysis.get_rdf!")
            else:
                raise ValueError(
                    "Unsupported type of elements given in ase.geometry.analysis.Analysis.get_rdf!")

            rdf = get_rdf(tmp_image, rmax, nbins, elements=el, no_dists=(not return_dists),
                          volume=volume)
            ls_rdf.append(rdf)

        return ls_rdf

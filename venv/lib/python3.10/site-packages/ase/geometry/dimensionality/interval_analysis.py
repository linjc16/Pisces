# fmt: off

"""Implements the dimensionality scoring parameter.

Method is described in:
Definition of a scoring parameter to identify low-dimensional materials
components
P.M. Larsen, M. Pandey, M. Strange, and K. W. Jacobsen
Phys. Rev. Materials 3 034003, 2019
https://doi.org/10.1103/PhysRevMaterials.3.034003
"""
from collections import namedtuple

import numpy as np

from ase.geometry.dimensionality import rank_determination, topology_scaling
from ase.geometry.dimensionality.bond_generator import next_bond

KInterval = namedtuple('KInterval', 'dimtype score a b h components cdim')


def f(x):
    if x == float("inf"):
        return 1
    k = 1 / 0.15**2
    return k * max(0, x - 1)**2 / (1. + k * max(0, x - 1)**2)


def calculate_score(a, b):
    return f(b) - f(a)


def reduced_histogram(h):
    h = [int(e > 0) for e in h]
    return tuple(h)


def build_dimtype(h):
    h = reduced_histogram(h)
    return ''.join([str(i) for i, e in enumerate(h) if e > 0]) + 'D'


def build_kinterval(a, b, h, components, cdim, score=None):
    if score is None:
        score = calculate_score(a, b)

    return KInterval(dimtype=build_dimtype(h), score=score,
                     a=a, b=b, h=h, components=components, cdim=cdim)


def merge_intervals(intervals):
    """Merges intervals of the same dimensionality type.

    For example, two histograms with component histograms [10, 4, 0, 0] and
    [6, 2, 0, 0] are both 01D structures so they will be merged.

    Intervals are merged by summing the scores, and taking the minimum and
    maximum k-values.  Component IDs in the merged interval are taken from the
    interval with the highest score.

    On rare occasions, intervals to be merged are not adjacent.  In this case,
    the score of the merged interval is not equal to the score which would be
    calculated from its k-interval.  This is necessary to maintain the property
    that the scores sum to 1.
    """
    dimtypes = {e.dimtype for e in intervals}

    merged_intervals = []
    for dimtype in dimtypes:
        relevant = [e for e in intervals if e.dimtype == dimtype]
        combined_score = sum(e.score for e in relevant)
        amin = min(e.a for e in relevant)
        bmax = max(e.b for e in relevant)
        best = max(relevant, key=lambda x: x.score)
        merged = build_kinterval(amin, bmax, best.h, best.components,
                                 best.cdim, score=combined_score)
        merged_intervals.append(merged)
    return merged_intervals


def build_kintervals(atoms, method_name):
    """The interval analysis is performed by inserting bonds one at a time
    until the component analysis finds a single component."""
    method = {'RDA': rank_determination.RDA,
              'TSA': topology_scaling.TSA}[method_name]

    assert all(e in [0, 1] for e in atoms.pbc)
    num_atoms = len(atoms)
    intervals = []
    kprev = 0
    calc = method(num_atoms)
    hprev = calc.check()
    components_prev, cdim_prev = calc.get_components()

    """
    The end state is a single component, whose dimensionality depends on
    the periodic boundary conditions:
    """
    end_state = np.zeros(4)
    end_dim = sum(atoms.pbc)
    end_state[end_dim] = 1
    end_state = tuple(end_state)

    # Insert each new bond into the component graph.
    for (k, i, j, offset) in next_bond(atoms):
        calc.insert_bond(i, j, offset)
        h = calc.check()
        if h == hprev:    # Test if any components were merged
            continue

        components, cdim = calc.get_components()

        # If any components were merged, create a new interval
        if k != kprev:
            # Only keep intervals of non-zero width
            intervals.append(build_kinterval(kprev, k, hprev,
                                             components_prev, cdim_prev))
        kprev = k
        hprev = h
        components_prev = components
        cdim_prev = cdim

        # Stop once all components are merged
        if h == end_state:
            intervals.append(build_kinterval(k, float("inf"), h,
                                             components, cdim))
            return intervals


def analyze_kintervals(atoms, method='RDA', merge=True):
    """Performs a k-interval analysis.

    In each k-interval the components (connected clusters) are identified.
    The intervals are sorted according to the scoring parameter, from high
    to low.

    Parameters:

    atoms: ASE atoms object
        The system to analyze. The periodic boundary conditions determine
        the maximum achievable component dimensionality, i.e. pbc=[1,1,0] sets
        a maximum dimensionality of 2.
    method: string
        Analysis method to use, either 'RDA' (default option) or 'TSA'.
        These correspond to the Rank Determination Algorithm of Mounet et al.
        and the Topological Scaling Algorithm (TSA) of Ashton et al.
    merge: boolean
        Decides if k-intervals of the same type (e.g. 01D or 3D) should be
        merged.  Default: true

    Returns:

    intervals: list
        List of KIntervals for each interval identified.  A KInterval is a
        namedtuple with the following field names:

        score: float
            Dimensionality score in the range [0, 1]
        a: float
            The start of the k-interval
        b: float
            The end of the k-interval
        dimtype: str
            The dimensionality type
        h: tuple
            The histogram of the number of components of each dimensionality.
            For example, (8, 0, 3, 0) means eight 0D and three 2D components.
        components: array
            The component ID of each atom.
        cdim: dict
            The component dimensionalities
    """
    intervals = build_kintervals(atoms, method)
    if merge:
        intervals = merge_intervals(intervals)

    # Sort intervals by score. Interval width resolves ambiguity when score=0.
    return sorted(intervals, reverse=True, key=lambda x: (x.score, x.b - x.a))

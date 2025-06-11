# fmt: off

"""
Implements the Rank Determination Algorithm (RDA)

Method is described in:
Definition of a scoring parameter to identify low-dimensional materials
components
P.M. Larsen, M. Pandey, M. Strange, and K. W. Jacobsen
Phys. Rev. Materials 3 034003, 2019
https://doi.org/10.1103/PhysRevMaterials.3.034003
"""
from collections import defaultdict

import numpy as np

from ase.geometry.dimensionality.disjoint_set import DisjointSet

# Numpy has a large overhead for lots of small vectors.  The cross product is
# particularly bad.  Pure python is a lot faster.


def dot_product(A, B):
    return sum(a * b for a, b in zip(A, B))


def cross_product(a, b):
    return [a[i] * b[j] - a[j] * b[i] for i, j in [(1, 2), (2, 0), (0, 1)]]


def subtract(A, B):
    return [a - b for a, b in zip(A, B)]


def rank_increase(a, b):
    if len(a) == 0:
        return True
    elif len(a) == 1:
        return a[0] != b
    elif len(a) == 4:
        return False

    L = a + [b]
    w = cross_product(subtract(L[1], L[0]), subtract(L[2], L[0]))
    if len(a) == 2:
        return any(w)
    elif len(a) == 3:
        return dot_product(w, subtract(L[3], L[0])) != 0
    else:
        raise Exception("This shouldn't be possible.")


def bfs(adjacency, start):
    """Traverse the component graph using BFS.

    The graph is traversed until the matrix rank of the subspace spanned by
    the visited components no longer increases.
    """
    visited = set()
    cvisited = defaultdict(list)
    queue = [(start, (0, 0, 0))]
    while queue:
        vertex = queue.pop(0)
        if vertex in visited:
            continue

        visited.add(vertex)
        c, p = vertex
        if not rank_increase(cvisited[c], p):
            continue

        cvisited[c].append(p)

        for nc, offset in adjacency[c]:

            nbrpos = (p[0] + offset[0], p[1] + offset[1], p[2] + offset[2])
            nbrnode = (nc, nbrpos)
            if nbrnode in visited:
                continue

            if rank_increase(cvisited[nc], nbrpos):
                queue.append(nbrnode)

    return visited, len(cvisited[start]) - 1


def traverse_component_graphs(adjacency):
    vertices = adjacency.keys()
    all_visited = {}
    ranks = {}
    for v in vertices:
        visited, rank = bfs(adjacency, v)
        all_visited[v] = visited
        ranks[v] = rank

    return all_visited, ranks


def build_adjacency_list(parents, bonds):
    graph = np.unique(parents)
    adjacency = {e: set() for e in graph}
    for (i, j, offset) in bonds:
        component_a = parents[i]
        component_b = parents[j]
        adjacency[component_a].add((component_b, offset))
    return adjacency


def get_dimensionality_histogram(ranks, roots):
    h = [0, 0, 0, 0]
    for e in roots:
        h[ranks[e]] += 1
    return tuple(h)


def merge_mutual_visits(all_visited, ranks, graph):
    """Find components with mutual visits and merge them."""
    merged = False
    common = defaultdict(list)
    for b, visited in all_visited.items():
        for offset in visited:
            for a in common[offset]:
                assert ranks[a] == ranks[b]
                merged |= graph.union(a, b)
            common[offset].append(b)

    if not merged:
        return merged, all_visited, ranks

    merged_visits = defaultdict(set)
    merged_ranks = {}
    parents = graph.find_all()
    for k, v in all_visited.items():
        key = parents[k]
        merged_visits[key].update(v)
        merged_ranks[key] = ranks[key]
    return merged, merged_visits, merged_ranks


class RDA:

    def __init__(self, num_atoms):
        """
        Initializes the RDA class.

        A disjoint set is used to maintain the component graph.

        Parameters:

        num_atoms: int    The number of atoms in the unit cell.
        """
        self.bonds = []
        self.graph = DisjointSet(num_atoms)
        self.adjacency = None
        self.hcached = None
        self.components_cached = None
        self.cdim_cached = None

    def insert_bond(self, i, j, offset):
        """
        Adds a bond to the list of graph edges.

        Graph components are merged if the bond does not cross a cell boundary.
        Bonds which cross cell boundaries can inappropriately connect
        components which are not connected in the infinite crystal.  This is
        tested during graph traversal.

        Parameters:

        i: int           The index of the first atom.
        n: int           The index of the second atom.
        offset: tuple    The cell offset of the second atom.
        """
        roffset = tuple(-np.array(offset))

        if offset == (0, 0, 0):    # only want bonds in aperiodic unit cell
            self.graph.union(i, j)
        else:
            self.bonds += [(i, j, offset)]
            self.bonds += [(j, i, roffset)]

    def check(self):
        """
        Determines the dimensionality histogram.

        The component graph is traversed (using BFS) until the matrix rank
        of the subspace spanned by the visited components no longer increases.

        Returns:
        hist : tuple         Dimensionality histogram.
        """
        adjacency = build_adjacency_list(self.graph.find_all(),
                                         self.bonds)
        if adjacency == self.adjacency:
            return self.hcached

        self.adjacency = adjacency
        self.all_visited, self.ranks = traverse_component_graphs(adjacency)
        res = merge_mutual_visits(self.all_visited, self.ranks, self.graph)
        _, self.all_visited, self.ranks = res

        self.roots = np.unique(self.graph.find_all())
        h = get_dimensionality_histogram(self.ranks, self.roots)
        self.hcached = h
        return h

    def get_components(self):
        """
        Determines the dimensionality and constituent atoms of each component.

        Returns:
        components: array    The component ID of every atom
        """
        component_dim = {e: self.ranks[e] for e in self.roots}
        relabelled_components = self.graph.find_all(relabel=True)
        relabelled_dim = {
            relabelled_components[k]: v for k, v in component_dim.items()
        }
        self.cdim_cached = relabelled_dim
        self.components_cached = relabelled_components

        return relabelled_components, relabelled_dim

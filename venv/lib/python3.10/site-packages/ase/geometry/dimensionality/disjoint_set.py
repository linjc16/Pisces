# fmt: off

import numpy as np


class DisjointSet:

    def __init__(self, n):
        self.sizes = np.ones(n, dtype=int)
        self.parents = np.arange(n)
        self.nc = n

    def _compress(self):
        a = self.parents
        b = a[a]
        while (a != b).any():
            a = b
            b = a[a]
        self.parents = a

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return False

        sizes = self.sizes
        parents = self.parents
        if sizes[a] < sizes[b]:
            parents[a] = b
            sizes[b] += sizes[a]
        else:
            parents[b] = a
            sizes[a] += sizes[b]

        self.nc -= 1
        return True

    def find(self, index):
        parents = self.parents
        parent = parents[index]
        while parent != parents[parent]:
            parent = parents[parent]
        parents[index] = parent
        return parent

    def find_all(self, relabel=False):
        self._compress()
        if not relabel:
            return self.parents

        # order elements by frequency
        _unique, inverse, counts = np.unique(self.parents,
                                            return_inverse=True,
                                            return_counts=True)
        indices = np.argsort(counts, kind='merge')[::-1]
        return np.argsort(indices)[inverse]

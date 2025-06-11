# fmt: off

import numpy as np


def dagger(matrix):
    return np.conj(matrix.T)


def rotate_matrix(h, u):
    return np.dot(u.T.conj(), np.dot(h, u))


def get_subspace(matrix, index):
    """Get the subspace spanned by the basis function listed in index"""
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    return matrix.take(index, 0).take(index, 1)


def normalize(matrix, S=None):
    """Normalize column vectors.

    ::

      <matrix[:,i]| S |matrix[:,i]> = 1

    """
    for col in matrix.T:
        if S is None:
            col /= np.linalg.norm(col)
        else:
            col /= np.sqrt(np.dot(col.conj(), np.dot(S, col)))


def subdiagonalize(h_ii, s_ii, index_j):
    nb = h_ii.shape[0]
    nb_sub = len(index_j)
    h_sub_jj = get_subspace(h_ii, index_j)
    s_sub_jj = get_subspace(s_ii, index_j)
    e_j, v_jj = np.linalg.eig(np.linalg.solve(s_sub_jj, h_sub_jj))
    normalize(v_jj, s_sub_jj)  # normalize: <v_j|s|v_j> = 1
    permute_list = np.argsort(e_j.real)
    e_j = np.take(e_j, permute_list)
    v_jj = np.take(v_jj, permute_list, axis=1)

    # Setup transformation matrix
    c_ii = np.identity(nb, complex)
    for i in range(nb_sub):
        for j in range(nb_sub):
            c_ii[index_j[i], index_j[j]] = v_jj[i, j]

    h1_ii = rotate_matrix(h_ii, c_ii)
    s1_ii = rotate_matrix(s_ii, c_ii)

    return h1_ii, s1_ii, c_ii, e_j


def cutcoupling(h, s, index_n):
    for i in index_n:
        s[:, i] = 0.0
        s[i, :] = 0.0
        s[i, i] = 1.0
        Ei = h[i, i]
        h[:, i] = 0.0
        h[i, :] = 0.0
        h[i, i] = Ei


def fermidistribution(energy, kt):
    # fermi level is fixed to zero
    # energy can be a single number or a list
    assert kt >= 0., 'Negative temperature encountered!'

    if kt == 0:
        if isinstance(energy, float):
            return int(energy / 2. <= 0)
        else:
            return (energy / 2. <= 0).astype(int)
    else:
        return 1. / (1. + np.exp(energy / kt))

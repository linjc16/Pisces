# fmt: off

import numpy as np


def cellvector_products(cell):
    cell = _pad_nonpbc(cell)
    g0 = np.empty(6, dtype=float)
    g0[0] = cell[0] @ cell[0]
    g0[1] = cell[1] @ cell[1]
    g0[2] = cell[2] @ cell[2]
    g0[3] = 2 * (cell[1] @ cell[2])
    g0[4] = 2 * (cell[2] @ cell[0])
    g0[5] = 2 * (cell[0] @ cell[1])
    return g0


def _pad_nonpbc(cell):
    # Add "infinitely long" lattice vectors for non-periodic directions,
    # perpendicular to the periodic ones.
    maxlen = max(cell.lengths())
    mask = cell.any(1)
    cell = cell.complete()
    cell[~mask] *= 2 * maxlen
    return cell


def niggli_reduce_cell(cell, epsfactor=None):
    from ase.cell import Cell

    cell = Cell.new(cell)
    npbc = cell.rank

    if epsfactor is None:
        epsfactor = 1e-5

    vol_normalization_exponent = 1 if npbc == 0 else 1 / npbc
    vol_normalization = cell.complete().volume**vol_normalization_exponent
    eps = epsfactor * vol_normalization

    g0 = cellvector_products(cell)
    g, C = _niggli_reduce(g0, eps)

    abc = np.sqrt(g[:3])
    # Prevent division by zero e.g. for cell==zeros((3, 3)):
    abcprod = max(abc.prod(), 1e-100)
    cosangles = abc * g[3:] / (2 * abcprod)
    angles = 180 * np.arccos(cosangles) / np.pi

    # Non-periodic directions have artificial infinitely long lattice vectors.
    # We re-zero their lengths before returning:
    abc[npbc:] = 0.0

    newcell = Cell.fromcellpar(np.concatenate([abc, angles]))

    newcell[npbc:] = 0.0
    return newcell, C


def lmn_to_ijk(lmn):
    if lmn.prod() == 1:
        ijk = lmn.copy()
        for idx in range(3):
            if ijk[idx] == 0:
                ijk[idx] = 1
    else:
        ijk = np.ones(3, dtype=int)
        if np.any(lmn != -1):
            r = None
            for idx in range(3):
                if lmn[idx] == 1:
                    ijk[idx] = -1
                elif lmn[idx] == 0:
                    r = idx
            if ijk.prod() == -1:
                ijk[r] = -1
    return ijk


def _niggli_reduce(g0, eps):
    I3 = np.eye(3, dtype=int)
    I6 = np.eye(6, dtype=int)

    C = I3.copy()
    D = I6.copy()

    g = D @ g0

    def lt(x, y, eps=eps):
        return x < y - eps

    def gt(x, y, eps=eps):
        return lt(y, x, eps)

    def eq(x, y, eps=eps):
        return not (lt(x, y, eps) or gt(x, y, eps))

    for _ in range(10000):
        if (gt(g[0], g[1])
                or (eq(g[0], g[1]) and gt(abs(g[3]), abs(g[4])))):
            C = C @ (-I3[[1, 0, 2]])
            D = I6[[1, 0, 2, 4, 3, 5]] @ D
            g = D @ g0
            continue
        elif (gt(g[1], g[2])
                or (eq(g[1], g[2]) and gt(abs(g[4]), abs(g[5])))):
            C = C @ (-I3[[0, 2, 1]])
            D = I6[[0, 2, 1, 3, 5, 4]] @ D
            g = D @ g0
            continue

        lmn = np.array(gt(g[3:], 0, eps=eps / 2), dtype=int)
        lmn -= np.array(lt(g[3:], 0, eps=eps / 2), dtype=int)

        ijk = lmn_to_ijk(lmn)

        C *= ijk[np.newaxis]

        D[3] *= ijk[1] * ijk[2]
        D[4] *= ijk[0] * ijk[2]
        D[5] *= ijk[0] * ijk[1]
        g = D @ g0

        if (gt(abs(g[3]), g[1])
                or (eq(g[3], g[1]) and lt(2 * g[4], g[5]))
                or (eq(g[3], -g[1]) and lt(g[5], 0))):
            s = int(np.sign(g[3]))

            A = I3.copy()
            A[1, 2] = -s
            C = C @ A

            B = I6.copy()
            B[2, 1] = 1
            B[2, 3] = -s
            B[3, 1] = -2 * s
            B[4, 5] = -s
            D = B @ D
            g = D @ g0
        elif (gt(abs(g[4]), g[0])
                or (eq(g[4], g[0]) and lt(2 * g[3], g[5]))
                or (eq(g[4], -g[0]) and lt(g[5], 0))):
            s = int(np.sign(g[4]))

            A = I3.copy()
            A[0, 2] = -s
            C = C @ A

            B = I6.copy()
            B[2, 0] = 1
            B[2, 4] = -s
            B[3, 5] = -s
            B[4, 0] = -2 * s
            D = B @ D
            g = D @ g0
        elif (gt(abs(g[5]), g[0])
                or (eq(g[5], g[0]) and lt(2 * g[3], g[4]))
                or (eq(g[5], -g[0]) and lt(g[4], 0))):
            s = int(np.sign(g[5]))

            A = I3.copy()
            A[0, 1] = -s
            C = C @ A

            B = I6.copy()
            B[1, 0] = 1
            B[1, 5] = -s
            B[3, 4] = -s
            B[5, 0] = -2 * s
            D = B @ D
            g = D @ g0
        elif (lt(g[[0, 1, 3, 4, 5]].sum(), 0)
                or (eq(g[[0, 1, 3, 4, 5]].sum(), 0)
                    and gt(2 * (g[0] + g[4]) + g[5], 0))):
            A = I3.copy()
            A[:, 2] = 1
            C = C @ A

            B = I6.copy()
            B[2, :] = 1
            B[3, 1] = 2
            B[3, 5] = 1
            B[4, 0] = 2
            B[4, 5] = 1
            D = B @ D
            g = D @ g0
        else:
            break
    else:
        raise RuntimeError('Niggli reduction not done in 10000 steps!\n'
                           'g={}\n'
                           'operation={}'
                           .format(g.tolist(), C.tolist()))

    return g, C

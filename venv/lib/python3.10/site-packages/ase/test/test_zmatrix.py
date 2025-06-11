import numpy as np
import pytest

from ase.io.zmatrix import parse_zmatrix

pos_ref = pytest.approx(np.array([
    [+0.000, +0.000, +0.000],
    [+1.310, +0.000, +0.000],
    [-0.160, +1.300, +0.000],
    [+1.150, +1.300, +0.000],
    [-0.394, -0.446, +1.031],
    [-0.394, -0.446, -1.031],
    [+1.545, +1.746, -1.031],
    [+1.545, +1.746, +1.031],
]), abs=1e-3)

tests = [
    (
        (
            'b;'
            'h 1 1.31;'
            'h 1 1.31 2 97;'
            'b 2 1.31 1 83 3 0;'
            'h 1 1.19 4 120 2 90;'
            'h 1 1.19 4 120 3 90;'
            'h 4 1.19 1 120 2 90;'
            'h 4 1.19 1 120 3 90'
        ), None,
    ),
    (
        [
            'B1',
            'H1 B1 bh1',
            'H2 B1 bh1 H1 hbh1',
            'B2 H1 bh1 B1 bhb H2 0',
            'H3 B1 bh2 B2 hbh2 H1 hbbh1',
            'H4 B1 bh2 B2 hbh2 H1 hbbh2',
            'H5 B2 bh2 B1 hbh2 H2 hbbh2',
            'H6 B2 bh2 B1 hbh2 H2 hbbh1',
        ], [
            'bh1 = 1.31',
            'bh2 = 1.19',
            'hbh1 = 97',
            'hbh2 = 120',
            'bhb = 83',
            'hbbh1 = 90',
            'hbbh2 = -hbbh1',
        ],
    ),
    (
        '\n'.join([
            'B',
            'H 1 rBH1',
            'H 1 rBH1 2 aHBH1',
            'B 1 rBB 2 aBBH 3 0',
            'H 1 rBH2 4 aHBH2 2 dHBBH',
            'H 1 rBH2 4 aHBH2 3 dHBBH',
            'H 4 rBH2 1 aHBH2 2 dHBBH',
            'H 4 rBH2 1 aHBH2 3 dHBBH',
        ]), '\n'.join([
            'rBH1 1.31',
            'rBH2 1.19',
            'rBB 1.736',
            'aHBH1 97',
            'aHBH2 120',
            'aBBH 48.5',
            'dHBBH 90',
        ]),
    ),
    (
        [
            'B 0 0.00 0.00 0.00',
            'H 0 1.31 0.00 0.00',
            'H 1 r1 2 a1',
            'B 2 r1 1 a2 3 d1',
            'H 1 r2 4 a3 2 d2',
            'H 1 r2 4 a3 2 -d2',
            'H 4 r2 1 a3 2 d2',
            'H 4 r2 1 a3 2 -d2',
        ], dict(
            r1=1.31,
            r2=1.19,
            a1=97,
            a2=83,
            a3=120,
            d1=0,
            d2=90,
        ),
    ),
]


@pytest.mark.parametrize('zmat, defs', tests)
def test_zmatrix_diborane(zmat, defs):
    assert parse_zmatrix(zmat, defs=defs).positions == pos_ref

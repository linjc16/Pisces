"""Test phasediagram code."""
import pytest

from ase.phasediagram import PhaseDiagram


def test_phasediagram():
    """Test example from docs."""
    refs = [('Cu', 0.0),
            ('Au', 0.0),
            ('CuAu2', -0.2),
            ('CuAu', -0.5),
            ('Cu2Au', -0.7)]
    pd = PhaseDiagram(refs)
    energy, indices, coefs = pd.decompose('Cu3Au')
    assert energy == pytest.approx(-0.7)
    assert (indices == [4, 0]).all()
    assert coefs == pytest.approx(1.0)


def test_phasediagram_1d():
    """Test 1D case."""
    refs = [('Cu', 0.0),
            ('Cu', 0.1)]
    pd = PhaseDiagram(refs)
    energy, indices, coefs = pd.decompose('Cu')
    assert energy == pytest.approx(0.0)
    assert (indices == [0]).all()
    assert coefs == pytest.approx(1.0)


def test_phasediagram_3d():
    """The triangualtion can have zero area slivers along the edges
    which can lead to problems."""
    pd = PhaseDiagram(refs3d.values())
    energy3, _indices, _coefs = pd.decompose(Mg=8, I=12)

    # Same calculation without Cu:
    pd = PhaseDiagram([ref for ref in refs3d.values() if 'Cu' not in ref[0]])
    energy2, _indices, _coefs = pd.decompose(Mg=8, I=12)

    assert energy2 == pytest.approx(energy3, abs=1e-6)
    assert energy2 == pytest.approx(-19.470, abs=0.001)


refs3d = {
    '1Cu2I3-1': ({'Cu': 2, 'I': 3}, -0.5856860458647013),
    '1Cu2I5-1': ({'Cu': 2, 'I': 5}, -0.20346192594182355),
    '1Cu3I4-1': ({'Cu': 3, 'I': 4}, -0.5115719741629157),
    '1Cu3I5-1': ({'Cu': 3, 'I': 5}, -0.706485561663893),
    '1Cu9I11-1': ({'Cu': 9, 'I': 11}, -2.0449589884729136),
    '1CuI-1': ({'Cu': 1, 'I': 1}, -0.16687020587193135),
    '1CuI2-1': ({'Cu': 1, 'I': 2}, -0.3865414415277115),
    '1CuI2-2': ({'Cu': 1, 'I': 2}, -0.15576437343060956),
    '1CuI2-3': ({'Cu': 1, 'I': 2}, 0.2822804394351808),
    '1Mg7I12-1': ({'Mg': 7, 'I': 12}, -18.078665663770753),
    '1MgI2-1': ({'Mg': 1, 'I': 2}, -3.2411795532768766),
    '1MgI2-2': ({'Mg': 1, 'I': 2}, -3.126299789327992),
    '1MgI2-3': ({'Mg': 1, 'I': 2}, -2.7644575559477076),
    '2Cu2I3-1': ({'Cu': 4, 'I': 6}, -1.1684201166847288),
    '2Cu2I3-2': ({'Cu': 4, 'I': 6}, -0.5293275722906259),
    '2CuI-1': ({'Cu': 2, 'I': 2}, -0.6644234604058941),
    '2CuI-2': ({'Cu': 2, 'I': 2}, -0.631196199599021),
    '2CuI2-1': ({'Cu': 2, 'I': 4}, -0.5555276551355437),
    '2CuI2-2': ({'Cu': 2, 'I': 4}, -0.33115927179888516),
    '2CuI3-1': ({'Cu': 2, 'I': 6}, 0.37253394890171876),
    '2CuI3-2': ({'Cu': 2, 'I': 6}, 0.8364517074113316),
    '2CuMgI3-1': ({'Cu': 2, 'Mg': 2, 'I': 6}, -6.949138440931463),
    '2MgI2-1': ({'Mg': 2, 'I': 4}, -5.954617527291109),
    '3CuI-1': ({'Cu': 3, 'I': 3}, -0.8956628969280995),
    '3CuI-2': ({'Cu': 3, 'I': 3}, -0.5947516410802418),
    '3CuI-3': ({'Cu': 3, 'I': 3}, -0.5836330832046226),
    '3CuI-4': ({'Cu': 3, 'I': 3}, -0.5571970366267802),
    '3CuI2-1': ({'Cu': 3, 'I': 6}, -0.9672564542235236),
    '3Mg3I4-1': ({'Mg': 9, 'I': 12}, -17.37064377209711),
    '4CuI-1': ({'Cu': 4, 'I': 4}, -1.0794320505303006),
    '4CuI-2': ({'Cu': 4, 'I': 4}, -0.6805380064770041),
    '4CuI2-1': ({'Cu': 4, 'I': 8}, -0.948933198571698),
    '4CuI2-2': ({'Cu': 4, 'I': 8}, -0.930761273479547),
    '4Mg2I3-1': ({'Mg': 8, 'I': 12}, -16.992460281159794),
    '5CuI-1': ({'Cu': 5, 'I': 5}, -1.4210519600116243),
    '6CuI-1': ({'Cu': 6, 'I': 6}, -1.6272941295370025),
    '6CuI-2': ({'Cu': 6, 'I': 6}, -1.220143848038628),
    '6CuI-3': ({'Cu': 6, 'I': 6}, -1.181677962399558),
    'Cu': ({'Cu': 1}, 0.0),
    'Cu4Mg8': ({'Cu': 4, 'Mg': 8}, -1.5030809260880602),
    'CuI': ({'Cu': 1, 'I': 1}, -0.31658662552506645),
    'I4': ({'I': 4}, 0.0),
    'Mg12': ({'Mg': 12}, 0.0),
    'Mg2Cu4': ({'Cu': 4, 'Mg': 2}, -0.9341752322551926),
    'MgI2': ({'I': 2, 'Mg': 1}, -3.2449179691304515)}

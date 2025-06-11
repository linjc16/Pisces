import numpy as np
import pytest

from ase.build import molecule
from ase.geometry.analysis import Analysis, get_max_volume_estimate


@pytest.fixture()
def images_without_cell():
    atoms1 = molecule('CH3CH2OH')
    atoms2 = molecule('CH3CH2OH')

    pos2 = atoms2.get_positions()

    xyz_argmin = np.argmin(pos2, axis=0)
    pos2[xyz_argmin] = pos2[xyz_argmin] - 0.1

    xyz_argmax = np.argmax(pos2, axis=0)
    pos2[xyz_argmax] = pos2[xyz_argmax] + 0.1

    atoms2.set_positions(pos2)
    return [atoms1, atoms2]


def test_analysis_max_volume_estimate(images_without_cell):
    volume1 = get_max_volume_estimate([images_without_cell[0]])
    volume2 = get_max_volume_estimate([images_without_cell[1]])
    volume_max = get_max_volume_estimate(images_without_cell)

    volume_max_ref = pytest.approx(110.61358934680972)

    assert volume1 == pytest.approx(97.11594231219294)
    assert volume2 == volume_max_ref
    assert volume_max == volume_max_ref


def test_analysis_rdf(images_without_cell):

    analysis = Analysis(images_without_cell)
    ls_rdf = analysis.get_rdf(
        5.0, 5, volume=analysis.get_max_volume_estimate())

    ls_rdf_ref = np.array(((0.65202591, 1.1177587, 0.54907445,
                            0.10573393, 0.01068895),
                           (0.0, 1.1177587, 0.5833916,
                            0.10573393, 0.01068895)))

    assert np.array(ls_rdf) == pytest.approx(ls_rdf_ref)


@pytest.mark.filterwarnings('ignore:the matrix subclass')
def test_analysis():
    # test the geometry.analysis module

    mol = molecule('CH3CH2OH')
    ana = Analysis(mol)
    assert np.shape(ana.adjacency_matrix[0].todense()) == (9, 9)
    for imI in range(len(ana.all_bonds)):
        l1 = sum(len(x) for x in ana.all_bonds[imI])
        l2 = sum(len(x) for x in ana.unique_bonds[imI])
        assert l1 == l2 * 2

    for imi in range(len(ana.all_angles)):
        l1 = sum(len(x) for x in ana.all_angles[imi])
        l2 = sum(len(x) for x in ana.unique_angles[imi])
        assert l1 == l2 * 2

    for imi in range(len(ana.all_dihedrals)):
        l1 = sum(len(x) for x in ana.all_dihedrals[imi])
        l2 = sum(len(x) for x in ana.unique_dihedrals[imi])
        assert l1 == l2 * 2

    angles1 = ana.get_angles('C', 'C', 'H', unique=False)[0]
    angles2 = ana.get_angles('C', 'C', 'H', unique=True)[0]
    assert len(angles1) == len(angles2) * 2

    csixty = molecule('C60')
    mol = molecule('C7NH5')

    ana = Analysis(csixty)
    ana2 = Analysis(mol)
    for imI in range(len(ana.all_bonds)):
        l1 = sum(len(x) for x in ana.all_bonds[imI])
        l2 = sum(len(x) for x in ana.unique_bonds[imI])
        assert l1 == l2 * 2
    for imI in range(len(ana.all_angles)):
        l1 = sum(len(x) for x in ana.all_angles[imI])
        l2 = sum(len(x) for x in ana.unique_angles[imI])
        assert l1 == l2 * 2
    for imI in range(len(ana.all_dihedrals)):
        l1 = sum(len(x) for x in ana.all_dihedrals[imI])
        l2 = sum(len(x) for x in ana.unique_dihedrals[imI])
        assert l1 == l2 * 2

    angles1 = ana2.get_angles('C', 'C', 'H', unique=False)[0]
    angles2 = ana2.get_angles('C', 'C', 'H', unique=True)[0]
    assert len(angles1) == len(angles2) * 2

    dihedrals1 = ana2.get_dihedrals('H', 'C', 'C', 'H', unique=False)[0]
    dihedrals2 = ana2.get_dihedrals('H', 'C', 'C', 'H', unique=True)[0]
    assert len(dihedrals1) == len(dihedrals2) * 2

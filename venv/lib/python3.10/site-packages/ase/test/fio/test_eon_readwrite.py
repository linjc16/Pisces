"""Check that reading and writing .con files is consistent."""

import numpy as np
import numpy.testing as npt
import pytest

import ase
import ase.io
import ase.symbols

# Error tolerance.
TOL = 1e-6

# The corresponding data as an ASE Atoms object.
DATA = ase.Atoms(
    "Cu3",
    cell=np.array([[7.22, 0, 0], [1, 10.83, 0], [1, 1, 14.44]]),
    positions=np.array(
        [
            [1.04833333, 0.965, 0.9025],
            [3.02, 2.77, 0.9025],
            [6.36666667, 10.865, 13.5375],
        ]
    ),
    pbc=(True, True, True),
)


def test_eon_read_single(datadir):
    box = ase.io.read(f"{datadir}/io/eon/single.con", format="eon")
    npt.assert_allclose(box.cell, DATA.cell, rtol=TOL, atol=0)
    assert (box.symbols == ase.symbols.string2symbols("Cu3")).all()
    npt.assert_allclose(box.get_masses(), np.array([63.5459999] * 3), rtol=TOL)
    npt.assert_allclose(box.positions, DATA.positions, rtol=TOL)


def test_eon_write_single(datadir):
    out_file = "out.con"
    ase.io.write(out_file, DATA, format="eon")
    data2 = ase.io.read(out_file, format="eon")
    npt.assert_allclose(data2.cell, DATA.cell, rtol=TOL, atol=0)
    npt.assert_allclose(data2.positions, DATA.positions, rtol=TOL)


def test_eon_roundtrip_multi(datadir):
    out_file = "out.con"
    images = ase.io.read(f"{datadir}/io/eon/multi.con", format="eon", index=":")
    ase.io.write(out_file, images, format="eon")
    data = ase.io.read(out_file, format="eon", index=":")
    assert len(data) == 10
    npt.assert_allclose(
        data[0].constraints[0].get_indices(),
        np.array([0, 1]), rtol=1e-5, atol=0
    )
    npt.assert_allclose(
        data[1].constraints[0].get_indices(),
        np.array([]), rtol=1e-5, atol=0
    )


def test_eon_read_multi(datadir):
    images = ase.io.read(f"{datadir}/io/eon/multi.con", format="eon", index=":")
    assert len(images) == 10
    npt.assert_allclose(
        images[0].constraints[0].get_indices(),
        np.array([0, 1]), rtol=1e-5, atol=0
    )
    npt.assert_allclose(
        images[1].constraints[0].get_indices(),
        np.array([]), rtol=1e-5, atol=0
    )


def test_eon_isotope_fail():
    out_file = "out.con"
    DATA.set_masses([33, 31, 22])
    with pytest.raises(RuntimeError):
        ase.io.write(out_file, DATA, format="eon")


def test_eon_masses():
    # Error tolerance.
    TOL = 1e-8

    data = ase.lattice.compounds.B2(['Cs', 'Cl'], latticeconstant=4.123,
                                    size=(3, 3, 3))

    m_Cs = ase.data.atomic_masses[ase.data.atomic_numbers['Cs']]
    m_Cl = ase.data.atomic_masses[ase.data.atomic_numbers['Cl']]

    con_file = 'pos.con'
    # Write and read the .con file.
    ase.io.write(con_file, data, format='eon')
    data2 = ase.io.read(con_file, format='eon')
    # Check masses.
    symbols = np.asarray(data2.get_chemical_symbols())
    masses = np.asarray(data2.get_masses())
    assert (abs(masses[symbols == 'Cs'] - m_Cs)).sum() < TOL
    assert (abs(masses[symbols == 'Cl'] - m_Cl)).sum() < TOL

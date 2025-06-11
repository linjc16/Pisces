import pickle

import pytest

from ase.build import molecule
from ase.calculators.emt import EMT
from ase.vibrations import Vibrations
from ase.vibrations.pickle2json import main as pickle2json_main


def test_pickle2json(testdir):
    atoms = molecule('H2O')
    atoms.calc = EMT()
    name = 'vib'
    vib = Vibrations(atoms, name=name)
    vib.run()

    forces_dct = dict(vib.cache)
    assert len(forces_dct) > 0

    # Create old-style pickle cache:
    for key, value in vib.cache.items():
        with (testdir / f'vib.{key}.pckl').open('wb') as fd:
            array = value['forces']
            pickle.dump(array, fd)

    vib.cache.clear()
    assert dict(vib.cache) == {}

    # When there are old pickles but no JSON files, run() should complain:
    with pytest.raises(RuntimeError, match='Found old pickle'):
        vib.run()

    picklefiles = [str(path) for path in testdir.glob('vib.*.pckl')]

    pickle2json_main(picklefiles)

    # Read forces after back-conversion:
    newforces_dct = dict(vib.cache)
    assert len(newforces_dct) > 0

    assert set(forces_dct) == set(newforces_dct)

    for key in forces_dct:
        assert forces_dct[key]['forces'] == pytest.approx(
            newforces_dct[key]['forces'])

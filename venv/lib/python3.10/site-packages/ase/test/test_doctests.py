import doctest
import importlib

import numpy as np
import pytest

module_names = """\
ase.atoms
ase.build.tools
ase.cell
ase.calculators.vasp.vasp
ase.collections.collection
ase.dft.kpoints
ase.eos
ase.formula
ase.geometry.cell
ase.geometry.geometry
ase.io.ulm
ase.lattice
ase.neighborlist
ase.spacegroup.spacegroup
ase.spacegroup.xtal
ase.symbols
""".split()


@pytest.mark.parametrize('modname', module_names)
def test_doctest(testdir, modname):
    mod = importlib.import_module(modname)
    with np.printoptions(legacy='1.13'):
        doctest.testmod(mod, raise_on_error=True, verbose=True)

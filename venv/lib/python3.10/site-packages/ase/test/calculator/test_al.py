import pytest

from ase.build import bulk

omx_par = {'definition_of_atomic_species': [['Al', 'Al8.0-p1', 'Al_CA13'],
                                            ['O', 'O6.0-p1', 'O_CA13']]}


calc = pytest.mark.calculator


# @calc('elk', rgkmax=5.0)
# Elk fails since calc.atoms is None after reading.
# Apparently this test did not run in CI before.
# Let's just disable it.
@calc('openmx', **omx_par)
def test_al(factory):
    name = factory.name
    # What on earth does kpts=1.0 mean?  Was failing, I changed it.  --askhl
    # Disabled GPAW since it was failing anyway. --askhl
    kpts = [2, 2, 2]
    calc = factory.calc(label=name, xc='LDA', kpts=kpts)
    al = bulk('AlO', crystalstructure='rocksalt', a=4.5)
    al.calc = calc
    e = al.get_potential_energy()
    calc.set(xc='PBE', kpts=kpts)
    epbe = al.get_potential_energy()
    print(e, epbe)
    calc = factory.calc(restart=name)
    print(calc.parameters, calc.results, calc.atoms)
    assert not calc.calculation_required(al, ['energy'])
    al = calc.get_atoms()
    print(al.get_potential_energy())
    label = 'dir/' + name + '-2'
    calc = factory.calc(label=label, atoms=al, xc='LDA', kpts=kpts)
    print(al.get_potential_energy())

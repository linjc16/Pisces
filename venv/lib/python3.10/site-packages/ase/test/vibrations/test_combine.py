from ase.build import molecule
from ase.test.utils import RandomCalculator
from ase.utils import workdir
from ase.vibrations import Infrared, Vibrations


def test_combine(testdir):
    dirname = 'subdir'
    vibname = 'ir'
    with workdir(dirname, mkdir=True):
        atoms = molecule('C2H6')
        ir = Infrared(atoms)
        assert ir.name == vibname
        ir.calc = RandomCalculator()
        ir.run()
        freqs = ir.get_frequencies()
        ints = ir.intensities
        assert ir.combine() == 49

        ir = Infrared(atoms)
        assert (freqs == ir.get_frequencies()).all()
        assert (ints == ir.intensities).all()

        vib = Vibrations(atoms, name=vibname)
        assert (freqs == vib.get_frequencies()).all()

        # Read the data from other working directory
        with workdir('..'):
            ir = Infrared(atoms, name=f'{dirname}/{vibname}')
            assert (freqs == ir.get_frequencies()).all()

        ir = Infrared(atoms)
        assert ir.split() == 1
        assert (freqs == ir.get_frequencies()).all()
        assert (ints == ir.intensities).all()

        vib = Vibrations(atoms, name=vibname)
        assert (freqs == vib.get_frequencies()).all()

        assert ir.clean() == 49

from ase import Atoms
from ase.io import Trajectory


def test_atoms_info_in_traj():
    info = dict(creation_date='2011-06-27',
                chemical_name='Hydrogen',
                # custom classes also works provided that it is
                # imported and pickleable...
                foo={'seven': 7})

    molecule = Atoms('H2', positions=[(0., 0., 0.), (0., 0., 1.1)], info=info)
    assert molecule.info == info

    atoms = molecule.copy()
    assert atoms.info == info

    with Trajectory('info.traj', 'w', atoms=molecule) as traj:
        traj.write()

    with Trajectory('info.traj') as traj:
        atoms = traj[-1]

    print(atoms.info)
    assert atoms.info == info

from ase import Atoms


def test_atoms_info_copy():

    at1 = Atoms('H2', positions=[[0., 0., 0.],
                                 [1., 0., 0.]])

    at1.info['str'] = "str"
    at1.info['int'] = 42

    at2 = Atoms(at1)

    assert at2.info == at1.info

# Stripped (minimal) version of nomad entry with 3 images.
# The images are actually identical for some reason, but we want to be sure
# that they are extracted correctly.
from ase.io import iread


def test_nomad(datadir):

    path = datadir / 'nomad-images.nomad-json'
    images = list(iread(path))
    assert len(images) == 3

    for atoms in images:
        assert all(atoms.pbc)
        assert (atoms.cell > 0).sum() == 3
        assert atoms.get_chemical_formula() == 'As24Sr32'


# Code for cleaning up nomad files so their size is reasonable for inclusion
# in test suite:
"""
ourkeys = {'section_run', 'section_system', 'name', 'atom_species',
           'atom_positions', 'flatData', 'uri', 'gIndex',
           'configuration_periodic_dimensions', 'lattice_vectors'}

includekeys = lambda k: k in ourkeys

fname = ...
with open(fname) as fd:
    d = read(fd, includekeys=includekeys)
print(json.dumps(d))
"""

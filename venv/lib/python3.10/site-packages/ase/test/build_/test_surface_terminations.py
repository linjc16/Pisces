from ase.build import surface
from ase.build.surfaces_with_termination import surfaces_with_termination
from ase.spacegroup import crystal


def test_surface_terminations():

    a = 4.6
    c = 2.95

    # Rutile:
    rutile = crystal(['Ti', 'O'], basis=[(0, 0, 0), (0.3, 0.3, 0.0)],
                     spacegroup=136, cellpar=[a, a, c, 90, 90, 90])

    slb = surface(rutile, indices=(1, 1, 0), layers=4, vacuum=10)
    slb *= (1, 2, 1)

    def check_surf_composition(images, formula):
        for atoms in images:
            zmax = atoms.positions[:, 2].max()
            sym = atoms.symbols[abs(atoms.positions[:, 2] - zmax) < 1e-2]
            red_formula, _ = sym.formula.reduce()
            assert red_formula == formula

    images = surfaces_with_termination(rutile,
                                       indices=(1, 1, 0),
                                       layers=4,
                                       vacuum=10,
                                       termination='O')

    check_surf_composition(images, 'O')

    images = surfaces_with_termination(rutile,
                                       indices=(1, 1, 0),
                                       layers=4, vacuum=10,
                                       termination='TiO')

    check_surf_composition(images, 'TiO')

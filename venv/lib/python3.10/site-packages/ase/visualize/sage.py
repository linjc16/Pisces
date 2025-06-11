# fmt: off

from ase.data import covalent_radii
from ase.data.colors import jmol_colors


def view_sage_jmol(atoms, repeat=None, *args, **kwargs):
    try:
        from .sage.plot.plot3d.shapes import ColorCube, Sphere
    except Exception:  # XXX Which kind of exception exactly?
        raise ImportError(
            'view_sage_jmol requires sage (http://www.sagemath.org/) ' +
            'and is intended to be used directly in the browser')

    if repeat is not None:
        atoms = atoms.repeat(repeat)

    cell = atoms.cell.diagonal() / 2
    model = ColorCube(list(cell), ['blue', 'blue', 'blue'], opacity=0.1)
    for atom in atoms:
        atomic_number = atom.number
        color = tuple(jmol_colors[atomic_number])
        radius = covalent_radii[atomic_number]
        model += Sphere(radius, color=color).translate(
            *(atom.position - atoms.cell.diagonal() / 2))
    model.show(aspect_ratio=1, frame=False)


view_sage = view_sage_jmol

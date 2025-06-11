# fmt: off

import warnings
from math import acos, pi, sqrt

import numpy as np

from ase.data import atomic_names as names
from ase.data import chemical_symbols as symbols
from ase.gui.i18n import _
from ase.gui.utils import get_magmoms


def formula(Z):
    hist = {}
    for z in Z:
        if z in hist:
            hist[z] += 1
        else:
            hist[z] = 1
    Z = sorted(hist.keys())
    strings = []
    for z in Z:
        n = hist[z]
        s = ('' if n == 1 else str(n)) + symbols[z]
        strings.append(s)
    return '+'.join(strings)


class Status:  # Status is used as a mixin in GUI
    def __init__(self):
        self.ordered_indices = []

    def status(self, atoms):
        # use where here:  XXX
        natoms = len(atoms)
        indices = np.arange(natoms)[self.images.selected[:natoms]]
        ordered_indices = [i for i in self.images.selected_ordered
                           if i < len(atoms)]
        n = len(indices)

        if n == 0:
            line = ''
            if atoms.calc:
                calc = atoms.calc

                def getresult(name, get_quantity):
                    # ase/io/trajectory.py line 170 does this by using
                    # the get_property(prop, atoms, allow_calculation=False)
                    # so that is an alternative option.
                    try:
                        if calc.calculation_required(atoms, [name]):
                            quantity = None
                        else:
                            quantity = get_quantity()
                    except Exception as err:
                        quantity = None
                        errmsg = ('An error occurred while retrieving {} '
                                  'from the calculator: {}'.format(name, err))
                        warnings.warn(errmsg)
                    return quantity

                energy = getresult('energy', atoms.get_potential_energy)
                forces = getresult('forces', atoms.get_forces)

                if energy is not None:
                    line += f'Energy = {energy:.3f} eV'

                if forces is not None:
                    maxf = np.linalg.norm(forces, axis=1).max()
                    line += f'   Max force = {maxf:.3f} eV/Å'
            self.window.update_status_line(line)
            return

        Z = atoms.numbers[indices]
        R = atoms.positions[indices]

        if n == 1:
            tag = atoms.get_tags()[indices[0]]
            text = (' #%d %s (%s): %.3f Å, %.3f Å, %.3f Å ' %
                    ((indices[0], names[Z[0]], symbols[Z[0]]) + tuple(R[0])))
            text += _(' tag=%(tag)s') % dict(tag=tag)
            magmoms = get_magmoms(self.atoms)
            if magmoms.any():
                # TRANSLATORS: mom refers to magnetic moment
                text += _(' mom={:1.2f}'.format(
                    magmoms[indices][0]))
            charges = self.atoms.get_initial_charges()
            if charges.any():
                text += _(' q={:1.2f}'.format(
                    charges[indices][0]))
            haveit = {'numbers', 'positions', 'forces', 'momenta',
                      'initial_charges', 'initial_magmoms', 'tags'}
            for key in atoms.arrays:
                if key not in haveit:
                    val = atoms.get_array(key)[indices[0]]
                    if val is not None:
                        if isinstance(val, int):
                            text += f' {key}={val:g}'
                        else:
                            text += f' {key}={val}'
        elif n == 2:
            D = R[0] - R[1]
            d = sqrt(np.dot(D, D))
            text = f' {symbols[Z[0]]}-{symbols[Z[1]]}: {d:.3f} Å'
        elif n == 3:
            d = []
            for c in range(3):
                D = R[c] - R[(c + 1) % 3]
                d.append(np.dot(D, D))
            a = []
            for c in range(3):
                t1 = 0.5 * (d[c] + d[(c + 1) % 3] - d[(c + 2) % 3])
                t2 = sqrt(d[c] * d[(c + 1) % 3])
                try:
                    t3 = acos(t1 / t2)
                except ValueError:
                    if t1 > 0:
                        t3 = 0
                    else:
                        t3 = pi
                a.append(t3 * 180 / pi)
            text = (' %s-%s-%s: %.1f°, %.1f°, %.1f°' %
                    tuple([symbols[z] for z in Z] + a))
        elif len(ordered_indices) == 4:
            angle = self.atoms.get_dihedral(*ordered_indices, mic=True)
            text = ('%s %s → %s → %s → %s: %.1f°' %
                    tuple([_('dihedral')] + [symbols[z] for z in Z] + [angle]))
        else:
            text = ' ' + formula(Z)

        self.window.update_status_line(text)

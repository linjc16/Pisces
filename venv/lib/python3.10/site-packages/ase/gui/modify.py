# fmt: off

from functools import partial

import numpy as np

import ase.gui.ui as ui
from ase.gui.i18n import _
from ase.gui.utils import get_magmoms
from ase.gui.widgets import Element


class ModifyAtoms:
    """Presents a dialog box where the user is able to change the
    atomic type, the magnetic moment and tags of the selected atoms.
    """

    def __init__(self, gui):
        self.gui = gui
        selected = self.selection()
        if not selected.any():
            ui.error(_('No atoms selected!'))
            return

        win = ui.Window(_('Modify'), wmtype='utility')
        element = Element(callback=self.set_element)
        win.add(element)
        win.add(ui.Button(_('Change element'),
                          partial(self.set_element, element)))
        self.tag = ui.SpinBox(0, -1000, 1000, 1, self.set_tag)
        win.add([_('Tag'), self.tag])
        self.magmom = ui.SpinBox(0.0, -10, 10, 0.1, self.set_magmom)
        win.add([_('Moment'), self.magmom])

        atoms = self.gui.atoms
        sym = atoms.symbols[selected]
        if len(sym.species()) == 1:
            element.symbol = sym[0]

        tags = atoms.get_tags()[selected]
        if np.ptp(tags) == 0:
            self.tag.value = tags[0]

        magmoms = get_magmoms(atoms)[selected]
        if np.ptp(magmoms.round(2)) == 0.0:
            self.magmom.value = round(magmoms[0], 2)

    def selection(self):
        return self.gui.images.selected[:len(self.gui.atoms)]

    def set_element(self, element):
        self.gui.atoms.numbers[self.selection()] = element.Z
        self.gui.draw()

    def set_tag(self):
        tags = self.gui.atoms.get_tags()
        tags[self.selection()] = self.tag.value
        self.gui.atoms.set_tags(tags)
        self.gui.draw()

    def set_magmom(self):
        magmoms = get_magmoms(self.gui.atoms)
        magmoms[self.selection()] = self.magmom.value
        self.gui.atoms.set_initial_magnetic_moments(magmoms)
        self.gui.draw()

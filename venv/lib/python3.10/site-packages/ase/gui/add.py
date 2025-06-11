# fmt: off

import os

import numpy as np

import ase.gui.ui as ui
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.gui.i18n import _

current_selection_string = _('(selection)')


class AddAtoms:
    def __init__(self, gui):
        self.gui = gui
        win = self.win = ui.Window(_('Add atoms'), wmtype='utility')
        win.add(_('Specify chemical symbol, formula, or filename.'))

        def choose_file():
            chooser = ui.ASEFileChooser(self.win.win)
            filename = chooser.go()
            if filename is None:  # No file selected
                return

            self.combobox.value = filename

            # Load the file immediately, so we can warn now in case of error
            self.readfile(filename, format=chooser.format)

        if self.gui.images.selected.any():
            default = current_selection_string
        else:
            default = 'H2'

        self._filename = None
        self._atoms_from_file = None

        from ase.collections import g2
        labels = sorted(name for name in g2.names
                        if len(g2[name]) > 1)
        values = labels

        combobox = ui.ComboBox(labels, values)
        win.add([_('Add:'), combobox,
                 ui.Button(_('File ...'), callback=choose_file)])
        combobox.widget.bind('<Return>', lambda e: self.add())

        combobox.value = default
        self.combobox = combobox

        spinners = [ui.SpinBox(0.0, -1e3, 1e3, 0.1, rounding=2, width=3)
                    for __ in range(3)]

        win.add([_('Coordinates:')] + spinners)
        self.spinners = spinners
        win.add(_('Coordinates are relative to the center of the selection, '
                  'if any, else absolute.'))
        self.picky = ui.CheckButton(_('Check positions'), True)
        win.add([ui.Button(_('Add'), self.add),
                 self.picky])
        self.focus()

    def readfile(self, filename, format=None):
        if filename == self._filename:
            # We have this file already
            return self._atoms_from_file

        from ase.io import read
        try:
            atoms = read(filename)
        except Exception as err:
            ui.show_io_error(filename, err)
            atoms = None
            filename = None

        # Cache selected Atoms/filename (or None) for future calls
        self._atoms_from_file = atoms
        self._filename = filename
        return atoms

    def get_atoms(self):
        # Get the text, whether it's a combobox item or not
        val = self.combobox.widget.get()

        if val == current_selection_string:
            selection = self.gui.images.selected.copy()
            if selection.any():
                atoms = self.gui.atoms.copy()
                return atoms[selection[:len(self.gui.atoms)]]

        if val in atomic_numbers:  # Note: This means val is a symbol!
            return Atoms(val)

        if val.isdigit() and int(val) < len(chemical_symbols):
            return Atoms(numbers=[int(val)])

        from ase.collections import g2
        if val in g2.names:
            return g2[val]

        if os.path.exists(val):
            return self.readfile(val)  # May show UI error

        ui.showerror(_('Cannot add atoms'),
                     _('{} is neither atom, molecule, nor file')
                     .format(val))

        return None

    def getcoords(self):
        addcoords = np.array([spinner.value for spinner in self.spinners])

        pos = self.gui.atoms.positions
        if self.gui.images.selected[:len(pos)].any():
            pos = pos[self.gui.images.selected[:len(pos)]]
            center = pos.mean(0)
            addcoords += center

        return addcoords

    def focus(self):
        self.combobox.widget.focus_set()

    def add(self):
        newatoms = self.get_atoms()
        if newatoms is None:  # Error dialog was shown
            return

        newcenter = self.getcoords()

        # Not newatoms.center() because we want the same centering method
        # used for adding atoms relative to selections (mean).
        previous_center = newatoms.positions.mean(0)
        newatoms.positions += newcenter - previous_center

        atoms = self.gui.atoms
        if len(atoms) and self.picky.value:
            from ase.geometry import get_distances
            _disps, dists = get_distances(atoms.positions,
                                         newatoms.positions)
            mindist = dists.min()
            if mindist < 0.5:
                ui.showerror(_('Bad positions'),
                             _('Atom would be less than 0.5 Å from '
                               'an existing atom.  To override, '
                               'uncheck the check positions option.'))
                return

        self.gui.add_atoms_and_select(newatoms)

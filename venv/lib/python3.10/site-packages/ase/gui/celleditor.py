# fmt: off

'''celleditor.py - Window for editing the cell of an atoms object
'''
import numpy as np

import ase.gui.ui as ui
from ase.cell import Cell
from ase.gui.i18n import _


class CellEditor:
    '''Window for editing the cell of an atoms object.'''

    def __init__(self, gui):
        self.gui = gui
        self.gui.register_vulnerable(self)

        # Create grid control for cells
        # xx xy xz ||x|| pbc
        # yx yy yz ||y|| pbc
        # zx zy zz ||z|| pbc
        self.cell_grid = []
        self.pbc = []
        self.angles = []

        atoms = self.gui.atoms

        cell = atoms.cell
        mags = cell.lengths()
        angles = cell.angles()
        pbc = atoms.pbc

        for i in [0, 1, 2]:  # x_ y_ z_
            row = []
            for j in [0, 1, 2]:  # _x _y _z
                row.append(ui.SpinBox(cell[i][j], -30, 30, 0.1,
                           self.apply_vectors, rounding=7, width=9))
            row.append(ui.SpinBox(mags[i], -30, 30, 0.1, self.apply_magnitudes,
                                  rounding=7, width=9))
            self.cell_grid.append(row)
            self.pbc.append(ui.CheckButton('', bool(pbc[i]), self.apply_pbc))
            self.angles.append(ui.SpinBox(angles[i], -360, 360, 15,
                                          self.apply_angles,
                                          rounding=7, width=9))

        self.scale_atoms = ui.CheckButton('', False)
        self.vacuum = ui.SpinBox(5, 0, 15, 0.1, self.apply_vacuum)

        # TRANSLATORS: This is a title of a window.
        win = self.win = ui.Window(_('Cell Editor'), wmtype='utility')

        x, y, z = self.cell_grid

        win.add([_('A:'), x[0], x[1], x[2], _('||A||:'), x[3],
                 _('periodic:'), self.pbc[0]])
        win.add([_('B:'), y[0], y[1], y[2], _('||B||:'), y[3],
                 _('periodic:'), self.pbc[1]])
        win.add([_('C:'), z[0], z[1], z[2], _('||C||:'), z[3],
                 _('periodic:'), self.pbc[2]])
        win.add([_('∠BC:'), self.angles[0], _('∠AC:'), self.angles[1],
                 _('∠AB:'), self.angles[2]])
        win.add([_('Scale atoms with cell:'), self.scale_atoms])
        win.add([ui.Button(_('Apply Vectors'), self.apply_vectors),
                 ui.Button(_('Apply Magnitudes'), self.apply_magnitudes),
                 ui.Button(_('Apply Angles'), self.apply_angles)])
        win.add([_('Pressing 〈Enter〉 as you enter values will '
                   'automatically apply correctly')])
        # TRANSLATORS: verb
        win.add([ui.Button(_('Center'), self.apply_center),
                 ui.Button(_('Wrap'), self.apply_wrap),
                 _('Vacuum:'), self.vacuum,
                 ui.Button(_('Apply Vacuum'), self.apply_vacuum)])

    def apply_center(self, *args):
        atoms = self.gui.atoms.copy()
        atoms.center()
        self.gui.new_atoms(atoms)

    def apply_wrap(self, *args):
        atoms = self.gui.atoms.copy()
        atoms.wrap()
        self.gui.new_atoms(atoms)

    def apply_vacuum(self, *args):
        atoms = self.gui.atoms.copy()

        axis = []
        for index, pbc in enumerate(atoms.pbc):
            if not pbc:
                axis.append(index)

        atoms.center(vacuum=self.vacuum.value, axis=axis)
        self.gui.new_atoms(atoms)

    def apply_vectors(self, *args):
        atoms = self.gui.atoms.copy()

        atoms.set_cell(self.get_vectors(),
                       scale_atoms=self.scale_atoms.var.get())
        self.gui.new_atoms(atoms)

    def get_vectors(self):
        x, y, z = self.cell_grid
        cell = np.array(
            [[x[0].value, x[1].value, x[2].value],
             [y[0].value, y[1].value, y[2].value],
             [z[0].value, z[1].value, z[2].value]]
        )
        return Cell(cell)

    def get_magnitudes(self):
        x, y, z = self.cell_grid
        return np.array([x[3].value, y[3].value, z[3].value])

    def apply_magnitudes(self, *args):
        atoms = self.gui.atoms.copy()

        old_mags = atoms.cell.lengths()
        new_mags = self.get_magnitudes()

        newcell = atoms.cell.copy()
        for i in range(3):
            newcell[i] *= new_mags[i] / old_mags[i]

        atoms.set_cell(newcell,
                       scale_atoms=self.scale_atoms.var.get())

        self.gui.new_atoms(atoms)

    def apply_angles(self, *args):
        atoms = self.gui.atoms.copy()

        cell_data = atoms.cell.cellpar()
        cell_data[3:7] = [self.angles[0].value, self.angles[1].value,
                          self.angles[2].value]

        atoms.set_cell(cell_data, scale_atoms=self.scale_atoms.var.get())

        self.gui.new_atoms(atoms)

    def apply_pbc(self, *args):
        atoms = self.gui.atoms.copy()

        pbc = [pbc.var.get() for pbc in self.pbc]
        atoms.set_pbc(pbc)

        self.gui.new_atoms(atoms)

    def notify_atoms_changed(self):
        atoms = self.gui.atoms
        self.update(atoms.cell, atoms.pbc)

    def update(self, cell, pbc):
        cell = Cell(cell)
        mags = cell.lengths()
        angles = cell.angles()

        for i in range(3):
            for j in range(3):
                if np.isnan(cell[i][j]):
                    cell[i][j] = 0
                self.cell_grid[i][j].value = cell[i][j]

            if np.isnan(mags[i]):
                mags[i] = 0
            self.cell_grid[i][3].value = mags[i]

            if np.isnan(angles[i]):
                angles[i] = 0
            self.angles[i].value = angles[i]

            self.pbc[i].var.set(bool(pbc[i]))

# fmt: off

import ase.gui.ui as ui
from ase.gui.i18n import _
from ase.utils import irotate, rotate


class Rotate:
    update = True

    def __init__(self, gui):
        self.gui = gui
        win = ui.Window(_('Rotate'), wmtype='utility')
        win.add(_('Rotation angles:'))
        self.rotate = [ui.SpinBox(42.0, -360, 360, 1, self.change)
                       for _ in '123']
        win.add(self.rotate)
        win.add(ui.Button(_('Update'), self.update_angles))
        win.add(_('Note:\nYou can rotate freely\n'
                  'with the mouse, by holding\n'
                  'down mouse button 2.'))
        self.update_angles()

    def change(self):
        x, y, z = (float(a.value) for a in self.rotate)
        self.gui.axes = rotate(f'{x:f}x,{y:f}y,{z:f}z')
        self.gui.set_frame()

    def update_angles(self):
        angles = irotate(self.gui.axes)
        for r, a in zip(self.rotate, angles):
            r.value = a

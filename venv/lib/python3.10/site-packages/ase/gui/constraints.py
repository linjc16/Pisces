# fmt: off

import ase.gui.ui as ui
from ase.gui.i18n import _


class Constraints:
    def __init__(self, gui):
        win = ui.Window(_('Constraints'), wmtype='utility')
        win.add([ui.Button(_('Fix'), self.selected),
                 _('selected atoms')])
        win.add([ui.Button(_('Release'), self.unconstrain),
                 _('selected atoms')])
        win.add(ui.Button(_('Clear all constraints'), self.clear))
        self.gui = gui

    def selected(self):
        self.gui.images.set_dynamic(self.gui.images.selected, False)
        self.gui.draw()

    def unconstrain(self):
        self.gui.images.set_dynamic(self.gui.images.selected, True)
        self.gui.draw()

    def clear(self):
        # This clears *all* constraints.  But when we constrain, we
        # only add FixAtoms....
        for atoms in self.gui.images:
            atoms.constraints = []

        # Also, these methods are repeated from settings.py *grumble*
        self.gui.draw()

# fmt: off

import ase.data
import ase.gui.ui as ui
from ase import Atoms
from ase.gui.i18n import _


class Element(list):
    def __init__(self, symbol='', callback=None):
        list.__init__(self,
                      [_('Element:'),
                       ui.Entry(symbol, 3, self.enter),
                       ui.Button(_('Help'), self.show_help),
                       ui.Label('', 'red')])
        self.callback = callback

    @property
    def z_entry(self):
        return self[1]

    def grab_focus(self):
        self.z_entry.entry.focus_set()

    def show_help(self):
        msg = _('Enter a chemical symbol or the atomic number.')
        # Title of a popup window
        ui.showinfo(_('Info'), msg)

    @property
    def Z(self):
        atoms = self.get_atoms()
        if atoms is None:
            return None
        assert len(atoms) == 1
        return atoms.numbers[0]

    @property
    def symbol(self):
        Z = self.Z
        return None if Z is None else ase.data.chemical_symbols[Z]

    # Used by tests...
    @symbol.setter
    def symbol(self, value):
        self.z_entry.value = value

    def get_atoms(self):
        val = self._get()
        if val is not None:
            self[2].text = ''
        return val

    def _get(self):
        txt = self.z_entry.value

        if not txt:
            self.error(_('No element specified!'))
            return None

        if txt.isdigit():
            txt = int(txt)
            try:
                txt = ase.data.chemical_symbols[txt]
            except KeyError:
                self.error()
                return None

        if txt in ase.data.atomic_numbers:
            return Atoms(txt)

        self.error()

    def enter(self):
        self.callback(self)

    def error(self, text=_('ERROR: Invalid element!')):
        self[2].text = text


def pybutton(title, callback):
    """A button for displaying Python code.

    When pressed, it opens a window displaying some Python code, or an error
    message if no Python code is ready.
    """
    return ui.Button('Python', pywindow, title, callback)


def pywindow(title, callback):
    code = callback()
    if code is None:
        ui.error(
            _('No Python code'),
            _('You have not (yet) specified a consistent set of parameters.'))
    else:
        win = ui.Window(title, wmtype='utility')
        win.add(ui.Text(code))

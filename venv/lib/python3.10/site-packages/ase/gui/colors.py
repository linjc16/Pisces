# fmt: off

"""colors.py - select how to color the atoms in the GUI."""
import numpy as np

import ase.gui.ui as ui
from ase.gui.i18n import _
from ase.gui.utils import get_magmoms


class ColorWindow:
    """A window for selecting how to color the atoms."""

    def __init__(self, gui):
        self.reset(gui)

    def reset(self, gui):
        """create a new color window"""
        self.win = ui.Window(_('Colors'), wmtype='utility')
        self.gui = gui
        self.win.add(ui.Label(_('Choose how the atoms are colored:')))
        values = ['jmol', 'tag', 'force', 'velocity',
                  'initial charge', 'magmom', 'neighbors']
        labels = [_('By atomic number, default "jmol" colors'),
                  _('By tag'),
                  _('By force'),
                  _('By velocity'),
                  _('By initial charge'),
                  _('By magnetic moment'),
                  _('By number of neighbors'), ]

        haveit = ['numbers', 'positions', 'forces', 'momenta',
                  'initial_charges', 'initial_magmoms']
        for key in self.gui.atoms.arrays:
            if key not in haveit:
                values.append(key)
                labels.append(f'By user-defined "{key}"')

        self.radio = ui.RadioButtons(labels, values, self.toggle,
                                     vertical=True)
        self.radio.value = gui.colormode
        self.win.add(self.radio)
        self.activate()
        self.label = ui.Label()
        self.win.add(self.label)

        if hasattr(self, 'mnmx'):
            self.win.add(self.cmaps)
            self.win.add(self.mnmx)

    def change_mnmx(self, mn=None, mx=None):
        """change min and/or max values for colormap"""
        if mn:
            self.mnmx[1].value = mn
        if mx:
            self.mnmx[3].value = mx
        mn, mx = self.mnmx[1].value, self.mnmx[3].value
        colorscale, _, _ = self.gui.colormode_data
        self.gui.colormode_data = colorscale, mn, mx
        self.gui.draw()

    def activate(self):
        images = self.gui.images
        atoms = self.gui.atoms
        radio = self.radio
        radio['tag'].active = atoms.has('tags')

        # XXX not sure how to deal with some images having forces,
        # and other images not.  Same goes for below quantities
        F = images.get_forces(atoms)
        radio['force'].active = F is not None
        radio['velocity'].active = atoms.has('momenta')
        radio['initial charge'].active = atoms.has('initial_charges')
        radio['magmom'].active = get_magmoms(atoms).any()
        radio['neighbors'].active = True

    def toggle(self, value):
        self.gui.colormode = value
        if value == 'jmol' or value == 'neighbors':
            if hasattr(self, 'mnmx'):
                "delete the min max fields by creating a new window"
                del self.mnmx
                del self.cmaps
                self.win.close()
                self.reset(self.gui)
            text = ''
        else:
            scalars = np.ma.array([self.gui.get_color_scalars(i)
                                   for i in range(len(self.gui.images))])
            mn = np.min(scalars)
            mx = np.max(scalars)
            self.gui.colormode_data = None, mn, mx

            cmaps = ['default', 'old']
            try:
                import pylab as plt
                cmaps += [m for m in plt.cm.datad if not m.endswith("_r")]
            except ImportError:
                pass
            self.cmaps = [_('cmap:'),
                          ui.ComboBox(cmaps, cmaps, self.update_colormap),
                          _('N:'),
                          ui.SpinBox(26, 0, 100, 1, self.update_colormap)]
            self.update_colormap('default')

            try:
                unit = {'tag': '',
                        'force': 'eV/Ang',
                        'velocity': '(eV/amu)^(1/2)',
                        'charge': '|e|',
                        'initial charge': '|e|',
                        'magmom': 'μB'}[value]
            except KeyError:
                unit = ''
            text = ''

            rng = mx - mn  # XXX what are optimal allowed range and steps ?
            self.mnmx = [_('min:'),
                         ui.SpinBox(mn, mn - 10 * rng, mx + rng, rng / 10.,
                                    self.change_mnmx, width=20),
                         _('max:'),
                         ui.SpinBox(mx, mn - 10 * rng, mx + rng, rng / 10.,
                                    self.change_mnmx, width=20),
                         _(unit)]
            self.win.close()
            self.reset(self.gui)

        self.label.text = text
        self.radio.value = value
        self.gui.draw()
        return text  # for testing

    def notify_atoms_changed(self):
        "Called by gui object when the atoms have changed."
        self.activate()
        mode = self.gui.colormode
        if not self.radio[mode].active:
            mode = 'jmol'
        self.toggle(mode)

    def update_colormap(self, cmap=None, N=26):
        "Called by gui when colormap has changed"
        import matplotlib
        if cmap is None:
            cmap = self.cmaps[1].value
        try:
            N = int(self.cmaps[3].value)
        except AttributeError:
            N = 26
        colorscale, mn, mx = self.gui.colormode_data
        if cmap == 'default':
            colorscale = ['#{0:02X}80{0:02X}'.format(int(red))
                          for red in np.linspace(0, 250, N)]
        elif cmap == 'old':
            colorscale = [f'#{int(red):02X}AA00'
                          for red in np.linspace(0, 230, N)]
        else:
            cmap_obj = matplotlib.colormaps[cmap]
            colorscale = [matplotlib.colors.rgb2hex(c[:3]) for c in
                          cmap_obj(np.linspace(0, 1, N))]
        self.gui.colormode_data = colorscale, mn, mx
        self.gui.draw()

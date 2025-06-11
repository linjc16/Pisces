# fmt: off

"""Inline viewer for jupyter notebook using X3D."""
import warnings

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from ase.io.x3d import write_x3d


def view_x3d(atoms, *args, **kwargs):
    """View atoms inline in a jupyter notebook. This command
    should only be used within a jupyter/ipython notebook.

    Args:
        atoms - ase.Atoms, atoms to be rendered"""
    try:
        from IPython.display import HTML
    except ImportError:
        warnings.warn('Please install IPython')
        return None

    notebook_style = {'width': '400px', 'height': '300px'}

    temp = StringIO()
    write_x3d(temp, atoms, format='X3DOM', style=notebook_style)
    data = temp.getvalue()
    temp.close()
    return HTML(data)

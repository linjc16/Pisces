# fmt: off

"""
Module for managing viewers

View plugins can be registered through the entrypoint system in with the
following in a module, such as a `viewer.py` file:

```python3
VIEWER_ENTRYPOINT = ExternalViewer(
    desc="Visualization using <my package>",
    module="my_package.viewer"
)
```

Where module `my_package.viewer` contains a `view_my_viewer` function taking
and `ase.Atoms` object as the first argument, and also `**kwargs`.

Then ones needs to register an entry point in `pyproject.toml` with

```toml
[project.entry-points."ase.visualize"]
my_viewer = "my_package.viewer:VIEWER_ENTRYPOINT"
```

After this, call to `ase.visualize.view(atoms, viewer='my_viewer')` will be
forwarded to `my_package.viewer.view_my_viewer` function.

"""
import pickle
import subprocess
import sys
import tempfile
import warnings
from contextlib import contextmanager
from importlib import import_module
from importlib.metadata import entry_points
from io import BytesIO
from pathlib import Path

from ase.io import write
from ase.io.formats import ioformats
from ase.utils.plugins import ExternalViewer


class UnknownViewerError(Exception):
    """The view tyep is unknown"""


class AbstractViewer:
    def view(self, *args, **kwargss):
        raise NotImplementedError


class PyViewer(AbstractViewer):
    def __init__(self, name: str, desc: str, module_name: str):
        """
        Instantiate an viewer
        """
        self.name = name
        self.desc = desc
        self.module_name = module_name

    def _viewfunc(self):
        """Return the function used for viewing the atoms"""
        return getattr(self.module, "view_" + self.name, None)

    @property
    def module(self):
        try:
            return import_module(self.module_name)
        except ImportError as err:
            raise UnknownViewerError(
                f"Viewer not recognized: {self.name}.  Error: {err}"
            ) from err

    def view(self, atoms, *args, **kwargs):
        return self._viewfunc()(atoms, *args, **kwargs)


class CLIViewer(AbstractViewer):
    """Generic viewer for"""

    def __init__(self, name, fmt, argv):
        self.name = name
        self.fmt = fmt
        self.argv = argv

    @property
    def ioformat(self):
        return ioformats[self.fmt]

    @contextmanager
    def mktemp(self, atoms, data=None):
        ioformat = self.ioformat
        suffix = "." + ioformat.extensions[0]

        if ioformat.isbinary:
            mode = "wb"
        else:
            mode = "w"

        with tempfile.TemporaryDirectory(prefix="ase-view-") as dirname:
            # We use a tempdir rather than a tempfile because it's
            # less hassle to handle the cleanup on Windows (files
            # cannot be open on multiple processes).
            path = Path(dirname) / f"atoms{suffix}"
            with path.open(mode) as fd:
                if data is None:
                    write(fd, atoms, format=self.fmt)
                else:
                    write(fd, atoms, format=self.fmt, data=data)
            yield path

    def view_blocking(self, atoms, data=None):
        with self.mktemp(atoms, data) as path:
            subprocess.check_call(self.argv + [str(path)])

    def view(
        self,
        atoms,
        data=None,
        repeat=None,
        **kwargs,
    ):
        """Spawn a new process in which to open the viewer."""
        if repeat is not None:
            atoms = atoms.repeat(repeat)

        proc = subprocess.Popen(
            [sys.executable, "-m", "ase.visualize.viewers"],
            stdin=subprocess.PIPE
        )

        pickle.dump((self, atoms, data), proc.stdin)
        proc.stdin.close()
        return proc


VIEWERS = {}


def _pipe_to_ase_gui(atoms, repeat, **kwargs):
    buf = BytesIO()
    write(buf, atoms, format="traj")

    args = [sys.executable, "-m", "ase", "gui", "-"]
    if repeat:
        args.append("--repeat={},{},{}".format(*repeat))

    proc = subprocess.Popen(args, stdin=subprocess.PIPE)
    proc.stdin.write(buf.getvalue())
    proc.stdin.close()
    return proc


def define_viewer(
    name, desc, *, module=None, cli=False, fmt=None, argv=None, external=False
):
    if not external:
        if module is None:
            module = name
        module = "ase.visualize." + module
    if cli:
        fmt = CLIViewer(name, fmt, argv)
    else:
        if name == "ase":
            # Special case if the viewer is named `ase` then we use
            # the _pipe_to_ase_gui as the viewer method
            fmt = PyViewer(name, desc, module_name=None)
            fmt.view = _pipe_to_ase_gui
        else:
            fmt = PyViewer(name, desc, module_name=module)
    VIEWERS[name] = fmt
    return fmt


def define_external_viewer(entry_point):
    """Define external viewer"""

    viewer_def = entry_point.load()
    if entry_point.name in VIEWERS:
        raise ValueError(f"Format {entry_point.name} already defined")
    if not isinstance(viewer_def, ExternalViewer):
        raise TypeError(
            "Wrong type for registering external IO formats "
            f"in format {entry_point.name}, expected "
            "ExternalViewer"
        )
    define_viewer(entry_point.name, **viewer_def._asdict(),
                  external=True)


def register_external_viewer_formats(group):
    if hasattr(entry_points(), "select"):
        viewer_entry_points = entry_points().select(group=group)
    else:
        viewer_entry_points = entry_points().get(group, ())

    for entry_point in viewer_entry_points:
        try:
            define_external_viewer(entry_point)
        except Exception as exc:
            warnings.warn(
                "Failed to register external "
                f"Viewer {entry_point.name}: {exc}"
            )


define_viewer("ase", "View atoms using ase gui.")
define_viewer("ngl", "View atoms using nglview.")
define_viewer("mlab", "View atoms using matplotlib.")
define_viewer("sage", "View atoms using sage.")
define_viewer("x3d", "View atoms using x3d.")

# CLI viweers that are internally supported
define_viewer(
    "avogadro", "View atoms using avogradro.", cli=True, fmt="cube",
    argv=["avogadro"]
)
define_viewer(
    "ase_gui_cli", "View atoms using ase gui.", cli=True, fmt="traj",
    argv=[sys.executable, '-m', 'ase.gui'],
)
define_viewer(
    "gopenmol",
    "View atoms using gopenmol.",
    cli=True,
    fmt="extxyz",
    argv=["runGOpenMol"],
)
define_viewer(
    "rasmol",
    "View atoms using rasmol.",
    cli=True,
    fmt="proteindatabank",
    argv=["rasmol", "-pdb"],
)
define_viewer("vmd", "View atoms using vmd.", cli=True, fmt="cube",
              argv=["vmd"])
define_viewer(
    "xmakemol",
    "View atoms using xmakemol.",
    cli=True,
    fmt="extxyz",
    argv=["xmakemol", "-f"],
)

register_external_viewer_formats("ase.visualize")

CLI_VIEWERS = {key: value for key, value in VIEWERS.items()
               if isinstance(value, CLIViewer)}
PY_VIEWERS = {key: value for key, value in VIEWERS.items()
              if isinstance(value, PyViewer)}


def cli_main():
    """
    This is mainly to facilitate launching CLI viewer in a separate python
    process
    """
    cli_viewer, atoms, data = pickle.load(sys.stdin.buffer)
    cli_viewer.view_blocking(atoms, data)


if __name__ == "__main__":
    cli_main()

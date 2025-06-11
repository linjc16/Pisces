"""Build ASE's web-page."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

cmds = """\
python3 -m venv venv
. venv/bin/activate
pip install -qq -U pip
pip install "sphinx<6.0"
pip install sphinx-rtd-theme pillow scriv
git clone -q https://gitlab.com/ase/ase.git
cd ase
pip install . -qq
(scriv collect --keep || echo Not compiling changelog then) &> scriv.out
cd doc
make
mv build/html ase-web-page"""


def build():
    root = Path('/scratch/jensj/ase-docs')
    if root.is_dir():
        sys.exit('Locked')
    root.mkdir()
    os.chdir(root)
    cmds2 = ' && '.join(cmds.splitlines())
    p = subprocess.run(cmds2, shell=True)
    if p.returncode == 0:
        status = 'ok'
    else:
        print('FAILED!', file=sys.stdout)
        status = 'error'
    f = root.with_name(f'ase-docs-{status}')
    if f.is_dir():
        shutil.rmtree(f)
    root.rename(f)
    return status


def build_all():
    assert build() == 'ok'
    webpage = Path('/scratch/jensj/ase-docs-ok/ase/doc/ase-web-page')
    home = Path.home() / 'web-pages'
    cmds = ' && '.join(
        [
            f'cd {webpage.parent}',
            'tar -czf ase-web-page.tar.gz ase-web-page',
            f'cp ase-web-page.tar.gz {home}',
        ]
    )
    subprocess.run(cmds, shell=True, check=True)


if __name__ == '__main__':
    build_all()

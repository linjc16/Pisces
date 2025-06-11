#!/usr/bin/env python3
# fmt: off


"""Generate new release of ASE.

This script does not attempt to import ASE - then it would depend on
which ASE is installed and how - but assumes that it is run from the
ASE root directory."""

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from time import strftime

import ase


def runcmd(cmd, output=False, error_ok=False):
    print('Executing:', cmd)
    try:
        if output:
            txt = subprocess.check_output(cmd, shell=True)
            return txt.decode('utf8')
        else:
            return subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as err:
        if error_ok:
            print(f'Failed: {err}')
            print('Continuing...')
        else:
            raise


bash = runcmd


def py(cmd):
    return runcmd(f'python3 {cmd}')


def git(cmd, error_ok=False):
    cmd = f'git {cmd}'
    return runcmd(cmd, output=True, error_ok=error_ok)


versionfile = Path(ase.__file__)

ase_toplevel = versionfile.parent.parent


def get_version():
    with open(versionfile) as fd:
        return re.search(r"__version__ = '(\S+)'", fd.read()).group(1)


def main():
    p = argparse.ArgumentParser(description='Generate new release of ASE.',
                                epilog='Run from the root directory of ASE.')
    p.add_argument('version', nargs=1,
                   help='version number for new release')
    # p.add_argument('nextversion', nargs=1,
    #                help='development version after release')
    p.add_argument('--clean', action='store_true',
                   help='delete release branch and tag')
    args = p.parse_args()

    assert versionfile.name == '__init__.py'
    assert ase_toplevel == Path.cwd()

    try:
        current_version = get_version()
    except Exception as err:
        p.error('Cannot get version: {}.  Are you in the root directory?'
                .format(err))

    print(f'Current version: {current_version}')

    version = args.version[0]

    branchname = f'ase-{version}'
    current_version = get_version()

    if args.clean:
        print(f'Cleaning {version}')
        git('checkout master')
        # git('tag -d {}'.format(version), error_ok=True)
        git(f'branch -D {branchname}', error_ok=True)
        # git('branch -D {}'.format('web-page'), error_ok=True)
        return

    print(f'New release: {version}')

    if shutil.which('scriv') is None:
        p.error('No "scriv" command in PATH.  Is scriv installed?')

    runcmd(f'scriv collect --add --title "Version {version}"')

    txt = git('status')
    branch = re.match(r'On branch (\S+)', txt).group(1)

    def match_and_edit_version(path, pattern, replacement):
        print(f'Editing {path}: version {version}')
        lines = []
        matches = 0

        with open(path) as fd:
            for line in fd:
                if line.startswith(pattern):
                    line = replacement.rstrip() + '\n'
                    matches += 1
                lines.append(line)

        assert matches == 1, 'Should only match one line!'
        path.write_text(''.join(lines))

    match_and_edit_version(
        versionfile,
        pattern='__version__ = ',
        replacement=f"__version__ = '{version}'")

    releasenotes = ase_toplevel / 'doc/releasenotes.rst'

    searchtxt = re.escape("""\
Git master branch
=================

:git:`master <>`.
""")

    replacetxt = """\
Git master branch
=================

:git:`master <>`.

* No changes yet


{header}
{underline}

{date}: :git:`{version} <../{version}>`
"""

    date = strftime('%d %B %Y').lstrip('0')
    header = f'Version {version}'
    underline = '=' * len(header)
    replacetxt = replacetxt.format(header=header, version=version,
                                   underline=underline, date=date)

    print(f'Editing {releasenotes}')
    with open(releasenotes) as fd:
        txt = fd.read()
    txt, n = re.subn(searchtxt, replacetxt, txt, re.MULTILINE)
    assert n == 1

    with open(releasenotes, 'w') as fd:
        fd.write(txt)

    searchtxt = """\
News
====
"""

    replacetxt = """\
News
====

* :ref:`ASE version {version} <releasenotes>` released ({date}).
"""

    replacetxt = replacetxt.format(version=version, date=date)

    frontpage = ase_toplevel / 'doc/index.rst'

    print(f'Editing {frontpage}')
    with open(frontpage) as fd:
        txt = fd.read()
    txt, n = re.subn(searchtxt, replacetxt, txt)
    assert n == 1
    with open(frontpage, 'w') as fd:
        fd.write(txt)

    installdoc = ase_toplevel / 'doc/install.rst'
    print(f'Editing {installdoc}')

    with open(installdoc) as fd:
        txt = fd.read()

    txt, nsub = re.subn(r'ase-\d+\.\d+\.\d+',
                        f'ase-{version}', txt)
    assert nsub > 0
    txt, nsub = re.subn(r'git clone -b \d+\.\d+\.\d+',
                        f'git clone -b {version}', txt)
    assert nsub == 1

    with open(installdoc, 'w') as fd:
        fd.write(txt)

    print(f'Creating new release from branch {branch!r}')
    git(f'checkout -b {branchname}')

    edited_paths = [versionfile, installdoc, frontpage, releasenotes]

    git('add {}'.format(' '.join(str(path) for path in edited_paths)))
    git(f'commit -m "ASE version {version}"')
    # git('tag -s {0} -m "ase-{0}"'.format(version))

    buildpath = Path('build')
    if buildpath.is_dir():
        print('Removing stale build directory, since it exists')
        assert Path('ase/__init__.py').exists()
        assert Path('setup.py').exists()
        shutil.rmtree('build')
    else:
        print('No stale build directory found; proceeding')

    py('-m build')
    # py('setup.py sdist > setup_sdist.log')
    # py('setup.py bdist_wheel > setup_bdist_wheel3.log')
    # bash(f'gpg --armor --yes --detach-sign dist/ase-{version}.tar.gz')

    print()
    print('Automatic steps done.')
    print()
    print('Now is a good time to:')
    print(' * check the diff')
    print(' * run the tests')
    print(' * verify the web-page build')
    print()
    print('Remaining steps')
    print('===============')
    print(f'git show {version}  # Inspect!')
    print('git checkout master')
    print(f'git merge {branchname}')
    print('twine upload '
          'dist/ase-{v}.tar.gz '
          'dist/ase-{v}-py3-none-any.whl '
          'dist/ase-{v}.tar.gz.asc'.format(v=version))
    print('git push --tags origin master  # Assuming your remote is "origin"')


if __name__ == '__main__':
    os.environ['LANGUAGE'] = 'C'

    main()

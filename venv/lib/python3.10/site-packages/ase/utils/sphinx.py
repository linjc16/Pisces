# fmt: off

import os
import re
import runpy
import traceback
import warnings
from os.path import join
from pathlib import Path
from stat import ST_MTIME
from subprocess import DEVNULL, CalledProcessError, check_call

from docutils import nodes
from docutils.parsers.rst.roles import set_classes


def mol_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    n = []
    t = ''
    while text:
        if text[0] == '_':
            n.append(nodes.inline(text=t))
            t = ''
            m = re.match(r'\d+', text[1:])
            if m is None:
                raise RuntimeError('Expected one or more digits after "_"')
            digits = m.group()
            n.append(nodes.subscript(text=digits))
            text = text[1 + len(digits):]
        else:
            t += text[0]
            text = text[1:]
    n.append(nodes.inline(text=t))
    return n, []


def git_role_tmpl(urlroot,
                  role,
                  rawtext, text, lineno, inliner, options={}, content=[]):
    if text[-1] == '>':
        i = text.index('<')
        name = text[:i - 1]
        text = text[i + 1:-1]
    else:
        name = text
        if name[0] == '~':
            name = name.split('/')[-1]
            text = text[1:]
        if '?' in name:
            name = name[:name.index('?')]
    # Check if the link is broken
    is_tag = text.startswith('..')  # Tags are like :git:`3.19.1 <../3.19.1>`
    path = os.path.join('..', text)
    do_exists = os.path.exists(path)
    if not (is_tag or do_exists):
        msg = f'Broken link: {rawtext}: Non-existing path: {path}'
        msg = inliner.reporter.error(msg, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]
    ref = urlroot + text
    set_classes(options)
    node = nodes.reference(rawtext, name, refuri=ref,
                           **options)
    return [node], []


def git_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    return git_role_tmpl('https://gitlab.com/ase/ase/blob/master/',
                         role,
                         rawtext, text, lineno, inliner, options, content)


def setup(app):
    app.add_role('mol', mol_role)
    app.add_role('git', git_role)
    create_png_files()


def creates():
    """Generator for Python scripts and their output filenames."""
    for dirpath, dirnames, filenames in sorted(os.walk('.')):
        if dirpath.startswith('./build'):
            # Skip files in the build/ folder
            continue

        for filename in sorted(filenames):
            if filename.endswith('.py'):
                path = join(dirpath, filename)
                with open(path) as fd:
                    lines = fd.readlines()
                if len(lines) == 0:
                    continue
                if 'coding: utf-8' in lines[0]:
                    lines.pop(0)
                outnames = []
                for line in lines:
                    if line.startswith('# creates:'):
                        outnames.extend([file.rstrip(',')
                                         for file in line.split()[2:]])
                    else:
                        break
                if outnames:
                    yield dirpath, filename, outnames


def create_png_files(raise_exceptions=False):
    from ase.utils import workdir
    try:
        check_call(['povray', '-h'], stderr=DEVNULL)
    except (FileNotFoundError, CalledProcessError):
        warnings.warn('No POVRAY!')
        # Replace write_pov with write_png:
        from ase.io import pov
        from ase.io.png import write_png

        def write_pov(filename, atoms,
                      povray_settings={}, isosurface_data=None,
                      **generic_projection_settings):

            write_png(Path(filename).with_suffix('.png'), atoms,
                      **generic_projection_settings)

            class DummyRenderer:
                def render(self):
                    pass

            return DummyRenderer()

        pov.write_pov = write_pov

    for dir, pyname, outnames in creates():
        path = join(dir, pyname)
        t0 = os.stat(path)[ST_MTIME]
        run = False
        for outname in outnames:
            try:
                t = os.stat(join(dir, outname))[ST_MTIME]
            except OSError:
                run = True
                break
            else:
                if t < t0:
                    run = True
                    break
        if run:
            print('running:', path)
            with workdir(dir):
                import matplotlib.pyplot as plt
                plt.figure()
                try:
                    runpy.run_path(pyname)
                except KeyboardInterrupt:
                    return
                except Exception:
                    if raise_exceptions:
                        raise
                    else:
                        traceback.print_exc()

            for n in plt.get_fignums():
                plt.close(n)

            for outname in outnames:
                print(dir, outname)


def clean():
    """Remove all generated files."""
    for dir, pyname, outnames in creates():
        for outname in outnames:
            if os.path.isfile(os.path.join(dir, outname)):
                os.remove(os.path.join(dir, outname))


def visual_inspection():
    """Manually inspect generated files."""
    import subprocess
    images = []
    text = []
    pdf = []
    for dir, pyname, outnames in creates():
        for outname in outnames:
            path = os.path.join(dir, outname)
            ext = path.rsplit('.', 1)[1]
            if ext == 'pdf':
                pdf.append(path)
            elif ext in ['csv', 'txt', 'out', 'css', 'LDA', 'rst']:
                text.append(path)
            else:
                images.append(path)
    subprocess.call(['eog'] + images)
    subprocess.call(['evince'] + pdf)
    subprocess.call(['more'] + text)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process generated files.')
    parser.add_argument('command', nargs='?', default='list',
                        choices=['list', 'inspect', 'clean', 'run'])
    args = parser.parse_args()
    if args.command == 'clean':
        clean()
    elif args.command == 'list':
        for dir, pyname, outnames in creates():
            for outname in outnames:
                print(os.path.join(dir, outname))
    elif args.command == 'run':
        create_png_files(raise_exceptions=True)
    else:
        visual_inspection()

import os
import sys
import threading

import numpy as np
import pytest

from ase.calculators.emt import EMT
from ase.calculators.socketio import SocketClient, SocketIOCalculator
from ase.cluster.icosahedron import Icosahedron
from ase.optimize import BFGS

# If multiple test suites are running, we don't want port clashes.
# Thus we generate a port from the pid.
# maxpid is commonly 32768, and max port number is 65536.
# But in case maxpid is much larger for some reason:
pid = os.getpid()
inet_port = (3141 + pid) % 65536
# We could also use a Unix port perhaps, but not yet implemented

# unixsocket = 'grumble'
timeout = 20.0


def getatoms():
    return Icosahedron('Au', 3)


def run_server(launchclient=True, sockettype='unix'):
    atoms = getatoms()

    port = None
    unixsocket = None

    if sockettype == 'unix':
        unixsocket = f'ase_ipi_protocol_bfgs_test_{pid}'
    else:
        assert sockettype == 'inet'
        port = inet_port

    with SocketIOCalculator(log=sys.stdout, port=port,
                            unixsocket=unixsocket,
                            timeout=timeout) as calc:
        if launchclient:
            thread = launch_client_thread(port=port, unixsocket=unixsocket)
        atoms.calc = calc
        with BFGS(atoms) as opt:
            opt.run()

    if launchclient:
        thread.join()

    forces = atoms.get_forces()
    energy = atoms.get_potential_energy()

    atoms.calc = EMT()
    ref_forces = atoms.get_forces()
    ref_energy = atoms.get_potential_energy()

    refatoms = run_normal()
    ref_energy = refatoms.get_potential_energy()
    eerr = abs(energy - ref_energy)
    ferr = np.abs(forces - ref_forces).max()

    perr = np.abs(refatoms.positions - atoms.positions).max()
    print(f'errs e={eerr} f={ferr} pos={perr}')
    assert eerr < 1e-11, eerr
    assert ferr < 1e-11, ferr
    assert perr < 1e-11, perr


def run_normal():
    atoms = getatoms()
    atoms.calc = EMT()
    with BFGS(atoms) as opt:
        opt.run()
    return atoms


def run_client(port, unixsocket):
    atoms = getatoms()
    atoms.calc = EMT()

    try:
        with open('client.log', 'w') as fd:
            client = SocketClient(log=fd, port=port,
                                  unixsocket=unixsocket,
                                  timeout=timeout)
            client.run(atoms, use_stress=False)
    except BrokenPipeError:
        # I think we can find a way to close sockets so as not to get an
        # error, but presently things are not like that.
        pass


def launch_client_thread(port, unixsocket):
    thread = threading.Thread(target=run_client, args=(port, unixsocket))
    thread.start()
    return thread


unix_only = pytest.mark.skipif(os.name != 'posix',
                               reason='requires unix platform')


@pytest.mark.optimize()
@pytest.mark.parametrize('sockettype', [
    'inet',
    pytest.param('unix', marks=unix_only),
])
def test_ipi_protocol(sockettype, testdir):
    try:
        run_server(sockettype=sockettype)
    except OSError as err:
        # The AppVeyor CI tests sometimes fail when we try to open sockets on
        # computers where this is forbidden.  For now we will simply skip
        # this test when that happens:
        if 'forbidden by its access permissions' in err.strerror:
            pytest.skip(err.strerror)
        else:
            raise

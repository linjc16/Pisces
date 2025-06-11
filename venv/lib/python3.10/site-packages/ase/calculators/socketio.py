# fmt: off

import os
import socket
from contextlib import contextmanager
from subprocess import PIPE, Popen

import numpy as np

import ase.units as units
from ase.calculators.calculator import (
    Calculator,
    OldShellProfile,
    PropertyNotImplementedError,
    StandardProfile,
    all_changes,
)
from ase.calculators.genericfileio import GenericFileIOCalculator
from ase.parallel import world
from ase.stress import full_3x3_to_voigt_6_stress
from ase.utils import IOContext


def actualunixsocketname(name):
    return f'/tmp/ipi_{name}'


class SocketClosed(OSError):
    pass


class IPIProtocol:
    """Communication using IPI protocol."""

    def __init__(self, socket, txt=None):
        self.socket = socket

        if txt is None:
            def log(*args):
                pass
        else:
            def log(*args):
                print('Driver:', *args, file=txt)
                txt.flush()
        self.log = log

    def sendmsg(self, msg):
        self.log('  sendmsg', repr(msg))
        # assert msg in self.statements, msg
        msg = msg.encode('ascii').ljust(12)
        self.socket.sendall(msg)

    def _recvall(self, nbytes):
        """Repeatedly read chunks until we have nbytes.

        Normally we get all bytes in one read, but that is not guaranteed."""
        remaining = nbytes
        chunks = []
        while remaining > 0:
            chunk = self.socket.recv(remaining)
            if len(chunk) == 0:
                # (If socket is still open, recv returns at least one byte)
                raise SocketClosed
            chunks.append(chunk)
            remaining -= len(chunk)
        msg = b''.join(chunks)
        assert len(msg) == nbytes and remaining == 0
        return msg

    def recvmsg(self):
        msg = self._recvall(12)
        if not msg:
            raise SocketClosed

        assert len(msg) == 12, msg
        msg = msg.rstrip().decode('ascii')
        # assert msg in self.responses, msg
        self.log('  recvmsg', repr(msg))
        return msg

    def send(self, a, dtype):
        buf = np.asarray(a, dtype).tobytes()
        # self.log('  send {}'.format(np.array(a).ravel().tolist()))
        self.log(f'  send {len(buf)} bytes of {dtype}')
        self.socket.sendall(buf)

    def recv(self, shape, dtype):
        a = np.empty(shape, dtype)
        nbytes = np.dtype(dtype).itemsize * np.prod(shape)
        buf = self._recvall(nbytes)
        assert len(buf) == nbytes, (len(buf), nbytes)
        self.log(f'  recv {len(buf)} bytes of {dtype}')
        # print(np.frombuffer(buf, dtype=dtype))
        a.flat[:] = np.frombuffer(buf, dtype=dtype)
        # self.log('  recv {}'.format(a.ravel().tolist()))
        assert np.isfinite(a).all()
        return a

    def sendposdata(self, cell, icell, positions):
        assert cell.size == 9
        assert icell.size == 9
        assert positions.size % 3 == 0

        self.log(' sendposdata')
        self.sendmsg('POSDATA')
        self.send(cell.T / units.Bohr, np.float64)
        self.send(icell.T * units.Bohr, np.float64)
        self.send(len(positions), np.int32)
        self.send(positions / units.Bohr, np.float64)

    def recvposdata(self):
        cell = self.recv((3, 3), np.float64).T.copy()
        icell = self.recv((3, 3), np.float64).T.copy()
        natoms = self.recv(1, np.int32)[0]
        positions = self.recv((natoms, 3), np.float64)
        return cell * units.Bohr, icell / units.Bohr, positions * units.Bohr

    def sendrecv_force(self):
        self.log(' sendrecv_force')
        self.sendmsg('GETFORCE')
        msg = self.recvmsg()
        assert msg == 'FORCEREADY', msg
        e = self.recv(1, np.float64)[0]
        natoms = self.recv(1, np.int32)[0]
        assert natoms >= 0
        forces = self.recv((int(natoms), 3), np.float64)
        virial = self.recv((3, 3), np.float64).T.copy()
        nmorebytes = self.recv(1, np.int32)[0]
        morebytes = self.recv(nmorebytes, np.byte)
        return (e * units.Ha, (units.Ha / units.Bohr) * forces,
                units.Ha * virial, morebytes)

    def sendforce(self, energy, forces, virial,
                  morebytes=np.zeros(1, dtype=np.byte)):
        assert np.array([energy]).size == 1
        assert forces.shape[1] == 3
        assert virial.shape == (3, 3)

        self.log(' sendforce')
        self.sendmsg('FORCEREADY')  # mind the units
        self.send(np.array([energy / units.Ha]), np.float64)
        natoms = len(forces)
        self.send(np.array([natoms]), np.int32)
        self.send(units.Bohr / units.Ha * forces, np.float64)
        self.send(1.0 / units.Ha * virial.T, np.float64)
        # We prefer to always send at least one byte due to trouble with
        # empty messages.  Reading a closed socket yields 0 bytes
        # and thus can be confused with a 0-length bytestring.
        self.send(np.array([len(morebytes)]), np.int32)
        self.send(morebytes, np.byte)

    def status(self):
        self.log(' status')
        self.sendmsg('STATUS')
        msg = self.recvmsg()
        return msg

    def end(self):
        self.log(' end')
        self.sendmsg('EXIT')

    def recvinit(self):
        self.log(' recvinit')
        bead_index = self.recv(1, np.int32)
        nbytes = self.recv(1, np.int32)
        initbytes = self.recv(nbytes, np.byte)
        return bead_index, initbytes

    def sendinit(self):
        # XXX Not sure what this function is supposed to send.
        # It 'works' with QE, but for now we try not to call it.
        self.log(' sendinit')
        self.sendmsg('INIT')
        self.send(0, np.int32)  # 'bead index' always zero for now
        # We send one byte, which is zero, since things may not work
        # with 0 bytes.  Apparently implementations ignore the
        # initialization string anyway.
        self.send(1, np.int32)
        self.send(np.zeros(1), np.byte)  # initialization string

    def calculate(self, positions, cell):
        self.log('calculate')
        msg = self.status()
        # We don't know how NEEDINIT is supposed to work, but some codes
        # seem to be okay if we skip it and send the positions instead.
        if msg == 'NEEDINIT':
            self.sendinit()
            msg = self.status()
        assert msg == 'READY', msg
        icell = np.linalg.pinv(cell).transpose()
        self.sendposdata(cell, icell, positions)
        msg = self.status()
        assert msg == 'HAVEDATA', msg
        e, forces, virial, morebytes = self.sendrecv_force()
        r = dict(energy=e,
                 forces=forces,
                 virial=virial,
                 morebytes=morebytes)
        return r


@contextmanager
def bind_unixsocket(socketfile):
    assert socketfile.startswith('/tmp/ipi_'), socketfile
    serversocket = socket.socket(socket.AF_UNIX)
    try:
        serversocket.bind(socketfile)
    except OSError as err:
        raise OSError(f'{err}: {socketfile!r}')

    try:
        with serversocket:
            yield serversocket
    finally:
        os.unlink(socketfile)


@contextmanager
def bind_inetsocket(port):
    serversocket = socket.socket(socket.AF_INET)
    serversocket.setsockopt(socket.SOL_SOCKET,
                            socket.SO_REUSEADDR, 1)
    serversocket.bind(('', port))
    with serversocket:
        yield serversocket


class FileIOSocketClientLauncher:
    def __init__(self, calc):
        self.calc = calc

    def __call__(self, atoms, properties=None, port=None, unixsocket=None):
        assert self.calc is not None
        cwd = self.calc.directory

        profile = getattr(self.calc, 'profile', None)
        if isinstance(self.calc, GenericFileIOCalculator):
            # New GenericFileIOCalculator:
            template = getattr(self.calc, 'template')

            self.calc.write_inputfiles(atoms, properties)
            if unixsocket is not None:
                argv = template.socketio_argv(
                    profile, unixsocket=unixsocket, port=None
                )
            else:
                argv = template.socketio_argv(
                    profile, unixsocket=None, port=port
                )
            return Popen(argv, cwd=cwd, env=os.environ)
        else:
            # Old FileIOCalculator:
            self.calc.write_input(atoms, properties=properties,
                                  system_changes=all_changes)

            if isinstance(profile, StandardProfile):
                return profile.execute_nonblocking(self.calc)

            if profile is None:
                cmd = self.calc.command.replace('PREFIX', self.calc.prefix)
                cmd = cmd.format(port=port, unixsocket=unixsocket)
            elif isinstance(profile, OldShellProfile):
                cmd = profile.command.replace("PREFIX", self.calc.prefix)
            else:
                raise TypeError(
                    f"Profile type {type(profile)} not supported for socketio")

            return Popen(cmd, shell=True, cwd=cwd)


class SocketServer(IOContext):
    default_port = 31415

    def __init__(self,  # launch_client=None,
                 port=None, unixsocket=None, timeout=None,
                 log=None):
        """Create server and listen for connections.

        Parameters:

        client_command: Shell command to launch client process, or None
            The process will be launched immediately, if given.
            Else the user is expected to launch a client whose connection
            the server will then accept at any time.
            One calculate() is called, the server will block to wait
            for the client.
        port: integer or None
            Port on which to listen for INET connections.  Defaults
            to 31415 if neither this nor unixsocket is specified.
        unixsocket: string or None
            Filename for unix socket.
        timeout: float or None
            timeout in seconds, or unlimited by default.
            This parameter is passed to the Python socket object; see
            documentation therof
        log: file object or None
            useful debug messages are written to this."""

        if unixsocket is None and port is None:
            port = self.default_port
        elif unixsocket is not None and port is not None:
            raise ValueError('Specify only one of unixsocket and port')

        self.port = port
        self.unixsocket = unixsocket
        self.timeout = timeout
        self._closed = False

        if unixsocket is not None:
            actualsocket = actualunixsocketname(unixsocket)
            conn_name = f'UNIX-socket {actualsocket}'
            socket_context = bind_unixsocket(actualsocket)
        else:
            conn_name = f'INET port {port}'
            socket_context = bind_inetsocket(port)

        self.serversocket = self.closelater(socket_context)

        if log:
            print(f'Accepting clients on {conn_name}', file=log)

        self.serversocket.settimeout(timeout)

        self.serversocket.listen(1)

        self.log = log

        self.proc = None

        self.protocol = None
        self.clientsocket = None
        self.address = None

        # if launch_client is not None:
        #    self.proc = launch_client(port=port, unixsocket=unixsocket)

    def _accept(self):
        """Wait for client and establish connection."""
        # It should perhaps be possible for process to be launched by user
        log = self.log
        if log:
            print('Awaiting client', file=self.log)

        # If we launched the subprocess, the process may crash.
        # We want to detect this, using loop with timeouts, and
        # raise an error rather than blocking forever.
        if self.proc is not None:
            self.serversocket.settimeout(1.0)

        while True:
            try:
                self.clientsocket, self.address = self.serversocket.accept()
                self.closelater(self.clientsocket)
            except socket.timeout:
                if self.proc is not None:
                    status = self.proc.poll()
                    if status is not None:
                        raise OSError('Subprocess terminated unexpectedly'
                                      ' with status {}'.format(status))
            else:
                break

        self.serversocket.settimeout(self.timeout)
        self.clientsocket.settimeout(self.timeout)

        if log:
            # For unix sockets, address is b''.
            source = ('client' if self.address == b'' else self.address)
            print(f'Accepted connection from {source}', file=log)

        self.protocol = IPIProtocol(self.clientsocket, txt=log)

    def close(self):
        if self._closed:
            return

        super().close()

        if self.log:
            print('Close socket server', file=self.log)
        self._closed = True

        # Proper way to close sockets?
        # And indeed i-pi connections...
        # if self.protocol is not None:
        #     self.protocol.end()  # Send end-of-communication string
        self.protocol = None
        if self.proc is not None:
            exitcode = self.proc.wait()
            if exitcode != 0:
                import warnings

                # Quantum Espresso seems to always exit with status 128,
                # even if successful.
                # Should investigate at some point
                warnings.warn('Subprocess exited with status {}'
                              .format(exitcode))
        # self.log('IPI server closed')

    def calculate(self, atoms):
        """Send geometry to client and return calculated things as dict.

        This will block until client has established connection, then
        wait for the client to finish the calculation."""
        assert not self._closed

        # If we have not established connection yet, we must block
        # until the client catches up:
        if self.protocol is None:
            self._accept()
        return self.protocol.calculate(atoms.positions, atoms.cell)


class SocketClient:
    def __init__(self, host='localhost', port=None,
                 unixsocket=None, timeout=None, log=None, comm=world):
        """Create client and connect to server.

        Parameters:

        host: string
            Hostname of server.  Defaults to localhost
        port: integer or None
            Port to which to connect.  By default 31415.
        unixsocket: string or None
            If specified, use corresponding UNIX socket.
            See documentation of unixsocket for SocketIOCalculator.
        timeout: float or None
            See documentation of timeout for SocketIOCalculator.
        log: file object or None
            Log events to this file
        comm: communicator or None
            MPI communicator object.  Defaults to ase.parallel.world.
            When ASE runs in parallel, only the process with world.rank == 0
            will communicate over the socket.  The received information
            will then be broadcast on the communicator.  The SocketClient
            must be created on all ranks of world, and will see the same
            Atoms objects."""
        # Only rank0 actually does the socket work.
        # The other ranks only need to follow.
        #
        # Note: We actually refrain from assigning all the
        # socket-related things except on master
        self.comm = comm

        if self.comm.rank == 0:
            if unixsocket is not None:
                sock = socket.socket(socket.AF_UNIX)
                actualsocket = actualunixsocketname(unixsocket)
                sock.connect(actualsocket)
            else:
                if port is None:
                    port = SocketServer.default_port
                sock = socket.socket(socket.AF_INET)
                sock.connect((host, port))
            sock.settimeout(timeout)
            self.host = host
            self.port = port
            self.unixsocket = unixsocket

            self.protocol = IPIProtocol(sock, txt=log)
            self.log = self.protocol.log
            self.closed = False

            self.bead_index = 0
            self.bead_initbytes = b''
            self.state = 'READY'

    def close(self):
        if not self.closed:
            self.log('Close SocketClient')
            self.closed = True
            self.protocol.socket.close()

    def calculate(self, atoms, use_stress):
        # We should also broadcast the bead index, once we support doing
        # multiple beads.
        self.comm.broadcast(atoms.positions, 0)
        self.comm.broadcast(np.ascontiguousarray(atoms.cell), 0)

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        if use_stress:
            stress = atoms.get_stress(voigt=False)
            virial = -atoms.get_volume() * stress
        else:
            virial = np.zeros((3, 3))
        return energy, forces, virial

    def irun(self, atoms, use_stress=None):
        if use_stress is None:
            use_stress = any(atoms.pbc)

        my_irun = self.irun_rank0 if self.comm.rank == 0 else self.irun_rankN
        return my_irun(atoms, use_stress)

    def irun_rankN(self, atoms, use_stress=True):
        stop_criterion = np.zeros(1, bool)
        while True:
            self.comm.broadcast(stop_criterion, 0)
            if stop_criterion[0]:
                return

            self.calculate(atoms, use_stress)
            yield

    def irun_rank0(self, atoms, use_stress=True):
        # For every step we either calculate or quit.  We need to
        # tell other MPI processes (if this is MPI-parallel) whether they
        # should calculate or quit.
        try:
            while True:
                try:
                    msg = self.protocol.recvmsg()
                except SocketClosed:
                    # Server closed the connection, but we want to
                    # exit gracefully anyway
                    msg = 'EXIT'

                if msg == 'EXIT':
                    # Send stop signal to clients:
                    self.comm.broadcast(np.ones(1, bool), 0)
                    # (When otherwise exiting, things crashed and we should
                    # let MPI_ABORT take care of the mess instead of trying
                    # to synchronize the exit)
                    return
                elif msg == 'STATUS':
                    self.protocol.sendmsg(self.state)
                elif msg == 'POSDATA':
                    assert self.state == 'READY'
                    cell, _icell, positions = self.protocol.recvposdata()
                    atoms.cell[:] = cell
                    atoms.positions[:] = positions

                    # User may wish to do something with the atoms object now.
                    # Should we provide option to yield here?
                    #
                    # (In that case we should MPI-synchronize *before*
                    #  whereas now we do it after.)

                    # Send signal for other ranks to proceed with calculation:
                    self.comm.broadcast(np.zeros(1, bool), 0)
                    energy, forces, virial = self.calculate(atoms, use_stress)

                    self.state = 'HAVEDATA'
                    yield
                elif msg == 'GETFORCE':
                    assert self.state == 'HAVEDATA', self.state
                    self.protocol.sendforce(energy, forces, virial)
                    self.state = 'NEEDINIT'
                elif msg == 'INIT':
                    assert self.state == 'NEEDINIT'
                    bead_index, initbytes = self.protocol.recvinit()
                    self.bead_index = bead_index
                    self.bead_initbytes = initbytes
                    self.state = 'READY'
                else:
                    raise KeyError('Bad message', msg)
        finally:
            self.close()

    def run(self, atoms, use_stress=False):
        for _ in self.irun(atoms, use_stress=use_stress):
            pass


class SocketIOCalculator(Calculator, IOContext):
    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']
    supported_changes = {'positions', 'cell'}

    def __init__(self, calc=None, port=None,
                 unixsocket=None, timeout=None, log=None, *,
                 launch_client=None, comm=world):
        """Initialize socket I/O calculator.

        This calculator launches a server which passes atomic
        coordinates and unit cells to an external code via a socket,
        and receives energy, forces, and stress in return.

        ASE integrates this with the Quantum Espresso, FHI-aims and
        Siesta calculators.  This works with any external code that
        supports running as a client over the i-PI protocol.

        Parameters:

        calc: calculator or None

            If calc is not None, a client process will be launched
            using calc.command, and the input file will be generated
            using ``calc.write_input()``.  Otherwise only the server will
            run, and it is up to the user to launch a compliant client
            process.

        port: integer

            port number for socket.  Should normally be between 1025
            and 65535.  Typical ports for are 31415 (default) or 3141.

        unixsocket: str or None

            if not None, ignore host and port, creating instead a
            unix socket using this name prefixed with ``/tmp/ipi_``.
            The socket is deleted when the calculator is closed.

        timeout: float >= 0 or None

            timeout for connection, by default infinite.  See
            documentation of Python sockets.  For longer jobs it is
            recommended to set a timeout in case of undetected
            client-side failure.

        log: file object or None (default)

            logfile for communication over socket.  For debugging or
            the curious.

        In order to correctly close the sockets, it is
        recommended to use this class within a with-block:

        >>> from ase.calculators.socketio import SocketIOCalculator

        >>> with SocketIOCalculator(...) as calc:  # doctest:+SKIP
        ...    atoms.calc = calc
        ...    atoms.get_forces()
        ...    atoms.rattle()
        ...    atoms.get_forces()

        It is also possible to call calc.close() after
        use.  This is best done in a finally-block."""

        Calculator.__init__(self)

        if calc is not None:
            if launch_client is not None:
                raise ValueError('Cannot pass both calc and launch_client')
            launch_client = FileIOSocketClientLauncher(calc)
        self.launch_client = launch_client
        self.timeout = timeout
        self.server = None

        self.log = self.openfile(file=log, comm=comm)

        # We only hold these so we can pass them on to the server.
        # They may both be None as stored here.
        self._port = port
        self._unixsocket = unixsocket

        # If there is a calculator, we will launch in calculate() because
        # we are responsible for executing the external process, too, and
        # should do so before blocking.  Without a calculator we want to
        # block immediately:
        if self.launch_client is None:
            self.server = self.launch_server()

    def todict(self):
        d = {'type': 'calculator',
             'name': 'socket-driver'}
        # if self.calc is not None:
        #    d['calc'] = self.calc.todict()
        return d

    def launch_server(self):
        return self.closelater(SocketServer(
            # launch_client=launch_client,
            port=self._port,
            unixsocket=self._unixsocket,
            timeout=self.timeout, log=self.log,
        ))

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        bad = [change for change in system_changes
               if change not in self.supported_changes]

        # First time calculate() is called, system_changes will be
        # all_changes.  After that, only positions and cell may change.
        if self.atoms is not None and any(bad):
            raise PropertyNotImplementedError(
                'Cannot change {} through IPI protocol.  '
                'Please create new socket calculator.'
                .format(bad if len(bad) > 1 else bad[0]))

        self.atoms = atoms.copy()

        if self.server is None:
            self.server = self.launch_server()
            proc = self.launch_client(atoms, properties,
                                      port=self._port,
                                      unixsocket=self._unixsocket)
            self.server.proc = proc  # XXX nasty hack

        results = self.server.calculate(atoms)
        results['free_energy'] = results['energy']
        virial = results.pop('virial')
        if self.atoms.cell.rank == 3 and any(self.atoms.pbc):
            vol = atoms.get_volume()
            results['stress'] = -full_3x3_to_voigt_6_stress(virial) / vol
        self.results.update(results)

    def close(self):
        self.server = None
        super().close()


class PySocketIOClient:
    def __init__(self, calculator_factory):
        self._calculator_factory = calculator_factory

    def __call__(self, atoms, properties=None, port=None, unixsocket=None):
        import pickle
        import sys

        # We pickle everything first, so we won't need to bother with the
        # process as long as it succeeds.
        transferbytes = pickle.dumps([
            dict(unixsocket=unixsocket, port=port),
            atoms.copy(),
            self._calculator_factory,
        ])

        proc = Popen([sys.executable, '-m', 'ase.calculators.socketio'],
                     stdin=PIPE)

        proc.stdin.write(transferbytes)
        proc.stdin.close()
        return proc

    @staticmethod
    def main():
        import pickle
        import sys

        socketinfo, atoms, get_calculator = pickle.load(sys.stdin.buffer)
        atoms.calc = get_calculator()
        client = SocketClient(host='localhost',
                              unixsocket=socketinfo.get('unixsocket'),
                              port=socketinfo.get('port'))
        # XXX In principle we could avoid calculating stress until
        # someone requests the stress, could we not?
        # Which would make use_stress boolean unnecessary.
        client.run(atoms, use_stress=True)


if __name__ == '__main__':
    PySocketIOClient.main()

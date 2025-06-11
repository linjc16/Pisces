import pytest

from ase.utils import Lock


def test_cannot_acquire_lock_twice(tmp_path):
    """Test timeout on Lock.acquire()."""

    lock = Lock(tmp_path / 'lockfile', timeout=0.3)
    with lock:
        with pytest.raises(TimeoutError):
            with lock:
                ...


def test_lock_close_file_descriptor(tmp_path):
    """Test that lock file descriptor is properly closed."""
    # The choice of timeout=1.0 is arbitrary but we don't want to use
    # something that is too large since it could mean that the test
    # takes long to fail.
    lock = Lock(tmp_path / 'lockfile', timeout=1.0)
    with lock:
        assert not lock.fd.closed

    assert lock.fd.closed

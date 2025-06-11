from io import BytesIO

from ase.io import iread, write


def to_bytes(images, format=None, **kwargs):
    """Convert atoms or multiple atoms objects to bytes."""
    buf = _to_buffer(images, format=format, **kwargs)
    btxt = buf.getvalue()
    return btxt


def _to_buffer(images, format=None, **kwargs):
    buf = BytesIO()
    write(buf, images, format=format, **kwargs)
    buf.seek(0)
    return buf


def parse_images(data, format=None, **kwargs):
    """Parse string or bytes into list of atoms objects."""
    buf = BytesIO(data)
    images = list(iread(buf, format=format, **kwargs))
    return images


def parse_atoms(data, format=None, **kwargs):
    images = parse_images(data, format=format, index=-1, **kwargs)
    assert len(images) == 1
    return images[0]

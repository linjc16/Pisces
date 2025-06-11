"""
Utilities for plugins to ase
"""

from typing import List, NamedTuple, Optional, Union


# Name is defined in the entry point
class ExternalIOFormat(NamedTuple):
    desc: str
    code: str
    module: Optional[str] = None
    glob: Optional[Union[str, List[str]]] = None
    ext: Optional[Union[str, List[str]]] = None
    magic: Optional[Union[bytes, List[bytes]]] = None
    magic_regex: Optional[bytes] = None


class ExternalViewer(NamedTuple):
    desc: str
    module: Optional[str] = None
    cli: Optional[bool] = False
    fmt: Optional[str] = None
    argv: Optional[List[str]] = None

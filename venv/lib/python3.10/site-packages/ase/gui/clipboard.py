from ase import Atoms
from ase.io.jsonio import decode, encode


class AtomsClipboard:
    def __init__(self, tk):
        self.tk = tk

    def get_text(self) -> str:
        return self.tk.clipboard_get()

    def set_text(self, text: str) -> None:
        self.tk.clipboard_clear()
        self.tk.clipboard_append(text)

    def get_atoms(self) -> Atoms:
        text = self.get_text()
        atoms = decode(text)
        if not isinstance(atoms, Atoms):
            typename = type(atoms).__name__
            raise ValueError(f'Cannot convert {typename} to Atoms')
        return atoms

    def set_atoms(self, atoms: Atoms) -> None:
        json_text = encode(atoms)
        self.set_text(json_text)

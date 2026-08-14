"""Readers for reaction-path files produced by common quantum chemistry codes.

Two formats show up:

* **Geometry blocks** -- a sequence of ``element x y z`` blocks separated by
  lines beginning with ``--``, one block per IRC step.  These are the
  ``*_clean_awk`` files.
* **IRC tables** -- two tab-separated columns, reaction coordinate and energy,
  written by GAMESS.  The header line reads ``X Pos<TAB>Energy``: ``"X Pos"``
  is a *single* label containing a space, so a whitespace-delimited reader sees
  three names for two columns and mislabels everything -- the coordinate arrives
  under ``"X"`` and the energy under ``"Pos"``.  The reader here ignores the
  header and returns the two columns under honest names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["read_geometry_blocks", "read_irc_table", "IRCTable"]


def read_geometry_blocks(path, unit: str = "A", separator: str = "--") -> list:
    """Parse a multi-geometry IRC file into a list of PySCF atom specifications.

    Each geometry is built into a throwaway :class:`~pyscf.gto.Mole` so that
    what comes back is ``mol._atom``: PySCF's internal representation, with
    coordinates in **Bohr**.  Downstream builders therefore use ``unit="B"``.

    Parameters
    ----------
    path:
        File containing the geometry blocks.
    unit:
        Length unit of the coordinates *in the file* (``"A"`` for Angstrom).
    separator:
        A line starting with this string ends a geometry block.

    Returns
    -------
    list
        One ``mol._atom`` entry per geometry, in file order.
    """
    from pyscf import gto

    text = Path(path).read_text()

    blocks, current = [], []
    for line in text.splitlines():
        if line.lstrip().startswith(separator):
            blocks.append("\n".join(current))
            current = []
        else:
            current.append(line)
    if any(line.strip() for line in current):
        blocks.append("\n".join(current))

    geometries = []
    for block in blocks:
        if not block.strip():
            continue
        mol = gto.Mole()
        mol.atom = block
        mol.unit = unit
        mol.build(verbose=0)
        geometries.append(mol._atom)
    return geometries


@dataclass(frozen=True)
class IRCTable:
    """Reaction coordinate and energy along an IRC, as written by GAMESS."""

    coordinate: np.ndarray
    energy: np.ndarray

    def __len__(self) -> int:
        return self.coordinate.size

    def reversed(self) -> IRCTable:
        """Traverse the path in the opposite direction, negating the coordinate.

        Half of an IRC is computed from the transition state outwards, so the
        two branches have to be stitched together with one of them flipped.
        """
        return IRCTable(coordinate=-self.coordinate[::-1], energy=self.energy[::-1])


def read_irc_table(path) -> IRCTable:
    """Read a two-column GAMESS IRC table, skipping its malformed header."""
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"{path}: expected at least 2 columns, got {data.shape[1]}")
    return IRCTable(coordinate=data[:, 0].copy(), energy=data[:, 1].copy())

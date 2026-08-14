"""Splitting the Green's function into orbital-symmetry blocks.

An orbital-symmetry-controlled reaction is exactly one where the active
orbitals fall into symmetry blocks that do not mix, so ``G`` is block diagonal
and ``det G`` factorises.  The winding number is then defined per block -- the
paper's ``G+`` and ``G-``.

Two ways to find the blocks:

``irrep_blocks``
    Group orbitals by their point-group irreducible representation.  This is
    the meaningful definition and is what should normally be used.

``parity_blocks``
    Take even- and odd-numbered active orbitals (``G[::2, ::2]`` and
    ``G[1::2, 1::2]``).  A fallback for when point-group symmetry is
    unavailable -- a geometry that is only approximately symmetric is assigned
    to C1, and then no symmetry adaptation happens at all.  It is correct only
    when the orbitals happen to come out alternating between two irreps; it
    carries no symmetry information of its own.

Whichever route is taken, :func:`block_leakage` says how good the decomposition
actually is.  Worth measuring rather than assuming, especially when the blocking
came from index parity: orbitals that keep their symmetry character on a
slightly distorted geometry still leave a small residual coupling, and how small
is an empirical question.
"""

from __future__ import annotations

import numpy as np

__all__ = ["parity_blocks", "irrep_blocks", "block_indices", "block_leakage"]


def parity_blocks(ncas: int) -> dict[str, np.ndarray]:
    """``{"even": [0, 2, 4, ...], "odd": [1, 3, 5, ...]}``."""
    return {"even": np.arange(0, ncas, 2), "odd": np.arange(1, ncas, 2)}


def irrep_blocks(orbsym, mol=None) -> dict[str, np.ndarray]:
    """Group active-orbital indices by irreducible representation.

    Parameters
    ----------
    orbsym:
        Symmetry ids of the active orbitals, as produced by
        :func:`casgf.active_space.active_orbsym`.
    mol:
        If given, its point group is used to turn the numeric ids into names
        (``"Ag"``, ``"B1u"``, ...).  Otherwise the ids are used as keys.
    """
    orbsym = np.asarray(orbsym)
    names: dict[int, str] = {}
    if mol is not None and getattr(mol, "groupname", None):
        try:
            from pyscf import symm

            names = {
                int(i): symm.irrep_id2name(mol.groupname, int(i)) for i in np.unique(orbsym)
            }
        except Exception:
            names = {}

    blocks: dict[str, np.ndarray] = {}
    for sym in np.unique(orbsym):
        key = names.get(int(sym), str(int(sym)))
        blocks[key] = np.flatnonzero(orbsym == sym)
    return blocks


def block_indices(block, ncas: int, orbsym=None, mol=None) -> np.ndarray | None:
    """Resolve a block specification into orbital indices.

    ``block`` may be

    * ``None`` or ``"all"`` -- the whole active space (returns ``None``, which
      every consumer reads as "no selection");
    * ``"even"`` / ``"odd"`` -- index parity;
    * an irrep name or id -- requires ``orbsym``;
    * an explicit sequence of orbital indices.
    """
    if block is None or (isinstance(block, str) and block == "all"):
        return None
    if isinstance(block, str):
        if block in ("even", "odd"):
            return parity_blocks(ncas)[block]
        if orbsym is None:
            raise ValueError(
                f"block={block!r} looks like an irrep, but no orbsym is available. "
                "Run the CASSCF with point group symmetry enabled, or use "
                "'even'/'odd' to fall back to index-parity blocking."
            )
        blocks = irrep_blocks(orbsym, mol)
        if block not in blocks:
            raise KeyError(f"no irrep {block!r} among the active orbitals; have {list(blocks)}")
        return blocks[block]

    idx = np.asarray(block, dtype=int)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("an explicit block must be a non-empty 1-D sequence of orbital indices")
    if idx.min() < 0 or idx.max() >= ncas:
        raise ValueError(f"block indices {idx} out of range for ncas={ncas}")
    return idx


def block_leakage(g, blocks) -> float:
    """How far a matrix is from being block diagonal, as a dimensionless number.

    Returns the largest ``|g[i, j]| / sqrt(|g[i, i]| * |g[j, j]|)`` over pairs
    ``i``, ``j`` in *different* blocks.  Zero means ``g`` is exactly block
    diagonal and ``det g`` factorises exactly; a small value means the block
    winding numbers are still well defined in practice.

    Parameters
    ----------
    g:
        A single Green's-function matrix, ``(ncas, ncas)``.  Evaluate it
        somewhere the blocks matter -- inside the gap, say.
    blocks:
        Index arrays, for example ``parity_blocks(ncas).values()``.

    Examples
    --------
    ``block_leakage(gf.at(0.0, eta=0.05), parity_blocks(gf.norb).values())``
    """
    g = np.asarray(g)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {g.shape}")

    blocks = [np.asarray(b, dtype=int) for b in blocks]
    scale = np.sqrt(np.abs(np.diag(g)))
    normalised = np.abs(g) / np.outer(scale, scale)

    worst = 0.0
    for a, block_a in enumerate(blocks):
        for block_b in blocks[a + 1 :]:
            if block_a.size and block_b.size:
                worst = max(worst, float(normalised[np.ix_(block_a, block_b)].max()))
    return worst

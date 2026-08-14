"""Walking a CASSCF active space along a reaction path.

The practical difficulty in a reaction-path calculation is not any single
CASSCF run -- it is keeping *the same* active space from one geometry to the
next.  The recipe: seed the first geometry from its MP2 natural orbitals, then
project each converged set of orbitals onto the next geometry with
``mcscf.project_init_guess``.  That loop is easy to get subtly wrong and tedious
to rewrite, so it lives here once.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .active_space import ActiveSpace, build_mol, run_casscf

__all__ = ["ScanStep", "irc_scan", "collect_integrals"]


@dataclass
class ScanStep:
    """One geometry of a scan."""

    index: int
    e_states: np.ndarray
    mo_coeff: np.ndarray
    active_space: ActiveSpace | None = None

    @property
    def e_tot(self) -> float:
        """Energy of the lowest state (the only one, unless state-averaged)."""
        return float(self.e_states[0])


def irc_scan(
    geometries: Sequence,
    ncas: int,
    nelecas,
    basis: str = "def2-SVP",
    unit: str = "B",
    stride: int = 1,
    mo_guess=None,
    state_average=None,
    symmetry: bool = False,
    verbose: int = 3,
    with_active_space: bool = True,
    fix_spin: bool = True,
) -> Iterator[ScanStep]:
    """Run CASSCF along a list of geometries, carrying the orbitals forward.

    Parameters
    ----------
    geometries:
        Atom specifications, for example from
        :func:`casgf.irc.read_geometry_blocks`.
    ncas, nelecas:
        Active space size.
    basis, unit, symmetry, verbose:
        Passed to :func:`casgf.active_space.build_mol`.  ``unit`` defaults to
        Bohr because that is what ``read_geometry_blocks`` returns.
    stride:
        Take every ``stride``-th geometry.  Dense IRCs are usually oversampled
        for this purpose.
    mo_guess:
        Orbitals for the first geometry visited.  ``None`` seeds from MP2
        natural orbitals.  Pass the final ``mo_coeff`` of another scan to
        continue a path across its transition state.
    state_average:
        Weights for a state-averaged CASSCF, for example ``[0.99, 0.01]`` to get
        the ground and first excited state.
    with_active_space:
        Also extract ``h1``/``eri`` at each geometry.  Turn off when only
        energies are wanted -- it saves building the two-electron integrals.

    Yields
    ------
    ScanStep
        One per visited geometry, in path order.  This is a generator so that a
        long scan can be written out incrementally rather than held in memory.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1")

    mo = mo_guess
    for index in range(0, len(geometries), stride):
        mol = build_mol(
            geometries[index], basis=basis, unit=unit, symmetry=symmetry, verbose=verbose
        )
        mc = run_casscf(
            mol,
            ncas,
            nelecas,
            mo_guess=mo,
            state_average=state_average,
            fix_spin=fix_spin,
        )
        mo = mc.mo_coeff
        yield ScanStep(
            index=index,
            e_states=np.atleast_1d(np.asarray(mc.e_states if state_average else mc.e_tot)),
            mo_coeff=mo,
            active_space=ActiveSpace.from_casscf(mc) if with_active_space else None,
        )


def collect_integrals(steps: Iterator[ScanStep]) -> dict[str, np.ndarray]:
    """Stack a scan's results into contiguous arrays.

    Returns a dict with ``h1`` ``(n_geom, ncas, ncas)``, ``eri``
    ``(n_geom, ncas, ncas, ncas, ncas)``, ``energies`` ``(n_geom, n_states)``
    and ``index`` ``(n_geom,)``.
    """
    h1, eri, energies, indices = [], [], [], []
    for step in steps:
        if step.active_space is None:
            raise ValueError("scan was run with with_active_space=False; no integrals to collect")
        h1.append(step.active_space.h1)
        eri.append(step.active_space.eri)
        energies.append(step.e_states)
        indices.append(step.index)
    return {
        "h1": np.asarray(h1),
        "eri": np.asarray(eri),
        "energies": np.asarray(energies),
        "index": np.asarray(indices),
    }

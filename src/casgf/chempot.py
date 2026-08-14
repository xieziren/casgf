"""Chemical potential conventions.

The Green's function is referenced to a chemical potential ``mu`` that puts
``w = 0`` inside the fundamental gap.  Two definitions are offered, and they
answer slightly different questions:

``mu_particle_hole``
    ``mu = (E^{N-1} - E^{N+1}) / 2`` from the ``N±1`` ground states *of the same
    active-space Hamiltonian*.  Nothing is re-optimised, so this is essentially
    free once the sectors are diagonalised, and it centres the gap exactly.
    This is the default.

``mu_relaxed_casscf``
    The same formula, but with ``E^{N±1}`` from separate CASSCF calculations on
    the cation and the anion, so the orbitals relax.  More expensive and no
    longer exactly gap-centring, but it is the physical ionisation
    potential/electron affinity midpoint.
"""

from __future__ import annotations

import numpy as np

from .active_space import build_mol, normalise_nelec, run_casscf

__all__ = ["mu_particle_hole", "mu_relaxed_casscf"]


def mu_particle_hole(e_gs_minus: float, e_gs_plus: float) -> float:
    """``mu = (E_gs^{N-1} - E_gs^{N+1}) / 2``.

    With this choice the lowest addition pole sits at ``+gap/2`` and the highest
    removal pole at ``-gap/2``.
    """
    return 0.5 * (float(e_gs_minus) - float(e_gs_plus))


def mu_relaxed_casscf(
    atom,
    ncas: int,
    nelecas,
    basis: str = "def2-SVP",
    unit: str = "B",
    verbose: int = 3,
    fix_spin: bool = False,
) -> float:
    """Chemical potential from separate cation and anion CASSCF calculations.

    ``mu = (E_CASSCF^{N-1} - E_CASSCF^{N+1}) / 2``.  The neutral energy appears
    in both the ionisation potential and the electron affinity and cancels
    exactly, so it is never needed.

    Both ions are doublets and are treated with the same ``ncas`` active space,
    each seeded from its own MP2 natural orbitals.

    Parameters
    ----------
    atom:
        Anything PySCF accepts as ``Mole.atom`` -- an xyz path, a coordinate
        block, or the ``mol._atom`` list returned by :mod:`casgf.irc`.
    ncas, nelecas:
        Active space of the *neutral* system; the ions use ``nelec ± 1``.
    unit:
        Length unit of ``atom``.  Defaults to Bohr, matching ``mol._atom``.
    """
    na, nb = normalise_nelec(nelecas, ncas)
    nelec = na + nb

    energies = {}
    for label, charge, nelec_ion in (("cation", 1, nelec - 1), ("anion", -1, nelec + 1)):
        mol = build_mol(atom, basis=basis, charge=charge, spin=1, unit=unit, verbose=verbose)
        mc = run_casscf(mol, ncas, nelec_ion, natorb=True, fix_spin=fix_spin)
        energies[label] = float(np.ravel(mc.e_tot)[0])

    return 0.5 * (energies["cation"] - energies["anion"])

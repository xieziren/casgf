"""The CASSCF active space that everything else is built on.

An :class:`ActiveSpace` is the small, self-contained object the rest of the
package needs: the effective one-electron integrals ``h1``, the two-electron
integrals ``eri`` in chemist notation, and how many electrons of each spin
occupy it.  It can come either from a live PySCF CASSCF calculation or straight
from stored arrays -- so a set of integrals computed once can be re-used
without re-running any quantum chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pyscf import ao2mo, gto, mcscf, mp, scf
from pyscf.mcscf.casci import h1e_for_cas

__all__ = [
    "ActiveSpace",
    "active_orbsym",
    "build_mol",
    "mp2_natural_orbitals",
    "normalise_nelec",
    "run_casscf",
]


def normalise_nelec(nelecas, ncas: int | None = None) -> tuple[int, int]:
    """Accept ``6`` or ``(3, 3)`` and always return ``(n_alpha, n_beta)``.

    A bare integer is split as evenly as possible, with the extra electron
    going to alpha -- the same convention PySCF uses.
    """
    if isinstance(nelecas, (int, np.integer)):
        total = int(nelecas)
        na = (total + 1) // 2
        nb = total // 2
    else:
        na, nb = (int(nelecas[0]), int(nelecas[1]))
    if ncas is not None and not (0 <= na <= ncas and 0 <= nb <= ncas):
        raise ValueError(f"nelecas={nelecas} does not fit in ncas={ncas}")
    return na, nb


@dataclass
class ActiveSpace:
    """Active-space Hamiltonian parameters.

    Attributes
    ----------
    h1:
        Effective one-electron integrals, shape ``(ncas, ncas)``.  These already
        include the mean field of the frozen core (PySCF's ``h1e_for_cas``).
    eri:
        Two-electron integrals in chemist notation ``(ij|kl)``, full 4-index
        form, shape ``(ncas,) * 4``.
    nelecas:
        ``(n_alpha, n_beta)`` in the active space.
    e_core:
        Frozen-core + nuclear repulsion energy.  It is a constant shared by
        every particle-number sector, so it cancels from every excitation
        energy the Green's function is built from; it is kept only so that
        total energies can be reported.
    orbsym:
        Symmetry labels of the active orbitals, if the calculation had point
        group symmetry switched on.  Used to split the Green's function into
        orbital-symmetry blocks.
    """

    h1: np.ndarray
    eri: np.ndarray
    nelecas: tuple[int, int]
    e_core: float = 0.0
    orbsym: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.h1 = np.asarray(self.h1, dtype=float)
        if self.h1.ndim != 2 or self.h1.shape[0] != self.h1.shape[1]:
            raise ValueError(f"h1 must be square, got shape {self.h1.shape}")
        ncas = self.h1.shape[0]
        self.eri = ao2mo.restore(1, np.asarray(self.eri, dtype=float), ncas)
        self.nelecas = normalise_nelec(self.nelecas, ncas)
        if self.orbsym is not None:
            self.orbsym = np.asarray(self.orbsym)
            if self.orbsym.size != ncas:
                raise ValueError(f"orbsym has {self.orbsym.size} entries, expected {ncas}")

    @property
    def ncas(self) -> int:
        return self.h1.shape[0]

    @property
    def nelec_total(self) -> int:
        return sum(self.nelecas)

    @classmethod
    def from_arrays(cls, h1, eri, nelecas, e_core: float = 0.0, orbsym=None, **meta):
        """Build directly from stored integrals, for example ``.npy`` files."""
        return cls(h1=h1, eri=eri, nelecas=nelecas, e_core=e_core, orbsym=orbsym, meta=meta)

    @classmethod
    def from_casscf(cls, mc):
        """Extract the active space from a converged PySCF CASSCF/CASCI object."""
        h1, e_core = h1e_for_cas(mc, mo_coeff=None, ncas=None, ncore=None)
        eri = ao2mo.restore(1, mc.get_h2eff(), mc.ncas)
        return cls(
            h1=h1,
            eri=eri,
            nelecas=mc.nelecas,
            e_core=float(e_core),
            orbsym=active_orbsym(mc),
            meta={"e_tot": float(np.ravel(mc.e_tot)[0])},
        )

    @classmethod
    def from_molecule(cls, mol, ncas: int, nelecas, mo_guess=None, **kwargs):
        """Run RHF -> MP2 natural orbitals -> CASSCF and take the active space.

        ``kwargs`` are passed to :func:`run_casscf`.
        """
        mc = run_casscf(mol, ncas, nelecas, mo_guess=mo_guess, **kwargs)
        return cls.from_casscf(mc)


def mp2_natural_orbitals(mf) -> np.ndarray:
    """MP2 natural orbitals, used to seed the active space at the first geometry."""
    mymp = mp.MP2(mf)
    mymp.kernel()
    _, natorbs = mcscf.addons.make_natural_orbitals(mymp)
    return natorbs


def run_casscf(
    mol,
    ncas: int,
    nelecas,
    mo_guess=None,
    natorb: bool = True,
    fix_spin: bool = True,
    state_average=None,
    conv_tol: float | None = None,
):
    """Run a CASSCF calculation with a natural-orbital-seeded active space.

    Parameters
    ----------
    mol:
        A built :class:`pyscf.gto.Mole`.
    ncas, nelecas:
        Active space size.
    mo_guess:
        Starting orbitals.  ``None`` seeds the active space from this
        geometry's MP2 natural orbitals; anything else is projected onto the
        current geometry with ``mcscf.project_init_guess`` -- which is what
        keeps the active space continuous along a reaction path.
    natorb:
        Rotate the converged active orbitals to natural orbitals.  Keep this on:
        the Green's function is expressed in the active-orbital basis, so the
        basis has to be defined reproducibly.
    fix_spin:
        Constrain the CI solver to a singlet.
    state_average:
        Weights for a state-averaged calculation, for example ``[0.99, 0.01]``.
        ``None`` runs a plain state-specific CASSCF.

    Returns
    -------
    The converged CASSCF object.
    """
    mf = scf.RHF(mol)
    mf.kernel()

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    if state_average is not None:
        mc = mc.state_average_(state_average)
    if fix_spin:
        mc.fcisolver.spin = 0
        mc.fix_spin_(ss=0)
    mc.natorb = natorb
    if conv_tol is not None:
        mc.conv_tol = conv_tol

    mo = mp2_natural_orbitals(mf) if mo_guess is None else mcscf.project_init_guess(mc, mo_guess)
    mc.kernel(mo)
    return mc


def active_orbsym(mc) -> np.ndarray | None:
    """Symmetry labels of the active orbitals, or ``None`` if unavailable.

    PySCF tags ``mo_coeff`` with ``orbsym`` when symmetry is on, but the tag is
    lost through some transformations (natural-orbital rotation among them), so
    fall back to relabelling the orbitals explicitly.
    """
    ncore, ncas = mc.ncore, mc.ncas
    active = slice(ncore, ncore + ncas)

    orbsym = getattr(mc.mo_coeff, "orbsym", None)
    if orbsym is not None:
        return np.asarray(orbsym)[active]

    mol = mc.mol
    if not getattr(mol, "symmetry", False):
        return None
    try:
        from pyscf import symm

        labels = symm.label_orb_symm(
            mol, mol.irrep_id, mol.symm_orb, np.asarray(mc.mo_coeff)[:, active]
        )
        return np.asarray(labels)
    except Exception:
        # Symmetry labelling fails when the converged orbitals are not symmetry
        # adapted to within the default tolerance. That is a legitimate outcome,
        # not an error: callers fall back to index-parity blocking.
        return None


def build_mol(atom, basis: str = "def2-SVP", charge: int = 0, spin: int = 0,
              unit: str = "B", symmetry=False, verbose: int = 3) -> gto.Mole:
    """Build a :class:`pyscf.gto.Mole` from an atom specification.

    ``unit="B"`` is the default because geometries parsed by :mod:`casgf.irc`
    come back as ``mol._atom``, which PySCF stores in Bohr.
    """
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.unit = unit
    mol.symmetry = symmetry
    mol.verbose = verbose
    mol.build()
    return mol

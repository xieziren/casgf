"""Exact diagonalisation of an active-space Hamiltonian.

The active-space Hamiltonian

    H = sum_{ij,s} h1[i,j] c^dag_{s,i} c_{s,j}
      + 1/2 sum_{ijkl,ss'} eri[i,j,k,l] c^dag_{s,i} c^dag_{s',k} c_{s',l} c_{s,j}

(``eri`` in chemist notation ``(ij|kl)``) is built and diagonalised completely
within a fixed particle-number/spin sector using PySCF's FCI machinery.

Active spaces used in the accompanying paper are small -- CAS(4,4), CAS(6,6),
CAS(8,8) -- so full dense diagonalisation is affordable and gives *every*
eigenpair, which is exactly what the Lehmann representation of the Green's
function needs.  CAS(8,8) is 4900 determinants and takes a few seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import ao2mo
from pyscf.fci import addons, cistring, direct_spin1

__all__ = ["Sector", "solve_sector", "n_determinants", "transition_amplitudes"]

#: Refuse to build a dense Hamiltonian larger than this many determinants.
#: 20000 determinants is a 3.2 GB matrix and an hour of ``eigh`` -- past the
#: point where dense diagonalisation is the right tool.
DEFAULT_MAX_DETS = 20_000


@dataclass(frozen=True)
class Sector:
    """Every eigenpair of the active-space Hamiltonian at fixed ``(na, nb)``.

    Attributes
    ----------
    nelec:
        ``(n_alpha, n_beta)`` of this sector.
    energies:
        Eigenvalues in ascending order, shape ``(ndet,)``.  These do *not*
        include the frozen-core energy, which is a constant common to every
        sector and therefore cancels in all the energy differences used below.
    vectors:
        Eigenvectors as columns, shape ``(ndet, ndet)``, in the determinant
        ordering given by ``addr``.
    addr:
        Determinant addresses relating the eigenvector rows to the flattened
        ``(n_alpha_strings, n_beta_strings)`` CI-vector layout PySCF uses:
        ``civec.ravel()[addr]`` is in the same order as ``vectors``.
    shape:
        ``(n_alpha_strings, n_beta_strings)`` of the CI vector.
    """

    nelec: tuple[int, int]
    energies: np.ndarray
    vectors: np.ndarray
    addr: np.ndarray
    shape: tuple[int, int]

    @property
    def e_gs(self) -> float:
        """Ground-state energy of this sector."""
        return float(self.energies[0])

    def civec(self, state: int = 0) -> np.ndarray:
        """Eigenvector ``state`` as a CI vector in PySCF's ``(na, nb)`` layout."""
        flat = np.zeros(self.shape[0] * self.shape[1])
        flat[self.addr] = self.vectors[:, state]
        return flat.reshape(self.shape)


def n_determinants(norb: int, nelec: tuple[int, int]) -> int:
    """Size of the FCI determinant space for ``nelec`` electrons in ``norb`` orbitals."""
    na, nb = nelec
    return cistring.num_strings(norb, na) * cistring.num_strings(norb, nb)


def solve_sector(
    h1: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    max_dets: int = DEFAULT_MAX_DETS,
) -> Sector:
    """Diagonalise the active-space Hamiltonian completely in one ``nelec`` sector.

    Parameters
    ----------
    h1:
        Effective one-electron integrals in the active space, shape ``(norb, norb)``.
    eri:
        Two-electron integrals in chemist notation.  Any of PySCF's packing
        conventions is accepted; it is restored to the full 4-index form.
    norb:
        Number of active orbitals.
    nelec:
        ``(n_alpha, n_beta)``.
    max_dets:
        Guard against accidentally requesting an unaffordable dense problem.

    Raises
    ------
    ValueError
        If ``nelec`` is unphysical for ``norb``, or the determinant space
        exceeds ``max_dets``.
    """
    na, nb = (int(nelec[0]), int(nelec[1]))
    if not (0 <= na <= norb and 0 <= nb <= norb):
        raise ValueError(f"nelec={nelec} is out of range for norb={norb}")

    ndet = n_determinants(norb, (na, nb))
    if ndet > max_dets:
        raise ValueError(
            f"sector {(na, nb)} of norb={norb} has {ndet} determinants, above "
            f"max_dets={max_dets}. Dense diagonalisation would need "
            f"{ndet**2 * 8 / 1e9:.1f} GB. Raise max_dets only if you mean it."
        )

    eri_full = ao2mo.restore(1, np.asarray(eri), norb)
    addr, ham = direct_spin1.pspace(np.asarray(h1), eri_full, norb, (na, nb), np=ndet)
    if len(addr) != ndet:
        # pspace returned a subspace; that would silently truncate the spectrum.
        raise RuntimeError(
            f"expected the full determinant space ({ndet}), got {len(addr)} from pspace"
        )
    energies, vectors = np.linalg.eigh(ham)
    return Sector(
        nelec=(na, nb),
        energies=energies,
        vectors=vectors,
        addr=addr,
        shape=(cistring.num_strings(norb, na), cistring.num_strings(norb, nb)),
    )


def transition_amplitudes(
    civec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    target: Sector,
    operator: str,
    spin: str = "alpha",
) -> np.ndarray:
    """Overlaps ``<n| O_i |civec>`` for every eigenstate ``n`` of ``target``.

    ``operator`` is ``"create"`` (``O_i = c^dag_i``, so ``target`` must be the
    N+1 sector) or ``"destroy"`` (``O_i = c_i``, N-1 sector).

    Returns an array of shape ``(norb, n_states)``; row ``i`` holds the overlaps
    of ``O_i |civec>`` with all eigenstates of ``target``.
    """
    if spin not in ("alpha", "beta"):
        raise ValueError(f"spin must be 'alpha' or 'beta', got {spin!r}")
    if operator == "create":
        apply = addons.cre_a if spin == "alpha" else addons.cre_b
    elif operator == "destroy":
        apply = addons.des_a if spin == "alpha" else addons.des_b
    else:
        raise ValueError(f"operator must be 'create' or 'destroy', got {operator!r}")

    amps = np.empty((norb, target.vectors.shape[1]))
    for i in range(norb):
        psi = apply(civec, norb, tuple(nelec), i).ravel()
        amps[i] = psi[target.addr] @ target.vectors
    return amps

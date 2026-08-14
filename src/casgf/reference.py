"""A deliberately naive Green's function, for cross-checking the fast one.

:func:`casgf.lehmann` diagonalises the ``N+-1`` sectors completely and assembles
``G`` from poles and residues.  That is efficient -- the spectrum is computed once
and every later frequency is nearly free -- but it puts a lot of bookkeeping
between the Hamiltonian and the answer: which eigenvector goes with which
transition amplitude, which branch gets which sign of the chemical-potential
shift, and so on.

This module computes the same object by a route that shares none of that
bookkeeping.  It never forms a pole or a residue.  For each frequency it solves

    (z - mu + E_0 - H_{N+1}) x_j = c^dag_j |0>          (particle part)
    (z - mu - E_0 + H_{N-1}) y_i = c_i |0>              (hole part)

Note the ``-mu`` on both lines: it is the same rigid shift of the frequency
axis in each branch, which is the resolvent-side statement of the ``+mu`` that
appears on every pole in the Lehmann form.

and contracts the solutions back, so the resolvent is inverted directly.  The
reference state comes from PySCF's Davidson solver rather than from the dense
diagonalisation used elsewhere.

It costs one dense linear solve per frequency, which makes it far too slow for
spectra or contours -- its job is to be an independent second opinion.  The two
implementations agree to ~1e-12 in ``tests/test_reference.py``; if they ever
stop agreeing, one of them has a bug.
"""

from __future__ import annotations

import numpy as np
from pyscf import ao2mo
from pyscf.fci import addons, direct_spin1

from .ed import DEFAULT_MAX_DETS, n_determinants

__all__ = ["green_function_by_resolvent"]


def _dense_hamiltonian(h1, eri, norb, nelec, max_dets):
    """Full FCI Hamiltonian of one sector, plus the determinant addressing."""
    ndet = n_determinants(norb, nelec)
    if ndet > max_dets:
        raise ValueError(
            f"sector {nelec} has {ndet} determinants, above max_dets={max_dets}"
        )
    addr, ham = direct_spin1.pspace(h1, eri, norb, nelec, np=ndet)
    return addr, ham


def green_function_by_resolvent(
    active_space,
    z,
    spin: str = "alpha",
    mu: float | None = None,
    max_dets: int = DEFAULT_MAX_DETS,
) -> np.ndarray:
    """``G(z)`` by direct resolvent inversion, shape ``(n_z, ncas, ncas)``.

    Parameters mirror :func:`casgf.lehmann`, except that ``z`` is an array of
    complex frequencies rather than a real grid plus a broadening.

    Slow by construction: one dense complex solve per frequency, in each of the
    two ``N+-1`` determinant spaces.
    """
    norb = active_space.ncas
    na, nb = active_space.nelecas
    h1 = active_space.h1
    eri = ao2mo.restore(1, active_space.eri, norb)

    if spin == "alpha":
        nelec_add, nelec_rem = (na + 1, nb), (na - 1, nb)
        create, destroy = addons.cre_a, addons.des_a
    elif spin == "beta":
        nelec_add, nelec_rem = (na, nb + 1), (na, nb - 1)
        create, destroy = addons.cre_b, addons.des_b
    else:
        raise ValueError(f"spin must be 'alpha' or 'beta', got {spin!r}")

    # Reference state from PySCF's Davidson solver, not from a dense eigh.
    e0, ci0 = direct_spin1.kernel(h1, eri, norb, (na, nb))

    addr_add, ham_add = _dense_hamiltonian(h1, eri, norb, nelec_add, max_dets)
    addr_rem, ham_rem = _dense_hamiltonian(h1, eri, norb, nelec_rem, max_dets)

    if mu is None:
        e_add = direct_spin1.kernel(h1, eri, norb, nelec_add)[0]
        e_rem = direct_spin1.kernel(h1, eri, norb, nelec_rem)[0]
        mu = 0.5 * (e_rem - e_add)
    mu = float(mu)

    # Columns are c^dag_i|0> and c_i|0>, in each sector's determinant ordering.
    amps_add = np.column_stack(
        [create(ci0, norb, (na, nb), i).ravel()[addr_add] for i in range(norb)]
    )
    amps_rem = np.column_stack(
        [destroy(ci0, norb, (na, nb), i).ravel()[addr_rem] for i in range(norb)]
    )

    eye_add = np.eye(ham_add.shape[0])
    eye_rem = np.eye(ham_rem.shape[0])

    z = np.atleast_1d(np.asarray(z, dtype=complex))
    out = np.empty((z.size, norb, norb), dtype=complex)
    for k, zk in enumerate(z):
        particle = amps_add.T @ np.linalg.solve((zk - mu + e0) * eye_add - ham_add, amps_add)
        hole = amps_rem.T @ np.linalg.solve((zk - mu - e0) * eye_rem + ham_rem, amps_rem)
        out[k] = particle + hole
    return out

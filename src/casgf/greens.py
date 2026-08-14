"""One-particle Green's function of a CAS active space, in Lehmann form.

The retarded Green's function of the active space is

    G_ij(w) = sum_n <0|c_i|n><n|c^dag_j|0> / (w + i*eta - (E_n^{N+1} - E_0^N + mu))
            + sum_m <0|c^dag_j|m><m|c_i|0> / (w + i*eta - (E_0^N - E_m^{N-1} + mu))

Both branches carry **the same** ``+mu`` shift: adding ``mu * N_op`` to the
Hamiltonian sends ``E^{(M)} -> E^{(M)} + mu * M``, and both an addition and a
removal energy is a difference between sectors differing by exactly one
particle.  Getting one of the two signs wrong is the single easiest way to
break this calculation -- it makes the gap asymmetric about ``w = 0`` while
leaving every other diagnostic (sum rules, Hermiticity) intact.  See
``docs/theory.md`` and ``tests/test_chempot.py``.

Once the three sectors are diagonalised, evaluating ``G`` at any frequency is
just a rational sum: 30000 contour points cost no more than a few matrix
products, which is what makes the winding-number contours practical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ed import DEFAULT_MAX_DETS, solve_sector, transition_amplitudes

__all__ = ["Lehmann", "lehmann"]

# Target working-set size, in complex128 elements, for one evaluation chunk.
_CHUNK_ELEMENTS = 2_000_000


@dataclass(frozen=True)
class Lehmann:
    """Pole/residue representation of a one-particle Green's function.

    ``poles`` and ``residues`` hold the addition branch first (``n_add``
    entries) and then the removal branch, so that

        G_ij(z) = sum_n residues[i, n] * residues[j, n] / (z - poles[n])

    covers both.  The residues are real because the CI vectors are real.
    """

    poles: np.ndarray
    residues: np.ndarray
    n_add: int
    mu: float
    e0: float
    nelec: tuple[int, int]
    spin: str = "alpha"

    @property
    def norb(self) -> int:
        return self.residues.shape[0]

    @property
    def poles_add(self) -> np.ndarray:
        """Electron-addition (particle) poles, at positive frequency."""
        return self.poles[: self.n_add]

    @property
    def poles_rem(self) -> np.ndarray:
        """Electron-removal (hole) poles, at negative frequency."""
        return self.poles[self.n_add :]

    @property
    def residues_add(self) -> np.ndarray:
        return self.residues[:, : self.n_add]

    @property
    def residues_rem(self) -> np.ndarray:
        return self.residues[:, self.n_add :]

    @property
    def gap(self) -> float:
        """Fundamental gap: lowest addition pole minus highest removal pole."""
        if self.n_add == 0 or self.n_add == self.poles.size:
            raise ValueError("both an addition and a removal branch are needed for a gap")
        return float(self.poles_add.min() - self.poles_rem.max())

    def with_mu(self, mu: float) -> Lehmann:
        """Same spectrum, re-referenced to a different chemical potential.

        The chemical potential enters only as a rigid shift of every pole, so
        this needs no new diagonalisation.
        """
        return Lehmann(
            poles=self.poles - self.mu + float(mu),
            residues=self.residues,
            n_add=self.n_add,
            mu=float(mu),
            e0=self.e0,
            nelec=self.nelec,
            spin=self.spin,
        )

    # -- evaluation ---------------------------------------------------------

    def _select(self, orbitals) -> np.ndarray:
        if orbitals is None:
            return self.residues
        return self.residues[np.asarray(orbitals)]

    def on_grid(self, freqs, eta: float = 1e-3, orbitals=None) -> np.ndarray:
        """``G(w + i*eta)`` on a frequency grid, shape ``(n_freq, nsel, nsel)``.

        ``orbitals`` selects a sub-block of the active space (for example one
        orbital-symmetry block); ``None`` keeps all of them.
        """
        res = self._select(orbitals)
        nsel = res.shape[0]
        z = np.atleast_1d(np.asarray(freqs, dtype=float)) + 1j * float(eta)
        out = np.empty((z.size, nsel, nsel), dtype=complex)
        chunk = max(1, _CHUNK_ELEMENTS // max(1, nsel * self.poles.size))
        for start in range(0, z.size, chunk):
            zz = z[start : start + chunk]
            weight = 1.0 / (zz[:, None] - self.poles[None, :])
            out[start : start + chunk] = (res[None, :, :] * weight[:, None, :]) @ res.T
        return out

    def at(self, omega: float, eta: float = 1e-3, orbitals=None) -> np.ndarray:
        """``G`` at a single frequency, shape ``(nsel, nsel)``."""
        return self.on_grid([omega], eta=eta, orbitals=orbitals)[0]

    def at_complex(self, z, orbitals=None) -> np.ndarray:
        """``G`` at arbitrary complex frequencies, shape ``(n_z, nsel, nsel)``.

        Unlike :meth:`on_grid` this puts no restriction on the sign of the
        imaginary part, which the winding-number contour needs.
        """
        res = self._select(orbitals)
        nsel = res.shape[0]
        z = np.atleast_1d(np.asarray(z, dtype=complex))
        out = np.empty((z.size, nsel, nsel), dtype=complex)
        chunk = max(1, _CHUNK_ELEMENTS // max(1, nsel * self.poles.size))
        for start in range(0, z.size, chunk):
            zz = z[start : start + chunk]
            weight = 1.0 / (zz[:, None] - self.poles[None, :])
            out[start : start + chunk] = (res[None, :, :] * weight[:, None, :]) @ res.T
        return out

    def det(self, freqs, eta: float = 1e-3, orbitals=None) -> np.ndarray:
        """``det G(w + i*eta)``, shape ``(n_freq,)``, complex."""
        return np.linalg.det(self.on_grid(freqs, eta=eta, orbitals=orbitals))

    def log_abs_det(self, freqs, eta: float = 1e-3, orbitals=None) -> np.ndarray:
        """``log|det G(w + i*eta)|``.

        Poles of ``G`` show up as peaks and zeros of ``det G`` as dips, which is
        what makes this a map of the topological structure rather than just of
        the spectrum.
        """
        sign, logdet = np.linalg.slogdet(self.on_grid(freqs, eta=eta, orbitals=orbitals))
        del sign
        return np.real(logdet)

    def spectral(self, freqs, eta: float = 1e-3, orbitals=None) -> np.ndarray:
        """Spectral function ``-Im Tr G(w + i*eta)``, shape ``(n_freq,)``.

        Evaluated straight from the pole weights, so this is cheap even on very
        dense grids.
        """
        weights = self.pole_weights(orbitals)
        w = np.atleast_1d(np.asarray(freqs, dtype=float))
        eta = float(eta)
        out = np.empty(w.size)
        chunk = max(1, _CHUNK_ELEMENTS // max(1, self.poles.size))
        for start in range(0, w.size, chunk):
            ww = w[start : start + chunk]
            denom = (ww[:, None] - self.poles[None, :]) ** 2 + eta**2
            out[start : start + chunk] = (weights[None, :] * eta / denom).sum(axis=1)
        return out

    def pole_weights(self, orbitals=None) -> np.ndarray:
        """Spectral weight carried by each pole, shape ``(n_poles,)``."""
        res = self._select(orbitals)
        return np.einsum("in,in->n", res, res)

    def sum_rule(self, orbitals=None) -> float:
        """Total spectral weight, which must equal the number of orbitals.

        ``sum_n Tr(r_n r_n^T) = Tr(<0|{c_i, c^dag_j}|0>) = nsel``.
        """
        return float(self.pole_weights(orbitals).sum())


def lehmann(
    active_space,
    spin: str = "alpha",
    mu: float | None = None,
    max_dets: int = DEFAULT_MAX_DETS,
) -> Lehmann:
    """Build the Lehmann representation for an :class:`~casgf.ActiveSpace`.

    Three sectors are diagonalised completely: the reference ``N``, and the
    ``N+1`` / ``N-1`` sectors reached by adding or removing one electron of the
    requested ``spin``.

    Parameters
    ----------
    active_space:
        Carries ``h1``, ``eri``, ``ncas`` and ``nelecas``.
    spin:
        ``"alpha"`` or ``"beta"``.
    mu:
        Chemical potential.  ``None`` (the default) uses the particle-hole
        symmetric choice ``mu = (E_gs^{N-1} - E_gs^{N+1}) / 2``, which puts
        ``w = 0`` in the middle of the fundamental gap.  Note this reads the
        ``N-1`` and ``N+1`` ground states off the sectors reached by flipping
        one electron of ``spin``; that is the global ``N±1`` ground state
        whenever those are doublets, which is the usual case here.  Pass ``mu``
        explicitly if it is not.

    Raises
    ------
    ValueError
        If a needed sector does not exist (a full or empty active space) and no
        explicit ``mu`` was given.
    """
    h1 = active_space.h1
    eri = active_space.eri
    norb = active_space.ncas
    na, nb = active_space.nelecas

    if spin == "alpha":
        nelec_add, nelec_rem = (na + 1, nb), (na - 1, nb)
    elif spin == "beta":
        nelec_add, nelec_rem = (na, nb + 1), (na, nb - 1)
    else:
        raise ValueError(f"spin must be 'alpha' or 'beta', got {spin!r}")

    ref = solve_sector(h1, eri, norb, (na, nb), max_dets=max_dets)
    e0 = ref.e_gs
    civec = ref.civec(0)

    have_add = 0 <= nelec_add[0] <= norb and 0 <= nelec_add[1] <= norb
    have_rem = 0 <= nelec_rem[0] <= norb and 0 <= nelec_rem[1] <= norb

    sec_add = solve_sector(h1, eri, norb, nelec_add, max_dets=max_dets) if have_add else None
    sec_rem = solve_sector(h1, eri, norb, nelec_rem, max_dets=max_dets) if have_rem else None

    if mu is None:
        if sec_add is None or sec_rem is None:
            raise ValueError(
                "cannot infer mu without both an N+1 and an N-1 sector; pass mu explicitly"
            )
        mu = 0.5 * (sec_rem.e_gs - sec_add.e_gs)
    mu = float(mu)

    if sec_add is not None:
        poles_add = (sec_add.energies - e0) + mu
        res_add = transition_amplitudes(civec, norb, (na, nb), sec_add, "create", spin)
    else:
        poles_add = np.zeros(0)
        res_add = np.zeros((norb, 0))

    if sec_rem is not None:
        poles_rem = (e0 - sec_rem.energies) + mu
        res_rem = transition_amplitudes(civec, norb, (na, nb), sec_rem, "destroy", spin)
    else:
        poles_rem = np.zeros(0)
        res_rem = np.zeros((norb, 0))

    return Lehmann(
        poles=np.concatenate([poles_add, poles_rem]),
        residues=np.concatenate([res_add, res_rem], axis=1),
        n_add=poles_add.size,
        mu=mu,
        e0=e0,
        nelec=(na, nb),
        spin=spin,
    )

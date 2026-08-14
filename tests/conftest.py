import numpy as np
import pytest

from casgf import ActiveSpace


def random_h1(norb, seed=0):
    """A random real symmetric one-electron Hamiltonian."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(norb, norb))
    return 0.5 * (a + a.T)


def random_eri(norb, seed=0, scale=0.3):
    """Random two-electron integrals with the full chemist-notation symmetry.

    ``(ij|kl) = (ji|kl) = (ij|lk) = (kl|ij)`` -- the permutational symmetry PySCF
    assumes when it builds the FCI Hamiltonian.
    """
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(norb,) * 4) * scale
    e = 0.5 * (e + e.transpose(1, 0, 2, 3))
    e = 0.5 * (e + e.transpose(0, 1, 3, 2))
    e = 0.5 * (e + e.transpose(2, 3, 0, 1))
    return e


@pytest.fixture
def free_space():
    """Non-interacting 4-orbital active space at half filling."""
    norb = 4
    return ActiveSpace.from_arrays(random_h1(norb, seed=3), np.zeros((norb,) * 4), nelecas=4)


@pytest.fixture
def interacting_space():
    """Interacting 4-orbital active space at half filling."""
    norb = 4
    return ActiveSpace.from_arrays(random_h1(norb, seed=7), random_eri(norb, seed=11), nelecas=4)

"""Exact properties every one-particle Green's function must satisfy."""

import numpy as np

from casgf import lehmann


def test_total_spectral_weight_equals_number_of_orbitals(interacting_space):
    gf = lehmann(interacting_space)
    assert abs(gf.sum_rule() - interacting_space.ncas) < 1e-10


def test_spectral_weight_of_a_sub_block(interacting_space):
    gf = lehmann(interacting_space)
    assert abs(gf.sum_rule(orbitals=[0, 2]) - 2.0) < 1e-10


def test_hermiticity(interacting_space):
    """``G(w - i*eta) = G(w + i*eta)^dagger`` for real ``h1`` and ``eri``."""
    gf = lehmann(interacting_space)
    freqs = np.linspace(-2, 2, 17)
    upper = gf.at_complex(freqs + 0.1j)
    lower = gf.at_complex(freqs - 0.1j)
    assert np.abs(lower - upper.conj().transpose(0, 2, 1)).max() < 1e-12


def test_spectral_function_is_non_negative(interacting_space):
    gf = lehmann(interacting_space)
    assert gf.spectral(np.linspace(-5, 5, 201), eta=0.05).min() >= 0.0


def test_spectral_function_matches_the_explicit_trace(interacting_space):
    """The cheap pole-weight formula must agree with forming ``G`` and tracing it."""
    gf = lehmann(interacting_space)
    freqs = np.linspace(-3, 3, 41)
    explicit = -np.imag(np.trace(gf.on_grid(freqs, eta=0.05), axis1=1, axis2=2))
    assert np.abs(gf.spectral(freqs, eta=0.05) - explicit).max() < 1e-12


def test_spectral_function_is_correctly_normalised(interacting_space):
    """``(1/pi) * integral of -Im Tr G`` over a finite window, done exactly.

    Each Lorentzian integrates to ``pi`` over the whole real line, so the sum
    rule fixes the normalisation.  Comparing against the *closed form over the
    same finite window* -- rather than against ``ncas`` -- removes the tail that
    any finite grid cuts off, leaving only quadrature error.
    """
    gf = lehmann(interacting_space)
    window, eta = 50.0, 0.05
    freqs = np.linspace(-window, window, 100_001)
    integral = np.trapezoid(gf.spectral(freqs, eta=eta), freqs) / np.pi

    weights = gf.pole_weights()
    exact = np.sum(
        weights
        * (np.arctan((window - gf.poles) / eta) + np.arctan((window + gf.poles) / eta))
        / np.pi
    )
    assert abs(integral - exact) < 1e-8
    # ...and the truncated window already accounts for nearly all the weight.
    assert abs(exact - interacting_space.ncas) < 5e-3


def test_residues_are_real_and_finite(interacting_space):
    gf = lehmann(interacting_space)
    assert np.isrealobj(gf.residues)
    assert np.isfinite(gf.residues).all()
    assert np.isfinite(gf.poles).all()

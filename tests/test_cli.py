"""Command line interface."""

import numpy as np
import pytest
from conftest import random_eri, random_h1

from casgf.cli import main


@pytest.fixture
def integrals(tmp_path):
    """A stack of two geometries, as a scan writes them out."""
    norb = 4
    h1 = np.stack([random_h1(norb, seed=1), random_h1(norb, seed=2)])
    eri = np.stack([random_eri(norb, seed=1), random_eri(norb, seed=2)])
    h1_path, eri_path = tmp_path / "h1.npy", tmp_path / "eri.npy"
    np.save(h1_path, h1)
    np.save(eri_path, eri)
    return h1_path, eri_path


def test_gf_writes_a_curve(integrals, tmp_path, capsys):
    h1, eri = integrals
    out = tmp_path / "curve.npy"
    code = main(
        ["gf", "--h1", str(h1), "--eri", str(eri), "--index", "0", "--nelec", "4",
         "--points", "51", "--eta", "0.01", "--out", str(out)]
    )
    assert code == 0
    assert np.load(out).shape == (51,)

    printed = capsys.readouterr().out
    assert "CAS(4,4)" in printed
    assert "sum rule" in printed


def test_gf_spectral_quantity(integrals, tmp_path):
    h1, eri = integrals
    out = tmp_path / "spectrum.npy"
    main(
        ["gf", "--h1", str(h1), "--eri", str(eri), "--index", "1", "--nelec", "2,2",
         "--quantity", "spectral", "--points", "31", "--eta", "0.05", "--out", str(out)]
    )
    assert (np.load(out) >= 0).all()


def test_gf_on_a_single_block(integrals, tmp_path):
    h1, eri = integrals
    out = tmp_path / "even.npy"
    main(
        ["gf", "--h1", str(h1), "--eri", str(eri), "--index", "0", "--nelec", "4",
         "--block", "even", "--points", "21", "--eta", "0.05", "--out", str(out)]
    )
    assert np.load(out).shape == (21,)


def test_index_is_required_for_a_stacked_array(integrals):
    h1, eri = integrals
    with pytest.raises(SystemExit, match="--index"):
        main(["gf", "--h1", str(h1), "--eri", str(eri), "--nelec", "4"])


def test_index_is_rejected_for_a_single_geometry(tmp_path):
    h1_path, eri_path = tmp_path / "h1.npy", tmp_path / "eri.npy"
    np.save(h1_path, random_h1(4, seed=1))
    np.save(eri_path, random_eri(4, seed=1))
    with pytest.raises(SystemExit, match="does not apply"):
        main(["gf", "--h1", str(h1_path), "--eri", str(eri_path), "--index", "0", "--nelec", "4"])


def test_winding_reports_both_blocks(integrals, capsys):
    h1, eri = integrals
    code = main(
        ["winding", "--h1", str(h1), "--eri", str(eri), "--index", "0", "--nelec", "4",
         "--points-per-segment", "2000"]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "winding number [even]" in printed
    assert "winding number [odd]" in printed


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["nonsense"])

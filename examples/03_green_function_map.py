"""Map ``log|det G(w)|`` across the cyclobutadiene automerization.

Cyclobutadiene's two double bonds trade places through a square transition state — the
textbook orbital-symmetry-controlled process. This script walks the rectangle → square →
rectangle path, running a fresh CASSCF(4,4) at every geometry, and plots two things:

* the energy profile, whose barrier is a number you can check against the literature;
* ``log|det G(w)|`` as an image, reaction coordinate across and frequency up. Bright ridges
  are poles of ``G``; dark valleys are zeros of ``det G``. Plotting the determinant rather
  than the spectral function is what makes the zeros visible at all.

The path is parameterised by the short C-C bond ``a``, with ``a + b`` held fixed, so ``a``
runs from the ground-state rectangle through the square (``a = b``) to the mirror-image
rectangle. Everything is symmetric about the square, which is a free correctness check on
the whole pipeline.

Usage
-----
    python 03_green_function_map.py
    python 03_green_function_map.py --points 21 --out automerization.png

Takes about half a minute: CAS(4,4) is 36 determinants, so the cost is entirely the CASSCF
calculations.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from casgf import irc_scan, lehmann

NCAS, NELEC = 4, 4
BASIS = "def2-SVP"
CH = 1.08  # C-H bond length, Angstrom
PERIMETER = 2.90  # a + b, held fixed along the scan
SQUARE = PERIMETER / 2  # a = b: the D4h transition state
WINDOW, N_FREQ, ETA = (-1.2, 1.2), 481, 5e-3
HARTREE_TO_KCAL = 627.509


def cyclobutadiene(a: float) -> str:
    """Planar cyclobutadiene with C-C bonds of alternating length ``a`` and ``b``.

    Carbons sit on the corners of a rectangle; each C-H bond runs along the exterior
    angle bisector. Idealised rather than optimised, which keeps the script
    self-contained.
    """
    b = PERIMETER - a
    atoms = []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        cx, cy = sx * a / 2, sy * b / 2
        atoms.append(f"C {cx:.8f} {cy:.8f} 0.0")
        atoms.append(f"H {cx + sx * CH / np.sqrt(2):.8f} {cy + sy * CH / np.sqrt(2):.8f} 0.0")
    return "; ".join(atoms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bond", type=float, nargs=2, default=(1.34, 1.56),
                        metavar=("MIN", "MAX"), help="range of the short C-C bond, Angstrom")
    parser.add_argument("--points", type=int, default=13, help="geometries along the path")
    parser.add_argument("--out", help="save the figure here instead of showing it")
    args = parser.parse_args()

    bonds = np.linspace(*args.bond, args.points)
    freqs = np.linspace(*WINDOW, N_FREQ)

    rows = np.empty((bonds.size, N_FREQ))
    energies = np.empty(bonds.size)
    gaps = np.empty(bonds.size)

    started = time.time()
    # irc_scan carries the converged orbitals from one geometry to the next, which is
    # what keeps the same four pi orbitals in the active space along the whole path.
    for position, step in enumerate(
        irc_scan([cyclobutadiene(a) for a in bonds], ncas=NCAS, nelecas=NELEC,
                 basis=BASIS, unit="A", verbose=0)
    ):
        gf = lehmann(step.active_space)
        rows[position] = gf.log_abs_det(freqs, eta=ETA)
        energies[position] = step.e_tot
        gaps[position] = gf.gap
        print(f"[{position + 1:3d}/{bonds.size}] a = {bonds[position]:5.3f} A  "
              f"E = {step.e_tot:.8f} Ha  gap = {gf.gap:.6f}  "
              f"{time.time() - started:5.1f} s", flush=True)

    relative = (energies - energies.min()) * HARTREE_TO_KCAL
    at_square = int(np.abs(bonds - SQUARE).argmin())
    print(f"\nbarrier at the square geometry: {relative[at_square]:.2f} kcal/mol")
    print(f"smallest gap {gaps.min():.6f} at a = {bonds[gaps.argmin()]:.3f} A")

    # The path is symmetric about the square, so the profile has to be too.
    if args.points % 2 == 1 and np.isclose(bonds.mean(), SQUARE):
        asymmetry = np.abs(relative - relative[::-1]).max()
        print(f"profile asymmetry about the square: {asymmetry:.2e} kcal/mol "
              "(should be ~0 by symmetry)")

    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7, 7), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )

    top.plot(bonds, relative, "o-", color="tab:blue", ms=3)
    top.set_ylabel("energy (kcal/mol)")
    top.axvline(SQUARE, color="tab:red", lw=1, ls="--")

    mesh = bottom.pcolormesh(bonds, freqs, rows.T, cmap="gray_r", shading="nearest")
    bottom.axvline(SQUARE, color="tab:red", lw=1, ls="--", label="square (D4h)")
    bottom.set_xlabel("short C-C bond (Å)")
    bottom.set_ylabel(r"$\omega$ (a.u.)")
    bottom.legend(loc="upper right", fontsize=9)
    bar = fig.colorbar(mesh, ax=bottom)
    bar.set_ticks([rows.min(), rows.max()])
    bar.set_ticklabels(["zero", "pole"])

    fig.suptitle(r"Cyclobutadiene automerization: energy and $\log|\det G(\omega)|$")
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out)
        print(f"saved -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

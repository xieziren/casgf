"""Map ``log|det G(w)|`` along a geometric coordinate.

Scans a rectangular H4 from a narrow rectangle, through the square, to a wide one, running
a fresh CASSCF at every point and plotting the Green's-function determinant as an image:
reaction coordinate across, frequency up. Bright ridges are poles of ``G`` and dark valleys
are zeros of ``det G``. Watching how they move -- and whether a ridge and a valley ever
cross -- is the point of plotting the determinant rather than the spectral function.

H4 is a stand-in for any path along which the active space is deformed; the same routine
works on an IRC or a relaxed scan by feeding :func:`casgf.irc_scan` a list of geometries
instead.

Usage
-----
    python 03_green_function_map.py
    python 03_green_function_map.py --points 41 --out map.png

Takes a few seconds: CAS(4,4) is 36 determinants, and the plot is dominated by the CASSCF
calculations rather than by the Green's functions.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from pyscf import gto

from casgf import ActiveSpace, lehmann

NCAS, NELEC = 4, 4
BASIS = "6-31g"
HEIGHT = 2.6  # Bohr; width == HEIGHT is the square
WINDOW, N_FREQ, ETA = (-1.0, 1.0), 401, 5e-3


def h4(width: float, height: float = HEIGHT) -> str:
    """Four hydrogens on the corners of a rectangle, in Bohr."""
    x, y = width / 2, height / 2
    return "; ".join(
        f"H {sx * x:.8f} {sy * y:.8f} 0.0" for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=float, nargs=2, default=(1.8, 3.4),
                        metavar=("MIN", "MAX"), help="range of rectangle widths, in Bohr")
    parser.add_argument("--points", type=int, default=25, help="geometries along the scan")
    parser.add_argument("--out", help="save the figure here instead of showing it")
    args = parser.parse_args()

    widths = np.linspace(*args.width, args.points)
    freqs = np.linspace(*WINDOW, N_FREQ)

    rows = np.empty((widths.size, N_FREQ))
    gaps = np.empty(widths.size)
    started = time.time()
    for position, width in enumerate(widths):
        mol = gto.M(atom=h4(width), basis=BASIS, unit="B", verbose=0)
        gf = lehmann(ActiveSpace.from_molecule(mol, ncas=NCAS, nelecas=NELEC))
        rows[position] = gf.log_abs_det(freqs, eta=ETA)
        gaps[position] = gf.gap
        print(f"[{position + 1:3d}/{widths.size}] width {width:5.3f} Bohr  "
              f"gap {gf.gap:.6f}  {time.time() - started:6.1f} s", flush=True)

    narrowest = widths[gaps.argmin()]
    print(f"\nsmallest gap {gaps.min():.6f} at width {narrowest:.3f} Bohr "
          f"(the square is at {HEIGHT:.3f})")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    mesh = ax.pcolormesh(widths, freqs, rows.T, cmap="gray_r", shading="nearest")
    ax.axvline(HEIGHT, color="tab:red", lw=1, ls="--", label="square")
    ax.set_xlabel("rectangle width (Bohr)")
    ax.set_ylabel(r"$\omega$ (a.u.)")
    ax.set_title(r"$\log|\det G(\omega)|$ across an H4 rectangle-to-square scan")
    ax.legend(loc="upper right", fontsize=9)
    bar = fig.colorbar(mesh, ax=ax)
    bar.set_ticks([rows.min(), rows.max()])
    bar.set_ticklabels(["zero", "pole"])
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out)
        print(f"saved -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

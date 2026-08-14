"""Command line front end.

Three things are worth doing without writing a script:

* ``casgf gf`` -- a Green's function curve from stored active-space integrals;
* ``casgf winding`` -- the winding number of one symmetry block;
* ``casgf scan`` -- a CASSCF walk along an IRC, writing out the integrals.

The last of these turns a reaction path into stored integrals, so the Green's
functions can be rebuilt later without redoing any quantum chemistry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .active_space import ActiveSpace
from .greens import lehmann
from .irc import read_geometry_blocks
from .scan import collect_integrals, irc_scan
from .symmetry import block_indices
from .winding import keyhole_contour, min_pole_distance, winding_number_of

__all__ = ["main"]


def _parse_nelec(text: str):
    """``"8"`` or ``"5,3"``."""
    if "," in text:
        a, b = text.split(",", 1)
        return int(a), int(b)
    return int(text)


def _load_active_space(args) -> ActiveSpace:
    h1 = np.load(args.h1)
    eri = np.load(args.eri)
    if h1.ndim == 3:
        if args.index is None:
            raise SystemExit(
                f"{args.h1} holds {h1.shape[0]} geometries; pass --index to pick one"
            )
        h1, eri = h1[args.index], eri[args.index]
    elif args.index is not None:
        raise SystemExit(f"{args.h1} holds a single geometry; --index does not apply")
    return ActiveSpace.from_arrays(h1, eri, nelecas=args.nelec)


def _report(label: str, values: np.ndarray, out: Path | None) -> None:
    print(f"{label}: {values.size} points, range [{values.min():.6g}, {values.max():.6g}]")
    if out is not None:
        np.save(out, values)
        print(f"saved -> {out}")


def cmd_gf(args) -> int:
    space = _load_active_space(args)
    gf = lehmann(space, spin=args.spin)
    orbitals = block_indices(args.block, space.ncas, orbsym=space.orbsym)

    print(f"CAS({space.nelec_total},{space.ncas})  mu = {gf.mu:.8f}  gap = {gf.gap:.8f}")
    print(f"sum rule = {gf.sum_rule():.10f} (expected {space.ncas})")

    freqs = np.linspace(args.window[0], args.window[1], args.points)
    if args.quantity == "logdet":
        _report("log|det G|", gf.log_abs_det(freqs, eta=args.eta, orbitals=orbitals), args.out)
    else:
        _report("-Im Tr G", gf.spectral(freqs, eta=args.eta, orbitals=orbitals), args.out)
    return 0


def cmd_winding(args) -> int:
    space = _load_active_space(args)
    gf = lehmann(space, spin=args.spin)
    contour = keyhole_contour(radius=args.radius, n_per_segment=args.points_per_segment)

    clearance = min_pole_distance(gf, contour)
    print(f"CAS({space.nelec_total},{space.ncas})  mu = {gf.mu:.8f}  gap = {gf.gap:.8f}")
    print(f"contour radius {args.radius}, closest pole {clearance:.3e} away")
    if clearance < 1e-4:
        print("warning: the contour nearly touches a pole; the result may be unreliable")

    blocks = ["even", "odd"] if args.block == "blocks" else [args.block]
    for block in blocks:
        orbitals = block_indices(block, space.ncas, orbsym=space.orbsym)
        print(f"  winding number [{block}] = {winding_number_of(gf, contour, orbitals=orbitals)}")
    return 0


def cmd_scan(args) -> int:
    geometries = read_geometry_blocks(args.geometries, unit=args.unit)
    print(f"{len(geometries)} geometries in {args.geometries}, stride {args.stride}")

    data = collect_integrals(
        irc_scan(
            geometries,
            ncas=args.ncas,
            nelecas=args.nelec,
            basis=args.basis,
            stride=args.stride,
            state_average=args.state_average,
            verbose=args.verbose,
        )
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for key, array in data.items():
        np.save(out / f"{key}.npy", array)
    print(f"wrote {', '.join(sorted(data))} to {out}/")
    print(f"energies: {data['energies'][:, 0].min():.8f} .. {data['energies'][:, 0].max():.8f} Ha")
    return 0


def _add_integral_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--h1", required=True, help="one-electron integrals (.npy)")
    parser.add_argument("--eri", required=True, help="two-electron integrals (.npy)")
    parser.add_argument("--index", type=int, help="geometry to take from a stacked array")
    parser.add_argument(
        "--nelec", type=_parse_nelec, required=True,
        help="active electrons, e.g. 8 or 4,4",
    )
    parser.add_argument("--spin", choices=("alpha", "beta"), default="alpha")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="casgf", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    gf = sub.add_parser("gf", help="Green's function curve from stored integrals")
    _add_integral_arguments(gf)
    gf.add_argument("--window", type=float, nargs=2, default=(-1.0, 1.0), metavar=("LO", "HI"))
    gf.add_argument("--points", type=int, default=301)
    gf.add_argument("--eta", type=float, default=1e-5, help="Lorentzian broadening")
    gf.add_argument("--quantity", choices=("logdet", "spectral"), default="logdet")
    gf.add_argument("--block", default="all", help="all, even, odd, or an irrep label")
    gf.add_argument("--out", type=Path, help="save the curve as .npy")
    gf.set_defaults(func=cmd_gf)

    wind = sub.add_parser("winding", help="winding number of det G around a closed contour")
    _add_integral_arguments(wind)
    wind.add_argument("--radius", type=float, default=1.5)
    wind.add_argument("--points-per-segment", type=int, default=10_000)
    wind.add_argument("--block", default="blocks", help="blocks, all, even, odd, or an irrep label")
    wind.set_defaults(func=cmd_winding)

    scan = sub.add_parser("scan", help="CASSCF along an IRC, writing out the integrals")
    scan.add_argument("--geometries", required=True, help="IRC file of '--'-separated blocks")
    scan.add_argument("--ncas", type=int, required=True)
    scan.add_argument("--nelec", type=_parse_nelec, required=True)
    scan.add_argument("--basis", default="def2-SVP")
    scan.add_argument("--unit", default="A", help="length unit in the IRC file")
    scan.add_argument("--stride", type=int, default=1)
    scan.add_argument("--state-average", type=float, nargs="+", metavar="W")
    scan.add_argument("--verbose", type=int, default=3, help="PySCF verbosity")
    scan.add_argument("--out", default="scan", help="output directory")
    scan.set_defaults(func=cmd_scan)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

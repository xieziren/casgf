"""IRC file readers."""

import numpy as np
import pytest

from casgf import read_geometry_blocks, read_irc_table

GEOMETRY_FILE = """\
H\t0.0000000000\t0.0000000000\t0.0000000000
H\t0.0000000000\t0.0000000000\t0.7400000000
--\t\t\t
H\t0.0000000000\t0.0000000000\t0.0000000000
H\t0.0000000000\t0.0000000000\t0.8000000000
--\t\t\t
H\t0.0000000000\t0.0000000000\t0.0000000000
H\t0.0000000000\t0.0000000000\t0.9000000000
"""

# The header is the real thing: "X Pos" is one label containing a space, so a
# whitespace-delimited reader sees three names for two columns.
TABLE_FILE = "X Pos\tEnergy\n0.000000\t-795.166879\n0.299910\t-795.166985\n0.599820\t-795.167316\n"


@pytest.fixture
def geometry_path(tmp_path):
    path = tmp_path / "irc_clean_awk"
    path.write_text(GEOMETRY_FILE)
    return path


@pytest.fixture
def table_path(tmp_path):
    path = tmp_path / "IRC-towards-product.txt"
    path.write_text(TABLE_FILE)
    return path


def test_geometry_blocks_are_split_on_the_separator(geometry_path):
    geometries = read_geometry_blocks(geometry_path)
    assert len(geometries) == 3
    assert all(len(g) == 2 for g in geometries)
    assert [atom[0] for atom in geometries[0]] == ["H", "H"]


def test_geometries_come_back_in_bohr(geometry_path):
    """``mol._atom`` stores coordinates in Bohr, whatever the file used."""
    geometries = read_geometry_blocks(geometry_path, unit="A")
    bond_length = geometries[0][1][1][2]
    assert bond_length == pytest.approx(0.74 / 0.52917721092, rel=1e-6)


def test_geometries_stay_in_file_order(geometry_path):
    geometries = read_geometry_blocks(geometry_path)
    z = [g[1][1][2] for g in geometries]
    assert z == sorted(z)


def test_trailing_separator_does_not_add_an_empty_geometry(tmp_path):
    path = tmp_path / "trailing"
    path.write_text(GEOMETRY_FILE + "--\t\t\t\n")
    assert len(read_geometry_blocks(path)) == 3


def test_irc_table_columns(table_path):
    table = read_irc_table(table_path)
    assert len(table) == 3
    assert np.allclose(table.coordinate, [0.0, 0.29991, 0.59982])
    assert np.allclose(table.energy, [-795.166879, -795.166985, -795.167316])


def test_irc_table_reversed(table_path):
    table = read_irc_table(table_path).reversed()
    assert np.allclose(table.coordinate, [-0.59982, -0.29991, 0.0])
    assert table.energy[-1] == pytest.approx(-795.166879)

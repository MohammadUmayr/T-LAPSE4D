"""
Functions to test the point-cloud I/O tools.

The Nx9 ``(X, Y, Z, R, G, B, NX, NY, NZ)`` layout is the contract every other module relies on, so the
round-trip tests here pin the column order, the 0-255 RGB scaling and the ``normal x/y/z`` extra
dimensions — a drift in any of those would corrupt every downstream co-registration silently.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pytest
from conftest import UTM45N, make_cloud
from shapely.geometry import Polygon

from cntp.io import apply_glacier_mask, load_las, read_las_bounds, save_las


@pytest.fixture
def las_path(tmp_path: Path, cloud: np.ndarray) -> Path:
    """Write the shared synthetic cloud to a .las file and return its path."""
    p = tmp_path / "cloud.las"
    save_las(cloud, p)
    return p


def _write_mask(path: Path, polygons: list[Polygon]) -> Path:
    """Write *polygons* to a vector file readable by :func:`cntp.io.apply_glacier_mask`."""
    gpd.GeoDataFrame(geometry=polygons, crs=f"EPSG:{UTM45N}").to_file(path)
    return path


class TestLasRoundTrip:
    """Writing an Nx9 cloud out and reading it back must be lossless within the stored precision."""

    def test_save_las__roundtrip_shape(self, las_path: Path, cloud: np.ndarray) -> None:
        assert load_las(las_path).shape == cloud.shape

    def test_save_las__roundtrip_xyz(self, las_path: Path, cloud: np.ndarray) -> None:
        # Header scale is derived as range/1e6, giving ~1e-5 m precision on a 20 m cloud.
        back = load_las(las_path)

        np.testing.assert_allclose(back[:, :3], cloud[:, :3], atol=1e-4)

    def test_save_las__roundtrip_rgb(self, las_path: Path, cloud: np.ndarray) -> None:
        # RGB is stored as uint16 (x257) and read back divided, so integer 0-255 values survive exactly.
        back = load_las(las_path)

        np.testing.assert_allclose(back[:, 3:6], cloud[:, 3:6], atol=1e-6)
        assert back[:, 3:6].max() <= 255.0

    def test_save_las__roundtrip_normals(self, las_path: Path, cloud: np.ndarray) -> None:
        # Normals are float32 extra dimensions, so float32 precision is the bound.
        back = load_las(las_path)

        np.testing.assert_allclose(back[:, 6:9], cloud[:, 6:9], atol=1e-6)

    def test_save_las__normals_stay_unit_length(self, las_path: Path) -> None:
        # extract_stable_terrain derives slope from these, so a scaling error would shift every slope.
        back = load_las(las_path)

        np.testing.assert_allclose(np.linalg.norm(back[:, 6:9], axis=1), 1.0, atol=1e-6)

    def test_save_las__writes_normal_extra_dimensions(self, las_path: Path) -> None:
        with laspy.open(las_path) as f:
            names = set(f.header.point_format.extra_dimension_names)

        assert {"normal x", "normal y", "normal z"} <= names

    def test_save_las__without_crs(self, las_path: Path) -> None:
        with laspy.open(las_path) as f:
            assert f.header.parse_crs() is None

    def test_save_las__with_crs(self, tmp_path: Path, cloud: np.ndarray) -> None:
        # When set, downstream readers recover the CRS via laspy's header.parse_crs().
        p = tmp_path / "with_crs.las"
        save_las(cloud, p, crs=UTM45N)

        with laspy.open(p) as f:
            assert f.header.parse_crs().to_epsg() == UTM45N

    def test_load_las__downsample(self, las_path: Path, cloud: np.ndarray) -> None:
        back = load_las(las_path, downsample_factor=0.25)

        assert 0 < len(back) < len(cloud)
        assert back.shape[1] == 9

    def test_load_las__downsample_factor_one(self, las_path: Path, cloud: np.ndarray) -> None:
        assert len(load_las(las_path, downsample_factor=1.0)) == len(cloud)


class TestReadLasBounds:
    """Reading the XYZ bounding box from the header, without loading the points."""

    def test_read_las_bounds(self, las_path: Path, cloud: np.ndarray) -> None:
        lo, hi = read_las_bounds(las_path)

        np.testing.assert_allclose(lo, cloud[:, :3].min(axis=0), atol=1e-4)
        np.testing.assert_allclose(hi, cloud[:, :3].max(axis=0), atol=1e-4)

    def test_read_las_bounds__shape(self, las_path: Path) -> None:
        lo, hi = read_las_bounds(las_path)

        assert lo.shape == (3,)
        assert hi.shape == (3,)
        assert np.all(hi >= lo)


class TestApplyGlacierMask:
    """
    Removing on-glacier points using the glacier outline. What survives is the stable terrain the
    co-registration is solved on, so a mask applied the wrong way round would align to moving ice.
    """

    def test_apply_glacier_mask(self, tmp_path: Path) -> None:
        # A polygon covering the X < 10 half of the make_cloud footprint.
        mask = _write_mask(tmp_path / "glacier.geojson", [Polygon([(-1, -1), (10, -1), (10, 21), (-1, 21)])])
        cloud = make_cloud(n=500)

        masked = apply_glacier_mask(cloud, mask)

        # Points inside the outline are removed; only the off-glacier half survives.
        assert len(masked) < len(cloud)
        assert np.all(masked[:, 0] >= 10)

    def test_apply_glacier_mask__preserves_nine_columns(self, tmp_path: Path, cloud: np.ndarray) -> None:
        mask = _write_mask(tmp_path / "glacier.geojson", [Polygon([(-1, -1), (10, -1), (10, 21), (-1, 21)])])

        assert apply_glacier_mask(cloud, mask).shape[1] == 9

    def test_apply_glacier_mask__disjoint_polygon(self, tmp_path: Path, cloud: np.ndarray) -> None:
        # A polygon nowhere near the cloud must remove nothing.
        mask = _write_mask(tmp_path / "elsewhere.geojson", [Polygon([(1000, 1000), (1010, 1000), (1010, 1010)])])

        assert len(apply_glacier_mask(cloud, mask)) == len(cloud)

    def test_apply_glacier_mask__unions_multiple_polygons(self, tmp_path: Path) -> None:
        # Multi-part glacier outlines must all mask, not just the first feature in the file.
        mask = _write_mask(
            tmp_path / "two_parts.geojson",
            [
                Polygon([(-1, -1), (5, -1), (5, 21), (-1, 21)]),
                Polygon([(15, -1), (21, -1), (21, 21), (15, 21)]),
            ],
        )
        cloud = make_cloud(n=500)

        masked = apply_glacier_mask(cloud, mask)

        # Only the strip between the two parts is left.
        assert np.all((masked[:, 0] >= 5) & (masked[:, 0] <= 15))

"""
Functions to test the rasterisation tools.

Answers are knowable in advance by construction: a plane interpolates to itself, two DEMs 5 m apart
difference to exactly 5 m, and a ramp with gradient 5 over 1 m pixels has slope arctan(5) = 78.7
degrees, either side of whatever threshold a test picks.

``build_dem_and_ortho_p2d`` is the one entry point not covered here — it shells out to ASP's
``point2dem``, so it cannot run without the binaries installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import rasterio
from affine import Affine
from conftest import GRID_SHAPE, GRID_TRANSFORM, UTM45N
from rasterio.crs import CRS

from cntp.raster import (
    build_dem_and_ortho,
    build_dod,
    build_reference_dem_and_ortho,
    extract_stable_terrain_from_dem,
    interpolate_and_mask,
    m3c2_to_raster,
    save_dem,
    save_ortho,
    stable_m3c2_raster,
)


def _scatter(n: int = 400, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return points on the tilted plane ``z = 2x + 3y``, so interpolation has an exact answer."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    return x, y, 2 * x + 3 * y


class TestSaveDem:
    """Writing a float array out as a georeferenced single-band GeoTIFF."""

    def test_save_dem__roundtrips_values(self, tmp_path: Path, crs: CRS) -> None:
        arr = np.arange(np.prod(GRID_SHAPE), dtype="float32").reshape(GRID_SHAPE)
        out = tmp_path / "dem.tif"

        save_dem(arr, out, crs, GRID_TRANSFORM)

        with rasterio.open(out) as src:
            np.testing.assert_allclose(src.read(1), arr)

    def test_save_dem__preserves_georeferencing(self, tmp_path: Path, crs: CRS) -> None:
        out = tmp_path / "dem.tif"

        save_dem(np.zeros(GRID_SHAPE, dtype="float32"), out, crs, GRID_TRANSFORM)

        with rasterio.open(out) as src:
            assert src.transform == GRID_TRANSFORM
            assert src.crs.to_epsg() == UTM45N
            assert src.count == 1

    def test_save_dem__declares_nan_nodata(self, tmp_path: Path, crs: CRS) -> None:
        # Downstream code relies on nodata being NaN so arithmetic propagates gaps naturally.
        out = tmp_path / "dem.tif"

        save_dem(np.full(GRID_SHAPE, np.nan, dtype="float32"), out, crs, GRID_TRANSFORM)

        with rasterio.open(out) as src:
            assert np.isnan(src.nodata)


class TestInterpolateAndMask:
    """Cubic griddata interpolation with a KD-tree gap mask — the cloud-to-DEM gridding step."""

    def test_interpolate_and_mask__recovers_a_plane(self) -> None:
        x, y, z = _scatter()
        xi, yi = np.meshgrid(np.linspace(2, 8, 20), np.linspace(2, 8, 20))

        zi = interpolate_and_mask(x, y, z, xi, yi, res=1.0, max_gap_pixels=5)

        # Well inside the point footprint, the interpolant must reproduce the source plane.
        np.testing.assert_allclose(zi, 2 * xi + 3 * yi, atol=0.5)

    def test_interpolate_and_mask__far_pixels_are_nan(self) -> None:
        x, y, z = _scatter()
        # A grid deliberately offset well away from the input footprint.
        xi, yi = np.meshgrid(np.linspace(40, 50, 10), np.linspace(40, 50, 10))

        zi = interpolate_and_mask(x, y, z, xi, yi, res=1.0, max_gap_pixels=2)

        # Pixels further than max_gap_pixels * res from any real point must not be invented.
        assert np.all(np.isnan(zi))

    def test_interpolate_and_mask__clipped_to_input_z_range(self) -> None:
        # Clough-Tocher cubic does not preserve monotonicity: at cliffs and cone walls it overshoots
        # far past either endpoint. The clip pulls those overshoots back to physically present values.
        x, y, z = _scatter()
        xi, yi = np.meshgrid(np.linspace(1, 9, 25), np.linspace(1, 9, 25))

        zi = interpolate_and_mask(x, y, z, xi, yi, res=1.0, max_gap_pixels=5)

        finite = zi[np.isfinite(zi)]
        assert finite.min() >= z.min()
        assert finite.max() <= z.max()

    def test_interpolate_and_mask__output_shape(self) -> None:
        x, y, z = _scatter()
        xi, yi = np.meshgrid(np.linspace(2, 8, 13), np.linspace(2, 8, 7))

        zi = interpolate_and_mask(x, y, z, xi, yi, res=1.0, max_gap_pixels=5)

        assert zi.shape == xi.shape


class TestBuildDod:
    """
    DEMs of difference. The sign convention is standard glaciology — positive means the day surface is
    above the reference (gain), negative means below (loss) — and matches the py4dgeo M3C2 sign on
    roughly horizontal terrain. Getting it backwards would invert every melt map.
    """

    def test_build_dod__day_minus_reference(self, tmp_path: Path, write_dem: Callable[..., Path]) -> None:
        ref = write_dem(np.full(GRID_SHAPE, 100.0), "ref.tif")
        day = write_dem(np.full(GRID_SHAPE, 105.0), "day.tif")

        out = build_dod(ref, day, tmp_path / "dod.tif")

        with rasterio.open(out) as src:
            np.testing.assert_allclose(src.read(1), 5.0)

    def test_build_dod__negative_where_surface_dropped(self, tmp_path: Path, write_dem: Callable[..., Path]) -> None:
        ref = write_dem(np.full(GRID_SHAPE, 100.0), "ref.tif")
        day = write_dem(np.full(GRID_SHAPE, 97.5), "day.tif")

        out = build_dod(ref, day, tmp_path / "dod.tif")

        with rasterio.open(out) as src:
            np.testing.assert_allclose(src.read(1), -2.5)

    def test_build_dod__default_output_path(self, write_dem: Callable[..., Path]) -> None:
        ref = write_dem(np.full(GRID_SHAPE, 100.0), "ref.tif")
        day = write_dem(np.full(GRID_SHAPE, 101.0), "day.tif")

        out = build_dod(ref, day)

        assert out == day.parent / "DOD.tif"
        assert out.exists()

    def test_build_dod__propagates_nan(self, tmp_path: Path, write_dem: Callable[..., Path]) -> None:
        # Nodata in either source must mask the output, so the DoD covers only the shared footprint.
        ref_arr = np.full(GRID_SHAPE, 100.0)
        day_arr = np.full(GRID_SHAPE, 105.0)
        ref_arr[0, 0] = np.nan
        day_arr[1, 1] = np.nan

        out = build_dod(write_dem(ref_arr, "ref.tif"), write_dem(day_arr, "day.tif"), tmp_path / "dod.tif")

        with rasterio.open(out) as src:
            dod = src.read(1)
        assert np.isnan(dod[0, 0])
        assert np.isnan(dod[1, 1])
        assert np.isfinite(dod[5, 5])

    def test_build_dod__converts_numeric_nodata(self, tmp_path: Path, write_dem: Callable[..., Path]) -> None:
        # ASP writes -9999 rather than NaN; left unconverted that becomes a -10099 m elevation change.
        ref_arr = np.full(GRID_SHAPE, 100.0)
        ref_arr[0, 0] = -9999.0
        ref = write_dem(ref_arr, "ref.tif", nodata=-9999.0)
        day = write_dem(np.full(GRID_SHAPE, 105.0), "day.tif")

        out = build_dod(ref, day, tmp_path / "dod.tif")

        with rasterio.open(out) as src:
            dod = src.read(1)
        assert np.isnan(dod[0, 0])
        np.testing.assert_allclose(dod[5, 5], 5.0)

    def test_build_dod__rejects_mismatched_shapes(self, tmp_path: Path, write_dem: Callable[..., Path]) -> None:
        ref = write_dem(np.full(GRID_SHAPE, 100.0), "ref.tif")
        day = write_dem(np.full((8, 8), 105.0), "day.tif")

        with pytest.raises(ValueError, match="shapes differ"):
            build_dod(ref, day, tmp_path / "dod.tif")

    def test_build_dod__rejects_misaligned_transforms(self, tmp_path: Path, write_dem: Callable[..., Path]) -> None:
        # Same shape but shifted pixels — differencing these silently is the real hazard, since the
        # output looks perfectly plausible while every pixel compares the wrong two locations.
        shifted = GRID_TRANSFORM * Affine.translation(3, 3)
        ref = write_dem(np.full(GRID_SHAPE, 100.0), "ref.tif")
        day = write_dem(np.full(GRID_SHAPE, 105.0), "day.tif", transform=shifted)

        with pytest.raises(ValueError, match="transforms differ"):
            build_dod(ref, day, tmp_path / "dod.tif")

    def test_build_dod__is_cached_unless_overwrite(self, tmp_path: Path, write_dem: Callable[..., Path]) -> None:
        ref = write_dem(np.full(GRID_SHAPE, 100.0), "ref.tif")
        day = write_dem(np.full(GRID_SHAPE, 105.0), "day.tif")
        out = build_dod(ref, day, tmp_path / "dod.tif")

        # Re-running against a different day DEM returns the cached file untouched ...
        day2 = write_dem(np.full(GRID_SHAPE, 200.0), "day2.tif")
        build_dod(ref, day2, out)
        with rasterio.open(out) as src:
            np.testing.assert_allclose(src.read(1), 5.0)

        # ... unless overwrite is asked for explicitly.
        build_dod(ref, day2, out, overwrite=True)
        with rasterio.open(out) as src:
            np.testing.assert_allclose(src.read(1), 100.0)


def _steep_dem(gradient: float = 5.0) -> np.ndarray:
    """A ramp with a known gradient, so slope = arctan(gradient) is exact and controllable."""
    rows, cols = np.indices(GRID_SHAPE)
    return (gradient * cols).astype("float32")


def _ortho(tmp_path: Path, crs: CRS, blue: float, name: str = "ortho.tif") -> Path:
    """Write a 3-band RGB ortho on the shared grid with a controllable blue channel."""
    import rasterio

    path = tmp_path / name
    with rasterio.open(
        path, "w", driver="GTiff", height=GRID_SHAPE[0], width=GRID_SHAPE[1], count=3,
        dtype="uint8", crs=crs, transform=GRID_TRANSFORM,
    ) as dst:
        dst.write(np.full(GRID_SHAPE, 120, dtype="uint8"), 1)
        dst.write(np.full(GRID_SHAPE, 120, dtype="uint8"), 2)
        dst.write(np.full(GRID_SHAPE, int(blue), dtype="uint8"), 3)
    return path


class TestSaveOrtho:
    """Nearest-neighbour RGB rasterisation of a cloud's colour onto the DEM grid."""

    def test_save_ortho(self, tmp_path: Path, crs: CRS) -> None:
        import rasterio

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 10, 500)
        y = rng.uniform(0, 10, 500)
        rgb = np.tile([200.0, 100.0, 50.0], (500, 1))
        xi, yi = np.meshgrid(np.linspace(1, 9, 12), np.linspace(1, 9, 12))
        out = tmp_path / "ortho.tif"

        save_ortho(x, y, rgb, xi, yi, 1.0, 5, out, crs, GRID_TRANSFORM)

        with rasterio.open(out) as src:
            # Three bands, uint8, nodata 0 — the format the pipeline's orthos use.
            assert src.count == 3
            assert src.dtypes[0] == "uint8"
            assert src.nodata == 0
            assert src.read(1).max() == 200

    def test_save_ortho__far_pixels_left_as_nodata(self, tmp_path: Path, crs: CRS) -> None:
        import rasterio

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 10, 300)
        y = rng.uniform(0, 10, 300)
        rgb = np.tile([200.0, 100.0, 50.0], (300, 1))
        # Grid far from every point, so no colour may be invented for it.
        xi, yi = np.meshgrid(np.linspace(80, 90, 8), np.linspace(80, 90, 8))
        out = tmp_path / "ortho.tif"

        save_ortho(x, y, rgb, xi, yi, 1.0, 2, out, crs, GRID_TRANSFORM)

        with rasterio.open(out) as src:
            assert src.read(1).max() == 0


class TestBuildDemAndOrtho:
    """
    Building a day's DEM and orthoimage on the reference's grid. Anchoring to the reference bbox is
    what makes every day's raster land on the same pixels, so they difference and stack directly.
    """

    def test_build_dem_and_ortho(self, tmp_path: Path, write_cloud_las) -> None:
        import rasterio

        cloud = write_cloud_las("cloud.las")
        dem_path, ortho_path = build_dem_and_ortho(cloud, cloud, tmp_path / "out", "2024-06-23")

        assert dem_path.name == "2024-06-23_dem.tif"
        assert ortho_path.name == "2024-06-23_ortho.tif"
        with rasterio.open(dem_path) as src:
            assert src.count == 1
            assert np.isfinite(src.read(1)).any()
        with rasterio.open(ortho_path) as src:
            assert src.count == 3

    def test_build_dem_and_ortho__crs_read_from_las_header(self, tmp_path: Path, write_cloud_las) -> None:
        import rasterio

        # Metashape writes the EPSG on export, so it need not be passed explicitly.
        cloud = write_cloud_las("cloud.las")
        dem_path, _ = build_dem_and_ortho(cloud, cloud, tmp_path / "out", "d")

        with rasterio.open(dem_path) as src:
            assert src.crs.to_epsg() == UTM45N

    def test_build_dem_and_ortho__explicit_epsg_wins(self, tmp_path: Path, write_cloud_las) -> None:
        import rasterio

        cloud = write_cloud_las("cloud.las")
        dem_path, _ = build_dem_and_ortho(cloud, cloud, tmp_path / "out", "d", utm_epsg=32644)

        with rasterio.open(dem_path) as src:
            assert src.crs.to_epsg() == 32644

    def test_build_dem_and_ortho__grid_anchored_to_reference(self, tmp_path: Path, write_cloud_las) -> None:
        import rasterio

        # A day cloud covering a different area must still land on the reference's grid origin.
        ref = write_cloud_las("ref.las")
        day = write_cloud_las("day.las", dz=2.0, seed=7)

        ref_dem, _ = build_dem_and_ortho(ref, ref, tmp_path / "a", "ref")
        day_dem, _ = build_dem_and_ortho(day, ref, tmp_path / "b", "day")

        with rasterio.open(ref_dem) as r, rasterio.open(day_dem) as d:
            assert r.transform == d.transform
            assert r.shape == d.shape

    def test_build_dem_and_ortho__resolution_sets_pixel_count(self, tmp_path: Path, write_cloud_las) -> None:
        import rasterio

        cloud = write_cloud_las("cloud.las")

        fine, _ = build_dem_and_ortho(cloud, cloud, tmp_path / "a", "d", res=1.0)
        coarse, _ = build_dem_and_ortho(cloud, cloud, tmp_path / "b", "d", res=2.0)

        with rasterio.open(fine) as f, rasterio.open(coarse) as c:
            assert c.width < f.width
            assert c.height < f.height

    def test_build_dem_and_ortho__is_cached_unless_overwrite(self, tmp_path: Path, write_cloud_las) -> None:
        cloud = write_cloud_las("cloud.las")
        out = tmp_path / "out"
        dem_path, _ = build_dem_and_ortho(cloud, cloud, out, "d")
        mtime = dem_path.stat().st_mtime_ns

        build_dem_and_ortho(cloud, cloud, out, "d")
        assert dem_path.stat().st_mtime_ns == mtime

        build_dem_and_ortho(cloud, cloud, out, "d", overwrite=True)
        assert dem_path.stat().st_mtime_ns != mtime

    def test_build_dem_and_ortho__downsample(self, tmp_path: Path, write_cloud_las) -> None:
        # Downsampling caps RAM for griddata; it must still produce a complete raster.
        import rasterio

        cloud = write_cloud_las("cloud.las")
        dem_path, _ = build_dem_and_ortho(cloud, cloud, tmp_path / "out", "d", cloud_downsample=0.5)

        with rasterio.open(dem_path) as src:
            assert np.isfinite(src.read(1)).any()

    def test_build_dem_and_ortho__no_epsg_anywhere(self, tmp_path: Path, cloud: np.ndarray) -> None:
        from cntp.io import save_las

        # Without a CRS in the header and none passed, the output would be unreferenced — refuse.
        bare = tmp_path / "bare.las"
        save_las(cloud, bare)

        with pytest.raises(ValueError, match="No EPSG"):
            build_dem_and_ortho(bare, bare, tmp_path / "out", "d")

    def test_build_reference_dem_and_ortho(self, tmp_path: Path, write_cloud_las) -> None:
        # Thin wrapper: same cloud as both source and grid anchor, fixed output filenames.
        cloud = write_cloud_las("ref.las")
        cache = tmp_path / "_ref_cache"

        dem_path, ortho_path = build_reference_dem_and_ortho(cloud, cache)

        assert dem_path == cache / "reference_dem.tif"
        assert ortho_path == cache / "reference_ortho.tif"
        assert dem_path.exists() and ortho_path.exists()


class TestExtractStableTerrainFromDem:
    """
    The raster analogue of the point-cloud stable-terrain filter: slope, NDWI/intensity and the
    glacier polygon, combined at the DEM's pixel grid. Only pixels passing all three keep a value.
    """

    def test_extract_stable_terrain_from_dem__steep_terrain_kept(
        self, tmp_path: Path, write_dem, crs: CRS
    ) -> None:
        import rasterio

        # gradient 5 over 1 m pixels -> slope 78.7 degrees, comfortably over the 60 degree threshold.
        dem = write_dem(_steep_dem(5.0), "dem.tif")

        out = extract_stable_terrain_from_dem(dem, slope_threshold=60.0)

        with rasterio.open(out) as src:
            assert np.isfinite(src.read(1)).any()

    def test_extract_stable_terrain_from_dem__flat_terrain_dropped(self, tmp_path: Path, write_dem) -> None:
        import rasterio

        dem = write_dem(np.full(GRID_SHAPE, 100.0), "dem.tif")

        out = extract_stable_terrain_from_dem(dem, slope_threshold=60.0)

        # Flat ground is glacier or valley fill, never the bedrock coregistration is solved on.
        with rasterio.open(out) as src:
            assert not np.isfinite(src.read(1)).any()

    def test_extract_stable_terrain_from_dem__default_output_path(self, tmp_path: Path, write_dem) -> None:
        dem = write_dem(_steep_dem(), "dem.tif")

        out = extract_stable_terrain_from_dem(dem)

        assert out == dem.with_name("dem_stable.tif")

    def test_extract_stable_terrain_from_dem__ndwi_drops_water(
        self, tmp_path: Path, write_dem, crs: CRS
    ) -> None:
        import rasterio

        dem = write_dem(_steep_dem(), "dem.tif")
        water = _ortho(tmp_path, crs, blue=200)  # blue-dominant, above the separation line

        out = extract_stable_terrain_from_dem(dem, ortho_path=water, out_path=tmp_path / "s.tif")

        with rasterio.open(out) as src:
            assert not np.isfinite(src.read(1)).any()

    def test_extract_stable_terrain_from_dem__ndwi_keeps_rock(
        self, tmp_path: Path, write_dem, crs: CRS
    ) -> None:
        import rasterio

        dem = write_dem(_steep_dem(), "dem.tif")
        rock = _ortho(tmp_path, crs, blue=60)  # R > B, negative NDWI

        out = extract_stable_terrain_from_dem(dem, ortho_path=rock, out_path=tmp_path / "s.tif")

        with rasterio.open(out) as src:
            assert np.isfinite(src.read(1)).any()

    def test_extract_stable_terrain_from_dem__ortho_must_be_rgb(
        self, tmp_path: Path, write_dem, crs: CRS
    ) -> None:
        dem = write_dem(_steep_dem(), "dem.tif")
        single_band = write_dem(np.zeros(GRID_SHAPE), "notortho.tif")

        with pytest.raises(ValueError, match="Expected an RGB ortho"):
            extract_stable_terrain_from_dem(dem, ortho_path=single_band, out_path=tmp_path / "s.tif")

    def test_extract_stable_terrain_from_dem__ortho_grid_must_match(
        self, tmp_path: Path, write_dem, crs: CRS
    ) -> None:
        import rasterio

        dem = write_dem(_steep_dem(), "dem.tif")
        small = tmp_path / "small_ortho.tif"
        with rasterio.open(
            small, "w", driver="GTiff", height=4, width=4, count=3,
            dtype="uint8", crs=crs, transform=GRID_TRANSFORM,
        ) as dst:
            for b in range(1, 4):
                dst.write(np.zeros((4, 4), dtype="uint8"), b)

        with pytest.raises(ValueError, match="shapes differ"):
            extract_stable_terrain_from_dem(dem, ortho_path=small, out_path=tmp_path / "s.tif")

    def test_extract_stable_terrain_from_dem__glacier_mask_excludes(
        self, tmp_path: Path, write_dem
    ) -> None:
        import geopandas as gpd
        import rasterio
        from shapely.geometry import box

        dem = write_dem(_steep_dem(), "dem.tif")
        # A polygon covering the whole raster footprint — everything is on-glacier.
        H, W = GRID_SHAPE
        poly = box(GRID_TRANSFORM.c, GRID_TRANSFORM.f - H, GRID_TRANSFORM.c + W, GRID_TRANSFORM.f)
        mask = tmp_path / "glacier.geojson"
        gpd.GeoDataFrame(geometry=[poly], crs=f"EPSG:{UTM45N}").to_file(mask)

        out = extract_stable_terrain_from_dem(dem, glacier_mask_path=mask, out_path=tmp_path / "s.tif")

        with rasterio.open(out) as src:
            assert not np.isfinite(src.read(1)).any()

    def test_extract_stable_terrain_from_dem__is_cached_unless_overwrite(
        self, tmp_path: Path, write_dem
    ) -> None:
        dem = write_dem(_steep_dem(), "dem.tif")
        out = extract_stable_terrain_from_dem(dem, out_path=tmp_path / "s.tif")
        mtime = out.stat().st_mtime_ns

        extract_stable_terrain_from_dem(dem, out_path=out)
        assert out.stat().st_mtime_ns == mtime

        extract_stable_terrain_from_dem(dem, out_path=out, overwrite=True)
        assert out.stat().st_mtime_ns != mtime


class TestM3c2ToRaster:
    """
    Rasterised cloud-to-cloud M3C2 distance. Unlike a vertical DoD this measures perpendicular
    separation along the local normal, so it is immune to the slope-projection inflation that
    pollutes vertical differencing on steep terrain.
    """

    def test_m3c2_to_raster(self, tmp_path: Path, write_cloud_las) -> None:
        import rasterio

        ref = write_cloud_las("ref.las")
        day = write_cloud_las("day.las", dz=0.5, seed=7)

        out = m3c2_to_raster(ref, day, tmp_path / "m3c2.tif")

        with rasterio.open(out) as src:
            arr = src.read(1)
            assert src.crs.to_epsg() == UTM45N
            assert np.isfinite(arr).any()

    def test_m3c2_to_raster__recovers_the_offset_sign(self, tmp_path: Path, write_cloud_las) -> None:
        import rasterio

        # Sign convention: positive means the day surface sits above the reference (gain).
        ref = write_cloud_las("ref.las")
        day = write_cloud_las("day.las", dz=0.5, seed=7)

        out = m3c2_to_raster(ref, day, tmp_path / "m3c2.tif")

        with rasterio.open(out) as src:
            arr = src.read(1)
        assert np.nanmedian(arr) > 0

    def test_m3c2_to_raster__explicit_epsg(self, tmp_path: Path, write_cloud_las) -> None:
        import rasterio

        ref = write_cloud_las("ref.las")
        day = write_cloud_las("day.las", seed=7)

        out = m3c2_to_raster(ref, day, tmp_path / "m3c2.tif", utm_epsg=32644)

        with rasterio.open(out) as src:
            assert src.crs.to_epsg() == 32644

    def test_m3c2_to_raster__is_cached_unless_overwrite(self, tmp_path: Path, write_cloud_las) -> None:
        ref = write_cloud_las("ref.las")
        day = write_cloud_las("day.las", seed=7)
        out = m3c2_to_raster(ref, day, tmp_path / "m3c2.tif")
        mtime = out.stat().st_mtime_ns

        m3c2_to_raster(ref, day, out)
        assert out.stat().st_mtime_ns == mtime

    def test_m3c2_to_raster__no_epsg_anywhere(self, tmp_path: Path, cloud: np.ndarray) -> None:
        from cntp.io import save_las

        bare = tmp_path / "bare.las"
        save_las(cloud, bare)

        with pytest.raises(ValueError, match="No EPSG"):
            m3c2_to_raster(bare, bare, tmp_path / "m3c2.tif")


class TestStableM3c2Raster:
    """Rasterising already-computed stable-terrain distances, without re-running M3C2."""

    def test_stable_m3c2_raster(self, tmp_path: Path) -> None:
        import rasterio

        rng = np.random.default_rng(42)
        xy = np.column_stack([rng.uniform(486000, 486020, 500), rng.uniform(3099980, 3100000, 500)])
        distances = rng.normal(0.1, 0.02, 500)

        out = stable_m3c2_raster(xy, distances, tmp_path / "stable.tif", UTM45N)

        with rasterio.open(out) as src:
            arr = src.read(1)
            assert src.crs.to_epsg() == UTM45N
            # Each pixel is the median of the corepoints in it, so it tracks the input level.
            assert np.nanmedian(arr) == pytest.approx(0.1, abs=0.02)

    def test_stable_m3c2_raster__ignores_nan_corepoints(self, tmp_path: Path) -> None:
        import rasterio

        rng = np.random.default_rng(42)
        xy = np.column_stack([rng.uniform(486000, 486020, 300), rng.uniform(3099980, 3100000, 300)])
        distances = rng.normal(0.1, 0.02, 300)
        distances[:100] = np.nan  # corepoints with no valid M3C2 that date

        out = stable_m3c2_raster(xy, distances, tmp_path / "stable.tif", UTM45N)

        with rasterio.open(out) as src:
            assert np.isfinite(src.read(1)).any()

    def test_stable_m3c2_raster__coarser_res_gives_fewer_pixels(self, tmp_path: Path) -> None:
        import rasterio

        rng = np.random.default_rng(42)
        xy = np.column_stack([rng.uniform(486000, 486020, 500), rng.uniform(3099980, 3100000, 500)])
        d = rng.normal(0.1, 0.02, 500)

        fine = stable_m3c2_raster(xy, d, tmp_path / "a.tif", UTM45N, res=1.0)
        coarse = stable_m3c2_raster(xy, d, tmp_path / "b.tif", UTM45N, res=5.0)

        with rasterio.open(fine) as f, rasterio.open(coarse) as c:
            assert c.width < f.width

    def test_stable_m3c2_raster__all_nan_raises(self, tmp_path: Path) -> None:
        xy = np.column_stack([np.arange(10.0), np.arange(10.0)])

        with pytest.raises(ValueError, match="No valid .* distances"):
            stable_m3c2_raster(xy, np.full(10, np.nan), tmp_path / "s.tif", UTM45N)

    def test_stable_m3c2_raster__is_cached_unless_overwrite(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        xy = np.column_stack([rng.uniform(486000, 486020, 200), rng.uniform(3099980, 3100000, 200)])
        d = rng.normal(0.1, 0.02, 200)
        out = stable_m3c2_raster(xy, d, tmp_path / "s.tif", UTM45N)
        mtime = out.stat().st_mtime_ns

        stable_m3c2_raster(xy, d, out, UTM45N)
        assert out.stat().st_mtime_ns == mtime

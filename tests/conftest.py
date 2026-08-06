"""
Fixtures shared across the test suite.

Everything here is synthetic and lives in ``tmp_path`` — no Metashape licence, no ASP binaries, no real
glacier data. That is deliberate: the suite has to run on a bare CI runner, so the parts of the stack that
are proprietary or heavyweight are tested through their pure-Python seams instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

# Draw to an in-memory buffer, never to a window. Must be set before pyplot is imported anywhere,
# or every figure-producing test would block waiting for a display that CI does not have.
matplotlib.use("Agg")

# UTM 45N — the zone covering the Changri glaciers, and the CRS used throughout the tests.
UTM45N = 32645

# A small, well-conditioned grid reused by the raster and stack fixtures: 1 m pixels, origin at a
# plausible Khumbu easting/northing.
GRID_TRANSFORM = Affine(1.0, 0.0, 486000.0, 0.0, -1.0, 3100000.0)
GRID_SHAPE = (12, 16)


@pytest.fixture(autouse=True)
def isolate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Point ``Path.home()`` at a throwaway directory for every test.

    Several entry points cache to ``~/.cache/cntp_signalstack`` without exposing a ``cache_dir``
    argument — :func:`cntp.postprocessing.pixel_relative_accuracy` is one. Without this, running the
    suite would read and write the developer's real multi-gigabyte cube cache.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))


@pytest.fixture
def crs() -> CRS:
    """Return the CRS shared by every synthetic raster."""
    return CRS.from_epsg(UTM45N)


@pytest.fixture
def transform() -> Affine:
    """Return the affine transform shared by every synthetic raster."""
    return GRID_TRANSFORM


@pytest.fixture
def write_dem(tmp_path: Path, crs: CRS) -> Callable[..., Path]:
    """
    Return a factory writing a float32 GeoTIFF: ``write_dem(array, name) -> Path``.

    Every DEM defaults to the shared grid, so any two DEMs written by one test are pixel-aligned —
    which is what :func:`cntp.raster.build_dod` requires of its inputs.
    """

    def _write(
        array: Any,
        name: str = "dem.tif",
        transform: Affine = GRID_TRANSFORM,
        nodata: float = np.nan,
    ) -> Path:
        path = tmp_path / name
        arr = np.asarray(array, dtype="float32")
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=arr.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(arr, 1)
        return path

    return _write


def make_cloud(
    n: int = 2000,
    *,
    slope_deg: float = 75.0,
    grey: float = 120.0,
    blue: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Build an Nx9 point cloud (X, Y, Z, R, G, B, NX, NY, NZ) with controlled geometry and colour.

    Normals are synthesised so every point has *exactly* ``slope_deg`` slope under
    :func:`cntp.coreg.calculate_aspect_slope`. That lets a test place points deliberately either side of a
    slope threshold, rather than drawing a random spread and hoping it lands where the test needs it.

    RGB defaults to grey-brown with R > B, giving a negative NDWI, so the cloud survives the water filter
    in :func:`cntp.coreg.extract_stable_terrain`.
    """
    rng = np.random.default_rng(seed)
    xyz = rng.uniform([0, 0, 0], [20, 20, 10], (n, 3))

    rgb = np.empty((n, 3))
    rgb[:, 0] = grey
    rgb[:, 1] = grey
    rgb[:, 2] = grey * 0.5 if blue is None else blue

    # slope = atan(sqrt(nx^2 + ny^2) / nz), so fixing the horizontal:vertical ratio fixes the slope.
    theta = np.deg2rad(slope_deg)
    azimuth = rng.uniform(0, 2 * np.pi, n)
    horiz = np.sin(theta)
    normals = np.column_stack([horiz * np.cos(azimuth), horiz * np.sin(azimuth), np.full(n, np.cos(theta))])

    return np.column_stack([xyz, rgb, normals])


@pytest.fixture
def cloud() -> np.ndarray:
    """Return a 2000-point cloud that passes both the slope and the NDWI filter."""
    return make_cloud()


@pytest.fixture
def signal_stack_dir(tmp_path: Path, crs: CRS) -> tuple[Path, list[str], list[float]]:
    """
    Build an ``output/<date>/single_day/<date>_M3C2_raster.tif`` tree and return ``(root, dates, values)``.

    Three dates on one shared grid, each a constant plane plus one always-NaN pixel, so the temporal
    reductions have both a hand-computable answer and a known gap to blank out.
    """
    dates = ["2024-06-23", "2024-06-24", "2024-07-15"]
    values = [1.0, 2.0, 3.0]
    root = tmp_path / "site"

    for date, val in zip(dates, values):
        d = root / "output" / date / "single_day"
        d.mkdir(parents=True)
        arr = np.full(GRID_SHAPE, val, dtype="float32")
        arr[0, 0] = np.nan
        with rasterio.open(
            d / f"{date}_M3C2_raster.tif",
            "w",
            driver="GTiff",
            height=GRID_SHAPE[0],
            width=GRID_SHAPE[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=GRID_TRANSFORM,
            nodata=np.nan,
        ) as dst:
            dst.write(arr, 1)

    return root, dates, values


@pytest.fixture
def pipeline_output_dir(tmp_path: Path, crs: CRS) -> tuple[Path, list[str]]:
    """
    Build a complete synthetic pipeline output tree and return ``(root, dates)``.

    Mirrors what a finished run leaves on disk, which is what the whole of
    :mod:`cntp.postprocessing` reads::

        output/_ref_cache/ref_0.15_stable.las      cached stable-terrain reference
        output/_ref_cache/reference_ortho.tif      UAV + TLC orthomosaic
        output/<date>/single_day/<date>_M3C2_raster.tif
        output/<date>/coreg/<date>_m3c2_distances.npz    before/after corepoint distances
        output/<date>/coreg/<date>_m3c2_stats.csv        the reported med/nmad/std

    The stable reference and the distance arrays share one corepoint count by construction, which
    is the invariant every loader in that module checks before stacking.
    """
    from cntp.io import save_las

    rng = np.random.default_rng(42)
    dates = ["2024-06-23", "2024-07-07", "2024-07-21"]
    n_core = 400
    root = tmp_path / "site"

    # Cached stable reference: corepoints spread over ~20 x 20 m so a 1 m grid gets many cells.
    ref_cache = root / "output" / "_ref_cache"
    ref_cache.mkdir(parents=True)
    stable = make_cloud(n=n_core, slope_deg=75.0, seed=42)
    stable[:, 0] = rng.uniform(486000, 486020, n_core)
    stable[:, 1] = rng.uniform(3099980, 3100000, n_core)
    save_las(stable, ref_cache / "ref_0.15_stable.las", crs=UTM45N)

    # Reference orthomosaic (RGBA uint8) on the shared grid.
    with rasterio.open(
        ref_cache / "reference_ortho.tif",
        "w",
        driver="GTiff",
        height=GRID_SHAPE[0],
        width=GRID_SHAPE[1],
        count=4,
        dtype="uint8",
        crs=crs,
        transform=GRID_TRANSFORM,
    ) as dst:
        for b in range(1, 5):
            dst.write(np.full(GRID_SHAPE, 60 * b % 255, dtype="uint8"), b)

    for i, date in enumerate(dates):
        # Per-date signal raster, with a NaN gap so the coverage map has structure.
        arr = np.full(GRID_SHAPE, float(i + 1), dtype="float32")
        arr[0, :2] = np.nan
        write_raster(root / "output" / date / "single_day" / f"{date}_M3C2_raster.tif", arr, crs)

        # Per-date stable-terrain corepoint distances, 1:1 with the reference above.
        coreg = root / "output" / date / "coreg"
        coreg.mkdir(parents=True, exist_ok=True)
        before = rng.normal(0.4, 0.3, n_core).astype("float32")
        after = rng.normal(0.0, 0.05 * (i + 1), n_core).astype("float32")
        after[:5] = np.nan  # corepoints this date did not observe
        np.savez(coreg / f"{date}_m3c2_distances.npz", before=before, after=after)

        nmad_after = float(1.4826 * np.nanmedian(np.abs(after - np.nanmedian(after))))
        (coreg / f"{date}_m3c2_stats.csv").write_text(
            "coreg,med,nmad,std\n"
            f"before,0.4,0.44,0.30\n"
            f"after,0.0,{nmad_after},{float(np.nanstd(after))}\n"
        )

    return root, dates


@pytest.fixture
def write_cloud_las(tmp_path: Path) -> Callable[..., Path]:
    """
    Return a factory writing a georeferenced LAS over the shared grid footprint.

    ``write_cloud_las(name, dz=0.0, n=3000) -> Path``. The surface is a gentle tilted plane so
    cubic interpolation has a well-posed answer, and the EPSG is embedded in the header because
    :func:`cntp.raster.build_dem_and_ortho` reads the CRS from there when none is passed.
    """
    from cntp.io import save_las

    x0, y0 = GRID_TRANSFORM.c, GRID_TRANSFORM.f - GRID_SHAPE[0]

    def _write(name: str = "cloud.las", dz: float = 0.0, n: int = 3000, seed: int = 42) -> Path:
        rng = np.random.default_rng(seed)
        x = rng.uniform(x0, x0 + GRID_SHAPE[1], n)
        y = rng.uniform(y0, y0 + GRID_SHAPE[0], n)
        z = 5000.0 + 0.1 * (x - x0) + 0.05 * (y - y0) + dz

        rgb = np.empty((n, 3))
        rgb[:, 0] = 120.0
        rgb[:, 1] = 120.0
        rgb[:, 2] = 60.0

        theta = np.deg2rad(75.0)
        az = rng.uniform(0, 2 * np.pi, n)
        normals = np.column_stack(
            [np.sin(theta) * np.cos(az), np.sin(theta) * np.sin(az), np.full(n, np.cos(theta))]
        )

        path = tmp_path / name
        save_las(np.column_stack([x, y, z, rgb, normals]), path, crs=UTM45N)
        return path

    return _write


def write_raster(path: Path, array: np.ndarray, crs: CRS, transform: Affine = GRID_TRANSFORM) -> Path:
    """Write a single-band float32 GeoTIFF, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(arr, 1)
    return path

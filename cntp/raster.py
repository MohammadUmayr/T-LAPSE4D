"""Raster-domain operations on point clouds and DEMs.

Everything that produces or consumes a 2-D georeferenced raster lives here:

- GeoTIFF I/O           : :func:`save_dem`, :func:`save_ortho`
- Cubic interpolation   : :func:`interpolate_and_mask`
- Cloud → DEM + ortho   : :func:`build_dem_and_ortho`,
                          :func:`build_reference_dem_and_ortho`
- DEM differencing      : :func:`build_dod`
- Raster stable terrain : :func:`extract_stable_terrain_from_dem`
- M3C2 → raster         : :func:`m3c2_to_raster`

Point-cloud I/O (LAS/LAZ) stays in :mod:`cntp.io`. Point-cloud coreg + M3C2
distance computation stays in :mod:`cntp.coreg`. ASP ``point2dem`` wrapper
lives in ``contributors/umayr/tools.py``.
"""

from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import py4dgeo
import rasterio
import rasterio.features
import rasterio.transform
from rasterio.errors import RasterioIOError
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from cntp.coreg import _NDWI_A, _NDWI_B, run_m3c2
from cntp.io import load_las, read_las_bounds

# ---------------------------------------------------------------------------
# GeoTIFF I/O
# ---------------------------------------------------------------------------

def save_dem(
    array: np.ndarray,
    filename: str | Path,
    crs_epsg: "rasterio.crs.CRS | str | int",
    transform: "rasterio.Affine",
) -> None:
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs_epsg,
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(array, 1)
    print("Saved:", filename)


def save_ortho(
    x: np.ndarray,
    y: np.ndarray,
    rgb: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
    res: float,
    max_gap_pixels: float,
    filename: str | Path,
    crs_epsg: "rasterio.crs.CRS | str | int",
    transform: "rasterio.Affine",
) -> None:
        # Precompute DEM pixel centers
        pixels_xy = np.column_stack([xi.ravel(), yi.ravel()])

        # Build KD-tree in XY
        tree = cKDTree(np.column_stack([x, y]))

        # Nearest neighbour lookup (multithreaded query — same result)
        dist, idx = tree.query(pixels_xy, k=1, distance_upper_bound=res * max_gap_pixels,
                               workers=-1)
        # Mask out empty pixels
        mask = np.isfinite(dist)

        # Prepare output arrays
        r_img = np.zeros(idx.shape, dtype=np.uint8)
        g_img = np.zeros(idx.shape, dtype=np.uint8)
        b_img = np.zeros(idx.shape, dtype=np.uint8)

        r_img[mask] = rgb[idx[mask], 0]
        g_img[mask] = rgb[idx[mask], 1]
        b_img[mask] = rgb[idx[mask], 2]

        # Stack into a single image array
        H, W = xi.shape
        ortho = np.dstack([
            r_img.reshape(H, W),
            g_img.reshape(H, W),
            b_img.reshape(H, W)
        ])

        with rasterio.open(
            filename,
            "w",
            driver="GTiff",
            height=ortho.shape[0],
            width=ortho.shape[1],
            count=3,                           # three bands!
            dtype=ortho.dtype,
            crs=crs_epsg,
            transform=transform,
            nodata=0
        ) as dst:
            dst.write(ortho[:, :, 0], 1)  # Red
            dst.write(ortho[:, :, 1], 2)  # Green
            dst.write(ortho[:, :, 2], 3)  # Blue

        print("Saved orthoimage:", filename)


# ---------------------------------------------------------------------------
# Cubic griddata + KDTree gap mask
# ---------------------------------------------------------------------------

def interpolate_and_mask(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
    res: float,
    max_gap_pixels: float,
) -> np.ndarray:
    # Interpolation (cubic)
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    # Clough-Tocher cubic doesn't preserve monotonicity — at steep features
    # (cliffs, cone walls) the interpolant can overshoot far beyond either
    # endpoint. Cap to the input cloud's actual Z range so overshoots that
    # exceed physically present elevations are pulled back in.
    zi = np.clip(zi, np.nanmin(z), np.nanmax(z))

    # Distance to nearest real point
    tree = cKDTree(np.column_stack((x, y)))
    dist, _ = tree.query(np.column_stack((xi.ravel(), yi.ravel())), k=1)
    dist = dist.reshape(xi.shape)

    # Mask far pixels
    mask = dist > (max_gap_pixels * res)
    zi_masked = np.where(mask, np.nan, zi)

    return zi_masked


# ---------------------------------------------------------------------------
# Cloud → DEM + orthoimage
# ---------------------------------------------------------------------------

def build_dem_and_ortho(
    cloud_las: str | Path,
    ref_las: str | Path,
    out_dir: str | Path,
    name_stem: str,
    res: float = 1.0,
    max_gap_pixels: int = 1,
    utm_epsg: int | None = None,
    cloud_downsample: float = 1.0,
    overwrite: bool = False,
) -> tuple:
    """Build a DEM and orthoimage from a co-registered point cloud on the reference's grid.

    The output grid is anchored to the reference cloud's XY bounding box
    (read from the LAS header — no points loaded), so every day's DEM/ortho
    lands on the same pixel raster and stacks/differences directly.

    Both rasters are written in one pass — the cloud is loaded once and
    `(x, y)` is shared between :func:`interpolate_and_mask` (DEM, cubic
    griddata + KDTree gap mask) and :func:`save_ortho` (RGB nearest-
    neighbour).

    Parameters
    ----------
    cloud_las : str | Path
        Co-registered LAS/LAZ — typically ``aligned_las`` from Step 3 of
        :func:`cntp.pipeline_4dsfm.run_4dsfm_day`.
    ref_las : str | Path
        Original (non-downsampled) reference cloud. Only its header bbox is
        read — no points are loaded.
    out_dir : str | Path
        Destination directory. Files written are
        ``<name_stem>_dem.tif`` and ``<name_stem>_ortho.tif``.
    name_stem : str
        Filename stem (e.g. the date string ``"2023-09-15"``).
    res : float
        Pixel size in metres (default 1.0).
    max_gap_pixels : int
        Pixels further than ``max_gap_pixels * res`` from any cloud point are
        nulled (DEM) or left as nodata (ortho). Default 1.
    utm_epsg : int, optional
        EPSG code of the output raster CRS. When ``None``, parsed from
        ``cloud_las``'s LAS header (Metashape writes it on export).
    cloud_downsample : float
        Fraction of cloud points to keep (0 < f ≤ 1). Default 1.0. Pass the
        same value as ``tba_downsample`` in the coreg step for parity with
        the ICP basis.
    overwrite : bool
        When False (default), skip computation entirely if both
        ``<name_stem>_dem.tif`` and ``<name_stem>_ortho.tif`` already exist
        in *out_dir*. When True, force a rebuild.

    Returns
    -------
    (dem_path, ortho_path) : tuple
        Paths to the written GeoTIFFs. If a write fails with permission
        denied, that path is returned as ``None``.
    """
    cloud_las = Path(cloud_las)
    ref_las   = Path(ref_las)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dem_path   = out_dir / f"{name_stem}_dem.tif"
    ortho_path = out_dir / f"{name_stem}_ortho.tif"

    if not overwrite and dem_path.exists() and ortho_path.exists():
        print(f"  DEM + ortho cached → {dem_path.name}, {ortho_path.name}")
        return dem_path, ortho_path

    # ── 1. Grid from the reference cloud's header bbox (no points loaded) ──
    ref_min, ref_max = read_las_bounds(ref_las)
    xmin, xmax = float(ref_min[0]), float(ref_max[0])
    ymin, ymax = float(ref_min[1]), float(ref_max[1])

    xi = np.arange(xmin, xmax, res)
    yi = np.arange(ymax, ymin, -res)
    xi, yi = np.meshgrid(xi, yi)

    transform = rasterio.transform.from_origin(xmin, ymax, res, res)

    # ── 2. Resolve CRS (param wins, else read from cloud LAS header) ───────
    if utm_epsg is None:
        with laspy.open(cloud_las) as f:
            crs = f.header.parse_crs()
        if crs is None or crs.to_epsg() is None:
            raise ValueError(
                f"No EPSG in {cloud_las.name} header; pass utm_epsg explicitly."
            )
        utm_epsg = crs.to_epsg()
    crs_epsg = f"EPSG:{utm_epsg}"

    # ── 3. Load + XY-clip the slave cloud to the reference footprint ───────
    cloud = load_las(cloud_las, downsample_factor=cloud_downsample)
    in_box = ((cloud[:, 0] >= xmin) & (cloud[:, 0] <= xmax) &
              (cloud[:, 1] >= ymin) & (cloud[:, 1] <= ymax))
    cloud = cloud[in_box]
    print(f"  Loaded {len(cloud):,} pts after XY clip "
          f"(downsample={cloud_downsample}, grid {xi.shape[1]}×{xi.shape[0]} @ {res} m, "
          f"{crs_epsg})")

    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    rgb     = cloud[:, 3:6]

    # A write refused by a read-only mount is reported as None rather than raising, so a batch run
    # over many dates isn't aborted by one unwritable destination.
    dem_out:   Path | None = dem_path
    ortho_out: Path | None = ortho_path

    # ── 4. DEM (cubic griddata + KDTree gap mask) ──────────────────────────
    zi = interpolate_and_mask(x, y, z, xi, yi, res, max_gap_pixels)
    try:
        save_dem(zi, dem_path, crs_epsg, transform)
    except RasterioIOError as e:
        if "Permission denied" in str(e):
            print(f"  Skipping DEM (permission denied): {dem_path}")
            dem_out = None
        else:
            raise

    # ── 5. Orthoimage (KDTree nearest-neighbour RGB) ───────────────────────
    try:
        save_ortho(x, y, rgb, xi, yi, res, max_gap_pixels,
                   ortho_path, crs_epsg, transform)
    except RasterioIOError as e:
        if "Permission denied" in str(e):
            print(f"  Skipping ortho (permission denied): {ortho_path}")
            ortho_out = None
        else:
            raise

    return dem_out, ortho_out


def build_dem_and_ortho_p2d(
    cloud_las: str | Path,
    ref_las: str | Path,
    out_dir: str | Path,
    name_stem: str,
    res: float = 1.0,
    max_gap_pixels: int = 1,
    utm_epsg: int | None = None,
    cloud_downsample: float = 1.0,
    overwrite: bool = False,
    verbose: bool = False,
) -> tuple:
    """DEM via ASP ``point2dem`` (HSfM method) + ortho via KDTree NN, shared grid.

    Drop-in alternative to :func:`build_dem_and_ortho` that rasterises the DEM
    with ASP ``point2dem`` instead of cubic ``griddata``. This is the published
    HSfM DEM method: point2dem's default ``weighted_average`` filter (Gaussian
    distance weighting — what the HSfM paper calls "IDW") with
    ``--search-radius-factor = max_gap_pixels`` (= 1 grid cell). Cells with no
    point inside the search radius are left as nodata rather than interpolated,
    avoiding artifacts over large data gaps. C++, multithreaded, streaming —
    seconds rather than minutes, no Delaunay / Clough-Tocher overshoot.

    The DEM grid is pinned to *ref_las*'s footprint (``--t_projwin``) so day and
    reference DEMs share a grid and difference without resampling. The ortho is
    built with the *same* :func:`save_ortho` (nearest-neighbour RGB) as
    :func:`build_dem_and_ortho`, on **point2dem's output grid** (read back from
    the DEM GeoTIFF), so DEM and ortho share an identical pixel grid (required
    by :func:`extract_stable_terrain_from_dem`).

    Note: ``point2dem`` yields a grid one row taller than
    :func:`build_dem_and_ortho`'s ``np.arange`` grid for the same bbox (cell-edge
    snapping). A DEM from this function must be differenced only against another
    DEM from this function — do not mix with cubic DEMs. ``cloud_downsample``
    applies only to the **ortho** load; ``point2dem`` streams the full cloud for
    the DEM regardless.

    Parameters / returns mirror :func:`build_dem_and_ortho`.
    """
    from cntp.asp import point2dem  # local import — avoids any import cycle

    cloud_las = Path(cloud_las)
    ref_las   = Path(ref_las)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dem_path   = out_dir / f"{name_stem}_dem.tif"
    ortho_path = out_dir / f"{name_stem}_ortho.tif"
    if not overwrite and dem_path.exists() and ortho_path.exists():
        print(f"  DEM + ortho cached → {dem_path.name}, {ortho_path.name}")
        return dem_path, ortho_path

    # ── 1. DEM via point2dem (HSfM), grid pinned to ref_las ──────────────
    asp_dem = point2dem(
        cloud_las, out_prefix=out_dir / name_stem, res=res, utm_epsg=utm_epsg,
        max_gap_pixels=max_gap_pixels, ref_las=ref_las, verbose=verbose,
    )
    Path(asp_dem).replace(dem_path)    # "<stem>-DEM.tif" → "<stem>_dem.tif"

    # ── 2. Read point2dem's grid → pixel-centre coordinates ──────────────
    with rasterio.open(dem_path) as src:
        transform = src.transform
        W, H      = src.width, src.height
        crs_epsg  = src.crs.to_string()
    xmin, ymax = transform.c, transform.f
    xs = xmin + (np.arange(W) + 0.5) * transform.a     # transform.a = +res
    ys = ymax + (np.arange(H) + 0.5) * transform.e     # transform.e = -res
    xi, yi = np.meshgrid(xs, ys)
    xmax = xmin + W * transform.a
    ymin = ymax + H * transform.e

    # ── 3. Load + XY-clip the cloud for the ortho ────────────────────────
    cloud = load_las(cloud_las, downsample_factor=cloud_downsample)
    inb = ((cloud[:, 0] >= xmin) & (cloud[:, 0] <= xmax) &
           (cloud[:, 1] >= ymin) & (cloud[:, 1] <= ymax))
    cloud = cloud[inb]
    print(f"  point2dem DEM {W}x{H} → ortho from {len(cloud):,} pts", flush=True)

    # ── 4. Ortho via the same (parallel) save_ortho, on point2dem's grid ─
    save_ortho(cloud[:, 0], cloud[:, 1], cloud[:, 3:6], xi, yi, res, max_gap_pixels,
               str(ortho_path), crs_epsg, transform)

    return dem_path, ortho_path


def build_reference_dem_and_ortho(
    ref_cloud_path: str | Path,
    cache_dir: str | Path,
    res: float = 1.0,
    max_gap_pixels: int = 1,
    utm_epsg: int | None = None,
    cloud_downsample: float = 1.0,
    overwrite: bool = False,
) -> tuple:
    """Build the reference DEM + orthoimage once, cache under *cache_dir*.

    Thin wrapper over :func:`build_dem_and_ortho` with ``cloud_las == ref_las``.
    The reference rasters live alongside the other shared reference artefacts
    in ``_ref_cache/`` and are reused across every day's processing — they
    only need to be computed once.

    Skips computation entirely if both output files already exist.

    Parameters
    ----------
    ref_cloud_path : str | Path
        Reference LAS/LAZ. Pass the original cloud for highest fidelity, or
        the pipeline's cached ``_ref_cache/<stem>_ds<f>.las`` for a faster
        build that's consistent with the ICP basis.
    cache_dir : str | Path
        Destination — typically ``output/_ref_cache/``.
    res, max_gap_pixels, utm_epsg :
        See :func:`build_dem_and_ortho`.
    cloud_downsample : float
        Additional downsample applied at load time on top of whatever the
        input file already represents (0 < f ≤ 1). Default 1.0 (no extra
        downsample). Use a smaller value when the input is large enough to
        OOM ``griddata``. The output grid spacing is set by ``res`` —
        downsampling further only thins the source points fed to the
        cubic interpolation.
    overwrite : bool
        When False (default), skip computation if both outputs already
        exist in *cache_dir*. When True, force a rebuild.

    Returns
    -------
    (dem_path, ortho_path) : tuple
        ``cache_dir/reference_dem.tif`` and ``cache_dir/reference_ortho.tif``.
        Filenames are fixed — if you change ``cloud_downsample`` or ``res``
        between runs, delete these manually (or set ``overwrite=True``) to
        regenerate.
    """
    return build_dem_and_ortho(
        cloud_las        = Path(ref_cloud_path),
        ref_las          = Path(ref_cloud_path),
        out_dir          = cache_dir,
        name_stem        = "reference",
        res              = res,
        max_gap_pixels   = max_gap_pixels,
        utm_epsg         = utm_epsg,
        cloud_downsample = cloud_downsample,
        overwrite        = overwrite,
    )


# ---------------------------------------------------------------------------
# DEM differencing
# ---------------------------------------------------------------------------

def build_dod(
    ref_dem_path: str | Path,
    day_dem_path: str | Path,
    out_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Compute a DEM of Difference: ``day_dem - ref_dem`` on the common grid.

    Both DEMs must share the same shape, affine transform, and CRS — which
    they do when both were built via :func:`build_dem_and_ortho` using the
    same reference cloud and the same ``res``. Pixels where *either* DEM
    holds nodata propagate to NaN in the output, so the DoD is naturally
    masked to the intersection of valid coverage (typically the day's
    footprint inside the reference's wider footprint).

    Sign convention (standard glaciology): **positive = day surface is
    higher than the reference** (accumulation / gain), **negative = day
    surface is lower** (melt / loss). Matches the py4dgeo M3C2 sign on
    roughly horizontal terrain.

    Parameters
    ----------
    ref_dem_path : str | Path
        Reference DEM GeoTIFF (e.g. ``_ref_cache/<stem>_dem.tif``).
    day_dem_path : str | Path
        Day's co-registered DEM GeoTIFF
        (e.g. ``<output>/<date>/single_day/<date>_dem.tif``).
    out_path : str | Path, optional
        Output GeoTIFF. Default ``<day_dem_path.parent>/DOD.tif``.
    overwrite : bool
        When False (default), skip if *out_path* exists.

    Returns
    -------
    Path
        Path to the written DoD GeoTIFF.
    """
    ref_dem_path = Path(ref_dem_path)
    day_dem_path = Path(day_dem_path)
    if out_path is None:
        out_path = day_dem_path.parent / "DOD.tif"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and out_path.exists():
        print(f"  DoD cached → {out_path.name}")
        return out_path

    with rasterio.open(ref_dem_path) as ref_src, rasterio.open(day_dem_path) as day_src:
        if ref_src.shape != day_src.shape:
            raise ValueError(
                f"DEM shapes differ — ref {ref_src.shape} vs day {day_src.shape}. "
                "Both must be built on the same reference cloud grid."
            )
        if ref_src.transform != day_src.transform:
            raise ValueError(
                f"DEM affine transforms differ — pixels are not aligned.\n"
                f"  ref : {ref_src.transform}\n  day : {day_src.transform}"
            )
        if ref_src.crs != day_src.crs:
            raise ValueError(
                f"DEM CRS differ: ref {ref_src.crs} vs day {day_src.crs}."
            )
        ref_arr   = ref_src.read(1).astype(np.float64)
        day_arr   = day_src.read(1).astype(np.float64)
        ref_nodata = ref_src.nodata
        day_nodata = day_src.nodata
        transform = day_src.transform
        crs_epsg  = day_src.crs.to_string()

    # Replace numeric nodata sentinels (e.g. ASP's -9999) with NaN so the
    # subtraction below propagates correctly. Sources already using NaN as
    # nodata are unaffected — `arr == NaN` is False everywhere, so np.where
    # is a no-op for them.
    if ref_nodata is not None and not np.isnan(ref_nodata):
        ref_arr = np.where(ref_arr == ref_nodata, np.nan, ref_arr)
    if day_nodata is not None and not np.isnan(day_nodata):
        day_arr = np.where(day_arr == day_nodata, np.nan, day_arr)

    # NaN propagates naturally — pixels with nodata in either source become NaN.
    dod = day_arr - ref_arr
    save_dem(dod, out_path, crs_epsg, transform)

    n_valid = int(np.isfinite(dod).sum())
    n_total = int(dod.size)
    print(f"  DoD valid pixels : {n_valid:,}/{n_total:,} "
          f"({100 * n_valid / n_total:.1f}%)")
    return out_path


# ---------------------------------------------------------------------------
# Raster stable-terrain extraction (slope + NDWI + glacier polygon)
# ---------------------------------------------------------------------------

def extract_stable_terrain_from_dem(
    dem_path: str | Path,
    ortho_path: str | Path | None = None,
    glacier_mask_path: str | Path | None = None,
    slope_threshold: float = 60.0,
    out_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Raster analogue of ``cntp.coreg.extract_stable_terrain`` + glacier mask.

    Combines three filters at the DEM's pixel grid and keeps only pixels that
    pass all three:

    1. **Slope filter** — slope, computed as ``arctan(|∇z|)``, must exceed
       ``slope_threshold`` (default 60°). Geometrically steep ⇒ bedrock /
       stable terrain.
    2. **NDWI + intensity filter** — uses the matching ``_ortho.tif`` (same
       grid). NDWI = ``(B - R) / (R + B)``; intensity = ``mean(RGB)``. A pixel
       passes if ``intensity - (NDWI*_NDWI_A + _NDWI_B) < 0`` — same line as
       :func:`cntp.coreg.extract_stable_terrain` (drops water + snow surfaces).
    3. **Glacier polygon mask** — pixels inside the glacier polygon are
       excluded.

    Pixels passing all three keep their DEM value; everything else becomes
    NaN.

    Parameters
    ----------
    dem_path : str | Path
        Single-band DEM GeoTIFF (e.g. ``reference_dem.tif`` or
        ``<date>_dem.tif``).
    ortho_path : str | Path, optional
        3-band RGB GeoTIFF on the **same grid** as ``dem_path`` — built
        together by :func:`build_dem_and_ortho`. When ``None`` the
        NDWI + intensity filter is skipped (only slope + glacier polygon
        are applied). Useful when the DEM was produced via ASP
        ``point2dem``, which does not emit a companion ortho.
    glacier_mask_path : str | Path, optional
        Shapefile with glacier polygon(s) in the same CRS as the raster.
        When ``None``, the glacier polygon filter is skipped.
    slope_threshold : float
        Minimum slope angle [°] for the slope filter. Default 60.
    out_path : str | Path, optional
        Output GeoTIFF. Default ``<dem_path.stem>_stable.tif`` next to the DEM.
    overwrite : bool
        When False (default), skip if *out_path* exists.

    Returns
    -------
    Path
        Path to the written stable-terrain DEM GeoTIFF.
    """
    dem_path = Path(dem_path)
    if out_path is None:
        out_path = dem_path.with_name(f"{dem_path.stem}_stable{dem_path.suffix}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and out_path.exists():
        print(f"  Stable DEM cached → {out_path.name}")
        return out_path

    # ── Read DEM ───────────────────────────────────────────────────────
    with rasterio.open(dem_path) as src_dem:
        z = src_dem.read(1).astype(np.float64)
        if src_dem.nodata is not None and not np.isnan(src_dem.nodata):
            z = np.where(z == src_dem.nodata, np.nan, z)
        transform = src_dem.transform
        profile = src_dem.profile

    # ── 1. Slope filter — gradient of DEM ──────────────────────────────
    # transform.a = x-pixel width, transform.e = y-pixel height (usually
    # negative because Y descends top→bottom in GeoTIFF). Use absolute
    # values so the slope magnitude is independent of axis direction.
    px = abs(transform.a)
    py = abs(transform.e)
    dz_dy, dz_dx = np.gradient(z, py, px)
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    slope_mask = slope_deg > slope_threshold

    # ── 2. NDWI + intensity filter (optional — needs ortho on same grid) ──
    if ortho_path is not None:
        ortho_path = Path(ortho_path)
        with rasterio.open(ortho_path) as src_ortho:
            if src_ortho.count < 3:
                raise ValueError(
                    f"Expected an RGB ortho (≥3 bands), got {src_ortho.count} "
                    f"in {ortho_path.name}."
                )
            if src_ortho.shape != z.shape:
                raise ValueError(
                    f"DEM and ortho shapes differ: DEM {z.shape} vs ortho "
                    f"{src_ortho.shape}. Both must share the same pixel grid "
                    "(built together by build_dem_and_ortho)."
                )
            r = src_ortho.read(1).astype(np.float64)
            g = src_ortho.read(2).astype(np.float64)
            b = src_ortho.read(3).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ndwi      = (b - r) / (r + b)
            grayscale = (r + g + b) / 3.0
        ndwi_mask = (grayscale - (ndwi * _NDWI_A + _NDWI_B)) < 0
    else:
        ndwi_mask = np.ones_like(z, dtype=bool)  # NDWI filter disabled

    # ── 3. Glacier polygon mask (optional) ─────────────────────────────
    if glacier_mask_path is not None:
        glacier_mask_path = Path(glacier_mask_path)
        gdf = gpd.read_file(glacier_mask_path)
        # geometry_mask: invert=True ⇒ True where pixel is INSIDE polygon.
        # We want pixels OUTSIDE the glacier (matches apply_glacier_mask's
        # "keep outside" semantics for clouds).
        inside_glacier = rasterio.features.geometry_mask(
            gdf.geometry,
            out_shape=z.shape,
            transform=transform,
            invert=True,
        )
        outside_glacier = ~inside_glacier
    else:
        outside_glacier = np.ones_like(z, dtype=bool)  # glacier filter disabled

    # ── Combine ─────────────────────────────────────────────────────────
    stable_mask = slope_mask & ndwi_mask & outside_glacier
    # Drop pixels that were already nodata in the DEM (NaN in z).
    stable_mask &= np.isfinite(z)

    z_stable = np.where(stable_mask, z, np.nan)

    profile.update(dtype="float64", nodata=np.nan, count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(z_stable, 1)

    n_valid = int(stable_mask.sum())
    n_total = int(stable_mask.size)
    print(f"  Stable terrain (DEM) : {n_valid:,}/{n_total:,} pixels "
          f"({100 * n_valid / n_total:.1f}%) — slope>{slope_threshold:.0f}°, "
          f"non-glacier, non-water/snow → {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# M3C2 distances → 1 m raster
# ---------------------------------------------------------------------------

def m3c2_to_raster(
    ref_las: str | Path,
    day_las: str | Path,
    out_path: str | Path,
    grid_anchor_las: str | Path | None = None,
    res: float = 1.0,
    utm_epsg: int | None = None,
    normal_radii: float = 2.5,
    cyl_radius: float = 2.5,
    max_distance: float = 30.0,
    ref_downsample: float = 1.0,
    day_downsample: float = 1.0,
    overwrite: bool = False,
) -> Path:
    """Rasterise M3C2 perpendicular distances between two clouds onto a 1 m grid.

    Companion to :func:`build_dod` — but instead of vertical ``day_z − ref_z``,
    each pixel holds the **median M3C2 distance** between the two clouds at
    that XY location. Because M3C2 measures perpendicular cloud-to-cloud
    separation along the local surface normal (not along Z), the result is
    **immune to the slope-projection inflation** that pollutes vertical DoD
    on steep terrain. Sign convention matches the flipped ``build_dod``:
    positive = day above ref = gain / accumulation; negative = day below
    ref = loss / melt.

    No masking is applied — both clouds are used in full so the output is
    the pure cloud-to-cloud distance field.

    Workflow:

    1. Load *ref_las* and *day_las* (optionally downsample).
    2. Build py4dgeo epochs and run :func:`cntp.coreg.run_m3c2`.
    3. Each corepoint (= reference cloud point) gets one M3C2 distance.
    4. Bin the corepoint XY positions into a 1 m grid anchored to
       *grid_anchor_las*'s XY bbox (or *ref_las*'s bbox if not given).
    5. Take the **median** distance per cell.
    6. Write a Float64 GeoTIFF with ``nodata=NaN`` via :func:`save_dem` —
       same format as the DoD raster, so the two stack cleanly in QGIS.

    Parameters
    ----------
    ref_las : str | Path
        Reference LAS/LAZ. Corepoints are this cloud's points.
    day_las : str | Path
        Day's coregistered LAS/LAZ.
    out_path : str | Path
        Output GeoTIFF.
    grid_anchor_las : str | Path, optional
        Cloud whose XY bbox defines the grid extent. Default: *ref_las*.
        Pass the original full-resolution reference cloud to get a grid
        identical to the one used by :func:`build_dem_and_ortho`, so the
        M3C2 raster sits cell-for-cell on the same grid as your DEMs.
    res : float
        Pixel size in metres. Default 1.0.
    utm_epsg : int, optional
        EPSG code for the output CRS. When ``None``, read from *ref_las*'s
        LAS header.
    normal_radii, cyl_radius, max_distance : float
        py4dgeo M3C2 parameters. Defaults match :func:`cntp.coreg.run_m3c2`
        — the same values used by every other M3C2 call in the project
        (Step 3b evaluate_coreg, Step 6b validation).
    ref_downsample, day_downsample : float
        Fraction of points to keep at load time. Default 1.0 (no extra
        downsampling). Use this to cap RAM — M3C2 internally builds KDTrees
        whose memory scales with the cloud size.
    overwrite : bool
        When False (default), skip if *out_path* exists.

    Returns
    -------
    Path
        *out_path*.
    """
    ref_las  = Path(ref_las)
    day_las  = Path(day_las)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and out_path.exists():
        print(f"  M3C2 raster cached → {out_path.name}")
        return out_path

    # ── Resolve grid + CRS ─────────────────────────────────────────────
    anchor = Path(grid_anchor_las) if grid_anchor_las is not None else ref_las
    a_min, a_max = read_las_bounds(anchor)
    # Defensive min/max — the LAS spec doesn't require header.mins <=
    # header.maxs (the original Reference_UAV_TLC_PCS.laz stores Y reversed).
    x0, x1 = float(a_min[0]), float(a_max[0])
    y0, y1 = float(a_min[1]), float(a_max[1])
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)

    if utm_epsg is None:
        with laspy.open(ref_las) as f:
            crs = f.header.parse_crs()
        if crs is None or crs.to_epsg() is None:
            raise ValueError(
                f"No EPSG in {ref_las.name} header; pass utm_epsg explicitly."
            )
        utm_epsg = crs.to_epsg()
    crs_epsg = f"EPSG:{utm_epsg}"

    nx = int(np.ceil((xmax - xmin) / res))
    ny = int(np.ceil((ymax - ymin) / res))
    transform = rasterio.transform.from_origin(xmin, ymax, res, res)

    # ── Load clouds ────────────────────────────────────────────────────
    print(f"  Loading reference cloud (downsample={ref_downsample}) …")
    ref_arr = load_las(ref_las, downsample_factor=ref_downsample)

    print(f"  Loading day cloud (downsample={day_downsample}) …")
    day_arr = load_las(day_las, downsample_factor=day_downsample)

    print(f"  Ref pts : {len(ref_arr):,}   |   Day pts : {len(day_arr):,}")

    # ── Run M3C2 ───────────────────────────────────────────────────────
    epoch_ref = py4dgeo.Epoch(ref_arr[:, :3])
    epoch_day = py4dgeo.Epoch(day_arr[:, :3])
    print(f"  Running M3C2 (normal_radii={normal_radii} m, "
          f"cyl_radius={cyl_radius} m, max_distance={max_distance} m) …")
    med, _, std, distances = run_m3c2(
        epoch_ref, epoch_day,
        normal_radii = normal_radii,
        cyl_radius   = cyl_radius,
        max_distance = max_distance,
    )
    print(f"  M3C2 over all corepoints: median={med:+.2f} m   std={std:.2f} m")

    # ── Bin corepoints → mean per cell ─────────────────────────────────
    xs = ref_arr[:, 0]
    ys = ref_arr[:, 1]
    rows = ((ymax - ys) / res).astype(np.int64)
    cols = ((xs - xmin) / res).astype(np.int64)

    in_grid = (
        (rows >= 0) & (rows < ny) &
        (cols >= 0) & (cols < nx) &
        np.isfinite(distances)
    )
    df = pd.DataFrame({
        "row": rows[in_grid],
        "col": cols[in_grid],
        "d":   distances[in_grid],
    })
    print(f"  Binning {len(df):,} valid corepoints into {ny}×{nx} grid (median per cell) …")
    medians = df.groupby(["row", "col"])["d"].median()

    grid = np.full((ny, nx), np.nan, dtype=np.float64)
    grid[medians.index.get_level_values(0).values,
         medians.index.get_level_values(1).values] = medians.values

    # ── Write GeoTIFF ──────────────────────────────────────────────────
    save_dem(grid, out_path, crs_epsg, transform)

    n_valid = int(np.isfinite(grid).sum())
    n_total = int(grid.size)
    print(f"  M3C2 raster valid pixels : {n_valid:,}/{n_total:,} "
          f"({100 * n_valid / n_total:.1f}%) → {out_path.name}")
    return out_path


def stable_m3c2_raster(
    ref_stable: np.ndarray,
    distances: np.ndarray,
    out_path: str | Path,
    utm_epsg: int,
    res: float = 1.0,
    overwrite: bool = False,
) -> Path:
    """Rasterise **pre-computed** stable-terrain M3C2 distances onto a grid.

    Companion to :func:`m3c2_to_raster`, but for an already-computed distance
    array (e.g. the after-co-registration residual from
    :func:`cntp.asp.evaluate_coreg`) — so **no M3C2 is recomputed**. Each pixel
    is the **median** distance of the corepoints that fall in it; empty cells
    are NaN. Use it to map co-registration **uncertainty over stable terrain**
    and stack the per-date rasters across a season.

    Parameters
    ----------
    ref_stable : np.ndarray
        (N, >=2) corepoint coordinates; columns 0/1 are easting / northing in
        the ``utm_epsg`` CRS (same array M3C2 was run on).
    distances : np.ndarray
        (N,) M3C2 distances aligned with ``ref_stable`` (NaN where invalid).
    out_path : str | Path
        Output GeoTIFF.
    utm_epsg : int
        EPSG code of the corepoint CRS (written into the raster).
    res : float
        Pixel size in metres. Default 1.0.
    overwrite : bool
        When False (default), skip if *out_path* already exists.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and out_path.exists():
        print(f"  Stable M3C2 raster cached → {out_path.name}")
        return out_path

    xs = np.asarray(ref_stable[:, 0], dtype=np.float64)
    ys = np.asarray(ref_stable[:, 1], dtype=np.float64)
    d  = np.asarray(distances, dtype=np.float64)
    valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(d)
    xs, ys, d = xs[valid], ys[valid], d[valid]
    if xs.size == 0:
        raise ValueError("No valid (finite) stable-terrain distances to rasterise.")

    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    nx = max(int(np.ceil((xmax - xmin) / res)), 1)
    ny = max(int(np.ceil((ymax - ymin) / res)), 1)
    transform = rasterio.transform.from_origin(xmin, ymax, res, res)

    rows = np.clip(((ymax - ys) / res).astype(np.int64), 0, ny - 1)
    cols = np.clip(((xs - xmin) / res).astype(np.int64), 0, nx - 1)
    df = pd.DataFrame({"row": rows, "col": cols, "d": d})
    med = df.groupby(["row", "col"])["d"].median()

    grid = np.full((ny, nx), np.nan, dtype=np.float64)
    grid[med.index.get_level_values(0).values,
         med.index.get_level_values(1).values] = med.values

    save_dem(grid, out_path, f"EPSG:{utm_epsg}", transform)
    print(f"  Stable M3C2 raster → {out_path.name}")
    return out_path

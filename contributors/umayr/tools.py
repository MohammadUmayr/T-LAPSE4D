"""Scratch space for new functions before they're promoted into cntp/.

Prototype here, verify in the notebook, then migrate into the matching
cntp module (io / coreg / asp / metashape / plot / pipeline_4dsfm).
"""

import os
import shutil
import subprocess

import numpy as np
import pandas as pd
from pathlib import Path

import rasterio
import rasterio.features
import geopandas as gpd
import laspy

import cntp
from cntp.coreg import _NDWI_A, _NDWI_B


def point2dem(
    cloud_las: str | Path,
    out_prefix: str | Path = None,
    res: float = 1.0,
    utm_epsg: int = None,
    max_gap_pixels: int = 1,
    nodata: float = -9999.0,
    ref_las: str | Path = None,
    extra_args: tuple = (),
    verbose: bool = False,
) -> Path:
    """Wrap ASP's ``point2dem`` CLI to rasterise a LAS cloud into a DEM.

    Thin wrapper around the same Ames Stereo Pipeline binary that
    :func:`cntp.asp.pc_align_p2p_sp2p` already requires. ``point2dem`` is
    streaming and multi-threaded: no Python-side RAM blow-up, no Delaunay
    triangulation, no Clough-Tocher overshoot. Per-pixel binning + median
    aggregation, written straight to a GeoTIFF.

    Parameters
    ----------
    cloud_las : str | Path
        Input LAS/LAZ point cloud (e.g. a coregistered cloud).
    out_prefix : str | Path, optional
        ASP output prefix passed to ``-o``. The DEM lands at
        ``<out_prefix>-DEM.tif`` (ASP appends the suffix). Default
        ``<cloud_dir>/<cloud_stem>``.
    res : float
        Output pixel size (``--tr``). Default 1.0 m.
    utm_epsg : int, optional
        EPSG code for ``--t_srs``. When ``None``, read from ``cloud_las``'s
        LAS header.
    max_gap_pixels : int
        Maps to ASP ``--search-radius-factor``: pixels farther than
        ``max_gap_pixels * res`` from any cloud point produce nodata.
        Default 1 (same convention as :func:`cntp.io.build_dem_and_ortho`).
    nodata : float
        ASP ``--nodata-value``. Default -9999.
    ref_las : str | Path, optional
        When set, the output grid is anchored to *ref_las*'s XY bounding
        box (read from header — no points loaded) via ASP
        ``--t_projwin xmin ymin xmax ymax`` (ASP convention — lower-left
        + upper-right, NOT GDAL's upper-left/lower-right). Mirrors
        :func:`cntp.io.build_dem_and_ortho`'s
        ``ref_las`` argument so the ASP DEM lands on the same footprint as
        the scipy DEM and the two can be differenced without resampling.
        Default ``None`` ⇒ ASP derives the grid from the input cloud's own
        bounding box.
    extra_args : tuple
        Additional CLI flags appended verbatim, e.g.
        ``("--remove-outliers", "--max-valid-triangulation-error", "5")``.
    verbose : bool
        Print the assembled CLI and ASP's stdout.

    Returns
    -------
    Path
        Path to ``<out_prefix>-DEM.tif``.
    """
    if shutil.which("point2dem") is None:
        raise RuntimeError(
            "point2dem not found on PATH.\n"
            "Install NASA Ames Stereo Pipeline:\n"
            "  https://stereopipeline.readthedocs.io/en/latest/installation.html"
        )

    cloud_las = Path(cloud_las)
    if out_prefix is None:
        out_prefix = cloud_las.parent / cloud_las.stem
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if utm_epsg is None:
        with laspy.open(cloud_las) as f:
            crs = f.header.parse_crs()
        if crs is None or crs.to_epsg() is None:
            raise ValueError(
                f"No EPSG in {cloud_las.name} header; pass utm_epsg explicitly."
            )
        utm_epsg = crs.to_epsg()

    threads = os.cpu_count() or 1

    cmd = [
        "point2dem",
        "--threads", str(threads),
        "--tr", str(res),
        "--t_srs", f"EPSG:{utm_epsg}",
        "--search-radius-factor", str(max_gap_pixels),
        "--nodata-value", str(nodata),
        "-o", str(out_prefix),
    ]
    if ref_las is not None:
        from cntp.io import read_las_bounds
        ref_min, ref_max = read_las_bounds(ref_las)
        # Defensive min/max — the LAS spec doesn't require header.mins <=
        # header.maxs, so some writers store them reversed (we hit this
        # with Reference_UAV_TLC_PCS.laz, which has Y min > Y max in its
        # header → degenerate projwin → empty DEM).
        x0, x1 = float(ref_min[0]), float(ref_max[0])
        y0, y1 = float(ref_min[1]), float(ref_max[1])
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)
        # ASP convention: --t_projwin xmin ymin xmax ymax (lower-left +
        # upper-right). Different from GDAL's --projwin ulx uly lrx lry.
        cmd.extend([
            "--t_projwin",
            str(xmin), str(ymin), str(xmax), str(ymax),
        ])
    cmd.extend(str(a) for a in extra_args)
    cmd.append(str(cloud_las))

    if verbose:
        print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if verbose:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"point2dem failed (exit {result.returncode}):\n{result.stdout}"
        )

    dem_path = out_prefix.parent / f"{out_prefix.name}-DEM.tif"
    if not dem_path.exists():
        raise RuntimeError(
            f"point2dem completed but expected output {dem_path} not found.\n"
            f"Stdout:\n{result.stdout}"
        )
    print(f"  ASP DEM → {dem_path}")
    return dem_path


def extract_stable_terrain_from_dem(
    dem_path: str | Path,
    ortho_path: str | Path = None,
    glacier_mask_path: str | Path = None,
    slope_threshold: float = 60.0,
    out_path: str | Path = None,
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
        together by :func:`cntp.io.build_dem_and_ortho`. When ``None`` the
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

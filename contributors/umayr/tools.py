"""Scratch space for new functions before they're promoted into cntp/.

Prototype here, verify in the notebook, then migrate into the matching
cntp module (io / coreg / asp / metashape / plot / pipeline_4dsfm / raster).

Current prototypes
------------------
- ``build_reference_tlc_cloud``      — once-per-glacier TLC-only reference cloud
  (registry EOP/IOP, fixed-IOP reconstruction, coregistered to the fused
  reference) → ``_ref_cache/reference_TLC_coreg.las``.
- ``run_4dsfm_day_with_rasters_tlc`` — per-date orchestrator identical to
  ``cntp.pipeline_4dsfm.run_4dsfm_day_with_rasters`` for SfM + coreg, but the
  DoD / stable-DoD / M3C2 rasters are built against the TLC reference so the
  change signal is TLC-vs-TLC (UAV↔TLC instrument bias cancels as common mode).
- ``plot_m3c2_spatial``              — 2-panel spatial QC map of M3C2 residuals
  before/after coregistration; analogous to demcoreg difference maps.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

import cntp  # noqa: F401


# ---------------------------------------------------------------------------
# Part 1 — TLC-only reference cloud (once per glacier)
# ---------------------------------------------------------------------------

def build_reference_tlc_cloud(
    registry_csv: Path,
    tlcam_dir: Path,
    ref_cloud: Path,
    glacier_mask: Path,
    output_dir: Path,
    ref_date: str = None,
    match_downscale: int = 1,
    depth_downscale: int = 2,
    ref_downsample: float = 0.4,
    tba_downsample: float = 1.0,
    p2p_max_disp: float = 10.0,
    sp2p_max_disp: float = 5.0,
    m_sp2p_max_disp: float = 0.5,
    use_ecef: bool = True,
    overwrite: bool = False,
    verbose: bool = False,
) -> Path:
    """Build the coregistered TLC-only reference cloud for change detection.

    The reference day's timelapse cameras already have validated IOP + EOP in
    the reference registry (the fused UAV+TLC bundle adjustment exported them,
    and ``bootstrap_registry`` filtered to TLC labels). So instead of bundle
    adjusting, we reuse that geometry: a fixed-IOP single-day reconstruction
    using *only the reference-day TLC images*, with EOP pinned tight to the
    registry values, then the **same** three-stage ICP coregistration every
    daily cloud receives. Because both the daily clouds and this reference are
    aligned to the *same* fused reference by the same procedure, the
    UAV-vs-TLC systematic offset is common-mode and cancels in
    ``day_TLC − refday_TLC``.

    The coreg target stays the fused ``ref_cloud`` (dense, lots of stable
    terrain → robust alignment); only the change-detection baseline becomes
    TLC-only.

    Parameters
    ----------
    registry_csv : Path
        Reference registry produced by ``bootstrap_registry``. The reference
        day, its TLC cameras' EOP (lon/lat/alt/yaw/pitch/roll), and their
        fixed-IOP ``calib_dir`` are all read from here.
    tlcam_dir : Path
        Standardised timelapse root (``<cam>_<date>_<time>.JPG``) — must hold
        the reference-day images for the registry's cameras.
    ref_cloud : Path
        Fused UAV+TLC reference cloud (UTM LAZ/LAS) — the coregistration target.
    glacier_mask : Path
        Glacier polygon shapefile (same CRS as the clouds).
    output_dir : Path
        Root output directory (parent of ``output/``).
    ref_date : str, optional
        Reference day (``YYYY-MM-DD``). When ``None`` (default) it is derived
        from the registry, which must contain exactly one date.
    match_downscale, depth_downscale, ref_downsample, tba_downsample,
    p2p_max_disp, sp2p_max_disp, m_sp2p_max_disp, use_ecef, overwrite, verbose :
        Same meaning as in :func:`cntp.pipeline_4dsfm.run_4dsfm_day`.

    Returns
    -------
    Path
        ``<output_dir>/output/_ref_cache/reference_TLC_coreg.las`` — the
        coregistered TLC reference, ready to pass as ``change_ref_cloud``.
    """
    from cntp.metashape import (
        discover_images, run_single_day_fixed_iop, _utm_epsg, _normalize_date,
    )
    from cntp.asp import extract_stable_reference, pc_align_p2p_sp2p, evaluate_coreg
    from cntp.io import load_las, save_las

    output_dir   = Path(output_dir)
    ref_cloud    = Path(ref_cloud)
    registry_csv = Path(registry_csv)

    ref_cache_dir = output_dir / "output" / "_ref_cache"
    out_path      = ref_cache_dir / "reference_TLC_coreg.las"
    if out_path.exists() and not overwrite:
        print(f"[ref-TLC] Skipping — {out_path.name} exists")
        return out_path

    # ── Reference day + cameras from the registry ────────────────────────
    # Normalise the date column — registry CSVs opened in Excel/LibreOffice
    # drift to '11/27/2023', but discover_images keys + output dirs are ISO.
    reg_df = pd.read_csv(registry_csv)
    reg_df["date"] = reg_df["date"].map(_normalize_date)
    if ref_date is None:
        dates = reg_df["date"].unique()
        if len(dates) != 1:
            raise ValueError(
                f"ref_date not given and registry has {len(dates)} dates "
                f"({dates.tolist()}); pass ref_date explicitly."
            )
        ref_date = dates[0]
    else:
        ref_date = _normalize_date(ref_date)
    ref_rows = reg_df[reg_df["date"] == ref_date]
    if ref_rows.empty:
        raise ValueError(f"No registry rows for ref_date {ref_date}.")
    print(f"[ref-TLC] Reference day {ref_date} — "
          f"{ref_rows['label'].str.split('_').str[0].unique().tolist()}")

    utm_epsg  = _utm_epsg(reg_df["lon"].mean())
    ecef_epsg = utm_epsg if use_ecef else None

    # Fixed IOP comes from the registry's calib_dir (the reference day's
    # adjusted_calib_4DSfM written by bootstrap).
    calib_dir = Path(ref_rows["calib_dir"].iloc[0])
    if not calib_dir.exists() or not list(calib_dir.glob("*.xml")):
        raise FileNotFoundError(
            f"calib_dir from registry has no XMLs: {calib_dir}"
        )

    # ── Synthesise the Step-1 cameras CSV from the registry EOP ──────────
    # run_single_day_fixed_iop expects Label/Lon/Lat/Alt/Yaw/Pitch/Roll.
    ref_tlc_dir = output_dir / "output" / ref_date / "ref_tlc"
    ref_tlc_dir.mkdir(parents=True, exist_ok=True)
    cameras_csv = ref_tlc_dir / f"{ref_date}_cameras_registry.csv"
    ref_rows.rename(columns={
        "label": "Label", "lon": "Lon", "lat": "Lat", "alt": "Alt",
        "yaw": "Yaw", "pitch": "Pitch", "roll": "Roll",
    })[["Label", "Lon", "Lat", "Alt", "Yaw", "Pitch", "Roll"]].to_csv(
        cameras_csv, index=False)

    # ── Reference-day TLC images ─────────────────────────────────────────
    by_date = discover_images(tlcam_dir)
    if ref_date not in by_date:
        raise ValueError(
            f"No images for ref_date {ref_date} under {tlcam_dir}. "
            f"Does tlcam_dir point at the right camera set?"
        )
    date_images = by_date[ref_date]

    # ── Fixed-IOP reconstruction with EOP pinned tight to the registry ───
    # Tight (0.001) priors snap alignCameras to the registry EOP — the
    # 'latter' approach (alignment runs, but cameras hold near registry pose).
    print(f"\n[ref-TLC] Single-day fixed IOP (EOP tight) — {ref_date}")
    tba_las_path, _ = run_single_day_fixed_iop(
        date            = ref_date,
        date_images     = date_images,
        calib_dir       = calib_dir,
        cameras_csv     = cameras_csv,
        output_dir      = output_dir,
        utm_epsg        = utm_epsg,
        match_downscale = match_downscale,
        depth_downscale = depth_downscale,
        loc_acc         = (0.001, 0.001, 0.001),
        rot_acc         = (0.001, 0.001, 0.001),
    )

    # ── Shared stable reference (fused) — same as the daily pipeline ─────
    ref_cache_dir.mkdir(parents=True, exist_ok=True)
    ref_ds_path = ref_cache_dir / f"{ref_cloud.stem}_ds{ref_downsample:.2f}.las"
    if not ref_ds_path.exists():
        print(f"[ref-TLC] Downsampled reference ({ref_downsample:.0%}) → {ref_ds_path.name}")
        save_las(load_las(ref_cloud, downsample_factor=ref_downsample), ref_ds_path,
                 crs=utm_epsg)
    stable_ref = extract_stable_reference(
        ref_cloud_path    = ref_ds_path,
        output_dir        = ref_cache_dir,
        glacier_mask_path = glacier_mask,
        plot_dir          = ref_cache_dir / "m3c2_plots" / "reference",
    )

    # ── Coregister the TLC reference to the fused reference ──────────────
    print(f"\n[ref-TLC] ASP 3-stage ICP — {ref_date}")
    coreg_dir = ref_tlc_dir / "coreg"
    aligned_las, _ = pc_align_p2p_sp2p(
        tba_las                 = tba_las_path,
        ref_las                 = ref_ds_path,
        output_dir              = coreg_dir,
        stable_ref_las          = stable_ref,
        p2p_max_displacement    = p2p_max_disp,
        sp2p_max_displacement   = sp2p_max_disp,
        m_sp2p_max_displacement = m_sp2p_max_disp,
        ref_downsample_factor   = 1.0,
        tba_downsample_factor   = tba_downsample,
        utm_epsg                = ecef_epsg,
        verbose                 = verbose,
    )

    # ── Co-registration QC: M3C2 before/after on stable terrain + plot ───
    # Same diagnostic the daily pipeline writes in Step 3b; lands the
    # before/after histogram at coreg/m3c2_plots/m3c2_distances.png.
    print(f"\n[ref-TLC] Evaluate co-registration — {ref_date}")
    eval_result = evaluate_coreg(
        ref_las               = ref_ds_path,
        tba_before_las        = tba_las_path,
        tba_after_las         = aligned_las,
        ref_downsample_factor = 1.0,
        tba_downsample_factor = tba_downsample,
        glacier_mask_path     = glacier_mask,
        stable_ref_las        = stable_ref,
        stable_dir            = coreg_dir / "stable_tba",
        plot_dir              = coreg_dir / "m3c2_plots",
    )
    print(f"  coreg M3C2 (stable) — before {eval_result['med_before']:+.4f} m "
          f"→ after {eval_result['med_after']:+.4f} m  (std {eval_result['std_after']:.4f})")

    # copyfile (not copy) — copy() also chmods the dest, which fails on the
    # /mnt/g drvfs mount (WSL → Windows fs): PermissionError on copymode.
    shutil.copyfile(aligned_las, out_path)
    print(f"\n[ref-TLC] Reference TLC cloud → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Part 1b — multi-threaded DEM + ortho (same algorithm as cntp.raster, tiled)
# ---------------------------------------------------------------------------

def _interp_tile(payload):
    """Worker: cubic griddata on one tile's points → tile DEM block.

    Runs in a separate process (griddata cubic is single-threaded, so we get
    parallelism by giving each process its own tile). Identical interpolation
    to ``cntp.raster.interpolate_and_mask`` — just on a sub-grid.
    """
    import numpy as np
    from scipy.interpolate import griddata
    pts, xi_t, yi_t = payload
    if len(pts) < 4:                       # need ≥4 pts for a 2-D triangulation
        return np.full(xi_t.shape, np.nan)
    return griddata((pts[:, 0], pts[:, 1]), pts[:, 2], (xi_t, yi_t), method="cubic")


def build_dem_and_ortho_mt(
    cloud_las,
    ref_las,
    out_dir,
    name_stem,
    res: float = 1.0,
    max_gap_pixels: int = 1,
    utm_epsg: int = None,
    cloud_downsample: float = 1.0,
    overwrite: bool = False,
    n_workers: int = None,
    n_tiles: tuple = None,
    margin: float = None,
) -> tuple:
    """Drop-in multi-threaded replacement for ``cntp.raster.build_dem_and_ortho``.

    Same output (cubic-griddata DEM + nearest-neighbour ortho on the reference
    grid), but the work is spread across cores:

    - **DEM**: the output grid is split into ``n_tiles`` blocks; each block runs
      the *identical* ``griddata(method='cubic')`` in its own process, on the
      tile's points **plus a `margin`** so the interpolant inside the tile
      matches the global one. (Clough-Tocher's gradient solve is global, so
      seams differ by sub-cm with an adequate margin — negligible for a DEM.)
    - **gap mask + ortho**: one shared ``cKDTree`` queried with ``workers=-1``
      (SciPy's built-in multi-threaded query) instead of two single-threaded
      builds.

    Extra params vs the original: ``n_workers`` (default = all cores),
    ``n_tiles`` (default ≈ one square tile per worker), ``margin`` metres
    (default ``10*res``).
    """
    import os
    import numpy as np
    import laspy
    import rasterio
    from concurrent.futures import ProcessPoolExecutor
    from scipy.spatial import cKDTree
    from cntp.io import load_las, read_las_bounds
    from cntp.raster import save_dem

    cloud_las, ref_las, out_dir = Path(cloud_las), Path(ref_las), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dem_path   = out_dir / f"{name_stem}_dem.tif"
    ortho_path = out_dir / f"{name_stem}_ortho.tif"
    if not overwrite and dem_path.exists() and ortho_path.exists():
        print(f"  DEM + ortho cached → {dem_path.name}, {ortho_path.name}")
        return dem_path, ortho_path

    # ── Grid from the reference cloud's header bbox (no points loaded) ────
    ref_min, ref_max = read_las_bounds(ref_las)
    xmin, xmax = float(ref_min[0]), float(ref_max[0])
    ymin, ymax = float(ref_min[1]), float(ref_max[1])
    xi = np.arange(xmin, xmax, res)
    yi = np.arange(ymax, ymin, -res)
    xi, yi = np.meshgrid(xi, yi)
    transform = rasterio.transform.from_origin(xmin, ymax, res, res)
    H, W = xi.shape

    # ── CRS ──────────────────────────────────────────────────────────────
    if utm_epsg is None:
        with laspy.open(cloud_las) as f:
            crs = f.header.parse_crs()
        utm_epsg = crs.to_epsg()
    crs_epsg = f"EPSG:{utm_epsg}"

    # ── Load + XY-clip the slave cloud ───────────────────────────────────
    cloud = load_las(cloud_las, downsample_factor=cloud_downsample)
    in_box = ((cloud[:, 0] >= xmin) & (cloud[:, 0] <= xmax) &
              (cloud[:, 1] >= ymin) & (cloud[:, 1] <= ymax))
    cloud = cloud[in_box]
    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    rgb     = cloud[:, 3:6]

    if n_workers is None:
        n_workers = os.cpu_count()
    if n_tiles is None:
        side = int(np.ceil(np.sqrt(n_workers)))
        n_tiles = (side, side)
    if margin is None:
        margin = 10.0 * res
    nty, ntx = n_tiles
    print(f"  DEM(mt): {len(cloud):,} pts → {W}×{H} grid, {nty}×{ntx} tiles, "
          f"{n_workers} workers, margin {margin:g} m")

    # ── 4. DEM — cubic griddata per tile, in parallel ────────────────────
    col_edges = np.linspace(0, W, ntx + 1, dtype=int)
    row_edges = np.linspace(0, H, nty + 1, dtype=int)
    tasks, slots = [], []
    for r in range(nty):
        r0, r1 = row_edges[r], row_edges[r + 1]
        if r1 <= r0:
            continue
        for c in range(ntx):
            c0, c1 = col_edges[c], col_edges[c + 1]
            if c1 <= c0:
                continue
            xi_t, yi_t = xi[r0:r1, c0:c1], yi[r0:r1, c0:c1]
            tx0, tx1 = xi_t.min() - margin, xi_t.max() + margin
            ty0, ty1 = yi_t.min() - margin, yi_t.max() + margin
            m = (x >= tx0) & (x <= tx1) & (y >= ty0) & (y <= ty1)
            tasks.append((np.column_stack([x[m], y[m], z[m]]), xi_t, yi_t))
            slots.append((r0, r1, c0, c1))

    zi = np.full((H, W), np.nan)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for (r0, r1, c0, c1), block in zip(slots, ex.map(_interp_tile, tasks)):
            zi[r0:r1, c0:c1] = block

    # Same global overshoot clip as cntp.raster.interpolate_and_mask
    zi = np.clip(zi, np.nanmin(z), np.nanmax(z))

    # Gap mask — one shared tree, parallel query
    tree = cKDTree(np.column_stack([x, y]))
    grid_xy = np.column_stack([xi.ravel(), yi.ravel()])
    dist, _ = tree.query(grid_xy, k=1, workers=-1)
    zi[dist.reshape(H, W) > (max_gap_pixels * res)] = np.nan
    save_dem(zi, dem_path, crs_epsg, transform)

    # ── 5. Orthoimage — reuse the tree, parallel NN query ────────────────
    dist2, idx = tree.query(grid_xy, k=1,
                            distance_upper_bound=res * max_gap_pixels, workers=-1)
    mask = np.isfinite(dist2)
    r_img = np.zeros(idx.shape, np.uint8)
    g_img = np.zeros(idx.shape, np.uint8)
    b_img = np.zeros(idx.shape, np.uint8)
    r_img[mask] = rgb[idx[mask], 0]
    g_img[mask] = rgb[idx[mask], 1]
    b_img[mask] = rgb[idx[mask], 2]
    ortho = np.dstack([r_img.reshape(H, W), g_img.reshape(H, W), b_img.reshape(H, W)])
    with rasterio.open(ortho_path, "w", driver="GTiff", height=H, width=W, count=3,
                       dtype=ortho.dtype, crs=crs_epsg, transform=transform, nodata=0) as dst:
        dst.write(ortho[:, :, 0], 1)
        dst.write(ortho[:, :, 1], 2)
        dst.write(ortho[:, :, 2], 3)

    print(f"  Saved DEM + ortho (mt): {dem_path.name}, {ortho_path.name}")
    return dem_path, ortho_path


# ---------------------------------------------------------------------------
# Part 1c — DEM via ASP point2dem (HSfM method) — PROMOTED to cntp.raster
# ---------------------------------------------------------------------------
# build_dem_and_ortho_p2d now lives in cntp.raster and is wired into
# run_4dsfm_day_with_rasters via dem_method="point2dem". Re-exported here so
# notebook cells importing it from `tools` keep working against one source.
from cntp.raster import build_dem_and_ortho_p2d  # noqa: F401


# ---------------------------------------------------------------------------
# Part 2 — per-date orchestrator with TLC-referenced change detection
# ---------------------------------------------------------------------------

def run_4dsfm_day_with_rasters_tlc(
    new_date: str,
    tlcam_dir: Path,
    ref_cloud: Path,
    change_ref_cloud: Path,
    glacier_mask: Path,
    registry_csv: Path,
    output_dir: Path,
    # ── SfM pipeline kwargs (forwarded to run_4dsfm_day) ──────────────────
    match_downscale: int = 1,
    depth_downscale: int = 2,
    loc_acc_new: tuple = (0.5, 0.5, 0.5),
    rot_acc_new: tuple = (5.0, 5.0, 5.0),
    ref_downsample: float = 0.4,
    tba_downsample: float = 1.0,
    p2p_max_disp: float = 10.0,
    sp2p_max_disp: float = 5.0,
    m_sp2p_max_disp: float = 2.0,
    use_ecef: bool = True,
    overwrite: bool = False,
    verbose: bool = False,
    add_to_registry: bool = True,
    # ── Raster knobs ──────────────────────────────────────────────────────
    res: float = 1.0,
    max_gap_pixels: int = 1,
    ref_cloud_downsample: float = 0.25,
    m3c2_ref_downsample: float = 0.25,
    slope_threshold: float = 60.0,
    utm_epsg: int = None,
    overwrite_ref_dem: bool = False,
    overwrite_day_dem: bool = False,
    overwrite_dod: bool = False,
    overwrite_stable: bool = False,
    overwrite_stable_dod: bool = False,
    overwrite_m3c2: bool = False,
) -> dict:
    """Per-date 4D SfM + change-detection rasters against the TLC reference.

    Identical to :func:`cntp.pipeline_4dsfm.run_4dsfm_day_with_rasters` for the
    SfM + coregistration half (Steps 1–7, coreg target = fused ``ref_cloud``).
    The raster half is repointed at ``change_ref_cloud`` (the coregistered
    TLC-only reference from :func:`build_reference_tlc_cloud`): the reference
    DEM/ortho, the per-day DEM grid anchor, and the M3C2 reference all use the
    TLC cloud, so DoD and M3C2 are measured TLC-vs-TLC over the TLC footprint.
    No fused-reference DEM/ortho is produced.
    """
    import laspy
    import rasterio
    from cntp.pipeline_4dsfm import run_4dsfm_day
    from cntp.raster import (
        build_reference_dem_and_ortho,
        build_dem_and_ortho,
        build_dod,
        extract_stable_terrain_from_dem,
        m3c2_to_raster,
    )
    from cntp.plot import plot_dod_histogram

    output_dir       = Path(output_dir)
    ref_cloud        = Path(ref_cloud)
    change_ref_cloud = Path(change_ref_cloud)
    registry_csv     = Path(registry_csv)

    if not change_ref_cloud.exists():
        raise FileNotFoundError(
            f"change_ref_cloud not found: {change_ref_cloud}\n"
            f"Run build_reference_tlc_cloud(...) first (setup_new_glacier)."
        )

    # ── Step 1–7: 4D SfM pipeline (coreg target = fused ref_cloud) ───────
    sfm_result = run_4dsfm_day(
        new_date        = new_date,
        tlcam_dir       = tlcam_dir,
        ref_cloud       = ref_cloud,
        glacier_mask    = glacier_mask,
        registry_csv    = registry_csv,
        output_dir      = output_dir,
        match_downscale = match_downscale,
        depth_downscale = depth_downscale,
        loc_acc_new     = loc_acc_new,
        rot_acc_new     = rot_acc_new,
        ref_downsample  = ref_downsample,
        tba_downsample  = tba_downsample,
        p2p_max_disp    = p2p_max_disp,
        sp2p_max_disp   = sp2p_max_disp,
        m_sp2p_max_disp = m_sp2p_max_disp,
        use_ecef        = use_ecef,
        overwrite       = overwrite,
        verbose         = verbose,
        add_to_registry = add_to_registry,
    )

    # ── Resolve shared paths + CRS ───────────────────────────────────────
    day_dir       = output_dir / "output" / new_date
    aligned_las   = day_dir / "coreg" / f"{new_date}_cloud_coreg_hsfm.las"
    single_day    = day_dir / "single_day"
    ref_cache_dir = output_dir / "output" / "_ref_cache"

    if utm_epsg is None:
        with laspy.open(change_ref_cloud) as _f:
            utm_epsg = _f.header.parse_crs().to_epsg()

    # ── Reference DEM + ortho — built from the TLC reference (cached) ────
    ref_dem, ref_ortho = build_reference_dem_and_ortho(
        ref_cloud_path   = change_ref_cloud,
        cache_dir        = ref_cache_dir,
        res              = res,
        max_gap_pixels   = max_gap_pixels,
        utm_epsg         = utm_epsg,
        cloud_downsample = ref_cloud_downsample,
        overwrite        = overwrite_ref_dem,
    )

    # ── Per-day DEM + ortho — anchored to the TLC grid (clips to TLC) ────
    dem, ortho = build_dem_and_ortho(
        cloud_las        = aligned_las,
        ref_las          = change_ref_cloud,
        out_dir          = single_day,
        name_stem        = new_date,
        res              = res,
        max_gap_pixels   = max_gap_pixels,
        utm_epsg         = utm_epsg,
        cloud_downsample = tba_downsample,
        overwrite        = overwrite_day_dem,
    )

    # ── DoD + histogram ──────────────────────────────────────────────────
    dod_path = build_dod(
        ref_dem_path = ref_dem,
        day_dem_path = dem,
        out_path     = single_day / "DOD.tif",
        overwrite    = overwrite_dod,
    )
    with rasterio.open(dod_path) as src:
        dod_values = src.read(1)
    dod_stats = plot_dod_histogram(
        dod_values,
        output_dir = single_day,
        title      = f"DoD — {new_date}",
        filename   = "dod_histogram.png",
    )

    # ── Stable-terrain DoD + histogram ───────────────────────────────────
    ref_ortho_path = ref_cache_dir / "reference_ortho.tif"
    day_ortho_path = single_day    / f"{new_date}_ortho.tif"
    ref_stable_dem = extract_stable_terrain_from_dem(
        dem_path          = ref_dem,
        ortho_path        = ref_ortho_path,
        glacier_mask_path = glacier_mask,
        slope_threshold   = slope_threshold,
        overwrite         = overwrite_stable,
    )
    day_stable_dem = extract_stable_terrain_from_dem(
        dem_path          = dem,
        ortho_path        = day_ortho_path,
        glacier_mask_path = glacier_mask,
        slope_threshold   = slope_threshold,
        overwrite         = overwrite_stable,
    )
    stable_dod_path = build_dod(
        ref_dem_path = ref_stable_dem,
        day_dem_path = day_stable_dem,
        out_path     = single_day / "DOD_stable.tif",
        overwrite    = overwrite_stable_dod,
    )
    with rasterio.open(stable_dod_path) as src:
        stable_dod_values = src.read(1)
    stable_stats = plot_dod_histogram(
        stable_dod_values,
        output_dir = single_day,
        title      = f"stable DoD — {new_date}",
        filename   = "dod_stable_histogram.png",
    )

    # ── M3C2 distance raster + histogram (TLC reference) ─────────────────
    m3c2_raster_path = m3c2_to_raster(
        ref_las         = change_ref_cloud,
        day_las         = aligned_las,
        out_path        = single_day / "M3C2_raster.tif",
        grid_anchor_las = change_ref_cloud,
        res             = res,
        utm_epsg        = utm_epsg,
        ref_downsample  = m3c2_ref_downsample,
        day_downsample  = 1.0,
        overwrite       = overwrite_m3c2,
    )
    with rasterio.open(m3c2_raster_path) as src:
        m3c2_values = src.read(1)
    m3c2_stats = plot_dod_histogram(
        m3c2_values,
        output_dir = single_day,
        title      = f"M3C2 raster — {new_date}",
        filename   = "m3c2_raster_histogram.png",
    )

    print(
        f"\n  [{new_date}] DoD med={dod_stats['median']:+.3f} m  |  "
        f"stable med={stable_stats['median']:+.3f} m  |  "
        f"M3C2 med={m3c2_stats['median']:+.3f} m"
    )

    return {
        "date":         new_date,
        "sfm":          sfm_result,
        "dod_stats":    dod_stats,
        "stable_stats": stable_stats,
        "m3c2_stats":   m3c2_stats,
    }


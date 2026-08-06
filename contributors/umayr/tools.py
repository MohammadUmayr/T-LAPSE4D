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


# ---------------------------------------------------------------------------
# Part 3 — avalanche-cone profile analysis (signal vs distance vs time)
# ---------------------------------------------------------------------------
# Post-hoc analysis of the per-date M3C2 reference→day signal rasters
# (``<date>/single_day/<date>_M3C2_raster.tif``). All rasters share one grid
# (fixed M3C2 corepoints), so they stack into a (T, H, W) cube. Profiles are
# QGIS LineStrings sampled by bilinear interpolation along the line; the signal
# is shown as a distance × time Hovmöller and as season-coloured profile lines.
#
# Method follows the Argentière profile workflow (M. Kneib), adapted to sample
# the M3C2 raster directly (it already IS reference→day dh, so no DEM
# differencing). Box-mean sampling is intentionally omitted — that was for his
# mass-balance point series, not the profiles. Prototype here; promote the
# plot fns to cntp.plot and the loaders/orchestrator to cntp.postprocessing
# once validated.

# -- Signal stack -- PROMOTED to cntp.postprocessing ------------------------
# SignalStack + load_signal_stack now live in cntp.postprocessing (its role is
# loading finished outputs from disk). Re-exported here so the Part-3 profile
# helpers below and notebook cells importing them from `tools` keep working.
from cntp.postprocessing import (  # noqa: F401
    SignalStack, load_signal_stack,
)


def load_profiles(profiles, *, name_field=None):
    """Read profile LineStrings → ``[(name, LineString), …]``.

    ``profiles`` may be a single shapefile, a **directory** (all ``*.shp`` in
    it), a glob string, or a list of paths. Each line feature is one profile.
    Naming priority: a valid ``name_field`` value → else the file stem (when the
    file holds one line) → else ``<stem>_<i>``. MultiLineStrings are merged.
    Files are processed in sorted order so Profile1, Profile2, … stay ordered.
    """
    import glob as _glob
    import geopandas as gpd
    from shapely.ops import linemerge

    if isinstance(profiles, (list, tuple)):
        files = [Path(p) for p in profiles]
    else:
        p = Path(profiles)
        if p.is_dir():
            files = sorted(p.glob("*.shp"))
        elif any(c in str(profiles) for c in "*?["):
            files = sorted(Path(x) for x in _glob.glob(str(profiles)))
        else:
            files = [p]
    if not files:
        raise FileNotFoundError(f"no shapefiles matched {profiles!r}")

    out = []
    for f in files:
        gdf = gpd.read_file(f).reset_index(drop=True)
        single = len(gdf) == 1
        for i, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            if geom.geom_type == "MultiLineString":
                geom = linemerge(geom)
            if geom.geom_type != "LineString":
                raise ValueError(
                    f"{f.name} feature {i} is {geom.geom_type}, expected "
                    f"LineString (draw profiles as single lines in QGIS)."
                )
            name = None
            if name_field and name_field in gdf.columns:
                val = row[name_field]
                if val is not None and str(val).strip() and str(val).lower() != "nan":
                    name = str(val)
            if name is None:
                name = f.stem if single else f"{f.stem}_{i + 1}"
            out.append((name, geom))
    if not out:
        raise ValueError(f"no line features found in {profiles!r}")
    return out


def load_transition_line(transition):
    """Read the Bergschrund / transition LineString from a shapefile.

    One line delineating the Bergschrund across the glacier. MultiLineStrings
    are merged; multiple features are merged into one geometry so a transition
    drawn as several segments still works.
    """
    import geopandas as gpd
    from shapely.ops import linemerge, unary_union

    gdf = gpd.read_file(Path(transition))
    geoms = [g for g in gdf.geometry if g is not None]
    if not geoms:
        raise ValueError(f"no geometry in {transition!r}")
    merged = linemerge(unary_union(geoms)) if len(geoms) > 1 else geoms[0]
    if merged.geom_type == "MultiLineString":
        merged = linemerge(merged)
    return merged


def profile_transition_distances(line, transition):
    """Along-profile distances [m] where *line* crosses the *transition* line.

    Returns a sorted list (empty when the profile never crosses). Normally one
    crossing per profile; a profile that re-crosses a sinuous Bergschrund gives
    several, and all are returned so the caller can draw each.
    """
    inter = line.intersection(transition)
    if inter.is_empty:
        return []
    geoms = [inter] if inter.geom_type == "Point" else list(getattr(inter, "geoms", []))
    # A grazing intersection can come back as a LineString; take its midpoint.
    pts = [g if g.geom_type == "Point" else g.interpolate(0.5, normalized=True)
           for g in geoms]
    return sorted(line.project(p) for p in pts)


def load_crevasse(crevasse):
    """Read the crevasse polygon(s) from a shapefile → one geometry.

    Multiple features are dissolved into a single (Multi)Polygon so a crevasse
    mapped as several patches still yields one set of spans per profile.
    """
    import geopandas as gpd
    from shapely.ops import unary_union

    gdf = gpd.read_file(Path(crevasse))
    geoms = [g for g in gdf.geometry if g is not None]
    if not geoms:
        raise ValueError(f"no geometry in {crevasse!r}")
    return unary_union(geoms)


def profile_crevasse_spans(line, crevasse):
    """Along-profile ``[(start, end), …]`` intervals [m] where *line* is inside
    the crevasse polygon.

    Empty when the profile never enters it. A profile that clips the polygon in
    several places gives several spans, and all are returned so each is shaded.
    """
    inter = line.intersection(crevasse)
    if inter.is_empty:
        return []
    parts = list(getattr(inter, "geoms", [inter]))
    spans = []
    for g in parts:
        if g.geom_type == "Point":       # grazing touch — no extent to shade
            continue
        d = [line.project(shapely_point) for shapely_point in
             (g.interpolate(0.0, normalized=True), g.interpolate(1.0, normalized=True))]
        spans.append((min(d), max(d)))
    return sorted(spans)


def _draw_crevasse(ax, spans, *, orient="v", zorder=0.3, alpha=0.18,
                   label="Crevasse"):
    """Shade the crevasse span(s), labelled once.

    ``orient='v'`` shades a band across the distance axis (profile lines);
    ``'h'`` shades it across the y axis (Hovmöller, where distance is y). The
    Hovmöller must pass a zorder above the pcolormesh, else the filled mesh
    hides the band.
    """
    if not spans:
        return False
    shade = ax.axvspan if orient == "v" else ax.axhspan
    for i, (a, b) in enumerate(spans):
        shade(a, b, color="0.55", alpha=alpha, lw=0, zorder=zorder,
              label=label if i == 0 else None)
    return True


def _as_positions(bergschrund):
    """``None`` / scalar / sequence → list of positions."""
    import numpy as np
    if bergschrund is None:
        return []
    if np.isscalar(bergschrund):
        return [float(bergschrund)]
    return [float(b) for b in bergschrund]


def _draw_bergschrund(ax, positions, *, orient="v", zorder=0.5,
                      label="Bergschrund"):
    """Draw the Bergschrund marker(s), labelled once.

    ``orient='v'`` for the profile lines (distance on x), ``'h'`` for the
    Hovmöller (distance on y). Default zorder puts it behind the curves; the
    Hovmöller must pass a zorder above the pcolormesh to stay visible.
    """
    if not positions:
        return False
    draw = ax.axvline if orient == "v" else ax.axhline
    for i, p in enumerate(positions):
        draw(p, color="0.30", lw=1.2, ls="--", zorder=zorder,
             label=label if i == 0 else None)
    return True


def smooth_along_profile(z, *, spacing=0.5, window=10.0):
    """Centred rolling mean of width *window* metres along a sampled profile.

    Shared by the dh curves and the slope curve so a feature's width means the
    same thing in both panels. ``window=None`` returns *z* untouched.
    """
    import numpy as np
    import pandas as pd
    if not window:
        return np.asarray(z, dtype="float64")
    w = max(1, int(round(window / spacing)))
    return (pd.Series(np.asarray(z, dtype="float64"))
            .rolling(w, center=True, min_periods=1).mean().to_numpy())


def slope_raster(dem):
    """Slope [deg] of *dem* (path or ``xdem.DEM``) via :func:`xdem.terrain.slope`.

    Use the **reference** DEM (``_ref_cache/reference_dem.tif``): it is the
    baseline every M3C2 is measured against, so one static slope curve per
    profile is directly comparable across all dates. Per-date DEMs would give a
    slope that drifts with each day's coreg residual and nodata gaps.
    """
    import xdem
    if not hasattr(dem, "data"):
        dem = xdem.DEM(str(dem))
    return xdem.terrain.slope(dem)


def sample_profile(raster, line, *, spacing=0.5):
    """Bilinear-sample *raster* along *line* every *spacing* metres.

    Returns ``(dist, z)``: distance along the line [m] and the raster value at
    each step (NaN where the line crosses nodata). Vertices are densified by
    linear interpolation of the polyline, then read with geoutils
    ``interp_points(method='linear')``.
    """
    import numpy as np

    x, y = line.xy
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dist_vert = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    length = dist_vert[-1]
    s = np.arange(0.0, length + 1e-8, spacing)
    xs = np.interp(s, dist_vert, x)
    ys = np.interp(s, dist_vert, y)
    z = raster.interp_points((xs, ys), method="linear", as_array=True)
    return s, np.asarray(z, dtype=float)


def sample_profile_timeseries(stack: SignalStack, line, *, spacing=0.5,
                              smoothing_window=10.0,
                              outlier_thresh=None, outlier_dist=0.0):
    """Sample *line* on every date in *stack* → ``(dist, matrix (T, S))``.

    Per date: bilinear-sample the line, optionally mask outliers
    (``|z| > outlier_thresh`` beyond ``outlier_dist`` m — set ``outlier_dist=0``
    to apply everywhere), then a centred rolling mean of width
    ``smoothing_window`` m (set ``None`` to skip). The distance axis is fixed by
    the line + spacing, so rows stack directly.
    """
    import numpy as np

    dist = None
    rows = []
    for i in range(len(stack)):
        s, z = sample_profile(stack.raster(i), line, spacing=spacing)
        if dist is None:
            dist = s
        if outlier_thresh is not None:
            m = np.abs(z) > outlier_thresh
            if outlier_dist and outlier_dist > 0:
                m &= (s > outlier_dist)
            z = np.where(m, np.nan, z)
        z = smooth_along_profile(z, spacing=spacing, window=smoothing_window)
        rows.append(z)
    return dist, np.vstack(rows)



def nmad_gate(nmad, max_nmad=0.5, *, times=None, from_date=None):
    """Keep-mask dropping acquisitions whose post-coreg stable nmad is too high.

    Same gate the biweekly box plot applies
    (:func:`cntp.postprocessing.absolute_accuracy_boxplots`): a failed
    coregistration is excluded outright rather than being left in to be
    out-voted. ``max_nmad=None`` keeps everything.

    ``from_date`` scopes the gate in time: dates **before** it are always kept,
    and only dates on/after it must pass the nmad test. Changri North needs this
    for the Hovmöller — coregistration degrades from mid-July (the monsoon), so
    the gate is applied from 2024-07-14 onward while the clean pre-monsoon
    record is kept whole. ``from_date=None`` gates the entire record.
    """
    import numpy as np

    nmad = np.asarray(nmad, dtype="float64")
    n = len(nmad)
    if max_nmad is None:
        return np.ones(n, dtype=bool)

    ok = np.isfinite(nmad)
    passes = ok & (nmad < max_nmad)
    if from_date is None:
        in_window = np.ones(n, dtype=bool)
    else:
        t = np.asarray(times, dtype="datetime64[D]")
        in_window = t >= np.datetime64(from_date)
    # Outside the window every date is kept; inside it, only the passing ones.
    gated = ~in_window | passes
    if not gated.any():
        print(f"  [warn] no acquisition has nmad < {max_nmad} m — gate ignored")
        return np.ones(n, dtype=bool)
    n_drop = int(n - gated.sum())
    if n_drop:
        scope = f" from {from_date}" if from_date else ""
        print(f"  Excluded {n_drop} acquisition(s) with post-coreg stable "
              f"nmad >= {max_nmad} m{scope}")
    return gated


def time_bin_average(times, matrix, bin_days=14):
    """Aggregate the (T, S) matrix into fixed ``bin_days`` windows along time.

    Returns ``(bin_times, bin_matrix)`` — one row per non-empty window (the
    NaN-skipping **median** of the dates in it) tagged with the window's centre
    date. Turns 324 daily profile curves into ~26 fortnightly curves so the
    profile-line plot is legible. Median (not mean) is used so the occasional
    bad-coreg / cloudy day in a window doesn't drag the curve — the daily M3C2
    has heavy-tailed outliers, so a robust centre is the right choice here. It
    also fills a single day's nodata gaps from the other days in the window,
    which a single-acquisition pick cannot do.
    Only the profile lines use this; the Hovmöller keeps the raw daily signal.
    """
    import numpy as np
    import pandas as pd

    idx = pd.to_datetime(np.asarray(times, dtype="datetime64[D]"))
    df = pd.DataFrame(np.asarray(matrix, dtype="float64"), index=idx)
    binned = df.resample(f"{bin_days}D").median()        # median skips NaN
    binned = binned[~binned.isna().all(axis=1)]          # drop empty windows
    centres = (binned.index + pd.Timedelta(days=bin_days / 2)).to_numpy(
    ).astype("datetime64[D]")
    return centres, binned.to_numpy()


def load_stable_nmad(output_dir, dates):
    """Per-date post-coregistration stable-terrain M3C2 nmad [m].

    Reads ``output/<date>/coreg/<date>_m3c2_stats.csv`` (the ``after`` row,
    ``nmad`` column) for each date — the co-registration noise floor the
    pipeline writes. Missing/unreadable → NaN.
    """
    import numpy as np
    import pandas as pd

    base = Path(output_dir) / "output"
    out = np.full(len(dates), np.nan)
    for i, d in enumerate(dates):
        f = base / d / "coreg" / f"{d}_m3c2_stats.csv"
        if f.exists():
            try:
                s = pd.read_csv(f, index_col="coreg")
                out[i] = float(s.loc["after", "nmad"])
            except Exception:
                pass
    return out



# ── Plot primitives (→ promote to cntp.plot once validated) ────────────────

_SEASONAL_DOY_COLORS = ['#4A86E8', '#1FD0B8', '#E69138', '#A61C00', '#4A86E8']

# Shared figure styling for both profile plots (Argentière serif look).
_PLOT_STYLE = {'font.family': 'serif', 'font.size': 12,
               'axes.titlesize': 13, 'axes.labelsize': 12,
               'xtick.labelsize': 10, 'ytick.labelsize': 10,
               'axes.linewidth': 0.8, 'mathtext.fontset': 'cm'}


def seasonal_doy_cmap():
    """Cyclic seasonal colour map keyed on day-of-year (1–366)."""
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list("seasonal_doy", _SEASONAL_DOY_COLORS)


def _tnum(times):
    """Datetime-like → matplotlib date numbers."""
    import numpy as np
    import matplotlib.dates as mdates
    return mdates.date2num(np.asarray(times, dtype="datetime64[s]").astype("O"))


def _robust_vmax(matrix, pct=98.0):
    """Per-profile symmetric colour limit from the spread of |dh|.

    Uses the ``pct``-th percentile of the finite |elevation-change| values so
    the scale follows each profile's own variability while ignoring the handful
    of blunder pixels, then rounds up to a clean colourbar step (0.5 m steps
    below 5 m, 1 m above).
    """
    import math
    import numpy as np
    a = np.abs(np.asarray(matrix, dtype="float64"))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 1.0
    v = float(np.percentile(a, pct))
    step = 0.5 if v < 5 else 1.0
    return max(math.ceil(v / step) * step, step)


def _cell_edges(centres):
    import numpy as np
    c = np.asarray(centres, dtype="float64")
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = (c[:-1] + c[1:]) / 2.0
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def plot_time_distance(times, distance, matrix, *, title=None, output_dir=None,
                       filename="time_distance.png", vmax=None, vmax_pct=98.0,
                       cmap="RdBu", shading="gouraud",
                       bergschrund=None, crevasse=None,
                       clabel="Elevation change (m)  reference→day", save_pdf=True):
    """Time × distance (Hovmöller) heatmap: date on x, distance on y, colour = dh.

    Diverging scale centred on 0 (blue = gain, red = loss), symmetric ±``vmax``.
    Leave ``vmax=None`` (default) to scale each profile to its own variability —
    the ``vmax_pct``-th percentile of ``|dh|`` (see :func:`_robust_vmax`) — or
    pass a number to fix it. ``shading='gouraud'`` interpolates smoothly
    (Argentière look); ``'flat'`` keeps cells/gaps honest.

    ``bergschrund`` / ``crevasse`` mark the transition-line crossing and the
    crevasse span along the profile. Distance is the y axis here, so they are
    drawn as a horizontal dashed line and a horizontal shaded band, both above
    the mesh (a background zorder would be hidden by the filled pcolormesh).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    distance = np.asarray(distance, dtype="float64")
    M = np.array(matrix, dtype="float64")
    tnum = _tnum(times)
    if vmax is None:
        vmax = _robust_vmax(M, pct=vmax_pct)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("0.85")
    plt.rcParams.update(_PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if shading == "flat":
        pcm = ax.pcolormesh(_cell_edges(tnum), _cell_edges(distance), M.T,
                            cmap=cmap_obj, vmin=-vmax, vmax=vmax, shading="flat")
    else:
        X, Y = np.meshgrid(tnum, distance)
        pcm = ax.pcolormesh(X, Y, M.T, cmap=cmap_obj, vmin=-vmax, vmax=vmax,
                            shading="gouraud")
    # Above the mesh: a background zorder would be hidden by the filled cells.
    has_cv = _draw_crevasse(ax, crevasse or [], orient="h", zorder=3, alpha=0.25)
    has_bs = _draw_bergschrund(ax, _as_positions(bergschrund), orient="h", zorder=3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Distance along profile (m)")
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    if title:
        ax.set_title(title)
    cb = fig.colorbar(pcm, ax=ax, extend="both")
    cb.set_label(clabel)
    plt.tight_layout()
    if has_bs or has_cv:
        # Outside the axes, under the colourbar. Anchored in figure coords after
        # tight_layout has settled the colourbar's final position, so the legend
        # can't overlap the mesh (a marker inside would hide the very data the
        # Bergschrund/crevasse are there to locate).
        handles, labels = ax.get_legend_handles_labels()
        box = cb.ax.get_position()
        fig.legend(handles, labels, loc="upper left",
                   bbox_to_anchor=(box.x0, box.y0 - 0.04),
                   bbox_transform=fig.transFigure,
                   frameon=False, fontsize=9, handlelength=1.6,
                   borderaxespad=0.0)
    _save_or_show(fig, output_dir, filename, save_pdf)


def plot_profile_lines(times, distance, matrix, *, title=None, output_dir=None,
                       filename="profile_lines.png",
                       ylabel="Elevation change (m)  reference→day",
                       ylim=None, cmap=None, bergschrund=None, crevasse=None,
                       slope=None, save_pdf=True):
    """Longitudinal profile, one curve per date, coloured chronologically.

    Shows the actual metres and shape of the deposit. Colour uses Marin's
    seasonal palette (:func:`seasonal_doy_cmap`) but is mapped across the actual
    acquisition span — first to last date — so the colourbar *starts at your
    first acquisition* rather than at a fixed Jan 1. Earliest curves are drawn
    first so later ones sit on top. Pass ``cmap`` to override the palette.

    ``bergschrund`` — along-profile distance(s) [m] where the profile crosses
    the transition line (:func:`profile_transition_distances`); drawn as a
    dashed vertical line behind the curves and named in the legend.
    ``crevasse`` — along-profile ``(start, end)`` span(s) [m] from
    :func:`profile_crevasse_spans`; shaded behind the curves and likewise
    named in the legend.
    ``slope`` — reference-DEM slope [deg] sampled on the same distance axis
    (:func:`sample_profile` on :func:`slope_raster`). Adds a lower panel sharing
    the distance axis, so dh can be read directly against local slope. The
    Bergschrund/crevasse markers are drawn on both panels.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable

    times = np.asarray(times, dtype="datetime64[s]")
    M = np.array(matrix, dtype="float64")
    distance = np.asarray(distance, dtype="float64")
    tnum = _tnum(times)
    norm = mcolors.Normalize(vmin=tnum.min(), vmax=tnum.max())
    if cmap is None:
        cmap_obj = seasonal_doy_cmap()           # Marin's seasonal colours
    elif hasattr(cmap, "N"):
        cmap_obj = cmap
    else:
        cmap_obj = plt.get_cmap(cmap)

    plt.rcParams.update(_PLOT_STYLE)
    if slope is None:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax_s, cb_axes = None, ax
    else:
        fig, (ax, ax_s) = plt.subplots(
            2, 1, figsize=(11, 6.5), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
        cb_axes = [ax, ax_s]                      # colourbar spans both panels
    has_cv = _draw_crevasse(ax, crevasse or [])
    has_bs = _draw_bergschrund(ax, _as_positions(bergschrund))
    for i in np.argsort(tnum):
        ax.plot(distance, M[i], color=cmap_obj(norm(tnum[i])), lw=1.4, alpha=0.9)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    if has_bs or has_cv:
        ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=10)
    if ax_s is not None:
        # Same markers on the slope panel so the eye can carry a distance across.
        _draw_crevasse(ax_s, crevasse or [])
        _draw_bergschrund(ax_s, _as_positions(bergschrund))
        ax_s.plot(distance, np.asarray(slope, dtype="float64"),
                  color="0.20", lw=1.3)
        ax_s.set_ylabel("Slope (°)")
        ax_s.set_xlabel("Distance along profile (m)")
        ax_s.grid(True, alpha=0.3)
        ax_s.set_ylim(bottom=0)
    else:
        ax.set_xlabel("Distance along profile (m)")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        finite = M[np.isfinite(M)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            ax.set_ylim(lo - pad, hi + pad)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=cb_axes)
    # Explicit bi-monthly ticks from the first full month of the record (Dec
    # 2023) so the colourbar actually shows Dec 2023 — the auto-locator drops
    # that edge tick and starts the labels at Feb.
    import pandas as pd
    first = pd.Timestamp(times.min()).to_period("M").to_timestamp()  # floor to month
    ticks = pd.date_range(first, pd.Timestamp(times.max()), freq="2MS")
    cb.set_ticks(mdates.date2num(ticks))
    cb.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    cb.set_label("Acquisition date")
    plt.tight_layout()
    _save_or_show(fig, output_dir, filename, save_pdf)


def plot_profile_grid(results, names, *, output_dir=None, filename=None,
                      site_label="Changri West", cmap=None, vmax=None,
                      ylim=None, vmax_pct=98.0, shading="flat", dh_cmap="RdBu",
                      clabel="Elevation change (m)  reference→day",
                      ylabel="Elevation change (m)  reference→day",
                      save_pdf=True):
    """3-row composite for one cone's profiles: biweekly / slope / Hovmöller.

    One column per profile in *names* (1 or 2). A cone sampled by a single
    profile — e.g. Changri North's CN_Profile5 — gives a 3x1 figure. Rows are
    the biweekly profile lines, the reference-DEM slope, and the Hovmöller.
    Rows 1-2 share the distance axis per column (the Hovmöller cannot — its x is
    date, with distance on y). Each row gets one colourbar spanning the columns.
    Bergschrund/crevasse are drawn on every panel when supplied.

    **Scaling rule:** the profiles in a figure are drawn over the *same
    avalanche cone*, so they share one ruler — dh (``ylim``), slope and the
    Hovmöller (``vmax``) are each set by whichever profile dominates. Scales are
    deliberately NOT shared across cones: the cones differ in magnitude, and
    forcing one global scale would flatten the smaller one. Pass ``ylim`` /
    ``vmax`` to override.

    *results* is the ``profiles`` dict returned by :func:`avalanche_cone_profiles`.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors
    import pandas as pd
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import MultipleLocator

    names = list(names)
    if not 1 <= len(names) <= 2:
        raise ValueError(f"names must be 1 or 2 profiles (one cone), got {len(names)}")
    missing = [n for n in names if n not in results]
    if missing:
        raise KeyError(f"no profile(s) {missing} in results (have {list(results)})")
    R = [results[n] for n in names]
    if any(r.get("slope") is None for r in R):
        raise ValueError("plot_profile_grid needs a slope curve — pass slope_dem "
                         "to avalanche_cone_profiles")

    # One symmetric scale for both Hovmöllers so the columns are comparable.
    if vmax is None:
        vmax = max(_robust_vmax(r["hmatrix"], pct=vmax_pct) for r in R)
    cmap_dh = plt.get_cmap(dh_cmap).copy()
    cmap_dh.set_bad("0.85")

    # Shared date normalisation for the biweekly curves, across both columns.
    all_t = np.concatenate([_tnum(r["line_times"]) for r in R])
    norm = mcolors.Normalize(vmin=all_t.min(), vmax=all_t.max())
    cmap_t = seasonal_doy_cmap() if cmap is None else (
        cmap if hasattr(cmap, "N") else plt.get_cmap(cmap))

    # The two profiles in a figure share one avalanche cone, so they share one
    # ruler: dh, slope and the Hovmöller scale are each set by whichever of the
    # pair dominates. (Across cones the scales are deliberately NOT shared — the
    # cones differ in magnitude and a global scale would flatten the smaller.)
    slope_top = max(float(np.nanmax(np.asarray(r["slope"], dtype="float64")))
                    for r in R)
    slope_top = 10.0 * np.ceil(slope_top * 1.05 / 10.0)     # clean 10° step

    # Distance axis for rows 1-2: one range + one tick step across the columns.
    # Left to matplotlib, each panel picks its own step from its own length
    # (e.g. 194 m -> 25 m ticks but 211 m -> 50 m), so the columns disagree and
    # a distance sits at a different place in each. The Hovmöller (row 3) is
    # deliberately excluded and keeps its own autoscaled distance axis.
    dist_max = max(float(np.asarray(r["dist"], dtype="float64").max()) for r in R)
    dist_step = next(s for s in (10, 20, 25, 50, 100, 200, 500)
                     if dist_max / s <= 8)

    if ylim is None:
        finite = np.concatenate([
            np.asarray(r["line_matrix"], dtype="float64")[
                np.isfinite(np.asarray(r["line_matrix"], dtype="float64"))]
            for r in R])
        lo, hi = float(finite.min()), float(finite.max())
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        ylim = (lo - pad, hi + pad)

    ncol = len(names)                            # 1 (lone profile) or 2 (pair)
    plt.rcParams.update(_PLOT_STYLE)
    fig = plt.figure(figsize=(7.5 * ncol, 11))
    # Every panel keeps its own x/y tick labels and axis labels, so the extra
    # h/wspace is what stops them colliding with the neighbouring panel. The
    # trailing column is the colourbar strip; its width is relative to a single
    # panel, so scale it by ncol to keep the bar the same absolute thickness.
    gs = fig.add_gridspec(3, ncol + 1,
                          width_ratios=[1] * ncol + [0.035 * 2 / ncol],
                          height_ratios=[3, 1.15, 2.4], hspace=0.42, wspace=0.26)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(ncol)] for i in range(3)]
    cax_line = fig.add_subplot(gs[0, ncol])
    # The slope row needs no colourbar, so its slot in the colourbar column is
    # free: it keeps the columns aligned AND hosts the Bergschrund/crevasse
    # legend, centred between the two colourbars.
    ax_leg = fig.add_subplot(gs[1, ncol])
    ax_leg.axis("off")
    cax_dh = fig.add_subplot(gs[2, ncol])

    for j, (name, r) in enumerate(zip(names, R)):
        dist, bs, cv = r["dist"], r["bergschrund"], r["crevasse"]
        ax_l, ax_s, ax_h = axes[0][j], axes[1][j], axes[2][j]
        ax_l.sharex(ax_s)                        # distance axis, rows 1-2

        # Row 1 — biweekly profile lines.
        _draw_crevasse(ax_l, cv)
        _draw_bergschrund(ax_l, _as_positions(bs))
        t = _tnum(r["line_times"])
        M = np.asarray(r["line_matrix"], dtype="float64")
        for i in np.argsort(t):
            ax_l.plot(dist, M[i], color=cmap_t(norm(t[i])), lw=1.3, alpha=0.9)
        ax_l.axhline(0, color="black", lw=0.8, ls=":")
        ax_l.set_title(f"{name} — {site_label}")
        ax_l.grid(True, alpha=0.3)
        ax_l.set_xlabel("Distance along profile (m)")
        ax_l.set_ylabel(ylabel)
        ax_l.set_ylim(*ylim)                     # shared across the cone's pair
        ax_l.set_xlim(0, dist_max)
        ax_l.xaxis.set_major_locator(MultipleLocator(dist_step))

        # Row 2 — reference-DEM slope.
        _draw_crevasse(ax_s, cv)
        _draw_bergschrund(ax_s, _as_positions(bs))
        ax_s.plot(dist, np.asarray(r["slope"], dtype="float64"),
                  color="0.20", lw=1.3)
        ax_s.set_xlabel("Distance along profile (m)")
        ax_s.set_ylim(0, slope_top)              # shared scale, both columns
        ax_s.yaxis.set_major_locator(MultipleLocator(10))
        ax_s.set_xlim(0, dist_max)
        ax_s.xaxis.set_major_locator(MultipleLocator(dist_step))
        ax_s.grid(True, alpha=0.3)
        ax_s.set_ylabel("Slope (°)")

        # Row 3 — Hovmöller (x = date, y = distance): cannot share x with above.
        tn = _tnum(r["htimes"])
        H = np.asarray(r["hmatrix"], dtype="float64")
        if shading == "flat":
            pcm = ax_h.pcolormesh(_cell_edges(tn), _cell_edges(dist), H.T,
                                  cmap=cmap_dh, vmin=-vmax, vmax=vmax,
                                  shading="flat")
        else:
            X, Y = np.meshgrid(tn, dist)
            pcm = ax_h.pcolormesh(X, Y, H.T, cmap=cmap_dh, vmin=-vmax, vmax=vmax,
                                  shading="gouraud")
        _draw_crevasse(ax_h, cv, orient="h", zorder=3, alpha=0.25)
        _draw_bergschrund(ax_h, _as_positions(bs), orient="h", zorder=3)
        ax_h.set_xlabel("Date")
        ax_h.xaxis_date()
        ax_h.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_h.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for lb in ax_h.get_xticklabels():
            lb.set_rotation(30)
            lb.set_horizontalalignment("right")
        # Distance axis deliberately NOT shared here: the Hovmöller keeps its
        # own autoscaled y range (rows 1-2 share theirs).
        ax_h.set_ylabel("Distance along profile (m)")

    # Colourbars: acquisition date for row 1, elevation change for row 3.
    sm = ScalarMappable(norm=norm, cmap=cmap_t)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax_line)
    first = pd.Timestamp(np.asarray(
        R[0]["line_times"], dtype="datetime64[s]").min()).to_period("M").to_timestamp()
    last = max(pd.Timestamp(np.asarray(r["line_times"],
                                       dtype="datetime64[s]").max()) for r in R)
    cb.set_ticks(mdates.date2num(pd.date_range(first, last, freq="2MS")))
    cb.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    cb.set_label("Acquisition date")
    fig.colorbar(pcm, cax=cax_dh, extend="both").set_label(clabel)

    # Legend in the free colourbar-column slot: outside every panel (never over
    # the mesh), centred between the two colourbars.
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        # Anchored at the column's left edge, spilling right into the margin —
        # centring it would push the wide labels left over the slope panel.
        ax_leg.legend(handles, labels, loc="center left",
                      bbox_to_anchor=(0.0, 0.5), frameon=False, fontsize=10,
                      handlelength=1.8, borderaxespad=0.0)

    if filename is None:
        filename = f"grid_{'_'.join(names)}.png"
    _save_or_show(fig, output_dir, filename, save_pdf)


def _save_or_show(fig, output_dir, filename, save_pdf):
    import matplotlib.pyplot as plt
    if output_dir is None:
        plt.show()
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / filename, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out / f"{Path(filename).stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / filename}")


def _daily_grid(times, matrix):
    """Reindex a (T, S) matrix onto a regular daily time axis.

    Non-acquisition days become full NaN columns so the Hovmöller shows the real
    calendar cadence — gaps between acquisitions appear as empty (masked) cells
    instead of being interpolated away. Pairs with ``shading='flat'``; gouraud
    can't span the NaN days. Returns ``(daily_times, daily_matrix)``.
    """
    import numpy as np
    import pandas as pd

    t = pd.to_datetime(np.asarray(times, dtype="datetime64[D]"))
    full = pd.date_range(t.min(), t.max(), freq="D")
    M = np.full((len(full), matrix.shape[1]), np.nan, dtype="float64")
    M[full.get_indexer(t)] = matrix          # place each date; empty days stay NaN
    return full.values.astype("datetime64[D]"), M


def avalanche_cone_profiles(output_dir, profiles_shp, *, plot_dir=None,
                            kind="M3C2_raster", date_from=None, date_to=None,
                            spacing=0.5, smoothing_window=10.0,
                            outlier_thresh=None, outlier_dist=0.0,
                            vmax=None, vmax_pct=98.0, shading="gouraud",
                            hovmoller_daily=False, line_bin_days=14,
                            line_ylim=None, line_nmad_max=0.5,
                            hov_nmad_max=None, hov_nmad_from=None,
                            name_field=None,
                            transition_shp=None, crevasse_shp=None,
                            slope_dem=None, slope_smoothing_window=None,
                            site_label="Changri West", show=False, save_pdf=True):
    """End-to-end: load the signal cube, sample each profile, write the plots.

    For every profile in *profiles_shp* writes, to *plot_dir* (default
    ``output/postprocessing/profiles/``):

    * ``<name>_time_distance.png``  — full-record Hovmöller (date x / distance y)
    * ``<name>_profile_lines.png``  — one curve per ``line_bin_days`` window
      (default 14 = biweekly). Acquisitions with post-coreg stable nmad >=
      ``line_nmad_max`` are gated out first (:func:`nmad_gate`, the same gate
      the biweekly box plot uses), then each window is reduced to its
      NaN-skipping median (:func:`time_bin_average`). Picking a single nearest
      acquisition per window instead — the box plot's rule — was tried and is
      visibly noisier: it forfeits both the outlier rejection and the gap
      filling the median gives. ``line_bin_days=None`` → one curve per date.

    The Hovmöller is unfiltered by default. Set ``hov_nmad_max`` to gate it too,
    and ``hov_nmad_from`` to scope that gate in time (dates before it are kept
    regardless). Changri North uses ``hov_nmad_max=0.5,
    hov_nmad_from="2024-07-14"``: coregistration degrades once the monsoon
    starts, so the failed days are dropped from mid-July on while the clean
    pre-monsoon record is kept whole. With ``hovmoller_daily=True`` the dropped
    days render as empty cells rather than being interpolated over — the gaps
    are visible, which is the point.

    Returns the cube + per-profile ``{dist, matrix}`` so the notebook can
    re-plot without re-sampling.

    Pass ``transition_shp`` (the Bergschrund line) and/or ``crevasse_shp`` (the
    crevasse polygon) to mark them on **both** figures — the Bergschrund as a
    dashed line, the crevasse as a shaded span (vertical on the profile lines,
    horizontal on the Hovmöller, where distance is the y axis). The crossing
    distance / spans are returned per profile as ``bergschrund``/``crevasse``.
    """
    import numpy as np

    stack = load_signal_stack(output_dir, kind=kind,
                              date_from=date_from, date_to=date_to)
    profiles = load_profiles(profiles_shp, name_field=name_field)
    transition = load_transition_line(transition_shp) if transition_shp else None
    crevasse = load_crevasse(crevasse_shp) if crevasse_shp else None
    slope_ras = slope_raster(slope_dem) if slope_dem else None
    out = None if show else Path(
        plot_dir or Path(output_dir) / "output" / "postprocessing" / "profiles")

    # Per-date post-coreg stable nmad → quality filter for the profile lines.
    # Always needed: gates failed coregs AND breaks nearest-slot ties.
    nmad = load_stable_nmad(output_dir, stack.dates)

    results = {}
    _pref = site_label.replace(" ", "") + "_"      # e.g. 'ChangriWest_'
    for name, line in profiles:
        disp = name[len(_pref):] if name.startswith(_pref) else name
        dist, matrix = sample_profile_timeseries(
            stack, line, spacing=spacing, smoothing_window=smoothing_window,
            outlier_thresh=outlier_thresh, outlier_dist=outlier_dist)
        bs = profile_transition_distances(line, transition) if transition else []
        if transition and not bs:
            print(f"  [{disp}] warning: never crosses the transition line")
        cv = profile_crevasse_spans(line, crevasse) if crevasse else []
        if crevasse and not cv:
            print(f"  [{disp}] note: does not reach the crevasse")
        # Same distance axis as `dist` — sampled from the reference DEM, so the
        # slope curve is static and shared by every date on this profile. By
        # default it uses the same rolling mean as the dh curves (feature widths
        # match across panels); slope_smoothing_window overrides it — slope is
        # noisier than dh, so a wider window (e.g. 20 m) can read cleaner.
        sl = None
        if slope_ras is not None:
            sl_win = (smoothing_window if slope_smoothing_window is None
                      else slope_smoothing_window)
            sl = smooth_along_profile(
                sample_profile(slope_ras, line, spacing=spacing)[1],
                spacing=spacing, window=sl_win)
        ttl = f"{disp} — {site_label}"
        # Hovmöller path: optionally gate failed coregs, scoped in time by
        # hov_nmad_from (North: monsoon onward only). Dropped days become empty
        # cells under hovmoller_daily rather than being interpolated across.
        if hov_nmad_max:
            hkeep = nmad_gate(nmad, hov_nmad_max,
                              times=stack.times, from_date=hov_nmad_from)
            htimes_r, hmat_r = stack.times[hkeep], matrix[hkeep]
        else:
            htimes_r, hmat_r = stack.times, matrix
        # Expand the time axis to a regular daily grid so non-acquisition days
        # (and gated-out ones) show as empty cells instead of being smeared over.
        htimes, hmatrix = (_daily_grid(htimes_r, hmat_r)
                           if hovmoller_daily else (htimes_r, hmat_r))
        plot_time_distance(htimes, dist, hmatrix, title=ttl, output_dir=out,
                           filename=f"{disp}_time_distance.png",
                           vmax=vmax, vmax_pct=vmax_pct, shading=shading,
                           bergschrund=bs, crevasse=cv, save_pdf=save_pdf)
        # Profile-line path: gate out failed coregs with the box plot's nmad
        # rule (whole record — NOT scoped by hov_nmad_from), then take the
        # per-window median over the survivors. The median is what keeps the
        # curves clean: it rejects the residual outliers the gate lets through
        # and fills a single day's nodata gaps from its neighbours.
        keep = nmad_gate(nmad, line_nmad_max)
        ltimes, lmat = stack.times[keep], matrix[keep]
        if line_bin_days:
            line_times, line_matrix = time_bin_average(ltimes, lmat,
                                                       bin_days=line_bin_days)
            line_ttl = f"{ttl}  ({line_bin_days}-day medians)"
            print(f"  [{disp}] {int(keep.sum())}/{keep.size} dates after nmad "
                  f"gate → {len(line_times)} {line_bin_days}-day medians")
        else:
            line_times, line_matrix, line_ttl = ltimes, lmat, ttl
        plot_profile_lines(line_times, dist, line_matrix, title=line_ttl,
                           output_dir=out, ylim=line_ylim, bergschrund=bs,
                           crevasse=cv, slope=sl,
                           filename=f"{disp}_profile_lines.png",
                           save_pdf=save_pdf)
        results[disp] = {"dist": dist, "matrix": matrix, "bergschrund": bs,
                         "crevasse": cv, "slope": sl,
                         # exactly what each panel drew, so a composite figure
                         # (:func:`plot_profile_grid`) can re-lay them out
                         # without re-sampling or re-deriving the gate/binning.
                         "line_times": line_times, "line_matrix": line_matrix,
                         "htimes": htimes, "hmatrix": hmatrix}
        print(f"  [{disp}] sampled {matrix.shape[0]} dates × {matrix.shape[1]} "
              f"steps; valid cells {100 * np.isfinite(matrix).mean():.0f}%")
    return {"stack": stack, "profiles": results, "times": stack.times}


# ---------------------------------------------------------------------------
# Part 4 -- relative / absolute accuracy -- PROMOTED to cntp
# ---------------------------------------------------------------------------
# Drawing primitives -> cntp.plot; loaders/compute/orchestrators ->
# cntp.postprocessing. Re-exported so notebook cells importing them from `tools`
# keep working against one source.
from cntp.plot import (  # noqa: F401
    plot_maps_row, plot_relative_accuracy_boxplot, plot_absolute_accuracy_boxes,
    data_window,
)
from cntp.postprocessing import (  # noqa: F401
    per_pixel_obs_count, per_pixel_nmad_map, load_stable_distance_stack,
    load_stable_grid_stack, stable_precision_arrays, pixel_relative_accuracy,
    absolute_accuracy_boxplots,
)

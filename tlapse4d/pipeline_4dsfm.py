"""End-to-end 4D SfM pipeline for a single new date.

Orchestrates the seven library calls that process one day's images from raw
JPEGs to a validated, co-registered point cloud + a registry entry:

  1.  Multi-temporal bundle adjustment   :func:`tlapse4d.metashape.run_multitemporal_ba`
  2.  Single-day fixed-IOP reconstruction :func:`tlapse4d.metashape.run_single_day_fixed_iop`
  3.  Three-stage ASP ICP coreg          :func:`tlapse4d.asp.pc_align_p2p_sp2p`
  3b. M3C2 evaluation on coreg cloud     :func:`tlapse4d.asp.evaluate_coreg`
  4.  Apply transform to camera EOPs     :func:`tlapse4d.asp.apply_coreg_to_cameras_ecef`
  6.  Rebuild cloud with corrected M      :func:`tlapse4d.metashape.rebuild_coreg_cloud`
  6b. M3C2 validation on rebuilt cloud   (inline — py4dgeo + matplotlib)
  7.  Update reference registry          :func:`tlapse4d.metashape.update_registry`

Each step checks for its key output on disk before running; pass
``overwrite=True`` to force a full re-run.
"""

from __future__ import annotations

from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py4dgeo
import rasterio

from tlapse4d.asp import (
    apply_coreg_to_cameras_ecef,
    evaluate_coreg,
    extract_stable_reference,
    pc_align_p2p_sp2p,
)
from tlapse4d.coreg import extract_stable_terrain, run_m3c2
from tlapse4d.io import apply_glacier_mask, load_las, save_las
from tlapse4d.metashape import (
    _normalize_date,
    _utm_epsg,
    discover_images,
    rebuild_coreg_cloud,
    run_multitemporal_ba,
    run_single_day_fixed_iop,
    update_registry,
)
from tlapse4d.plot import plot_dod_histogram
from tlapse4d.raster import (
    build_dem_and_ortho,
    build_dem_and_ortho_p2d,
    build_dod,
    build_reference_dem_and_ortho,
    m3c2_to_raster,
    stable_m3c2_raster,
)


def run_4dsfm_day(
    new_date: str,
    tlcam_dir: Path,
    ref_cloud: Path,
    glacier_mask: Path,
    registry_csv: Path,
    output_dir: Path,
    match_downscale: int = 1,
    depth_downscale: int = 2,
    filter_mode: str = "Mild",
    loc_acc_new: tuple = (0.5, 0.5, 0.5),
    rot_acc_new: tuple = (5.0, 5.0, 5.0),
    ref_downsample: float = 0.4,
    tba_downsample: float = 1.0,
    p2p_max_disp: float = 10.0,
    sp2p_max_disp: float = 5.0,
    m_sp2p_max_disp: float = 2.0,
    p2p_outlier_ratio: float = 0.75,
    sp2p_outlier_ratio: float = 0.75,
    m_sp2p_outlier_ratio: float = 0.75,
    use_ecef: bool = True,
    overwrite: bool = False,
    verbose: bool = False,
    stop_after_ba: bool = False,
    add_to_registry: bool = True,
    run_validation: bool = True,
    max_unaligned: int = 10,
    time_window: tuple[int, int] | None = None,
    exclude_cameras: list[dict] | None = None,
) -> dict | None:
    """Run the full 4D SfM pipeline for *new_date*.

    Parameters
    ----------
    new_date : str
        Date to process (``YYYY-MM-DD``).
    tlcam_dir : Path
        Root directory containing ``C1_renamed/``, ``C2_renamed/``, ….
    ref_cloud : Path
        Reference point cloud (LAZ/LAS) in UTM.
    glacier_mask : Path
        Glacier polygon shapefile (same CRS as point clouds).
    registry_csv : Path
        Reference registry CSV managed by :func:`tlapse4d.metashape.update_registry`.
        The UTM zone is derived from the mean ``lon`` column.
    output_dir : Path
        Root output directory (parent of ``output/``).
    match_downscale, depth_downscale : int
        Metashape ``matchPhotos`` and ``buildDepthMaps`` downscales.
    loc_acc_new, rot_acc_new : tuple
        Position (m) and rotation (°) accuracy priors for new-day cameras.
    ref_downsample : float
        Fraction of reference points kept for ICP (0 < f ≤ 1).
    tba_downsample : float
        Fraction of TBA points kept for ICP (1.0 = full).
    p2p_max_disp, sp2p_max_disp, m_sp2p_max_disp : float
        Max ICP correspondence distance [m] for stages 1, 2, 3.
    use_ecef : bool
        Run ICP in ECEF space (recommended — avoids flat-Cartesian distortion).
    overwrite : bool
        When False (default) each step is skipped if its key output already
        exists; useful for resuming after a crash.
    verbose : bool
        Print each ``pc_align`` command and stdout.
    stop_after_ba : bool
        When True, run only Step 1 (multi-temporal bundle adjustment) and
        return immediately so the user can inspect the BA outputs
        (``4D_SfM/<date>_cameras_4DSfM.csv`` and ``4D_SfM/adjusted_calib_4DSfM/``)
        before deciding whether to continue. Default False runs the full
        pipeline.
    add_to_registry : bool
        When True (default), Step 7 appends the day's validated cameras to
        the reference registry CSV so future multi-temporal BA runs in Step 1
        bundle-adjust against them. Set False to keep the registry frozen
        at its current state — useful when the day's coreg is shaky and you
        don't want to pollute the reference baseline that downstream dates
        will read. All other steps (1–6 + 6b) still run normally, so the
        DEM / ortho / DoD / M3C2 outputs for this date are produced
        regardless.
    run_validation : bool
        When True (default), run Step 6 (Metashape rebuild of the co-registered
        cloud) + Step 6b (M3C2 validation that the ASP transform propagated).
        Set False to skip both — the rasters use the Step 3 coreg cloud
        (``aligned_las``), not the rebuilt ``validated_laz``, and with the
        registry frozen nothing else consumes it, so validation is dead weight.
        The coreg M3C2 plot (Step 3b) is independent and always runs. When
        skipped, ``validation_med`` / ``validation_nmad`` / ``validation_std`` stay NaN.
    time_window : tuple[int, int], optional
        Inclusive capture-hour window ``(start_hour, end_hour)`` forwarded to
        :func:`tlapse4d.metashape.discover_images`. Keeps only frames captured in
        that daytime window (e.g. ``(9, 17)``) and drops off-schedule night /
        motion-triggered captures before they reach Step 1/Step 2. Because the
        dropped night frames never align, they no longer inflate the Step-1
        4D SfM ``max_unaligned`` gate, so ``max_unaligned`` can be tightened
        without falsely skipping good days. ``None`` (default) keeps every frame.
    exclude_cameras : list[dict], optional
        Camera-exclusion rules forwarded to
        :func:`tlapse4d.metashape.discover_images`, dropping specific cameras over
        date ranges (e.g. ``[{"camera": "C8", "from": "2023-08-01"}]`` for a
        boulder-displaced camera) before Step 1. The camera keeps its earlier
        good frames as reference anchors but contributes no new-day pose, sensor
        or group. ``None`` (default) keeps every camera. Site-specific rules
        belong in the site config, not in library code.

    Returns
    -------
    dict
        ``date, tba_las_path, aligned_las, validated_laz, cameras_coreg_csv,
        transform_path, coreg_med_before, coreg_med_after, coreg_nmad_after, coreg_std_after,
        validation_med, validation_nmad, validation_std``.
    """
    output_dir   = Path(output_dir)
    ref_cloud    = Path(ref_cloud)
    registry_csv = Path(registry_csv)

    date_dir   = output_dir / "output" / new_date
    sfm_dir    = date_dir / "4D_SfM"
    single_dir = date_dir / "single_day"
    coreg_dir  = date_dir / "coreg"
    val_dir    = single_dir / "validation"

    # ── Discover images and derive UTM zone ──────────────────────────────
    reg_df   = pd.read_csv(registry_csv)
    # Via to_numpy() so the mean is a plain scalar rather than a pandas object.
    utm_epsg = _utm_epsg(float(reg_df["lon"].to_numpy().mean()))

    by_date = discover_images(tlcam_dir, time_window=time_window,
                              exclude_cameras=exclude_cameras)
    if new_date not in by_date:
        raise ValueError(f"No images found for date {new_date} under {tlcam_dir}")
    date_images = by_date[new_date]
    if time_window is not None:
        n_kept = sum(len(v) for v in date_images.values())
        print(f"  [time_window {time_window[0]:02d}:00–{time_window[1]:02d}:59] "
              f"{n_kept} daytime frame(s) kept for {new_date}")

    ecef_epsg = utm_epsg if use_ecef else None

    # Pre-derive every output path so skip logic works regardless of order.
    cameras_4dsfm_csv = sfm_dir    / f"{new_date}_cameras_4DSfM.csv"
    calib_dir_out     = sfm_dir    / "adjusted_calib_4DSfM"
    tba_las_path      = single_dir / f"{new_date}_cloud.las"
    cameras_csv       = single_dir / f"{new_date}_cameras.csv"
    psx_path          = single_dir / f"{new_date}.psx"
    aligned_las       = coreg_dir  / f"{new_date}_cloud_coreg_hsfm.las"
    transform_path    = coreg_dir  / "stage3" / "run-transform.txt"
    cameras_coreg_csv = coreg_dir  / f"{new_date}_cameras_coreg.csv"
    stable_tba_path   = coreg_dir  / "stable_tba" / f"{new_date}_cloud_coreg_hsfm_stable.laz"
    validated_laz     = val_dir    / f"{new_date}_cloud_validated.laz"
    validated_stable  = val_dir    / f"{new_date}_cloud_validated_stable.laz"

    # Shared reference cache — downsampled ref + stable ref + ref diagnostic
    # plots all depend only on (ref_cloud, ref_downsample, glacier_mask), so
    # they're built once and reused for every day in output/.
    ref_cache_dir = output_dir / "output" / "_ref_cache"
    ref_ds_stem   = ref_cloud.stem + f"_ds{ref_downsample:.2f}"
    ref_ds_path   = ref_cache_dir / f"{ref_ds_stem}.las"
    stable_ref    = ref_cache_dir / f"{ref_ds_stem}_stable.las"
    ref_plot_dir  = ref_cache_dir / "m3c2_plots" / "reference"

    coreg_med_before = coreg_med_after = coreg_nmad_after = coreg_std_after = float("nan")
    validation_med   = validation_nmad = validation_std  = float("nan")

    # ── Step 1: multi-temporal bundle adjustment ─────────────────────────
    # The 4D SfM alignment gate lives here: when >= max_unaligned new-day
    # cameras fail to align, run_multitemporal_ba writes this skip marker and
    # raises, abandoning the whole date. Honour the marker on re-runs so the
    # futile BA isn't repeated; overwrite=True forces a retry and clears it.
    ba_skip_marker = sfm_dir / f"{new_date}_unaligned_skip.txt"
    if cameras_4dsfm_csv.exists() and not overwrite:
        print(f"[Step 1] Skipping — {cameras_4dsfm_csv.name} exists")
    elif ba_skip_marker.exists() and not overwrite:
        raise RuntimeError(
            f"{new_date}: skipped earlier by the 4D SfM alignment gate "
            f"({ba_skip_marker.name}); pass overwrite=True to retry."
        )
    else:
        print(f"\n[Step 1] Multi-temporal BA — {new_date}")
        if ba_skip_marker.exists():
            ba_skip_marker.unlink()   # retrying (overwrite) — clear stale marker
        cameras_4dsfm_csv, calib_dir_out = run_multitemporal_ba(
            date                   = new_date,
            date_images            = date_images,
            reference_registry_csv = registry_csv,
            output_dir             = output_dir,
            utm_epsg               = utm_epsg,
            match_downscale        = match_downscale,
            loc_acc_new            = loc_acc_new,
            rot_acc_new            = rot_acc_new,
            verbose                = verbose,
            max_unaligned          = max_unaligned,
        )

    if stop_after_ba:
        print("\n[stop_after_ba=True] Halting after Step 1.")
        return None

    # ── Step 2: single-day fixed-IOP reconstruction ──────────────────────
    # No alignment gate here — the only cloud-cover gate is in Step 1 (4D SfM).
    # If Step 1 passed, this date is worth reconstructing.
    if tba_las_path.exists() and not overwrite:
        print(f"[Step 2] Skipping — {tba_las_path.name} exists")
    else:
        print(f"\n[Step 2] Single-day fixed IOP — {new_date}")
        tba_las_path, cameras_csv = run_single_day_fixed_iop(
            date            = new_date,
            date_images     = date_images,
            calib_dir       = calib_dir_out,
            cameras_csv     = cameras_4dsfm_csv,
            output_dir      = output_dir,
            utm_epsg        = utm_epsg,
            match_downscale = match_downscale,
            depth_downscale = depth_downscale,
            filter_mode     = filter_mode,
            loc_acc         = loc_acc_new,
            rot_acc         = rot_acc_new,
            verbose         = verbose,
        )

    # ── Stable reference (shared cache across all days) ──────────────────
    # Downsampled ref + glacier-masked, slope/NDWI-filtered stable ref live
    # under output/_ref_cache/. extract_stable_reference also writes the
    # reference NDWI + RGB diagnostic plots there (once, reused across days).
    ref_cache_dir.mkdir(parents=True, exist_ok=True)
    if not ref_ds_path.exists():
        print(f"  Saving downsampled reference ({ref_downsample:.0%}) → {ref_ds_path.name}")
        save_las(load_las(ref_cloud, downsample_factor=ref_downsample), ref_ds_path,
                 crs=utm_epsg)
    stable_ref = extract_stable_reference(
        ref_cloud_path    = ref_ds_path,
        output_dir        = ref_cache_dir,
        glacier_mask_path = glacier_mask,
        plot_dir          = ref_plot_dir,
    )

    # ── Step 3: ASP three-stage ICP ──────────────────────────────────────
    # Pass the pre-downsampled ref + ref_downsample_factor=1.0 so pc_align
    # doesn't re-create a per-day downsample under coreg/downsampled/.
    if overwrite or not aligned_las.exists():
        print(f"\n[Step 3] ASP 3-stage ICP — {new_date}")
        aligned_las, transform_path = pc_align_p2p_sp2p(
            tba_las                 = tba_las_path,
            ref_las                 = ref_ds_path,
            output_dir              = coreg_dir,
            stable_ref_las          = stable_ref,
            p2p_max_displacement    = p2p_max_disp,
            sp2p_max_displacement   = sp2p_max_disp,
            m_sp2p_max_displacement = m_sp2p_max_disp,
            p2p_outlier_ratio       = p2p_outlier_ratio,
            sp2p_outlier_ratio      = sp2p_outlier_ratio,
            m_sp2p_outlier_ratio    = m_sp2p_outlier_ratio,
            ref_downsample_factor   = 1.0,
            tba_downsample_factor   = tba_downsample,
            utm_epsg                = ecef_epsg,
            verbose                 = verbose,
        )
    else:
        print(f"[Step 3] Skipping — {aligned_las.name} exists")

    # ── Step 3b: M3C2 on coreg cloud (uses cached stable reference) ──────
    if overwrite or not stable_tba_path.exists():
        print(f"\n[Step 3b] Evaluate co-registration — {new_date}")
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
        coreg_med_before  = eval_result["med_before"]
        coreg_med_after   = eval_result["med_after"]
        coreg_nmad_after  = eval_result["nmad_after"]
        coreg_std_after   = eval_result["std_after"]

        # Georeferenced raster of the after-coreg stable-terrain residual
        # (co-registration uncertainty) → single_day, for stacking over a season.
        stable_m3c2_raster(
            ref_stable = eval_result["ref_stable"],
            distances  = eval_result["dist_after"],
            out_path   = single_dir / f"{new_date}_stable_m3c2.tif",
            utm_epsg   = utm_epsg,
        )
    else:
        print("[Step 3b] Skipping — stable TBA exists")
        # Reload the persisted coreg stats so the stable-terrain M3C2 numbers
        # aren't NaN on a cached re-run (Step 3b is what normally sets them;
        # evaluate_coreg wrote them to coreg/<date>_m3c2_stats.csv).
        stats_csv = coreg_dir / f"{new_date}_m3c2_stats.csv"
        if stats_csv.exists():
            _s = pd.read_csv(stats_csv, index_col="coreg")
            coreg_med_before = float(_s.loc["before", "median"])
            coreg_med_after  = float(_s.loc["after",  "median"])
            coreg_nmad_after = float(_s.loc["after",  "nmad"])
            coreg_std_after  = float(_s.loc["after",  "std"])

    # ── Step 4: apply transform to camera EOPs ───────────────────────────
    if overwrite or not cameras_coreg_csv.exists():
        print(f"\n[Step 4] Apply transform to cameras — {new_date}")
        # ECEF mode: utm_epsg=None in the camera helper → WGS84→ECEF→T→ECEF→WGS84
        cam_epsg = None if use_ecef else utm_epsg
        apply_coreg_to_cameras_ecef(
            cameras_csv    = cameras_csv,
            transform_path = transform_path,
            out_csv        = cameras_coreg_csv,
            utm_epsg       = cam_epsg,
        )
    else:
        print(f"[Step 4] Skipping — {cameras_coreg_csv.name} exists")

    # ── Step 6: rebuild cloud in Metashape with corrected matrix ─────────
    # buildDepthMaps + buildPointCloud + export must run in the same session
    # as the matrix assignment, otherwise the corrected M is silently lost
    # on the next doc.open() (Metashape recomputes M from GPS priors).
    if run_validation and (overwrite or not validated_laz.exists()):
        print(f"\n[Step 6] Rebuild co-registered cloud — {new_date}")
        rebuild_coreg_cloud(
            psx_path        = psx_path,
            transform_path  = transform_path,
            output_laz      = validated_laz,
            depth_downscale = depth_downscale,
            utm_epsg        = utm_epsg,
            filter_mode     = filter_mode,
        )
    elif not run_validation:
        print("\n[Step 6] Skipping — run_validation=False")
    else:
        print(f"[Step 6] Skipping — {validated_laz.name} exists")

    # ── Step 6b: M3C2 between stable TBA (Step 3b) and validated cloud ───
    # Near-zero → ASP transform was correctly propagated into the rebuild.
    if run_validation and (overwrite or not validated_stable.exists()):
        print(f"\n[Step 6b] Validate rebuilt cloud — {new_date}")
        val_cloud = load_las(validated_laz, downsample_factor=tba_downsample)
        if glacier_mask is not None:
            val_cloud = apply_glacier_mask(val_cloud, glacier_mask)
        _, val_stable_arr = extract_stable_terrain(val_cloud)

        val_dir.mkdir(parents=True, exist_ok=True)
        save_las(val_stable_arr, validated_stable, crs=utm_epsg)

        stable_tba_cloud = load_las(stable_tba_path)
        epoch_tba = py4dgeo.Epoch(stable_tba_cloud[:, :3])
        epoch_val = py4dgeo.Epoch(val_stable_arr[:, :3])
        validation_med, validation_nmad, validation_std, dist_val = run_m3c2(epoch_tba, epoch_val)
        print(f"  M3C2  median: {validation_med:+.2f} m  nmad: {validation_nmad:.2f} m  std: {validation_std:.2f} m")

        val_plot_dir = val_dir / "validation_plots"
        val_plot_dir.mkdir(parents=True, exist_ok=True)
        # Clip only the histogram display so a few outliers don't squash the
        # x-axis. Stats shown in the label come from run_m3c2 over the full
        # (un-clipped) distance array, so they match the printed values.
        d      = dist_val[~np.isnan(dist_val)]
        d_plot = np.clip(d, -3 * validation_nmad, 3 * validation_nmad)
        fig, ax = plt.subplots()
        ax.hist(d_plot, bins=60, color="steelblue", alpha=0.8)
        ax.axvline(validation_med, color="tomato", linestyle="--", linewidth=1.5,
                   label=f"median = {validation_med:+.2f} m  nmad = {validation_nmad:.2f} m")
        ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel("M3C2 distance (m)")
        ax.set_ylabel("Count")
        ax.set_title(f"Validation — stable TBA vs validated cloud ({new_date})")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(val_plot_dir / "m3c2_validation.png", dpi=150)
        plt.close(fig)
    elif not run_validation:
        print("[Step 6b] Skipping — run_validation=False")
    else:
        print(f"[Step 6b] Skipping — {validated_stable.name} exists")

    # ── Step 7: update reference registry ────────────────────────────────
    if not add_to_registry:
        print("\n[Step 7] Skipping — add_to_registry=False (registry kept frozen)")
    else:
        reg_df_fresh = pd.read_csv(registry_csv)
        if new_date not in reg_df_fresh["date"].astype(str).values:
            print(f"\n[Step 7] Update registry — {new_date}")
            update_registry(
                registry_csv      = registry_csv,
                date              = new_date,
                date_images       = date_images,
                cameras_coreg_csv = cameras_coreg_csv,
                calib_dir         = calib_dir_out,
            )
        else:
            print(f"[Step 7] Skipping — {new_date} already in registry")

    return {
        "date":              new_date,
        "tba_las_path":      str(tba_las_path),
        "aligned_las":       str(aligned_las),
        "validated_laz":     str(validated_laz),
        "cameras_coreg_csv": str(cameras_coreg_csv),
        "transform_path":    str(transform_path),
        "coreg_med_before":  coreg_med_before,
        "coreg_med_after":   coreg_med_after,
        "coreg_nmad_after":  coreg_nmad_after,
        "coreg_std_after":   coreg_std_after,
        "validation_med":    validation_med,
        "validation_nmad":   validation_nmad,
        "validation_std":    validation_std,
    }


# ---------------------------------------------------------------------------
# Per-date orchestrator: 4D SfM pipeline + DEM / ortho / DoD / stable / M3C2 raster
# ---------------------------------------------------------------------------

def run_4dsfm_day_with_rasters(
    new_date: str,
    tlcam_dir: Path,
    ref_cloud: Path,
    glacier_mask: Path,
    registry_csv: Path,
    output_dir: Path,
    # ── SfM pipeline kwargs (forwarded to run_4dsfm_day) ──────────────────
    match_downscale: int = 1,
    depth_downscale: int = 2,
    filter_mode: str = "Mild",
    loc_acc_new: tuple = (0.5, 0.5, 0.5),
    rot_acc_new: tuple = (5.0, 5.0, 5.0),
    ref_downsample: float = 0.4,
    tba_downsample: float = 1.0,
    p2p_max_disp: float = 10.0,
    sp2p_max_disp: float = 5.0,
    m_sp2p_max_disp: float = 2.0,
    p2p_outlier_ratio: float = 0.75,
    sp2p_outlier_ratio: float = 0.75,
    m_sp2p_outlier_ratio: float = 0.75,
    use_ecef: bool = True,
    overwrite: bool = False,
    verbose: bool = False,
    add_to_registry: bool = True,
    run_validation: bool = True,
    max_unaligned: int = 10,
    time_window: tuple[int, int] | None = None,
    exclude_cameras: list[dict] | None = None,
    # ── Raster knobs ──────────────────────────────────────────────────────
    res: float = 1.0,
    max_gap_pixels: int = 1,
    ref_cloud_downsample: float = 0.25,
    m3c2_ref_downsample: float = 0.25,
    slope_threshold: float = 60.0,
    utm_epsg: int | None = None,
    overwrite_ref_dem: bool = False,
    overwrite_day_dem: bool = False,
    overwrite_dod: bool = False,
    overwrite_stable: bool = False,
    overwrite_stable_dod: bool = False,
    overwrite_m3c2: bool = False,
    dem_method: str = "point2dem",
) -> dict:
    """Run 4D SfM Steps 1–7 + build every per-date raster output.

    Wrapper that bundles :func:`run_4dsfm_day` with the per-date raster
    pipeline (DEM + orthoimage + DoD + stable-terrain DoD + M3C2 raster,
    each with its histogram PNG). Intended as the single library entry
    point that a minimal batch notebook (see ``4d_sfm_dem_monthly.ipynb``)
    calls in a loop.

    Outputs (all under ``<output_dir>/output/<new_date>/single_day/``
    unless otherwise noted):

    - ``<date>_dem.tif`` + ``<date>_ortho.tif`` (per-day DEM + ortho)
    - ``<date>_DOD.tif`` + ``<date>_dod_histogram.png``
    - ``<date>_M3C2_raster.tif`` + ``<date>_m3c2_raster_histogram.png``

    Stable-terrain co-registration accuracy is the Step-3b M3C2 (before/after),
    saved under ``<date>/coreg/m3c2_plots/`` as ``<date>_m3c2_distances.png``
    (histogram) and ``<date>_m3c2_spatial.png`` (+ PDF) — this replaces the
    former stable-DoD raster.

    The shared reference rasters land in ``<output_dir>/output/_ref_cache/``
    (built once, skipped on subsequent dates):

    - ``reference_dem.tif`` + ``reference_ortho.tif``

    Parameters
    ----------
    new_date, tlcam_dir, ref_cloud, glacier_mask, registry_csv, output_dir :
        See :func:`run_4dsfm_day`.
    match_downscale, depth_downscale, filter_mode, loc_acc_new, rot_acc_new, ref_downsample,
    tba_downsample, p2p_max_disp, sp2p_max_disp, m_sp2p_max_disp, use_ecef,
    overwrite, verbose, add_to_registry, max_unaligned, time_window, exclude_cameras :
        Forwarded to :func:`run_4dsfm_day`. ``time_window=(start_hour,
        end_hour)`` keeps only daytime frames (drops night / motion-triggered
        captures) so the Step-1 4D SfM ``max_unaligned`` gate can be tightened safely.
        ``exclude_cameras=[{"camera": "C8", "from": "2023-08-01"}]`` drops a
        compromised camera over a date range before it reaches Step 1.
    res : float
        Output raster pixel size in metres. Default 1.0.
    max_gap_pixels : int
        Forwarded to :func:`tlapse4d.raster.build_dem_and_ortho`.
    ref_cloud_downsample : float
        Extra load-time downsample fed to ``griddata`` when building the
        reference DEM. Default 0.25.
    m3c2_ref_downsample : float
        Load-time downsample for the reference cloud fed to py4dgeo's M3C2
        KDTree. Default 0.25 (matches the typical 50 M-pt cache → ~12.5 M
        corepoints; keeps M3C2 runtime/memory tractable).
    slope_threshold : float
        Stable-terrain slope cutoff [°]. Default 60.
    utm_epsg : int, optional
        EPSG code for the raster CRS. When ``None``, parsed from
        ``ref_cloud``'s LAS header.
    overwrite_ref_dem, overwrite_day_dem, overwrite_dod, overwrite_m3c2 :
        Per-raster overwrite flags. Default ``False`` skips if the file is
        on disk.
    overwrite_stable, overwrite_stable_dod :
        Accepted for backward compatibility but unused — the stable-DoD raster
        was removed in favour of the Step-3b M3C2 on stable terrain.
    dem_method : str
        DEM rasteriser. ``"point2dem"`` (default) — ASP ``point2dem``, the
        published HSfM method (Gaussian distance-weighted average / "IDW",
        1-cell search radius); fast, multithreaded, leaves large gaps as
        nodata. ``"cubic"`` — scipy Clough-Tocher ``griddata``; slow and
        interpolates across gaps. Both reference and day DEMs use the chosen
        method, anchored to the same grid so the DoD differences cleanly.

    Returns
    -------
    dict
        ``{"date": str, "sfm": <run_4dsfm_day result>, "dod_stats": dict,
        "stable_stats": dict, "m3c2_stats": dict}``. Each ``*_stats`` dict
        comes from :func:`tlapse4d.plot.plot_dod_histogram` and contains
        ``median``, ``mean``, ``std``, ``n`` over finite raster pixels.
    """

    output_dir   = Path(output_dir)
    ref_cloud    = Path(ref_cloud)
    registry_csv = Path(registry_csv)

    # ── Step 1–7: 4D SfM pipeline ────────────────────────────────────────
    sfm_result = run_4dsfm_day(
        new_date        = new_date,
        tlcam_dir       = tlcam_dir,
        ref_cloud       = ref_cloud,
        glacier_mask    = glacier_mask,
        registry_csv    = registry_csv,
        output_dir      = output_dir,
        match_downscale = match_downscale,
        depth_downscale = depth_downscale,
        filter_mode     = filter_mode,
        loc_acc_new     = loc_acc_new,
        rot_acc_new     = rot_acc_new,
        ref_downsample  = ref_downsample,
        tba_downsample  = tba_downsample,
        p2p_max_disp         = p2p_max_disp,
        sp2p_max_disp        = sp2p_max_disp,
        m_sp2p_max_disp      = m_sp2p_max_disp,
        p2p_outlier_ratio    = p2p_outlier_ratio,
        sp2p_outlier_ratio   = sp2p_outlier_ratio,
        m_sp2p_outlier_ratio = m_sp2p_outlier_ratio,
        use_ecef             = use_ecef,
        overwrite       = overwrite,
        verbose         = verbose,
        add_to_registry = add_to_registry,
        run_validation  = run_validation,
        max_unaligned   = max_unaligned,
        time_window     = time_window,
        exclude_cameras = exclude_cameras,
    )
    if sfm_result is None:
        # run_4dsfm_day only returns None for stop_after_ba, which this function does not expose.
        # Guarded anyway: the raster steps below index sfm_result and would otherwise fail with an
        # opaque "NoneType is not subscriptable" three hundred lines later.
        raise RuntimeError(
            f"{new_date}: the SfM step returned no result, so the raster products cannot be built."
        )

    # ── Resolve shared paths + CRS ───────────────────────────────────────
    day_dir       = output_dir / "output" / new_date
    aligned_las   = day_dir / "coreg" / f"{new_date}_cloud_coreg_hsfm.las"
    single_day    = day_dir / "single_day"
    ref_cache_dir = output_dir / "output" / "_ref_cache"
    ref_ds_path   = ref_cache_dir / f"{ref_cloud.stem}_ds{ref_downsample:.2f}.las"
    ref_input     = ref_ds_path if ref_ds_path.exists() else ref_cloud

    if utm_epsg is None:
        with laspy.open(ref_cloud) as _f:
            utm_epsg = _f.header.parse_crs().to_epsg()

    # ── Reference + per-day DEM + ortho ──────────────────────────────────
    # Both DEMs are anchored to ref_input so their grids match and the DoD
    # differences without resampling. (The cubic day DEM previously anchored
    # to the full ref_cloud, whose bbox is ~1 px taller than the ds ref_input
    # → a 1-row DoD shape mismatch; anchoring both to ref_input fixes it.)
    if dem_method == "point2dem":
        ref_dem, ref_ortho = build_dem_and_ortho_p2d(
            cloud_las        = ref_input,
            ref_las          = ref_input,
            out_dir          = ref_cache_dir,
            name_stem        = "reference",
            res              = res,
            max_gap_pixels   = max_gap_pixels,
            utm_epsg         = utm_epsg,
            cloud_downsample = ref_cloud_downsample,
            overwrite        = overwrite_ref_dem,
            verbose          = verbose,
        )
        dem, ortho = build_dem_and_ortho_p2d(
            cloud_las        = aligned_las,
            ref_las          = ref_input,
            out_dir          = single_day,
            name_stem        = new_date,
            res              = res,
            max_gap_pixels   = max_gap_pixels,
            utm_epsg         = utm_epsg,
            cloud_downsample = tba_downsample,
            overwrite        = overwrite_day_dem,
            verbose          = verbose,
        )
    elif dem_method == "cubic":
        ref_dem, ref_ortho = build_reference_dem_and_ortho(
            ref_cloud_path   = ref_input,
            cache_dir        = ref_cache_dir,
            res              = res,
            max_gap_pixels   = max_gap_pixels,
            utm_epsg         = utm_epsg,
            cloud_downsample = ref_cloud_downsample,
            overwrite        = overwrite_ref_dem,
        )
        dem, ortho = build_dem_and_ortho(
            cloud_las        = aligned_las,
            ref_las          = ref_input,
            out_dir          = single_day,
            name_stem        = new_date,
            res              = res,
            max_gap_pixels   = max_gap_pixels,
            utm_epsg         = utm_epsg,
            cloud_downsample = tba_downsample,
            overwrite        = overwrite_day_dem,
        )
    else:
        raise ValueError(
            f"dem_method must be 'point2dem' or 'cubic', got {dem_method!r}"
        )

    # ── DoD + histogram ──────────────────────────────────────────────────
    dod_path = build_dod(
        ref_dem_path = ref_dem,
        day_dem_path = dem,
        out_path     = single_day / f"{new_date}_DOD.tif",
        overwrite    = overwrite_dod,
    )
    with rasterio.open(dod_path) as src:
        dod_values = src.read(1)
    dod_stats = plot_dod_histogram(
        dod_values,
        output_dir = single_day,
        title      = f"DoD — {new_date}",
        filename   = f"{new_date}_dod_histogram.png",
    )

    # ── Stable-terrain accuracy = Step-3b M3C2 (before/after co-registration) ──
    # The M3C2 on stable terrain (histogram + spatial map in coreg/m3c2_plots/)
    # supersedes the old stable-DoD raster, so no DOD_stable.tif is produced.
    stable_stats = {
        "median": sfm_result["coreg_med_after"],
        "nmad":   sfm_result["coreg_nmad_after"],
        "std":    sfm_result["coreg_std_after"],
    }

    # ── M3C2 distance raster + histogram ─────────────────────────────────
    m3c2_raster_path = m3c2_to_raster(
        ref_las         = ref_ds_path if ref_ds_path.exists() else ref_cloud,
        day_las         = aligned_las,
        out_path        = single_day / f"{new_date}_M3C2_raster.tif",
        grid_anchor_las = ref_cloud,
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
        filename   = f"{new_date}_m3c2_raster_histogram.png",
    )

    print(
        f"\n  [{new_date}] DoD med={dod_stats['median']:+.2f} m  |  "
        f"stable M3C2 med={stable_stats['median']:+.2f} m  |  "
        f"M3C2 raster med={m3c2_stats['median']:+.2f} m"
    )

    return {
        "date":         new_date,
        "sfm":          sfm_result,
        "dod_stats":    dod_stats,
        "stable_stats": stable_stats,
        "m3c2_stats":   m3c2_stats,
    }


# ---------------------------------------------------------------------------
# Multi-date orchestration: pick the dates, run them, summarise
# ---------------------------------------------------------------------------

def select_dates(
    tlcam_dir: str | Path,
    registry_csv: str | Path,
    date_from: str | None = None,
    date_to: str | None = None,
    time_window: tuple[int, int] | None = None,
    exclude_cameras: list[dict] | None = None,
) -> list[str]:
    """Return the dates present in *tlcam_dir* that are worth processing.

    A date qualifies when it has time-lapse imagery, is not already a reference
    day in the registry, and falls inside the requested bounds. Reference days
    are the baseline the multi-temporal bundle adjustment aligns *against*, so
    reprocessing them as new dates would be circular.

    Pass the same ``time_window`` / ``exclude_cameras`` the pipeline will run
    with. Discovering dates without them can select a date whose only frames are
    then filtered away, which fails later inside :func:`run_4dsfm_day` with
    "No images found for date".

    Parameters
    ----------
    tlcam_dir : str | Path
        Directory of standardised time-lapse images (searched recursively).
    registry_csv : str | Path
        Reference registry CSV; its ``date`` column lists the reference days.
    date_from, date_to : str, optional
        Inclusive ``YYYY-MM-DD`` bounds. ISO dates compare chronologically as
        plain strings, so no parsing is needed. ``None`` leaves that end open.
    time_window, exclude_cameras :
        Forwarded to :func:`tlapse4d.metashape.discover_images`.

    Returns
    -------
    list[str]
        Dates to process, ascending.
    """
    by_date = discover_images(tlcam_dir, time_window=time_window,
                              exclude_cameras=exclude_cameras)
    all_dates = sorted(by_date)

    ref_dates = set(pd.read_csv(Path(registry_csv))["date"].map(_normalize_date))

    dates = [
        d for d in all_dates
        if d not in ref_dates
        and (date_from is None or d >= date_from)
        and (date_to   is None or d <= date_to)
    ]

    print(f"Discovered {len(all_dates)} dated image set(s); "
          f"{len(ref_dates)} reference day(s) excluded.")
    print(f"{len(dates)} date(s) to process"
          + (f": {dates[0]} … {dates[-1]}" if dates else "."))
    return dates


def run_batch(
    dates: list[str],
    tlcam_dir: str | Path,
    ref_cloud: str | Path,
    glacier_mask: str | Path,
    registry_csv: str | Path,
    output_dir: str | Path,
    summary_csv: str | Path | None = None,
    **params,
) -> "pd.DataFrame":
    """Run :func:`run_4dsfm_day_with_rasters` over *dates* and summarise the results.

    A date that fails does not abort the batch: the error is recorded against
    that date and the run continues, the same way
    :func:`tlapse4d.preprocess.homogenize_images` records a skipped image rather
    than abandoning the folder. A season's processing is hours long and one bad
    date — a failed alignment gate, a cloudy day — should not cost the rest.

    Parameters
    ----------
    dates : list[str]
        Dates to process, ``YYYY-MM-DD``. See :func:`select_dates`.
    tlcam_dir, ref_cloud, glacier_mask, registry_csv, output_dir :
        See :func:`run_4dsfm_day`.
    summary_csv : str | Path, optional
        Where to write the summary table. Default
        ``<output_dir>/output/batch_summary.csv``; pass ``None``-equivalent by
        giving an explicit path elsewhere.
    **params
        Forwarded verbatim to :func:`run_4dsfm_day_with_rasters` — every SfM and
        raster knob. Keeping them in one mapping lets a caller hold a named
        parameter set per site.

    Returns
    -------
    pd.DataFrame
        One row per date: ``date`` plus the DoD, stable-terrain and M3C2 raster
        median/std, or an ``error`` column for dates that failed.
    """
    import traceback

    output_dir = Path(output_dir)
    summary_csv = (Path(summary_csv) if summary_csv is not None
                   else output_dir / "output" / "batch_summary.csv")

    results: list[dict] = []
    for d in dates:
        print(f"\n{'#' * 70}\n#  {d}\n{'#' * 70}")
        try:
            results.append(run_4dsfm_day_with_rasters(
                new_date     = d,
                tlcam_dir    = tlcam_dir,
                ref_cloud    = ref_cloud,
                glacier_mask = glacier_mask,
                registry_csv = registry_csv,
                output_dir   = output_dir,
                **params,
            ))
        except Exception as e:  # noqa: BLE001 — any failure → record and continue
            print(f"\n  {d} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append({"date": d, "error": repr(e)})

    rows = []
    for r in results:
        if "error" in r:
            rows.append({"date": r["date"], "error": r["error"]})
        else:
            rows.append({
                "date":          r["date"],
                "dod_median":    r["dod_stats"]["median"],    "dod_std":    r["dod_stats"]["std"],
                "stable_median": r["stable_stats"]["median"], "stable_std": r["stable_stats"]["std"],
                "m3c2_median":   r["m3c2_stats"]["median"],   "m3c2_std":   r["m3c2_stats"]["std"],
            })
    df = pd.DataFrame(rows)

    n_failed = sum("error" in r for r in results)
    print(f"\n{'=' * 70}\n  Batch summary — {len(results)} date(s), {n_failed} failed\n{'=' * 70}")
    for r in results:
        if "error" in r:
            print(f"  {r['date']} : ERROR — {r['error']}")
        else:
            print(
                f"  {r['date']} : "
                f"DoD med={r['dod_stats']['median']:+.2f} m  std={r['dod_stats']['std']:.2f}  |  "
                f"stable med={r['stable_stats']['median']:+.2f} m  std={r['stable_stats']['std']:.2f}  |  "
                f"M3C2 med={r['m3c2_stats']['median']:+.2f} m  std={r['m3c2_stats']['std']:.2f}"
            )

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    print(f"\nSaved {summary_csv}")
    return df

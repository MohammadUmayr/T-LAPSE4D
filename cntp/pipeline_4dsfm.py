"""End-to-end 4D SfM pipeline for a single new date.

Orchestrates the seven library calls that process one day's images from raw
JPEGs to a validated, co-registered point cloud + a registry entry:

  1.  Multi-temporal bundle adjustment   :func:`cntp.metashape.run_multitemporal_ba`
  2.  Single-day fixed-IOP reconstruction :func:`cntp.metashape.run_single_day_fixed_iop`
  3.  Three-stage ASP ICP coreg          :func:`cntp.asp.pc_align_p2p_sp2p`
  3b. M3C2 evaluation on coreg cloud     :func:`cntp.asp.evaluate_coreg`
  4.  Apply transform to camera EOPs     :func:`cntp.asp.apply_coreg_to_cameras_ecef`
  6.  Rebuild cloud with corrected M      :func:`cntp.metashape.rebuild_coreg_cloud`
  6b. M3C2 validation on rebuilt cloud   (inline — py4dgeo + matplotlib)
  7.  Update reference registry          :func:`cntp.metashape.update_registry`

Each step checks for its key output on disk before running; pass
``overwrite=True`` to force a full re-run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cntp.metashape import (
    discover_images,
    run_multitemporal_ba,
    run_single_day_fixed_iop,
    rebuild_coreg_cloud,
    update_registry,
    _utm_epsg,
)
from cntp.asp import (
    extract_stable_reference,
    pc_align_p2p_sp2p,
    evaluate_coreg,
    apply_coreg_to_cameras_ecef,
)
from cntp.io import load_las, save_las, apply_glacier_mask


def run_4dsfm_day(
    new_date: str,
    tlcam_dir: Path,
    ref_cloud: Path,
    glacier_mask: Path,
    registry_csv: Path,
    output_dir: Path,
    match_downscale: int = 0,
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
    stop_after_ba: bool = False,
) -> dict:
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
        Reference registry CSV managed by :func:`cntp.metashape.update_registry`.
        The UTM zone is derived from the mean ``lon`` column.
    output_dir : Path
        Root output directory (parent of ``output_new/``).
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

    Returns
    -------
    dict
        ``date, tba_las_path, aligned_las, validated_laz, cameras_coreg_csv,
        transform_path, coreg_med_before, coreg_med_after, coreg_std_after,
        validation_med, validation_std``.
    """
    output_dir   = Path(output_dir)
    ref_cloud    = Path(ref_cloud)
    registry_csv = Path(registry_csv)

    date_dir   = output_dir / "output_new" / new_date
    sfm_dir    = date_dir / "4D_SfM"
    single_dir = date_dir / "single_day"
    coreg_dir  = date_dir / "coreg"
    val_dir    = single_dir / "validation"

    # ── Discover images and derive UTM zone ──────────────────────────────
    reg_df   = pd.read_csv(registry_csv)
    utm_epsg = _utm_epsg(reg_df["lon"].mean())

    by_date = discover_images(tlcam_dir)
    if new_date not in by_date:
        raise ValueError(f"No images found for date {new_date} under {tlcam_dir}")
    date_images = by_date[new_date]

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
    # they're built once and reused for every day in output_new/.
    ref_cache_dir = output_dir / "output_new" / "_ref_cache"
    ref_ds_stem   = ref_cloud.stem + f"_ds{ref_downsample:.2f}"
    ref_ds_path   = ref_cache_dir / f"{ref_ds_stem}.las"
    stable_ref    = ref_cache_dir / f"{ref_ds_stem}_stable.las"
    ref_plot_dir  = ref_cache_dir / "m3c2_plots" / "reference"

    coreg_med_before = coreg_med_after = coreg_std_after = float("nan")
    validation_med   = validation_std  = float("nan")

    # ── Step 1: multi-temporal bundle adjustment ─────────────────────────
    if overwrite or not cameras_4dsfm_csv.exists():
        print(f"\n[Step 1] Multi-temporal BA — {new_date}")
        cameras_4dsfm_csv, calib_dir_out = run_multitemporal_ba(
            date                   = new_date,
            date_images            = date_images,
            reference_registry_csv = registry_csv,
            output_dir             = output_dir,
            utm_epsg               = utm_epsg,
            match_downscale        = match_downscale,
            loc_acc_new            = loc_acc_new,
            rot_acc_new            = rot_acc_new,
        )
    else:
        print(f"[Step 1] Skipping — {cameras_4dsfm_csv.name} exists")

    if stop_after_ba:
        print(f"\n[stop_after_ba=True] Halting after Step 1.")
        return None

    # ── Step 2: single-day fixed-IOP reconstruction ──────────────────────
    if overwrite or not tba_las_path.exists():
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
            loc_acc         = loc_acc_new,
            rot_acc         = rot_acc_new,
        )
    else:
        print(f"[Step 2] Skipping — {tba_las_path.name} exists")

    # ── Stable reference (shared cache across all days) ──────────────────
    # Downsampled ref + glacier-masked, slope/NDWI-filtered stable ref live
    # under output_new/_ref_cache/. extract_stable_reference also writes the
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
        coreg_med_before = eval_result["med_before"]
        coreg_med_after  = eval_result["med_after"]
        coreg_std_after  = eval_result["std_after"]
    else:
        print(f"[Step 3b] Skipping — stable TBA exists")

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
    if overwrite or not validated_laz.exists():
        print(f"\n[Step 6] Rebuild co-registered cloud — {new_date}")
        rebuild_coreg_cloud(
            psx_path        = psx_path,
            transform_path  = transform_path,
            output_laz      = validated_laz,
            depth_downscale = depth_downscale,
            utm_epsg        = utm_epsg,
        )
    else:
        print(f"[Step 6] Skipping — {validated_laz.name} exists")

    # ── Step 6b: M3C2 between stable TBA (Step 3b) and validated cloud ───
    # Near-zero → ASP transform was correctly propagated into the rebuild.
    if overwrite or not validated_stable.exists():
        print(f"\n[Step 6b] Validate rebuilt cloud — {new_date}")
        import numpy as np
        import py4dgeo
        import matplotlib.pyplot as plt
        from cntp.coreg import extract_stable_terrain, run_m3c2

        val_cloud = load_las(validated_laz, downsample_factor=tba_downsample)
        if glacier_mask is not None:
            val_cloud = apply_glacier_mask(val_cloud, glacier_mask)
        _, val_stable_arr = extract_stable_terrain(val_cloud)

        val_dir.mkdir(parents=True, exist_ok=True)
        save_las(val_stable_arr, validated_stable, crs=utm_epsg)

        stable_tba_cloud = load_las(stable_tba_path)
        epoch_tba = py4dgeo.Epoch(stable_tba_cloud[:, :3])
        epoch_val = py4dgeo.Epoch(val_stable_arr[:, :3])
        validation_med, validation_std, dist_val = run_m3c2(epoch_tba, epoch_val)
        print(f"  M3C2 median : {validation_med:.4f} m  std : {validation_std:.4f} m")

        val_plot_dir = val_dir / "validation_plots"
        val_plot_dir.mkdir(parents=True, exist_ok=True)
        # Clip only the histogram display so a few outliers don't squash the
        # x-axis. Stats shown in the label come from run_m3c2 over the full
        # (un-clipped) distance array, so they match the printed values.
        d      = dist_val[~np.isnan(dist_val)]
        d_plot = np.clip(d, -3 * np.std(d), 3 * np.std(d))
        fig, ax = plt.subplots()
        ax.hist(d_plot, bins=60, color="steelblue", alpha=0.8)
        ax.axvline(validation_med, color="tomato", linestyle="--", linewidth=1.5,
                   label=f"median = {validation_med:.4f} m  std = {validation_std:.4f} m")
        ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel("M3C2 distance (m)")
        ax.set_ylabel("Count")
        ax.set_title(f"Validation — stable TBA vs validated cloud ({new_date})")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(val_plot_dir / "m3c2_validation.png", dpi=150)
        plt.close(fig)
    else:
        print(f"[Step 6b] Skipping — {validated_stable.name} exists")

    # ── Step 7: update reference registry ────────────────────────────────
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
        "coreg_std_after":   coreg_std_after,
        "validation_med":    validation_med,
        "validation_std":    validation_std,
    }

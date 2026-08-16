"""Pipeline parameters for the per-date runs.

Paths live in ``site_config*.py`` (which glacier, where the data is); this file holds *how* the
pipeline is run — the SfM and raster knobs forwarded to
:func:`tlapse4d.pipeline_4dsfm.run_4dsfm_day_with_rasters`.

Unlike ``site_config.py`` this file is tracked in git: these values are analysis decisions, so a
processed date is only reproducible if they are recorded.

Override per run rather than editing this file, e.g.::

    from run_config import params
    params = params | dict(verbose=True, max_unaligned=10)
"""

params = dict(
    # ── SfM pipeline knobs (forwarded to run_4dsfm_day) ──────────────────
    match_downscale       = 1,
    depth_downscale       = 2,
    filter_mode           = "Mild",        # "Mild" or "Aggressive"
    loc_acc_new           = (0.5, 0.5, 0.5),
    rot_acc_new           = (5.0, 5.0, 5.0),
    ref_downsample        = 0.4,   # per-glacier coreg knob (0.05 North dense / 0.40 West)
    tba_downsample        = 1.0,
    p2p_max_disp          = 10,
    sp2p_max_disp         = 5,
    m_sp2p_max_disp       = 2,
    p2p_outlier_ratio     = 0.75,
    sp2p_outlier_ratio    = 0.75,
    m_sp2p_outlier_ratio  = 0.75,  # lower to 0.5-0.6 to tighten Stage 3 vs glacier false matches
    use_ecef              = True,
    overwrite             = False,
    verbose               = False,
    # Registry frozen at the 2023-11-27 baseline (no feedback into BA).
    add_to_registry       = False,
    # Skip Step 6 rebuild + Step 6b validation (coreg M3C2 plot still runs).
    run_validation        = False,
    # Cloud-cover gate (now in Step 1 / 4D SfM): skip the date if >= this many
    # NEW-DAY cameras fail to align in the multi-temporal bundle adjustment.
    max_unaligned         = 6,
    # Keep only daytime frames; drop off-schedule night / motion captures BEFORE
    # the alignment gate counts them. None = keep every frame.
    time_window           = (9, 17),
    # Drop boulder-degraded C8+C9 from 2024-07-15 on. Site-specific — move to
    # site_config.py if this file is ever shared across glaciers.
    exclude_cameras       = [{"camera": "C8", "from": "2024-07-15"},
                             {"camera": "C9", "from": "2024-07-15"}],

    # ── Raster knobs ─────────────────────────────────────────────────────
    dem_method            = "point2dem",   # HSfM ASP point2dem (IDW); "cubic" = legacy
    res                   = 1.0,
    max_gap_pixels        = 1,
    ref_cloud_downsample  = 0.25,
    m3c2_ref_downsample   = 0.25,
    slope_threshold       = 60.0,
    overwrite_ref_dem     = False,
    overwrite_day_dem     = False,
    overwrite_dod         = False,
    overwrite_stable      = False,
    overwrite_stable_dod  = False,
    overwrite_m3c2        = False,
)

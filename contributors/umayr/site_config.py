"""site_config.py — the ONE place you set paths. Edit the 3 below for your glacier.
Both notebooks read these (import site_config as site), so they can't drift.
To switch glacier: change the 3 paths.
"""
from pathlib import Path

output_dir   = Path("/mnt/e/umayr/Changri/Changri_North")                                       # results go in output_dir/output/
tlcam_dir    = Path("/mnt/e/umayr/Changri/TLCAM/ChangriNorth_renamed")                          # timelapse images
glacier_mask = Path("/mnt/e/umayr/Changri/Changri_North/shapefile/Shapefile_ChangriNorth.shp")  # glacier outline

# ── derived automatically — leave these ──
out           = output_dir / "output"
ref_cloud     = out / "Reference_UAV_TLC_PCS.laz"
registry_csv  = out / "reference_registry.csv"
ref_tlc_cloud = out / "_ref_cache" / "reference_TLC_coreg.las"

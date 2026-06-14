"""Template — copy this to site_config.py, then edit the 3 paths for your glacier:

    cp site_config.example.py site_config.py

Both notebooks read site_config.py (import site_config as site), so their paths
can't drift. You process one glacier at a time; to switch glacier, change the 3 paths.
site_config.py is gitignored, so your paths stay local (never committed).
"""
from pathlib import Path

output_dir   = Path("/path/to/glacier")              # results go in output_dir/output/
tlcam_dir    = Path("/path/to/cameras_renamed")      # standardised timelapse images
glacier_mask = Path("/path/to/glacier_outline.shp")  # glacier outline shapefile

# ── derived automatically — leave these ──
out           = output_dir / "output"
ref_cloud     = out / "Reference_UAV_TLC_PCS.laz"
registry_csv  = out / "reference_registry.csv"

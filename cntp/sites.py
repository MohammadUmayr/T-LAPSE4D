"""Derive every per-glacier path from the few you choose.

You process **one glacier at a time**. A project-local ``site_config.py`` holds
the paths for the glacier you're currently working on as a single ``site``; to
switch glaciers you edit those paths. Every notebook does ``from site_config
import site``, so setup and per-date processing read the same source and can't
drift. You set only ``output_dir``, ``tlcam_dir`` and ``glacier_mask`` —
:func:`resolve_site` fills in the rest (everything lives under
``<output_dir>/output/`` by fixed convention).
"""

from pathlib import Path
from types import SimpleNamespace

OUTPUT_DIRNAME = "output"
REF_CLOUD_NAME = "Reference_UAV_TLC_PCS.laz"
REGISTRY_NAME  = "reference_registry.csv"


def resolve_site(output_dir, tlcam_dir, glacier_mask,
                 ref_cloud=None, check_exists: bool = False) -> SimpleNamespace:
    """Build all per-glacier paths from the three you pass in.

    Returns a namespace with ``output_dir, tlcam_dir, glacier_mask, out,
    ref_cloud, registry_csv, ref_tlc_cloud`` — pass ``site.tlcam_dir`` /
    ``site.ref_cloud`` / … straight into the pipeline.

    By convention ``ref_cloud`` is ``<output_dir>/output/<REF_CLOUD_NAME>`` (where
    ``bootstrap_registry`` exports it). Pass ``ref_cloud=`` to override for a
    legacy/non-standard layout. ``check_exists=True`` raises if tlcam/mask/registry
    are missing — leave False during setup (the registry doesn't exist until
    ``bootstrap_registry`` runs).
    """
    output_dir = Path(output_dir)
    out = output_dir / OUTPUT_DIRNAME
    site = SimpleNamespace(
        output_dir    = output_dir,
        tlcam_dir     = Path(tlcam_dir),
        glacier_mask  = Path(glacier_mask),
        out           = out,
        ref_cloud     = Path(ref_cloud) if ref_cloud else out / REF_CLOUD_NAME,
        registry_csv  = out / REGISTRY_NAME,
        ref_tlc_cloud = out / "_ref_cache" / "reference_TLC_coreg.las",
    )
    if check_exists:
        for label in ("tlcam_dir", "glacier_mask", "registry_csv"):
            p = getattr(site, label)
            if not p.exists():
                raise FileNotFoundError(f"{label} does not exist: {p}")
    return site


_TEMPLATE = '''\
"""Glacier setup — the ONLY file you edit to choose a glacier.
Every notebook reads `site` from here, so their paths can never go out of sync.
To set up or switch glacier: change the 3 paths below. That's the whole config.
"""
from cntp.sites import resolve_site

site = resolve_site(
    output_dir   = "/path/to/glacier",              # results go in <output_dir>/output/
    tlcam_dir    = "/path/to/cameras_renamed",       # standardised timelapse images
    glacier_mask = "/path/to/glacier_outline.shp",   # glacier outline shapefile
)
'''


def init_site_config(path="site_config.py", overwrite: bool = False) -> Path:
    """Write a ready-to-edit ``site_config.py`` template into the current folder."""
    path = Path(path)
    if path.exists() and not overwrite:
        print(f"{path} already exists — edit it, or pass overwrite=True to replace.")
        return path
    path.write_text(_TEMPLATE)
    print(f"Wrote {path} — edit the paths, then import it in your notebooks.")
    return path

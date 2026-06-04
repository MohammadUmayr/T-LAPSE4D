# CNTP

A library for generating and processing point clouds from the timelapse photogrammetry.

It turns fixed time-lapse photographs of a glacier into **co-registered 3-D point
clouds, DEMs, orthoimages, and surface-change maps**, one date at a time. `cntp`
wraps [Agisoft Metashape](https://www.agisoft.com/) and the
[NASA Ames Stereo Pipeline](https://stereopipeline.readthedocs.io/) behind a small
set of plain Python functions, so an entire multi-temporal (4D Structure-from-Motion)
monitoring workflow can be driven from a Jupyter notebook by passing paths as variables.

> Developed for time-lapse glacier monitoring on the Changri glaciers (Khumbu,
> Nepal), but **site-agnostic** — any time-lapse camera network works, with no
> code changes for a new site.

---

## What it does

![CNTP workflow — from raw time-lapse photos to per-date point clouds, DEMs, orthoimages, DoD and M3C2 change maps](docs/workflow.png)

The camera identity is read from the **filename** (`<camera>_<date>_<time>`), so the
on-disk folder layout is irrelevant and there is no hardcoded camera list — a new
glacier is just new data plus its own reference.

### Standard image format

The pipeline only needs image filenames in the form

```
<camera>_<YYYY-MM-DD>_<HHMMSS>.<ext>        e.g.  C1_2023-11-27_090001.JPG
```

where the camera id is everything before the first underscore. A folder per camera
is the convention, but **folder names are not parsed** — `discover_images` reads the
camera and date from the filename, so any layout works:

```
ChangriWest_renamed/              ← point tlcam_dir here
├── C1/
│   ├── C1_2023-11-27_090001.JPG
│   ├── C1_2023-11-27_100000.JPG
│   └── ...
├── C2/
│   └── C2_2023-11-27_090001.JPG ...
├── ...
└── C5/
    └── C5_2023-11-27_090001.JPG ...
```

> **Note —** the standardise step (`homogenize_images` / `ensure_standardized`) is only
> needed to **produce** this format from messy raw images. If your images already follow
> the pattern above, skip it entirely and point `tlcam_dir` straight at the folder.

---

## Installation

```bash
git clone git@github.com:MohammadUmayr/CNTP_hackathon.git
cd ./CNTP_hackathon
conda env create -f environment.yml
conda activate cntp
pip install -e .
```

### External (non-pip) dependencies

| Tool | Used for | How to get it |
|---|---|---|
| **Agisoft Metashape** Python 3 module — **v2.3.1** | Bundle adjustment & reconstruction | Proprietary — download the matching Linux wheel (`metashape-2.3.1-…-linux_x86_64.whl`) from agisoft.com and `pip install` it. Set `AGISOFT_LICENSE_PATH` **before** `import cntp`. |
| **NASA Ames Stereo Pipeline** (`pc_align`, `point2dem`) | Point-cloud co-registration + DEM rasterisation | [Install guide](https://stereopipeline.readthedocs.io/en/latest/installation.html) — must be on `PATH`. |
| **ImageMagick** (`identify`) | Reading EXIF capture time in `homogenize_images` | Included in `environment.yml` (conda-forge). |

---

## Quickstart

> **Worked examples live in the notebooks.** The clearest way to see how to use the
> library is the Jupyter notebooks under `contributors/umayr/` — chiefly
> **`setup_new_glacier.ipynb`** (one-time per-glacier setup) and
> **`4d_sfm_dem_monthly.ipynb`** (processing dates + producing all raster products);
> `4d_sfm_pipeline.ipynb` shows the per-date SfM run. The snippets below are condensed
> from them.

### 1. One-time setup per glacier — `setup_new_glacier.ipynb`

```python
from cntp.preprocess import ensure_standardized
from cntp.metashape  import bootstrap_registry

# (a) standardise raw images (copies, originals untouched; EXIF time → filename)
tlcam_dir = ensure_standardized("/data/SITE/raw")        # → /data/SITE/raw_renamed

# (b) [MANUAL] in Metashape: build a GCP-referenced reference cloud and
#     calibrate the cameras, then save the .psx and note the chunk label.

# (c) extract calibrations + camera positions into the registry AND export the
#     reference point cloud (UTM .laz) to ref_cloud_out
bootstrap_registry(
    ref_psx       = "/data/SITE/reference.psx",
    chunk_label   = "OptimiseCamera(woCP)",
    ref_date      = "2024-01-07",
    output_dir    = "/data/SITE",
    registry_csv  = "/data/SITE/output_new/reference_registry.csv",
    ref_cloud_out = "/data/SITE/output_new/Reference_UAV_TLC_PCS.laz",   # exported reference cloud (UTM .laz)
)
```

### 2. Process a date

```python
from cntp.pipeline_4dsfm import run_4dsfm_day_with_rasters

result = run_4dsfm_day_with_rasters(
    new_date     = "2024-01-18",
    tlcam_dir    = tlcam_dir,
    ref_cloud    = "/data/SITE/output_new/Reference_UAV_TLC_PCS.laz",   # the cloud bootstrap_registry exported
    glacier_mask = "/data/SITE/glacier.shp",
    registry_csv = "/data/SITE/output_new/reference_registry.csv",
    output_dir   = "/data/SITE",
)
print(result["dod_stats"], result["m3c2_stats"])
```

Every step **caches its output**, so re-running skips finished work. For a whole
season, just loop over dates — see `4d_sfm_dem_monthly.ipynb`.

---

## Inputs

| Input | What it is |
|---|---|
| **Standardised images** | `<camera>_<YYYY-MM-DD>_<HHMMSS>.JPG`. Produced by `homogenize_images`, or bring your own — `discover_images` reads the camera id and date from the filename, so any folder layout works. |
| **`reference_registry.csv`** | Per-image reference camera positions/orientations + calibration directories. Built once by `bootstrap_registry` from a GCP-referenced Metashape project. |
| **Reference point cloud** (LAZ, UTM) | GCP-aligned reconstruction from the UAV survey **+ the same-day time-lapse images** that the per-day clouds are co-registered to. **Exported automatically by `bootstrap_registry`** (returned as `result["ref_cloud"]`). |
| **Glacier mask** (shapefile, same CRS) | Separates glacier from stable off-glacier terrain; the stable (unchanging) terrain is what the per-day clouds are **co-registered on** (ICP / `pc_align` aligns on the off-glacier surface, since the glacier itself moves/melts between dates). |

## Coordinate systems

The pipeline works in **UTM**. Every distance/size parameter — M3C2 radii, slope
thresholds, DEM resolution, position-accuracy priors — is expressed in **metres**, so
the point clouds must be in a projected, metre-based CRS (not raw lon/lat degrees).

- Export the **reference point cloud from Metashape in the site's UTM zone**, so it
  matches what the library expects (the per-day clouds inherit this CRS too).
- Keep **camera reference positions in WGS84** (lon/lat) — the registry stores these,
  and the UTM zone is derived from them automatically.
- **Limitation:** the UTM zone is auto-derived for the **northern hemisphere only**
  (EPSG `326xx`); a southern-hemisphere site would need `_utm_epsg` generalised.

## Outputs

Everything lands under `<output_dir>/output_new/`. The reference products are built
**once** and shared; everything else is **per date**:

```
<output_dir>/output_new/
│
├── _ref_cache/                          shared reference artefacts (built once, reused for every date)
│   ├── reference_dem.tif                reference DEM …
│   ├── reference_ortho.tif              … and orthoimage
│   └── reference_dem_stable.tif         reference DEM on stable (off-glacier) terrain only
│
└── <date>/                              one folder per processed date
    │
    ├── 4D_SfM/                          multi-temporal bundle adjustment (Step 1)
    │   ├── <date>_cameras_4DSfM.csv     refined camera positions / orientations
    │   └── adjusted_calib_4DSfM/        per-camera calibration (XML)
    │
    ├── single_day/                      single-day reconstruction + raster products
    │   ├── <date>.psx                   Metashape project
    │   ├── <date>_cloud.las             day point cloud (before co-registration)
    │   ├── <date>_dem.tif               day DEM …
    │   ├── <date>_ortho.tif             … and orthoimage
    │   ├── DOD.tif                      DEM of Difference   (day − ref)
    │   ├── DOD_stable.tif               DoD on stable terrain only (co-reg accuracy check)
    │   ├── M3C2_raster.tif              slope-aware surface change
    │   ├── *_histogram.png              a histogram for each raster above
    │   └── validation/                  validated clouds (post-coreg quality check)
    │
    └── coreg/                           co-registration (ASP pc_align)
        ├── <date>_cloud_coreg_hsfm.las  ← the co-registered day cloud (the aligned product)
        ├── <date>_cameras_coreg.csv     camera positions after the alignment transform
        └── stage3/run-transform.txt     the 4×4 rigid-body transform that was applied
```

**The headline deliverables** are the three change maps in `single_day/` (plus the
co-registered cloud in `coreg/`):

| File | What it tells you |
|---|---|
| `DOD.tif` | DEM of Difference, `day − ref`. **Positive = gain/accumulation, negative = loss/melt.** Vertical change — biased on steep slopes. |
| `DOD_stable.tif` | The same difference but on stable, off-glacier terrain only. A **co-registration accuracy check** — it should be ≈ 0; a non-zero bias here flags an alignment problem. |
| `M3C2_raster.tif` | Surface change measured **perpendicular to the terrain** (slope-aware), so it stays reliable on steep slopes where the vertical DoD is inflated. |

Because each step caches its output, re-running a date just reads these back (fast)
instead of recomputing.

---

## Pipeline parameters

`run_4dsfm_day` / `run_4dsfm_day_with_rasters` take the six required paths above plus
the tunable flags below (defaults shown — most runs only set the paths).

**Bundle adjustment / reconstruction**

| Flag | Default | Meaning |
|---|---|---|
| `match_downscale` | `0` | Metashape `matchPhotos` image downscale: `0` = 2× upscale (most tie points, slowest), `1` = full res, `2`/`4`/`8` = ½/¼/⅛ (faster, fewer points). |
| `depth_downscale` | `2` | Metashape `buildDepthMaps` downscale: `1` = full res (slow), `2` = ½ (default), `4`/`8`/`16` = coarser & faster. |
| `loc_acc_new` | `(0.5, 0.5, 0.5)` | Position accuracy prior (m) for the new-day cameras — `(x, y, z)`. |
| `rot_acc_new` | `(5.0, 5.0, 5.0)` | Rotation accuracy prior (°) for the new-day cameras — `(yaw, pitch, roll)`. |

**Co-registration (3-stage ICP / ASP `pc_align`)**

| Flag | Default | Meaning |
|---|---|---|
| `use_ecef` | `True` | Run ICP in ECEF coordinates (recommended — avoids flat-Cartesian distortion). |
| `ref_downsample` | `0.4` | Fraction of reference-cloud points kept for ICP (0–1). |
| `tba_downsample` | `1.0` | Fraction of the day's cloud kept for ICP (`1.0` = all). |
| `p2p_max_disp` | `10.0` | Max ICP correspondence distance (m), stage 1 — coarsest. |
| `sp2p_max_disp` | `5.0` | Max ICP correspondence distance (m), stage 2. |
| `m_sp2p_max_disp` | `2.0` | Max ICP correspondence distance (m), stage 3 — finest. |

**Raster products** (`run_4dsfm_day_with_rasters` only)

| Flag | Default | Meaning |
|---|---|---|
| `res` | `1.0` | Output raster pixel size (m) for DEM / ortho / DoD / M3C2. |
| `max_gap_pixels` | `1` | Pixels farther than this many cells from any cloud point become nodata. |
| `ref_cloud_downsample` | `0.25` | Load-time downsample of the reference cloud when gridding the reference DEM. |
| `m3c2_ref_downsample` | `0.25` | Load-time downsample of the reference cloud fed to the M3C2 KDTree (caps RAM). |
| `slope_threshold` | `60.0` | Slope (°) above which terrain is excluded from the stable-terrain DoD. |
| `utm_epsg` | `None` | Override the output EPSG; `None` auto-derives the UTM zone (see Coordinate systems). |

**Control flags**

| Flag | Default | Meaning |
|---|---|---|
| `overwrite` | `False` | Re-run every step even if its cached output already exists. |
| `add_to_registry` | `True` | Append the day's validated cameras to `reference_registry.csv` (Step 7). Set `False` to keep the reference baseline frozen. |
| `stop_after_ba` | `False` | Stop after Step 1 (multi-temporal bundle adjustment) so the BA outputs can be inspected before continuing. |
| `verbose` | `False` | Print each `pc_align` command and its stdout. |
| `overwrite_ref_dem`, `overwrite_day_dem`, `overwrite_dod`, `overwrite_stable`, `overwrite_stable_dod`, `overwrite_m3c2` | `False` | Per-raster overwrite flags (rasters only). |

---

## Module map

| Module | Responsibility |
|---|---|
| `cntp.preprocess` | Raw → standardised image filenames (`homogenize_images`, `ensure_standardized`). |
| `cntp.metashape` | Metashape SfM engine: multi-temporal bundle adjustment, single-day reconstruction, sensor/calibration setup, and the reference registry (`bootstrap_registry`, `update_registry`). |
| `cntp.asp` | NASA Ames Stereo Pipeline wrappers — `pc_align` ICP co-registration (incl. ECEF). |
| `cntp.coreg` | py4dgeo point-cloud helpers — M3C2 distances, stable-terrain extraction. |
| `cntp.io` | LAS/LAZ point-cloud I/O + glacier masking. |
| `cntp.raster` | DEM / orthoimage / DoD / M3C2 → GeoTIFF rasterisation. |
| `cntp.plot` | Diagnostic plots + histograms. |
| `cntp.batch` | Batch co-registration helpers. |
| `cntp.pipeline_4dsfm` | Per-date orchestration — `run_4dsfm_day`, `run_4dsfm_day_with_rasters`. |

## Notebooks

The driver notebooks for the current library live in `contributors/umayr/`:

| Notebook | Purpose |
|---|---|
| **`setup_new_glacier.ipynb`** *(main)* | One-time per-glacier onboarding: standardise images → (manual Metashape reference) → bootstrap registry. |
| **`4d_sfm_dem_monthly.ipynb`** *(main)* | Multi-date batch with all raster products (`run_4dsfm_day_with_rasters`). |
| `4d_sfm_pipeline.ipynb` | Per-date 4D SfM pipeline, SfM only (`run_4dsfm_day`). |

Other notebooks in `contributors/umayr/` are exploratory/test notebooks written
while developing and validating individual functions — not part of the current
workflow.

---

## License

MIT — see [LICENSE](LICENSE).

## Hackathon

The agenda, objectives, and outcomes of the first hackathon can be found
[here](https://docs.google.com/document/d/1gz8LUjDSC-tZ4XqOcob5vvtnEDYOLn0UwT7iVJvNUWs/edit?usp=sharing).

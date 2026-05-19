# Project Knowledge — Development Log

Each entry records what was added or changed, where, and exactly what to remove or revert if needed.

---

## 2026-05-19 — Runtime / disk optimisations (round 1)

### Goal
Cut per-day disk usage and avoid recomputing artefacts that depend only on
the reference cloud, not on which day is being processed.

### Changes
1. **Shared reference cache.** Downsampled reference (`<stem>_ds{f}.las`) and
   stable reference (`<stem>_ds{f}_stable.las`) now live under
   `output_new/_ref_cache/`, not under each day's `coreg/downsampled/` and
   `coreg/stable_ref/`. They depend only on (ref_cloud, ref_downsample,
   glacier_mask), so day 2..N reuse what day 1 created.
   - `cntp/pipeline_4dsfm.py`: new `ref_cache_dir = output_dir / "output_new" / "_ref_cache"`.
   - `pc_align_p2p_sp2p` now receives `ref_las = ref_ds_path` and
     `ref_downsample_factor = 1.0` so it doesn't make a per-day duplicate.

2. **Single-day BA exports `.las`, not `.laz`.** `run_single_day_fixed_iop`
   in `cntp/metashape.py` was changed: it used to write `{date}_cloud.laz`,
   then the coreg step decompressed it into `coreg/downsampled/{date}_cloud_full.las`.
   Now the single-day output is `{date}_cloud.las` directly and the
   downsampled-TBA-cache step in `pc_align_p2p_sp2p` is **skipped entirely**
   when `tba_downsample_factor=1.0`. Saves ~250 MB of duplicate cloud per
   day; ASP reads the single-day `.las` directly (no decompression).
   - The pipeline orchestrator's `tba_las_path` (formerly `laz_path`) now
     points at the `.las`. Result-dict key renamed accordingly.

3. **Drop `slope.las` from `extract_stable_reference`.** Previously saved
   both `<stem>_stable.las` and `<stem>_slope.las`. Now only the stable
   cloud is saved. The pre-NDWI-filter cloud is consumed in-memory to
   render the reference NDWI + RGB diagnostic plots (under
   `_ref_cache/m3c2_plots/reference/`) before being discarded. Function
   signature changed from `tuple[Path, Path]` to `Path`.
   - Also added a fast-path: if both `_stable.las` and the requested PNGs
     already exist, skip the load + KDTree pass.
   - `evaluate_coreg`: removed the now-redundant `ref_slope_las` parameter
     and the reference plotting branch (reference plots are produced once,
     up-front, by `extract_stable_reference`). TBA NDWI + RGB plots still
     run per-day under `coreg/m3c2_plots/tba/`.

4. **Drop the 3D `stable_terrain_geometry.png` plot.** `plot_stable_terrain_diagnostics`
   in `cntp/plot.py` now produces only `ndwi_vs_intensity.png` and
   `stable_terrain_rgb.png`. The geometry-only 3D scatter wasn't carrying
   information that NDWI/RGB didn't already convey.

5. **`plot_m3c2_distances` label fix.** The label used to show
   `med = np.median(d_clipped)` computed on the ±3σ-clipped distribution,
   which silently differed from the median reported by `run_m3c2` over the
   full data. Now the label uses the un-clipped median **and std** for both
   before and after; clipping is applied only to the histogram bins so a
   few outliers don't squash the x-axis.

### To revert
- `cntp/metashape.py`: change `run_single_day_fixed_iop` export back to
  `{date}_cloud.laz` (rename `cloud_path` → `laz_path` + the format suffix).
- `cntp/asp.py`:
  - Restore the unconditional pre-write block in `pc_align_p2p_sp2p` (the
    `tba_ds_path = ds_dir / (tba_las.stem + "_full.las")` branch + the
    `if not tba_ds_path.exists(): save_las(...)` save).
  - Restore the old `extract_stable_reference` body that saved
    `<stem>_slope.las` and returned `(out_path, slope_path)`.
  - Re-add the `ref_slope_las` parameter to `evaluate_coreg` and the
    reference-plotting branch.
- `cntp/plot.py`:
  - Re-add the `_plot_if_missing(..., "stable_terrain_geometry.png", ...)`
    block in `plot_stable_terrain_diagnostics`.
  - Restore the old `plot_m3c2_distances` body (clip first, then compute
    median on clipped data, no std in label).
- `cntp/pipeline_4dsfm.py`:
  - Move `ref_ds_path` / `stable_ref` back under each day's
    `coreg/downsampled/` and `coreg/stable_ref/`.
  - Restore the destructured `stable_ref, _ = extract_stable_reference(...)`.
  - Restore the Step 3 call to use `ref_las = ref_cloud` and
    `ref_downsample_factor = ref_downsample`.
  - Rename `tba_las_path` back to `laz_path`.

---

## 2026-05-19 — Move pipeline logic into the library; slim the notebook

### Goal
Replace the long step-by-step notebook with a thin driver that just calls one
library function. All orchestration now lives in `cntp/`.

### Files added
- **`cntp/asp.py`** — moved verbatim from `contributors/umayr/tools.py`. Contains
  the three-stage ICP coreg (`pc_align_p2p_sp2p`, `extract_stable_reference`,
  `evaluate_coreg`), camera-transform helpers (`apply_coreg_to_cameras`,
  `apply_coreg_to_cameras_ecef`, `las_utm_to_ecef`, `wgs84_to_utm`,
  `utm_to_wgs84`), and internal helpers (`_check_asp`, `_run_command`,
  `_read_asp_transform`, `pc_align_stage`, `_apply_transform_to_las`).
- **`cntp/pipeline_4dsfm.py`** — single `run_4dsfm_day(...)` function that
  orchestrates Steps 1, 2, 3, 3b, 4, 6, 6b, 7. Each step skips itself if its
  key output file already exists; pass `overwrite=True` to force recompute.

### Files modified
- **`cntp/metashape.py`** — added `rebuild_coreg_cloud(psx_path, transform_path,
  output_laz, depth_downscale, utm_epsg)` at the end. Encapsulates the
  open-PSX → apply `T_ecef @ M_old` → `buildDepthMaps` → `buildPointCloud` →
  `exportPointCloud` flow that has to run in a single Metashape session (the
  matrix is recomputed from GPS priors on every `doc.open()`). Also added a
  tiny `_read_4x4_matrix` helper (avoids importing from `cntp.asp` to keep the
  module dependency direction clean) and the `numpy` import.
- **`contributors/umayr/4d_sfm_pipeline.ipynb`** — slimmed from 24 cells
  (inline Metashape + ASP + matplotlib) to 6 cells: title markdown, imports
  (`from cntp.pipeline_4dsfm import run_4dsfm_day`), config markdown, config
  (paths + `params` dict), run markdown, single `run_4dsfm_day(...)` call +
  summary print.

### Files removed
- **`contributors/umayr/tools.py`** — superseded by `cntp/asp.py`. Stale
  `__pycache__/tools.cpython-*.pyc` cleaned too.

### Pipeline param defaults captured
`run_4dsfm_day` defaults match the values that produced the validated
2023-12-15 run: `ref_downsample=0.4`, `p2p_max_disp=10`, `sp2p_max_disp=5`,
`m_sp2p_max_disp=2`, `use_ecef=True`, `match_downscale=0`, `depth_downscale=2`,
`loc_acc_new=(0.5,0.5,0.5)`, `rot_acc_new=(5.0,5.0,5.0)`.

### To revert
- Restore `contributors/umayr/tools.py` from `git show backup_4dsfm^:contributors/umayr/tools.py > contributors/umayr/tools.py`.
- Delete `cntp/asp.py` and `cntp/pipeline_4dsfm.py`.
- Remove `rebuild_coreg_cloud`, `_read_4x4_matrix`, and the `import numpy as np`
  line from `cntp/metashape.py` (everything below the
  `# 4D SfM pipeline — rebuild co-registered cloud …` divider).
- Restore the old notebook from `git show backup_4dsfm^:contributors/umayr/4d_sfm_pipeline.ipynb > contributors/umayr/4d_sfm_pipeline.ipynb`.

### Knock-on effects (not handled — call out for the user)
The following untracked notebooks under `contributors/umayr/` still have
`import tools` and `from tools import ...` lines, which will fail now that
`tools.py` is gone:
- `asp_coreg.ipynb`
- `asp_coreg_ecef.ipynb`
- `overview.ipynb`

Replace those imports with `from cntp.asp import ...` if you want to run them.
The notebooks under `contributors/marin/` and `contributors/friedrich/` use a
different `tools.py` in their own folder and are unaffected.

---

## 2026-05-19 — Fix registry date-format drift breaking sensor assignment

### Problem
Multi-temporal BA was merging all 20 reference images into a single Metashape sensor
("49C (7.45mm)") instead of 5 per-camera sensors. The resulting BA was severely
under-constrained on the 5 floating new-day sensors, so their focal lengths drifted
±10% between cameras (vs ±0.2% in a healthy run). The single-day cloud came out with
~84% of the correct XY extent, sat ~7 m off the reference, and pc_align Stage 2 hit
its 5 m `max_displacement` cap with a 13.6 m drift to absorb — full coreg failure.

### Root cause
`reference_registry.csv` had been opened in Excel/LibreOffice at some point and saved
back with the `date` column rewritten from `2023-11-27` to `11/27/2023`. The lookup in
`_assign_sensors_multitemporal` keys sensors by `(cam_prefix, _image_date(label))`
where `_image_date` always returns `YYYY-MM-DD` from the image label. With the CSV
in `M/D/YYYY`, every lookup returned `None`, no sensor was assigned, and Metashape
auto-grouped all ref images into one default sensor.

### Fix — Normalize dates in code so on-disk format can be anything

`cntp/metashape.py`:
- Added `_normalize_date(value)` helper that coerces any pandas-readable date string
  to canonical `YYYY-MM-DD` via `pd.to_datetime(value).strftime("%Y-%m-%d")`.
- `run_multitemporal_ba`: applied to `reg_df["date"]` right after `pd.read_csv`.
- `update_registry`: applied to both the existing CSV on re-read and the combined
  frame before `to_csv`, so a re-write always emits canonical format.

Also normalized the existing CSV in place (backup saved as
`reference_registry.csv.bak`). The code fix is what makes this permanent — future
Excel openings will be silently corrected on the next pipeline read.

### To revert
- Remove `_normalize_date` from `cntp/metashape.py` (just below `_image_date`).
- Remove the two `existing["date"] = ... .map(_normalize_date)` /
  `reg_df["date"] = ... .map(_normalize_date)` lines and the
  `combined["date"] = ... .map(_normalize_date)` line in `update_registry`.

---

## 2026-05-16 — Pipeline optimisations: skip redundant ref processing in `evaluate_coreg`

### Changes to `contributors/umayr/tools.py` — `evaluate_coreg()`

**Problem:** Step 3b was redundantly (1) loading the full reference UAV cloud and downsampling
it, and (2) running `apply_glacier_mask` + `extract_stable_terrain` on the reference, even
though `extract_stable_reference` (called just before Step 3) already produces and caches
exactly this result on disk.

**What changed:**
- Added `stable_ref_las: Path = None` parameter.
- When provided, the function loads the cached stable reference directly
  (`load_las(stable_ref_las)`) and skips loading `ref_las`, applying the glacier mask, and
  running `extract_stable_terrain` for the reference cloud — saving two KDTree builds per run.
- Glacier mask and `extract_stable_terrain` still run on the TBA "before" and "after" clouds.
- Plotting: reference diagnostic plot is skipped when `stable_ref_las` is provided (no
  `ref_slope` available); only the TBA diagnostic is plotted.

**To revert:**
- Remove `stable_ref_las: Path = None` from the signature and docstring.
- Restore the original loading block:
  ```python
  ref    = load_las(ref_las, downsample_factor=ref_downsample_factor)
  before = load_las(tba_before_las, downsample_factor=_tba_ds)
  after  = load_las(tba_after_las,  downsample_factor=_tba_ds)
  if glacier_mask_path is not None:
      ref    = apply_glacier_mask(ref,    glacier_mask_path)
      before = apply_glacier_mask(before, glacier_mask_path)
      after  = apply_glacier_mask(after,  glacier_mask_path)
  ref_slope,    ref_stable    = extract_stable_terrain(ref,    slope_threshold=slope_threshold)
  before_slope, before_stable = extract_stable_terrain(before, slope_threshold=slope_threshold)
  after_slope,  after_stable  = extract_stable_terrain(after,  slope_threshold=slope_threshold)
  ```
- Restore the original plotting loop:
  ```python
  for cloud_slope, cloud_stable, name in (
      (ref_slope,    ref_stable,    "reference"),
      (before_slope, before_stable, "tba"),
  ):
  ```

### Changes to `contributors/umayr/4d_sfm_pipeline.ipynb` — Step 3b cell (`c8f1e230`)

**What changed:**
- `ref_las` changed from `ref_cloud` (full-resolution) to `ref_ds_path` (pre-downsampled copy
  already on disk from the Step 3 setup block).
- `ref_downsample_factor` changed from `ref_downsample` (0.3) to `1.0` (no further subsampling
  inside the function).
- Added `stable_ref_las = stable_ref` (pre-built stable-terrain cloud from `extract_stable_reference`).

**To revert:**
```python
coreg_stats = evaluate_coreg(
    ref_las               = ref_cloud,
    tba_before_las        = laz_path,
    tba_after_las         = aligned_las,
    ref_downsample_factor = ref_downsample,
    glacier_mask_path     = glacier_mask,
    stable_dir            = coreg_dir / "stable_tba",
    plot_dir              = coreg_dir / "m3c2_plots",
)
```

---

## 2026-05-16 — Fix co-registration transform propagation into Metashape (core pipeline bug)

### Problem
`validated_laz` (Metashape-rebuilt cloud from Step 6) had ~0.875 m offset from `ref_cloud`,
same as the original unregistered cloud. The ASP transform was not propagating into the
Metashape-rebuilt cloud.

**Root cause 1 — `importReference` + `updateTransform()` gave only 0.0819 m shift:**
GPS priors in the PSX (from `cameras_4dsfm_csv`) were already close to the co-registered
positions. `updateTransform()` re-fits the chunk transform to match GPS priors, so the
effective shift was only 0.08 m instead of the required ~1 m.

**Root cause 2 — `chunk.transform.matrix` is not persisted across save/reload:**
Metashape recomputes `chunk.transform.matrix` from GPS priors every time a project is opened.
Any direct assignment to `M` was silently overwritten on the next `doc.open()`.

### Fix — Direct matrix composition in the same session as the cloud rebuild

`chunk.transform.matrix` (M) maps local chunk space → WGS84 geocentric ECEF. The ASP
transform `T_ecef` operates in the same ECEF space. The corrected matrix is simply:

```
M_new = T_ecef @ M_old
```

Both matrices share WGS84 geocentric ECEF — no CRS conversion needed. The fix must be
applied in the **same session** as `buildDepthMaps` + `buildPointCloud` + `exportPointCloud`,
with no `doc.save()` / `doc.open()` in between, so the corrected `M` is still active.

### Changes to `contributors/umayr/4d_sfm_pipeline.ipynb`

#### Step 5 cell (`step5-update-metashape`) — now verify-only
Reads and prints the ASP transform, stores it in `T_ecef`. No Metashape operations.

#### Step 6 cell (`step6-rebuild-cloud`) — applies transform + rebuilds in one session
```python
doc.open(str(single_day_psx), ...)
chunk = doc.chunk
# Apply T_ecef @ M_old
M_old = chunk.transform.matrix
M_np  = np.array([[M_old[r, c] for c in range(4)] for r in range(4)])
M_new = T_ecef @ M_np
chunk.transform.matrix = Metashape.Matrix([[float(M_new[r, c]) for c in range(4)] for r in range(4)])
# Build and export in same session — no reload
chunk.buildDepthMaps(...)
chunk.buildPointCloud()
chunk.exportPointCloud(...)
doc.save()
```

**To revert:** Replace with `importReference` + `updateTransform()` approach (see previous
entry "2026-05-15 — Fix `update_metashape_cameras_after_transform`").

#### Step 6 validation cell (`step6-validation-coreg`) — M3C2 between validated and aligned clouds
Compares `validated_laz` (Metashape MVS rebuild) vs `aligned_las` (ASP rigid transform).
Near-zero median confirms the transform was correctly propagated.
Validated result for 2023-12-15: **M3C2 median = −0.0045 m, std = 0.389 m**.

### Changes to `cntp/metashape.py` — `update_metashape_cameras_after_transform()`

Signature changed from `(psx_path, cameras_coreg_csv, chunk_label)` to
`(psx_path, transform_path, chunk_label)`. Now performs the same `T_ecef @ M_old` composition
and saves the PSX. Used as a standalone diagnostic; the actual cloud rebuild happens in the
notebook Step 6 as described above.

**To revert:** Restore old signature and `importReference` + `updateTransform()` body.

---

## 2026-05-16 — Notebook cleanup: remove 6 unnecessary cells from `4d_sfm_pipeline.ipynb`

Cells deleted (by ID):

| Cell ID | What it was |
|---------|-------------|
| `4913826c-9b06-4974-95e7-871709523786` | Redundant restore of `laz_path` / `cameras_single_csv` before Step 3 |
| `dda6a90e` | Restore cell only needed for the ASP validation test (now removed) |
| `d18f6672-b0cb-4e13-87fd-b3ff778f33ba` | Duplicate restore of `laz_path` / `cameras_single_csv` |
| `0f9ef0de` | ASP validation test (`pc_align_p2p_sp2p` on `validated_laz` vs `ref_cloud`) — replaced by M3C2 cell |
| `a4d9a35e-9cb7-4232-be75-fc81d99fbe72` | Empty cell |
| `bef3f576-3b74-49f3-99ce-54b4c5ddfd0c` | Empty cell |

Kept: `step6-validation-coreg` (M3C2 between `validated_laz` stable terrain and `aligned_las`
stable terrain from Step 3b).

**To revert:** Re-insert the deleted cells from git history.

---

## 2026-05-15 — Fix `update_metashape_cameras_after_transform` + Step 6 safeguard

### Problem
`validated_laz` (Metashape-rebuilt cloud from Step 6) had the same ~0.875 m offset from
`ref_cloud` as the original `laz_path` — the co-registration transform was not propagating
into the Metashape-rebuilt cloud, regardless of UTM or ECEF mode.

The function was confirmed to match all 44/44 cameras (no label mismatch), but the transform
was not changing. Root cause: manual assignment of `cam.reference.location` leaves Metashape's
internal reference-change flag unset, so `updateTransform()` sees no change and recomputes the
same (pre-coreg) chunk transform.

### Changes to `cntp/metashape.py` — `update_metashape_cameras_after_transform()`

**What changed:**
- Removed manual `cam.reference.location` assignment loop (and the `import pandas as pd` it needed)
- Replaced with `chunk.importReference(cameras_coreg_csv, format=..., columns='nxyzXYZ', delimiter=',', skip_rows=1, crs=wgs84, items=ReferenceItemsCameras)` — Metashape's native import pipeline, which correctly updates internal state so `updateTransform()` sees the change
- Post-import loop still sets `location_accuracy=0.001`, `rotation_accuracy=0.001`, `rotation_enabled=True` for all cameras where `reference.enabled`

**To revert:** Replace the `importReference` block with the old manual loop:
```python
import pandas as pd
df = pd.read_csv(cameras_coreg_csv)
loc_dict = {row["Label"]: row for _, row in df.iterrows()}
updated = 0
for cam in chunk.cameras:
    if cam.label in loc_dict:
        row = loc_dict[cam.label]
        cam.reference.location          = Metashape.Vector([row["Lon"], row["Lat"], row["Alt"]])
        cam.reference.rotation          = Metashape.Vector([row["Yaw"], row["Pitch"], row["Roll"]])
        cam.reference.location_accuracy = Metashape.Vector([0.001, 0.001, 0.001])
        cam.reference.rotation_accuracy = Metashape.Vector([0.001, 0.001, 0.001])
        cam.reference.enabled           = True
        cam.reference.rotation_enabled  = True
        updated += 1
```

### Changes to `contributors/umayr/4d_sfm_pipeline.ipynb` — cell `step6-rebuild-cloud`

**What changed:** Added `chunk.updateTransform()` immediately after `doc.open()` / `chunk = doc.chunk`,
before `buildDepthMaps()`. This ensures the chunk transform is freshly computed from the
saved co-registered camera references right before the point cloud is rebuilt and exported.

**To revert:** Remove the `chunk.updateTransform()` line and its comment.

---

## 2026-05-15 — Pipeline notebook: Step 3, 3b, 6 overhaul + tools.py fixes

### Changes to `contributors/umayr/4d_sfm_pipeline.ipynb`

#### Config cell
- Added `glacier_mask = base_dir / "glaciermask_new" / "glacier_mask_pcs.shp"`
- Added `use_ecef` flag (default `True`) controlling co-registration coordinate space
- Added `print(f"Co-reg mode : {'ECEF' if use_ecef else 'UTM'}")` for visibility

#### Imports cell
- Added `extract_stable_reference`, `evaluate_coreg` to `from tools import`

#### Step 3 cell — co-registration
- Pre-downsamples reference cloud to `coreg/downsampled/` before calling `extract_stable_reference`
  so only `ref_downsample × full-cloud` is ever in Python memory (fixes OOM kernel restart).
  `pc_align_p2p_sp2p` detects the cached file and skips re-downsampling.
- `extract_stable_reference` now receives the downsampled copy, not the full `ref_cloud`
- `stable_ref_las` wired into `pc_align_p2p_sp2p` so Stage 3 ICP runs on stable terrain only
- `utm_epsg = utm_epsg if use_ecef else None` — ECEF/UTM mode controlled by `use_ecef`

**To revert:** Remove the pre-downsampling block and `stable_ref_las`, restore hardcoded `utm_epsg`.

#### Step 3b cell — M3C2 accuracy (new)
- Calls `evaluate_coreg` with `ref_cloud`, `laz_path` (before), `aligned_las` (after)
- `stable_dir = coreg_dir / "stable_tba"` — saves stable TBA as `.laz`
- `plot_dir = coreg_dir / "m3c2_plots"` — saves before/after histogram plots

**To revert:** Delete the cell and its markdown header.

#### Step 4 cell — apply transform
- `utm_epsg = None if use_ecef else utm_epsg` — matches transform space to Step 3

#### Step 6 second cell — validation (redesigned, ASP-based)
- **Removed:** M3C2 comparison between `stable_tba` and `validated_stable`
  - Reason: comparing ASP-rigidly-transformed SfM cloud vs fresh SfM rebuild is fundamentally
    flawed — they are produced by different processes and will always have systematic offset
- **Replaced with:** second `pc_align_p2p_sp2p` run on `validated_laz` against `ref_cloud`
  - `val_coreg_dir = output_new/YYYY-MM-DD/validation/`
  - Same parameters as Step 3 (same `stable_ref_las`, same `use_ecef` flag)
  - `T_val = _read_asp_transform(val_transform_path)`
  - `translation = np.linalg.norm(T_val[:3, 3])`
  - Near-zero translation (< 0.5 m) → transform propagated correctly through Metashape cameras
  - Prints PASS / WARN and full transform matrix

**To revert:** Restore the M3C2 stable_tba vs validated_stable comparison cell.

---

### Changes to `contributors/umayr/tools.py`

#### `evaluate_coreg()` — fixed `stable_dir` parameter
- `stable_dir` was in the signature but never used — the stable TBA was incorrectly saved under `plot_dir/tba/`
- Fixed: when `stable_dir` is provided, saves `stable_dir/<tba_after_las.stem>_stable.laz`
- Extension changed from `.las` to `.laz`
- Removed the old `plot_dir/tba/` save path

**To revert:** Restore `tba_stable_out = plot_dir / "tba" / (Path(tba_after_las).stem + "_stable.las")` inside the `if plot_dir` block.

---

### Updated output folder structure

```
output_new/
  reference_registry.csv
  YYYY-MM-DD/
    4D_SfM/
      YYYY-MM-DD_4DSfM.psx
      YYYY-MM-DD_cameras_4DSfM.csv
      adjusted_calib_4DSfM/C1.xml … C5.xml
    single_day/
      YYYY-MM-DD.psx
      YYYY-MM-DD_cloud.laz
      YYYY-MM-DD_cameras.csv
      YYYY-MM-DD_cloud_validated.laz
      YYYY-MM-DD_cloud_validated_stable.laz
      validation_plots/
        m3c2_validation.png
    coreg/
      YYYY-MM-DD_cloud_coreg_hsfm.las
      YYYY-MM-DD_cameras_coreg.csv
      stable_ref/Reference_UAV_TLC_PCS_ds0.XX_stable.las
      stable_tba/YYYY-MM-DD_cloud_coreg_hsfm_stable.laz
      m3c2_plots/
      downsampled/
      ecef/          (only when use_ecef=True)
      stage1/ stage2/ stage3/
```

---

## 2026-05-15 — Cleanup: remove `calib_xmls` from `run_multitemporal_ba` + pipeline notebook

### Changes to `cntp/metashape.py`

#### `run_multitemporal_ba()` — removed `calib_xmls` parameter

**What changed:** The `calib_xmls: dict[str, Path]` parameter was removed entirely.
The function now loads new-day sensor IOP internally from the last registry day's
`adjusted_calib_4DSfM/` directory. If a camera's XML is missing, a `FileNotFoundError`
is raised immediately with a clear message.

**Why:** The registry always provides the most recent refined IOP — passing raw factory
calibrations via `calib_xmls` was dead code once the registry was bootstrapped. Removing
the parameter makes the API cleaner and ensures the correct (refined) IOP is always used.

**To revert:** Add `calib_xmls: dict[str, Path] = {}` parameter back and restore the
`elif cam in calib_xmls` fallback branch inside `_setup_sensors_multitemporal()`.

---

### Changes to `contributors/umayr/4d_sfm_pipeline.ipynb`

**What changed:** Removed `calib_dir`, `reference_excel`, `ref_df`, and `calib_xmls`
from the config cell. `utm_epsg` is now derived from the registry CSV:
```python
df_reg   = pd.read_csv(registry_csv)
utm_epsg = _utm_epsg(df_reg["lon"].mean())
```
Removed `load_reference` and `process_day` from the imports cell.
Removed `calib_xmls=calib_xmls` argument from the `run_multitemporal_ba()` call.

**To revert:** Add those config variables back and pass `calib_xmls` to the call.

---

## 2026-05-14 — `contributors/umayr/tools.py` — Full rewrite (ASP co-registration)

### What was implemented

`tools.py` was originally two import lines (`import xdem`, `import py4dgeo`).
It was replaced with a full ASP-based three-stage ICP co-registration library,
mirroring the `pc_align_p2p_sp2p` pipeline from Knuth et al. (2023) §4.3.3.

**To revert:** Replace the entire file contents with:
```python
import xdem
import py4dgeo
```

### Functions added

| Function | What it does |
|----------|-------------|
| `_check_asp()` | Raises `RuntimeError` if `pc_align` not on PATH. |
| `_run_command(cmd, verbose)` | Runs a subprocess, raises on non-zero exit. |
| `_read_asp_transform(transform_path)` | Parses ASP `*-transform.txt` → 4×4 numpy array. Note: Stage 3's file already contains the full composed transform T3 ∘ T2 ∘ T1. |
| `wgs84_to_utm(lon, lat, alt, utm_epsg)` | WGS84 → UTM via pyproj. |
| `utm_to_wgs84(easting, northing, elevation, utm_epsg)` | UTM → WGS84 via pyproj. |
| `apply_coreg_to_cameras(cameras_csv, transform_path, utm_epsg, out_csv)` | Applies 4×4 transform in UTM space to camera positions CSV. |
| `las_utm_to_ecef(utm_las, utm_epsg, out_path, chunk_size)` | Reprojects UTM LAZ → ECEF LAS in chunks. Uses scale=0.01 m (covers ±21 M m). |
| `apply_coreg_to_cameras_ecef(cameras_csv, transform_path, out_csv, utm_epsg)` | Applies transform to camera positions. When `utm_epsg=None` (default): ECEF mode (WGS84 → ECEF → T → ECEF → WGS84). When `utm_epsg` set: UTM mode. |
| `_apply_transform_to_las(las_path, T, out_path, utm_epsg, chunk_size)` | Applies 4×4 transform to LAS cloud in chunks. UTM mode or ECEF mode. Rotates normals. |
| `pc_align_stage(tba_las, ref_las, output_prefix, alignment_method, max_displacement, ...)` | Runs one ASP `pc_align` stage. Handles downsampling, ECEF conversion, initial transform chaining. |
| `extract_stable_reference(ref_cloud_path, output_dir, glacier_mask_path, slope_threshold, plot_dir)` | Builds glacier-masked + slope/NDWI-filtered stable terrain reference cloud for Stage 3. |
| `pc_align_p2p_sp2p(tba_las, ref_las, output_dir, ...)` | Three-stage ICP pipeline: p2p → sp2p → sp2p on stable terrain. Applies final composed transform to full-res TBA cloud. |
| `evaluate_coreg(ref_las, tba_before_las, tba_after_las, ...)` | M3C2 accuracy assessment on stable terrain before and after co-registration. |

### Key design decisions

- **ECEF mode**: When `utm_epsg` is passed to `pc_align_p2p_sp2p`, downsampled clouds are
  converted UTM → ECEF before ASP so ICP runs in true Cartesian space (avoids UTM distortion).
  The final transform is applied via UTM → ECEF → T → ECEF → UTM so the output stays in UTM.
- **Transform composition**: ASP compounds transforms when `--initial-transform` is passed, so
  Stage 3's `*-transform.txt` contains T3 ∘ T2 ∘ T1 — only that file needs to be applied.
- **Downsampled copies cached**: Written once to `output_dir/downsampled/` and reused across
  all three stages (and across validation runs).
- **TBA pre-written as .las**: Even at full resolution, written once to `downsampled/` so ASP
  doesn't decompress `.laz` on-the-fly every stage.

---

## 2026-05-14 — 4D SfM Pipeline

### What was implemented
Full 4D SfM pipeline for multi-temporal time-lapse camera processing, following Friedrich's
strategy: multi-temporal BA with fixed reference cameras → single-day re-run with fixed IOP
→ co-registration → transform application → validation → registry update.

---

### Changes to `cntp/metashape.py`

#### 1. Modified `_setup_sensors()`
**What changed:** Added `fixed_calibration: bool = False` parameter and `sensor.fixed_calibration = fixed_calibration` inside the loop.

**To revert:** Remove the `fixed_calibration` parameter and the `sensor.fixed_calibration` line.

---

#### 2. New helper `_camera_prefix()`
**Location:** Added just before `_setup_sensors()`.

**What it does:** Extracts camera prefix from image label — `"C1_2023-07-15_083000"` → `"C1"`.

**To revert:** Delete the entire function.

---

#### 3. New helper `_image_date()`
**Location:** Added just after `_camera_prefix()`.

**What it does:** Extracts date from image label — `"C1_2023-07-15_083000"` → `"2023-07-15"`.

**To revert:** Delete the entire function.

---

#### 4. New section `# 4D SfM helpers` (added at end of file)
Contains the following new functions — **delete the entire section to fully revert**:

| Function | What it does |
|----------|-------------|
| `_set_camera_references_from_csv()` | Sets per-image EOP from a cameras CSV with configurable accuracy. Used in `run_single_day_fixed_iop()` to load 4DSfM camera positions as EOP priors. |
| `_last_day_mean_eop()` | Computes mean EOP per camera group (C1…C5) from the most recently added day in the registry. Used in `run_multitemporal_ba()` to initialise new-day cameras. |
| `_export_camera_csv_filtered()` | Like `_export_camera_csv()` but only exports cameras whose label contains a date string. Used in `run_multitemporal_ba()` to export only new-day cameras from a multi-temporal chunk. |
| `_setup_sensors_multitemporal()` | Creates one fixed sensor per camera per reference day (e.g. `C1_2023-07-15`) from each day's validated `calib_dir`, plus one floating sensor per camera for the new day. |
| `_assign_sensors_multitemporal()` | Routes each camera to its per-day reference sensor or to the new-day sensor based on the date in the label. |

---

#### 5. New section `# Reference registry` (added at end of file)
Contains:

| Function | What it does |
|----------|-------------|
| `update_registry()` | Appends a validated day's co-registered EOP + IOP path + image paths to `reference_registry.csv`. Creates the file if it does not exist. |

**Registry CSV columns:** `date, label, image_path, lon, lat, alt, yaw, pitch, roll, calib_dir`

**To revert:** Delete the entire section.

---

#### 6. New section `# 4D SfM pipeline — multi-temporal bundle adjustment`
Contains:

| Function | What it does |
|----------|-------------|
| `run_multitemporal_ba()` | Main entry point for Step 1. Reads registry, creates chunk with all reference + new-day images, sets up per-day sensors, assigns EOP (tight 0.001 for reference, loose 0.5m/5° for new day), runs matchPhotos + alignCameras + optimizeCameras. Exports to `output/YYYY-MM-DD/4D_SfM/`. |

**Key design decisions:**
- Reference sensors: one per camera per day, fixed IOP loaded from that day's `adjusted_calib_4DSfM/`
- New-day sensors: IOP initialised from last registry day's refined calibration, falling back to raw `Calib_IOP/` XMLs
- New-day EOP prior: mean position per camera group from last registry day's rows

**To revert:** Delete the entire section.

---

#### 7. New section `# 4D SfM pipeline — single-day re-run with fixed IOP`
Contains:

| Function | What it does |
|----------|-------------|
| `run_single_day_fixed_iop()` | Main entry point for Step 2. Loads IOP from `adjusted_calib_4DSfM/` (fixed), loads EOP priors from `cameras_4DSfM.csv` (loose 0.5m/5°), runs BA, builds depth maps + point cloud. Exports to `output/YYYY-MM-DD/single_day/`. |

**To revert:** Delete the entire section.

---

### New file: `contributors/umayr/4d_sfm_pipeline.ipynb`

**What it does:** Notebook orchestrating the full 4D SfM pipeline for one new day.
Steps: multi-temporal BA → single-day re-run → co-registration → apply transform →
update Metashape → validation → update registry.

**To revert:** Delete the file.

---

### Output folder structure produced

See the 2026-05-15 entry for the current up-to-date structure (`output_new/`, no `validated/` folder).

---

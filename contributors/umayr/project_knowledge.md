# Project Knowledge — Development Log

Each entry records what was added or changed, where, and exactly what to remove or revert if needed.

---

## 2026-06-13 — Config simplified: self-contained `site_config.py`; removed `cntp/sites.py`

### Why
`cntp/sites.py` (`resolve_site`) was needless indirection — the config imported a
library function just to build paths, and nothing in the pipeline imported it.
Folded its job straight into `site_config.py` as plain, visible lines.

### Changes
- **Deleted `cntp/sites.py`** (only `site_config.py` + docs referenced it; no pipeline import).
- `contributors/umayr/site_config.py` is now self-contained: 3 `Path(...)` inputs at the
  top + a derived block (`out`, `ref_cloud`, `registry_csv`, `ref_tlc_cloud`). No
  `resolve_site`, no `cntp.sites`.
- Notebooks now `import site_config as site` (was `from site_config import site`).
- README config section + this log updated.

### To revert
Restore `cntp/sites.py` from git; put `from cntp.sites import resolve_site` +
`site = resolve_site(...)` back in `site_config.py`; notebooks back to
`from site_config import site`.

---

## 2026-06-11 — Single-source per-glacier paths (`site_config.py` + `cntp.sites`) + `output_new`→`output`

### Why
Setup and the monthly notebook each re-declared the same paths → they could drift
(the West/North `tlcam_dir`/`ref_cloud`/`registry_csv` mix-ups). Fix: one source of
truth per glacier, read by every notebook. Also renamed the output subfolder from
`output_new` to `output`.

### `cntp/sites.py` (new — generic, ships NO real paths)
`resolve_site(output_dir, tlcam_dir, glacier_mask, ref_cloud=None)` → a
`SimpleNamespace` of derived paths (`out`, `ref_cloud`, `registry_csv`,
`ref_tlc_cloud`) under `<output_dir>/output/`. You pass the 3 real choices; the
rest follow fixed conventions. `ref_cloud=` overrides the convention for legacy
layouts (West keeps its cloud in `Ref_PC/`). `ref_downsample` is a *param*, not a
path, so it's deliberately NOT here. Also `init_site_config()` writes a template.

### `contributors/umayr/site_config.py` (the project's values, single source)
One active `site = resolve_site(...)` for the glacier you're processing. Every
notebook does `from site_config import site` and uses `site.tlcam_dir` /
`site.ref_cloud` / … — setup and monthly read the *same object*, so they can't
disagree. **One glacier at a time**: switch by editing the 3 paths (a commented
spare block holds the other glacier). A new user edits this one file, never the library.

### `output_new` → `output`
- Code: `sed output_new→output` in `cntp/pipeline_4dsfm.py`, `cntp/metashape.py`,
  `cntp/raster.py` (docstrings), `contributors/umayr/tools.py`. The convention
  lives in `cntp.sites.OUTPUT_DIRNAME`.
- Disk: `mv <root>/output_new <root>/output` for **North** + **West**. (drvfs quirk:
  the direct rename to `output` failed with "Permission denied" while any other
  target name worked — had to stage via a temp name, `output_new → _outtmp → output`.
  Also needed the kernel/QGIS handles on the dir released first.)
- Registry: rewrote the `calib_dir` column (`output_new`→`output`) in both
  `reference_registry.csv` so BA/`build_reference_tlc_cloud` still resolve calibs.

### Notebooks rewired
`setup_new_glacier` (imports + Stage 3 + verify) and `4d_sfm_dem_monthly` (config +
run cells) now read `site`. Their paths blocks are gone — config cell is just the
`site` import + `monthly_dates` + `params`; the run call uses `site.*`.

### Verified
Both glaciers' `output_dir/tlcam_dir/ref_cloud/registry_csv/glacier_mask` resolve and
exist; both notebooks `nbformat`-valid; zero `output_new` left in code/active
notebooks; `cntp` compiles.

### To revert
Set the notebooks back to literal paths; delete `cntp/sites.py` + `site_config.py`;
`sed output→output_new` back in the four files; `mv` the two `output` dirs back to
`output_new` and re-fix the registry `calib_dir`. (The `output` name itself is a
fine convention to keep even if reverting `site_config`.)

---

## 2026-06-11 — `run_validation` flag to skip Step 6 + 6b

### Why
With the registry frozen (`add_to_registry=False`) the reference no longer grows,
so the Step 6 Metashape rebuild (`validated_laz`) + Step 6b M3C2 validation are
dead weight — the rasters difference the Step 3 coreg cloud (`aligned_las`), not
the rebuilt cloud, and nothing else consumes it. Step 6 (buildDepthMaps +
buildPointCloud) is expensive, so skipping it saves real time per date.

### `cntp/pipeline_4dsfm.py`
- New **`run_validation: bool = True`** on `run_4dsfm_day` (threaded through
  `run_4dsfm_day_with_rasters`). `False` skips **Step 6 + Step 6b** only.
- **Step 3b (coreg M3C2 before/after plot) is deliberately NOT gated** — it's the
  meaningful coreg QC and always runs (unless its own output is cached). This was
  an explicit requirement: "coreg plot has to be outside validation."
- Steps 1–4, 7 unaffected. When skipped, `validation_med`/`validation_std` stay NaN.

### To revert
Remove the `run_validation` param + the `if run_validation and (...)` / `elif not
run_validation` guards on Step 6 and 6b (restore plain `if overwrite or not ...`).

---

## 2026-06-11 — DEM via ASP point2dem (HSfM IDW) replaces cubic griddata in the pipeline

### Why
Cubic `griddata` (Clough-Tocher) for the DEM is single-threaded and triangulates
all ~14 M points to fill a 1.78 M-cell grid (~8 pts/cell) → ~30 min/day. It's
also *not* the method the HSfM workflow publishes — that's **ASP `point2dem`**
(default `weighted_average` = Gaussian distance weighting, the paper's "IDW",
`--search-radius-factor 1` = 1-cell radius; gaps larger than the radius left as
nodata, not interpolated). point2dem is C++, multithreaded, streaming → seconds.

Rejected alternatives: tiled-cubic multiprocessing (`build_dem_and_ortho_mt`,
kept in tools.py) is **not** equivalent to global cubic — seam test on the ref
DEM gave median |Δ|=8 mm but p99=0.55 m, max=73 m (Clough-Tocher's gradient
solve is global; tiling breaks it at every seam). Binning / PDAL `writers.gdal`
also work but are different methods (and PDAL isn't in the `cntp` env). point2dem
*is* the published tool, already wrapped in `cntp.asp`.

### `cntp/raster.py`
- **New `build_dem_and_ortho_p2d()`** — DEM via `cntp.asp.point2dem` (grid pinned
  to `ref_las` via `--t_projwin`, `search-radius-factor = max_gap_pixels`); ortho
  via the **same `save_ortho`** (NN RGB) read back onto point2dem's grid, so
  DEM↔ortho share a pixel grid (needed by `extract_stable_terrain_from_dem`).
  `cloud_downsample` affects only the ortho; point2dem streams the full cloud.
- **`save_ortho`** — KDTree query now `workers=-1` (multithreaded, **same result**).

### `cntp/asp.py`
- `point2dem` docstring: "median aggregation" → "Gaussian distance-weighted
  average (default `weighted_average` filter — HSfM's 'IDW')" (it never passed
  `--filter`, so it was always `weighted_average`, not median).

### `cntp/pipeline_4dsfm.py` — `run_4dsfm_day_with_rasters`
- New param **`dem_method: str = "point2dem"`** (`"point2dem"` | `"cubic"`).
  point2dem path builds **both** reference and day DEMs with
  `build_dem_and_ortho_p2d`; cubic path is the legacy scipy builders.
- **Bug fix (both paths):** day DEM now anchors to `ref_input` (the ds ref), not
  the full `ref_cloud`. The full cloud's bbox is ~1 px taller than the ds copy,
  so the old day DEM came out 1226 rows vs the 1225-row reference → `build_dod`
  "shapes differ". Anchoring both to `ref_input` fixes it.

### `contributors/umayr/tools.py`
- `build_dem_and_ortho_p2d` **promoted to `cntp.raster`**; tools.py now re-exports
  it (`from cntp.raster import build_dem_and_ortho_p2d`) so notebook cells keep
  working off one source. `build_dem_and_ortho_mt` stays a tools.py prototype.

### Validation (Changri North, 2024-01-18)
Reference + day DEMs both `(1226, 1452)`, `transforms_equal=True` → DoD clean.
Full DoD median **+0.167 m** (winter accumulation), **stable DoD median −0.019 m**
(≈ 0, matches coreg M3C2 stable +0.006 m → chain consistent). ~110 s/DEM vs
~30 min cubic / ~172 s mt. point2dem coverage is lower (day 24% valid vs cubic's
interpolated fill) — by design: it doesn't interpolate over gaps.

### Caveats
- **Don't mix methods:** point2dem grid is 1 row taller than cubic's `np.arange`
  grid for the same bbox. All DEMs in a DoD must use the same `dem_method`.
- Switching method reuses the same `reference_dem.tif` / `<date>_dem.tif` names —
  set `overwrite_ref_dem`/`overwrite_day_dem=True` (or clear `_ref_cache` +
  single_day rasters) once when switching, else stale DEMs are reused.

### To revert
Set `dem_method="cubic"` (restores the scipy path with the grid-anchor fix). To
fully remove: drop `build_dem_and_ortho_p2d` from `cntp/raster.py` + its import +
the `dem_method` branch in `pipeline_4dsfm.py`; revert `save_ortho`'s `workers=-1`
and the `point2dem` docstring; restore the tools.py function from git.

---

## 2026-06-11 — Stage-3 ICP `max_displacement` drives similarity-ICP divergence

### Symptom
On Changri North 2024-01-18, the 3-stage ASP coreg made the stable-terrain fit
**worse**: M3C2 stable median 1.065 → 0.959 m but **std 0.756 → 1.584 m**, with a
**bimodal** "after" residual (a tilt/scale signature, not a clean offset). Stage 3's
own ASP error went *up* (median 0.94 → 1.79 m) and it reported a displacement of
8.17 m against a 5 m cap ("max-displacement smaller than final observed
displacement" warning).

### Cause
Stage 3 is `similarity-point-to-point` (fits **scale** + rotation) against the
**stable-only** reference, but the **source is the full day cloud** (glacier
included). With the cap loose (`m_sp2p_max_disp = 5`), the free scale DOF + false
correspondences from the moving glacier let ICP settle on a scale/tilt (0.47 %
scale ≈ several m edge-to-edge) that lowers a robust subset while inflating the
rest → std blows up, residual goes bimodal.

### Fix / finding
Tightening **`m_sp2p_max_disp` 5 → 0.5** fixed it: stable median **+1.065 →
+0.006 m**, std **0.756 → 0.640 m**, single clean mode on 0. Stage-3 transform
now scale-1 = 0.0017, max disp 0.557 m. Locked `m_sp2p_max_disp = 0.5` in
`4d_sfm_dem_monthly.ipynb` config.

**Caveat (batch):** 0.5 m works because Stages 1–2 already brought stable terrain
within ~0.32 m (Stage-3 *input* median). For a date whose Stage-2 stable residual
is > ~0.5 m, a 0.5 m cap is *too tight* and will clip a needed correction — set the
cap a little above each date's Stage-2 stable residual, not far above it. Rule of
thumb surfaced while debugging: if "after" looks worse than "before", loosen
Stage 3 for that date.

### Tooling added (prototype, `contributors/umayr/`)
- A standalone **"redo coregistration + M3C2 plot"** cell (`pc_align_p2p_sp2p` +
  `evaluate_coreg`) that re-runs Step 3 and the Step-3b M3C2 before/after plot
  only — skips Step 4/6/6b validation. Used to iterate the Stage-3 cap.

---

## 2026-06-11 — `optimizeCameras`: fit k4 in both BA steps

### Goal
Solve the 4th radial-distortion coefficient (k4) during bundle adjustment, not
just k1–k3.

### `cntp/metashape.py` — `fit_k4=False` → `fit_k4=True`
Both `optimizeCameras` calls now fit k4:
- `run_multitemporal_ba()` (Step 1 multi-temporal BA) — [metashape.py:987](../../cntp/metashape.py)
- `run_single_day_fixed_iop()` (Step 2 single-day fixed-IOP) — [metashape.py:1134](../../cntp/metashape.py)

These are the only two `optimizeCameras` calls in the codebase; `tools.py`
(`build_reference_tlc_cloud`) reaches them via `run_single_day_fixed_iop`, so it
inherits the change. Note this only affects **new** reconstructions where the
sensor isn't fixed — reference-day IOP loaded fixed from the registry /
`adjusted_calib_4DSfM` XMLs is unchanged unless re-bootstrapped.

### To revert
Set `fit_k4=True` → `fit_k4=False` on both lines (they're identical, so a
single replace-all flips both back).

---

## 2026-06-10 — TLC-only reference for change detection (prototype in `tools.py`)

### Goal
Remove the systematic UAV↔TLC instrument bias from the DoD / M3C2 change signal.
The reference cloud `Reference_UAV_TLC_PCS.laz` is a **fused UAV + timelapse-camera**
cloud. It's a good *coregistration* target (dense, lots of stable terrain), but every
daily cloud is **TLC-only**, so differencing `day_TLC − fused_ref` measures the
instrument offset on top of real glacial change. Fix: build a **TLC-only reference**
for the baseline day and difference TLC-vs-TLC.

### Concept — split the two roles the reference plays
- **Coregistration** (Step 3 ICP, Step 3b QC) → stays on the **fused** `ref_cloud`.
- **Change detection** (reference DEM/ortho, per-day DEM grid anchor, DoD, M3C2) →
  switches to a **coregistered TLC-only reference**.

**Why coregister the TLC reference too (common-mode cancellation):** if both the daily
TLC clouds *and* the TLC reference are aligned to the *same* fused reference by the
*same* 3-stage ICP, any systematic TLC↔fused shift is applied identically to both and
**cancels** in `day_TLC − refday_TLC`. (Requires using the **same coreg knobs** —
`ref_downsample`, `p2p/sp2p/m_sp2p_max_disp` — for the reference build and the daily
runs. Different knobs reintroduce a residual offset.)

### How the TLC reference is built (the "latter" / tight-`alignCameras` approach)
The reference day's TLC cameras already have validated IOP + EOP in the registry
(the fused BA exported them; `bootstrap_registry` filtered to TLC labels via
`is_timelapse_label`). So we **reuse** that geometry instead of bundle-adjusting:
fixed-IOP single-day reconstruction from *only* the reference-day TLC images, EOP
pinned **tight (0.001)** to the registry values so `alignCameras` snaps to the
registry pose, then the same `pc_align_p2p_sp2p` coreg a daily cloud gets. Output is
saved as **`.las`** (not `.laz`) so `build_dem_and_ortho` ingests it without a
decompress step.

### New file content: `contributors/umayr/tools.py` (scratch — not yet promoted)
| Function | What it does |
|----------|-------------|
| `build_reference_tlc_cloud()` | Once-per-glacier. Derives `ref_date` from the registry (normalised via `_normalize_date` — registry dates drift to `11/27/2023` when opened in Excel), synthesises the Step-1 cameras CSV (`Label/Lon/…/Roll`) + `calib_dir` from the registry rows, calls `run_single_day_fixed_iop` with **tight** `loc_acc=rot_acc=(0.001,)*3`, builds the fused stable ref (`extract_stable_reference`), coregisters via `pc_align_p2p_sp2p`, and copies the result to `output_new/_ref_cache/reference_TLC_coreg.las`. |
| `run_4dsfm_day_with_rasters_tlc()` | Drop-in for the monthly loop. SfM + coreg half = unchanged `run_4dsfm_day` (fused target). Raster half repointed at `change_ref_cloud`: reference DEM/ortho, **per-day DEM grid anchor** (so the day DEM clips to the TLC footprint and `build_dod`'s same-grid assert holds), and M3C2 reference all use the TLC cloud. **No fused-reference DEM/ortho is produced.** |

### Notebook wiring (when promoted)
- **`setup_new_glacier`** (after bootstrap): `change_ref = build_reference_tlc_cloud(...)`.
- **`4d_sfm_dem_monthly`**: pass `change_ref_cloud = output_dir/"output_new"/"reference_TLC_coreg.las"` to `run_4dsfm_day_with_rasters_tlc(...)`; errors clearly if missing.

### Status / next
Prototyped in `tools.py`; pre-flight on **Changri West** passes (license active,
registry → `2023-11-27`, UTM 32645, 46 imgs C1–C5). Pending: smoke-test in the
notebook, then migrate both functions into `cntp.pipeline_4dsfm` (add
`change_ref_cloud` param to the real `run_4dsfm_day_with_rasters`).

### To revert
Reset `contributors/umayr/tools.py` to the scaffold (module docstring +
`import cntp  # noqa: F401`). Nothing in `cntp/` changed yet.

---

## 2026-06-04 — v0.1.0 release branch: typing/lint cleanup, point2dem promotion, install-guide README, fresh-install validation

### Goal
Finalise the library for the supervisor meeting on a versioned branch: clear IDE
warnings, promote a useful helper, complete the install docs, and prove the README
install actually works on a clean machine.

### Branch / release
All work below lives on branch **`v0.1.0`** (matches `pyproject.toml` version; not yet
released, hence the branch name not a tag). Pushed to `origin/v0.1.0`. The personal dev
log (`project_knowledge.md`) and scratch/output folders are deliberately **not** tracked
on the branch. Commit chain: `d27a9fc` → `cbe6ffe` → `ebbe729` → `67a7ea1` → `3fdb527`
→ `a6da90a` → `53763f8` → `6410896` → `671488c`.

### `cntp/metashape.py` — optional-import typing fix (`d27a9fc`)
Wrapped the optional Metashape import in a `TYPE_CHECKING` guard:
```python
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    import Metashape          # type checker always sees the real module
else:
    try: import Metashape
    except ImportError: Metashape = None
```
Clears the 10 Pylance *"Variable not allowed in type expression"* warnings on the
`"Metashape.Chunk"` / `"Metashape.Sensor"` annotations (the runtime `= None` fallback was
making `Metashape` look like a variable, not a type). Runtime behaviour unchanged
(`TYPE_CHECKING` is `False` at run time). **Revert:** restore the plain
`try/except ImportError: Metashape = None` block and drop `TYPE_CHECKING` from the import.

### `cntp/asp.py` + `contributors/umayr/tools.py` — promote `point2dem` (`d27a9fc`)
Moved the `point2dem()` ASP-DEM-rasterisation wrapper out of the scratch `tools.py` into
`cntp.asp` (not on the pipeline path — `cntp.raster.build_dem_and_ortho` is — but kept as
a faster, lower-memory alternative). Adapted to house style: uses the module's
`_run_command` helper instead of an inline `subprocess.run`; `read_las_bounds` imported at
top (added to the existing `from cntp.io import …`); fixed the docstring cross-refs that
wrongly pointed at `cntp.io.build_dem_and_ortho` (it lives in `cntp.raster`); final
"ASP DEM →" print gated on `verbose`. `tools.py` reset to just the module docstring +
`import cntp  # noqa: F401` scaffold. **Revert:** delete `point2dem` from `cntp/asp.py`,
drop `read_las_bounds` from the io import.

### `cntp/pipeline_4dsfm.py` + README — `match_downscale` default 0 → 1 (`ebbe729`)
Both entry points (`run_4dsfm_day`, `run_4dsfm_day_with_rasters`) defaulted
`match_downscale` to `0` (2× upscale) while the underlying `metashape.py` functions
defaulted to `1`. Aligned the pipeline defaults to `1` (full res) and updated the README
parameters table. **Revert:** set the two pipeline defaults back to `0`.

### `environment.yml` — add `libglu` (`67a7ea1`) — found via the fresh-install test
A clean env from `environment.yml` could **not import Metashape**:
`libGLU.so.1: cannot open shared object file`. The original `cntp` env had `libglu`
(conda-forge) installed manually but it was never declared, and nothing else pulls it in.
Added `libglu` to `environment.yml`. **Revert:** remove the `libglu` line.

### `README.md` — complete the install docs + structure (`cbe6ffe`, `3fdb527`, `a6da90a`, `53763f8`, `6410896`, `671488c`)
- **Platform support** subsection + table: `cntp` code is OS-independent, but **ASP has no
  native Windows build** → Linux native / macOS should-work / Windows-via-WSL2 only.
  Metashape dependency row softened (Windows/macOS wheels also exist, not Linux-only).
- **Installing the Metashape Python module** subsection: download wheel → `pip install
  <path>` into the activated env → `abi3` note (a `cp39` wheel installs on 3.14) →
  `AGISOFT_LICENSE_PATH` before import (notebook + shell forms) → verify one-liner.
- **Installing NASA Ames Stereo Pipeline (ASP)** subsection: 3 numbered steps — conda
  install (`nasa-ames-stereo-pipeline` channel, `stereo-pipeline 3.6.0`) or tarball; add
  ASP `bin/` to `PATH` via `~/.bashrc` + `source` (with "adjust to your own bin/" note,
  since ASP lives in a separate env and can't be `conda activate`d alongside `cntp`);
  verify with `which`. (Iterated from verbose → streamlined per feedback.)
- **Table of Contents** added; external-deps table Metashape/ASP rows now point at their
  "full steps below" subsections (dedup); **Pipeline parameters** opens with a
  six-required-arguments table; Quickstart gained a Note that step (a) standardise is
  optional when images are already standard.

### Emoji / sign cleanup (`cbe6ffe`)
Repo-wide audit for emoji/decorative signs. Removed the one true offender — `⚠` (WARNING
SIGN) — from `4d_sfm_dem_monthly.ipynb`'s per-date failure `print` (surgical byte-replace,
notebook outputs preserved). Kept legitimate scientific/typographic notation
(`° ± ≤ σ ∇ × → — …`) and the `# ── …` box-drawing section dividers. README kept
emoji-free throughout (swept after every edit).

### Fresh-install validation (throwaway, then cleaned up)
Clean-room test of the README install: cloned `v0.1.0` to `/tmp/cntp_fresh`, built a
**separate** `cntp_test` conda env from `environment.yml`, `pip install -e .`, added the
existing Metashape wheel from `/mnt/g/2023_11_Nepal/2023_Changri/`. Result: all 13
third-party imports, all 10 `cntp` submodules, and all entry points import; ASP found
(separate `asp` env on PATH); ImageMagick present (it only looked missing because the env
wasn't `conda activate`d). **One real bug:** Metashape import failed on missing `libGLU`
→ fixed by the `libglu` addition above (verified the fix in `cntp_test`). Afterwards
removed `/tmp/cntp_fresh` and the `cntp_test` env (1.9G); the working `cntp` env and the
user's Metashape wheel were untouched.

### To revert (this whole entry)
Per-section reverts listed above; or `git revert`/`checkout` the commit chain
`d27a9fc..671488c` on `v0.1.0`.

---

## 2026-06-04 — README rewrite + docs/workflow image + environment.yml cleanup

### Goal
Make the repo presentable as a real library (pre-supervisor-meeting): a complete
README, a workflow diagram image, and an `environment.yml` that matches reality.

### `README.md` — full rewrite
Replaced the 2-line stub with a complete README (the original tagline line is kept
verbatim). Sections: overview + workflow image; **Standard image format** (filename
pattern + directory tree + a Note that the standardise step is optional if images are
already standard); Installation (conda + `pip -e .` + external-deps table: Metashape
**v2.3.1**, NASA Ames Stereo Pipeline on `PATH`, ImageMagick `identify`); Quickstart
(one-time setup + per-date, condensed from the notebooks); Inputs; **Coordinate
systems** (UTM/metre requirement; the northern-hemisphere-only `_utm_epsg` limitation);
Outputs (annotated `output_new/` directory tree + a headline-deliverables table with
the DoD/M3C2 sign convention and meaning); **Pipeline parameters** (4 grouped tables of
the `run_4dsfm_day*` flags with defaults); Module map; Notebooks (only the 3 current
drivers — `setup_new_glacier`, `4d_sfm_dem_monthly`, `4d_sfm_pipeline` — with the first
two flagged *main*; the rest noted as exploratory/test notebooks); License; Hackathon.
Factual corrections folded in: reference cloud = UAV **+ same-day time-lapse**
(GCP-aligned), not UAV-only; the glacier mask is used **for** co-registration (align on
stable terrain), not just the accuracy check; reference input format stated as **LAZ**
only; reference cloud named **Reference_UAV_TLC_PCS.laz** (homogeneous with the per-day
notebooks). Used GitHub-renderable `> **Note —**` callouts (not `[!NOTE]`, which the IDE
preview doesn't render).

### `docs/workflow.png`
New `docs/` folder + workflow diagram image; the README "What it does" section now
references `![…](docs/workflow.png)` (replaced the inline ASCII diagram). The image
shows `bootstrap_registry` exporting the reference cloud alongside the registry.

### `contributors/umayr/setup_new_glacier.ipynb`
Stage 3 code cell now defines and passes
`ref_cloud_out = output_dir/"output_new"/"Reference_UAV_TLC_PCS.laz"`; the Verify cell
surfaces the exported reference-cloud path + the per-day inputs.

### `environment.yml`
- **Removed** `opencv` and `openpyxl` — unused (`cv2` imported nowhere; `read_excel`
  went away with `load_reference`; confirmed neither is used in any notebook).
- **Added** `pytest` (the `tests/` suite needs it; it is *not* pulled in by `xdem`).
- Deliberately **did not declare** the scientific/geospatial stack (`numpy`, `pandas`,
  `scipy`, `matplotlib`, `rasterio`, `geopandas`, `shapely`, `pyproj`) — they install
  transitively via `xdem`, and the user chose to keep relying on that. (Dependency
  audit: the only direct third-party imports across `cntp/` are these 13 — Metashape,
  geopandas, laspy, matplotlib, numpy, pandas, py4dgeo, pyproj, pytest, rasterio, scipy,
  shapely, tqdm — all installed in the env.)

### To revert
`git checkout` `README.md` / `environment.yml` /
`contributors/umayr/setup_new_glacier.ipynb`; delete `docs/workflow.png`.

---

## 2026-06-04 — bootstrap_registry now exports the reference point cloud

### Goal
Close a setup gap: the per-day pipeline (`run_4dsfm_day`) takes `ref_cloud` as an
**input**, but nothing in the setup produced it — the reference point cloud was an
undocumented manual Metashape export. Now `bootstrap_registry` exports it too, so one
call yields everything the per-day pipeline needs (registry + calibrations + ref cloud).

### `cntp/metashape.py` — `bootstrap_registry`
- New params: `ref_cloud_out: str | Path = None` (default `output_dir/output_new/reference.laz`)
  and `export_ref_cloud: bool = True`.
- After writing the registry, exports the chunk's dense point cloud via
  `chunk.exportPointCloud(..., format=Metashape.PointCloudFormatLAZ, crs=UTM, save_point_color=True)`
  — **`.laz`** (compressed, to match the existing `Reference_UAV_TLC_PCS.laz`, so the
  per-day `ref_cloud` path/format is unchanged). Skips if the file exists and `overwrite=False`.
- UTM zone derived from the time-lapse cameras' mean longitude (read back from the
  just-written cameras CSV via `_utm_epsg`) — the same zone `run_4dsfm_day` derives from
  the registry, so the per-day clouds co-register in a matching CRS.
- Return dict gains a `ref_cloud` key (the exported path, or `None` if skipped/disabled).
- Verified `Metashape.PointCloudFormatLAZ` exists in the installed module (v2.3.1).
  Compiles + ruff-F clean; the Metashape export call itself is byte-faithful to the
  proven pattern in `run_single_day_fixed_iop` but **untested here** (no license).

### Docs updated
- `README.md`: "Reference point cloud" input row now says it's exported automatically by
  `bootstrap_registry` (returned as `result["ref_cloud"]`); Quickstart threads
  `setup = bootstrap_registry(...)` → `ref_cloud = setup["ref_cloud"]`.
- `contributors/umayr/setup_new_glacier.ipynb`: intro, Stage 2 checklist, Stage 3, and
  Verify cells updated to mention the `.laz` export and `result["ref_cloud"]`.

### To revert
Remove the `ref_cloud_out`/`export_ref_cloud` params, the export block, and the
`ref_cloud` return key from `bootstrap_registry`; revert the README/notebook wording.

---

## 2026-06-04 — Codebase lint pass (ruff) + review findings

### Goal
Pre-meeting cleanliness sweep over the whole `cntp/` package. Installed `ruff`
into the `cntp` conda env (dev tool only — not added to `environment.yml`) and
ran `F` (pyflakes), `E9`, plus `B`/`SIM`/`C4`/`UP` style families.

### Applied (zero-risk, auto-fixed with `ruff --fix`)
- **Removed dead imports** left over from the legacy removal: `io`, `os`,
  `zipfile` in `cntp/metashape.py`; `LinearSegmentedColormap` in `cntp/plot.py`.
- **Removed pointless f-strings** (F541 — `f"..."` with no `{}` placeholder) at
  `cntp/pipeline_4dsfm.py` lines 184, 260, 339 (dropped the `f` prefix only).

Result: package is **F-clean** except one intentional, deferred warning (below).
Verified: `py_compile` + `import cntp` + entry-point imports all pass.

### Reviewed but deliberately NOT changed
- **5 unused functions kept** (referenced nowhere in package or notebooks):
  `downsample_point_cloud`, `filter_points`, `filter_points_outside_box`,
  `otsu_thresholding` (`cntp/coreg.py`), `plot_stable_terrain_geometry`
  (`cntp/plot.py`). Kept because `coreg.py` is the py4dgeo-based point-cloud
  coreg path, now superseded by ASP `pc_align` (`cntp/asp.py`); the whole module
  may be retired later, so no point selectively trimming now.
- **`__init__.py` F401** (`import cntp.preprocess` "unused") — left as-is. It's an
  intentional submodule import; the real fix is to declare a public API
  (`__all__` + `__version__` + re-export `run_4dsfm_day`, `homogenize_images`,
  `ensure_standardized`, `discover_images`, `bootstrap_registry`), which also
  silences the warning. Deferred to the README/`pyproject` polish step.

### Clean-bill checks (no findings)
No syntax errors / undefined names (E9/F), no bare `except`, no `TODO/FIXME`,
no hardcoded absolute paths in library logic, no empty section banners.

### Outstanding optional polish (not blockers)
Flesh out `README.md` + `pyproject.toml` (declare deps/metadata + a `[tool.ruff]`
block), expose the public API in `__init__.py`, and the `naming.py` / `registry.py`
split of `metashape.py`.

### To revert
`git checkout` the affected files; nothing structural changed (only removed unused
imports and `f` prefixes).

---

## 2026-06-04 — Remove legacy single-epoch pipeline from cntp/metashape.py

### Goal
Pre-supervisor-meeting cleanup. `metashape.py` carried the original single-epoch
pipeline (`process_day` + `run_pipeline`) that predates and is fully superseded by
the 4D SfM stack (`run_4dsfm_day` → `run_multitemporal_ba` / `run_single_day_fixed_iop`).
It was only ever used by the exploratory `test_metashape.ipynb`. Removed entirely.

### Removed from `cntp/metashape.py` (1701 → 1249 lines, −452)
- `process_day` and `run_pipeline` (legacy single-epoch pipeline).
- `_set_camera_references` (used only by `process_day`; distinct from the kept
  `_set_camera_references_from_csv`, which the 4D SfM stack uses).
- `load_reference` (read the per-camera `ref.xlsx`; only `run_pipeline` used it —
  the 4D workflow uses the reference *registry*, not an Excel sheet).
- `_build_image_ref_csv` (already orphaned — called by nothing).
- The legacy module docstring (a `run_pipeline` example) → rewritten to describe the
  module as the Metashape SfM engine for the 4D pipeline.
- Fixed a `run_pipeline` reference in the `update_metashape_cameras_after_transform`
  docstring.

### Kept (the 4D SfM stack still uses them)
`_setup_sensors`, `_assign_sensors`, `_setup_camera_groups`, `_assign_camera_groups`,
`_export_camera_csv`, `_export_calib_xml`, `_set_camera_references_from_csv`,
`_setup_sensors_multitemporal`, `run_multitemporal_ba`, `run_single_day_fixed_iop`,
`bootstrap_registry`, `update_registry`, `update_metashape_cameras_after_transform`,
`rebuild_coreg_cloud`, and all naming/registry helpers.

### Deleted files
- `contributors/umayr/test_metashape.ipynb` (+ its `.ipynb_checkpoints` copy) — the
  practice notebook for learning the Metashape API.
- Stale `.ipynb_checkpoints/4d_sfm_pipeline-checkpoint.ipynb` (orphaned old version
  that still imported `load_reference`; the live `4d_sfm_pipeline.ipynb` is unaffected).

### Verification
`py_compile` all modules clean; `import cntp` + all entry points import; a dangling-ref
sweep across `cntp/` and live notebooks finds zero references to the removed symbols;
`discover_images` / `is_timelapse_label` still pass their functional checks.

### To revert
`git checkout` the pre-removal `cntp/metashape.py` and restore the deleted notebooks
from git history (note: the notebooks were untracked, so they're only recoverable if
they were committed elsewhere).

---

## 2026-06-03 — Generalise library to any glacier/cameras (Changri North C6–C10)

### Goal
Run the same pipeline on a second glacier (Changri Nup North, cameras C6–C10) with **no code edits per site**. Stop hardcoding the camera set, the `_renamed` folder convention, and the drone whitelist; fold the one-time registry bootstrap into a library function; promote the raw-image renamer into the package.

### Core idea
The library now derives camera identity **from the data**, never from a constant. A time-lapse image is anything whose filename matches `<camera>_<YYYY-MM-DD>_<HHMMSS>.<ext>`; the camera id is everything before the first underscore (so `C10`'s two digits work, as does any name). Each glacier = one folder you point the library at.

### `cntp/metashape.py` changes
- **Removed** the module constant `CAMERAS = ["C1".."C5"]` and the `C[1-5]` regex.
- `_IMG_RE` is now `^(?P<cam>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})_\d{6}\.(?:jpg|jpeg|JPG|JPEG)$` (named groups, generic camera, any digit count).
- **New** `_LABEL_RE` + public `is_timelapse_label(label)` — "does this label look like a time-lapse photo?" Replaces the old `CAMERAS` whitelist for dropping drone/UAV images (drone labels like `FC6310R (8.8mm)` / `DJI_0457` don't match).
- `discover_images(tlcam_dir)` now `rglob`s the whole tree and keys off the **filename**, ignoring folder names — `C6_renamed/`, `C6/`, or all-in-one-folder all work. No more `{cam}_renamed` assumption. **Perf fix (2026-06-03):** removed the per-entry `is_file()` stat — the strict `<cam>_<date>_<time>.<ext>` regex already excludes directories, and on `/mnt/g` (drvfs) a stat per file made this ~1000x slower (200 s vs 0.2 s for 16k files). It runs once per `run_4dsfm_day` call, so in a monthly batch loop the stat version was adding ~100-200 s **per date** even when every compute step was cached. (Also surfaced: keep each glacier's `_renamed` set in its own folder — a `ChangriNorth_renamed/` left *inside* `TLCAM/` gets recursively scooped into a West scan that points at `TLCAM`, both mixing glaciers and doubling the scan.)
- Four `for cam in CAMERAS` loops made data-driven: `_setup_sensors` iterates `calib_xmls`; the `process_day` calib loader, both `_setup_sensors_multitemporal` calib loops, the `run_multitemporal_ba` new-calib loader, and the `run_single_day_fixed_iop` calib dict all now `glob("*.xml")` and take the camera id from the file stem. `len(CAMERAS)` in a log line → `len(new_calib_xmls)`.
- `_export_camera_csv(chunk, out_path, keep=None)` gained an optional `keep` predicate (label→bool) to filter cameras on export; default unchanged (exports all).
- **New** `bootstrap_registry(ref_psx, chunk_label, ref_date, output_dir, registry_csv, cameras_csv_out=None, calib_dir_out=None, overwrite=False)` — opens a reference `.psx`, exports EOP (via `_export_camera_csv(..., keep=is_timelapse_label)`) + IOP XMLs (sensors whose prefix is in the standardised cameras), reconstructs `date_images`, calls `update_registry`. Idempotent (skips if registry exists unless `overwrite=True`). This is the old `bootstrap_registry.ipynb` cells 6–11 as one call.

### New file — `cntp/preprocess.py`
`homogenize_images(base_dir, output_dir=None, cameras=None, subfolders=None, extensions=…, per_camera_subdir=True, manifest=None, verbose=True)` — the raw→standard "translator" (promotes the old standalone renamer script). **`output_dir` defaults to a sibling `<base_dir>_renamed`** next to the raw tree (e.g. `…/TLCAM/ChangriNorth` → `…/TLCAM/ChangriNorth_renamed`); kept outside `base_dir` so the output is never rescanned as a camera. Override only if you want it elsewhere. Auto-detects camera folders (immediate subdirs — the **folder name becomes the camera prefix**) and recurses for nested EK-card subfolders; reads EXIF `DateTimeOriginal` via ImageMagick `identify`; **copies** to `<cam>_<date>_<time>.JPG` leaving originals untouched; writes a `manifest.csv` (camera, original_path, standard_name, datetime, status). Fixes the old script's 3-way filename-collision bug (numeric suffix loop guarantees uniqueness). Optional — skip it if you already have standard-named images. (`cntp/__init__.py` already imported `cntp.preprocess`; the file was previously empty.)

**Parallelism — `n_jobs` (2026-06-03).** `homogenize_images` gained `n_jobs: int | None = None` (default `min(8, os.cpu_count())`; `1` = serial). The per-camera loop is now three phases: (1) read EXIF capture times in parallel (`_read_stamp`, read-only), (2) assign unique standard names **serially** (no name-collision race), (3) copy files in parallel (`_copy_task`, each writes a distinct pre-assigned dst). A nested `_run(fn, items)` maps via `ThreadPoolExecutor` when `n_jobs>1`, else serially — output is byte-identical regardless of `n_jobs` (verified: parallel vs serial give the same files, collision suffixes, and manifest; bad-EXIF images skipped). Measured ~3.6× on 40 real images at n_jobs=8; n_jobs=16 was *slower* (16 concurrent `identify` process spawns contend), hence the cap at 8. ThreadPool (not ProcessPool) because each `identify` is already a subprocess and `shutil.copy2` is I/O-bound, so the GIL is released during the waits.

**Notebook front door — `ensure_standardized(image_dir, cameras=None, **homogenize_kwargs) -> Path`** (in `cntp/preprocess.py`). Idempotent one-call setup cell for the 4D SfM notebook: (1) if `image_dir` already holds standard-named images → returns it; (2) elif a sibling `<image_dir>_renamed` with a completed run (its `manifest.csv` present + standard images) exists → returns that (re-run is cheap, no 3 h redo); (3) else runs `homogenize_images` and returns the new `<image_dir>_renamed`. Returns the dir to pass as `tlcam_dir`. Completion is detected via the manifest (written only at the end), so an interrupted run is flagged, not silently reused. Tested: messy→homogenise, re-run→reuse, already-standard→as-is.

**Dependency — ImageMagick.** `homogenize_images` shells out to the `identify` CLI for EXIF capture-time, so ImageMagick must be on PATH (i.e. the `cntp` conda env active). Added `imagemagick` to `environment.yml` (next to the other CLI tools) and installed into the `cntp` env on 2026-06-03 (`conda install -n cntp -c conda-forge imagemagick`; v7.1.2, where `identify` is a symlink to `magick`). Verified on real Changri North data: `/mnt/g/2023_11_Nepal/2023_Changri/TLCAM/ChangriNorth/{C6..C10}/<EK-card>/*.JPG` → `C6_2024-01-07_160001.JPG` etc., with a round-trip through `discover_images`. (Note: those images are dated Jan 2024, not Nov 2023.) **To revert:** remove the `imagemagick` line from `environment.yml`.

### Notebook — `contributors/umayr/bootstrap_registry.ipynb`
Replaced the hand-written cells with a thin driver calling `bootstrap_registry(...)`. Original preserved at **`contributors/umayr/bootstrap_registry_LEGACY.ipynb.bak`** (both untracked in git). The 3 old `from cntp.metashape import CAMERAS` drone-skip checks are gone (logic now inside the library function).

### New notebook — `contributors/umayr/setup_new_glacier.ipynb`
One-time, per-glacier onboarding notebook that supersedes `bootstrap_registry.ipynb`. Three ordered stages reflecting the real workflow seam (one-time setup vs repeated per-day): **Stage 1** standardize images (`tlcam_dir = ensure_standardized(raw_dir)`); **Stage 2** a markdown-only checklist for the MANUAL Metashape step (build GCP reference cloud + calibrate cameras, save `.psx`, note chunk label) — the notebook deliberately stalls here for GUI work; **Stage 3** `bootstrap_registry(...)` to write the registry, then a verify cell. Rationale: the manual Metashape step sits between two automated steps, so a single run-all notebook is impossible; keeping one-time setup (incl. the ~3 h homogenize) out of the per-day `4d_sfm` notebook avoids accidental reruns. Pure ASCII, no emojis. `bootstrap_registry.ipynb` is now redundant (its logic == Stage 3); kept for now, safe to delete.

### Verification
Synthetic tests (cntp env, no Metashape needed) pass: `is_timelapse_label` cases (incl. `C10`, `CAM1`, drone, junk); `discover_images` across `_renamed`/plain/flat layouts + nested + two-digit camera + multi-date, excluding drone/junk; `homogenize_images` auto-detect/recursion/collision-uniqueness/manifest (EXIF reader patched). `py_compile` clean; `grep` confirms zero `CAMERAS` code refs remain (one docstring mention only). Changri West behaviour is unchanged — `C1…C5` still match the generic detection, and the real reference project's sensors (`FC6310R (8.8mm)`, `C1…C5`) filter identically.

### To revert
- `cntp/metashape.py`: restore `CAMERAS` constant + `C[1-5]` `_IMG_RE`; revert `discover_images`, the calib loops, `_export_camera_csv` signature; delete `is_timelapse_label`, `_LABEL_RE`, `bootstrap_registry`.
- Delete `cntp/preprocess.py` contents (back to empty).
- Restore the notebook from `bootstrap_registry_LEGACY.ipynb.bak`.

### To run Changri North
1. `homogenize_images("/mnt/e/2023_11_Nepal/2023_Changri_North/TLCAM", "/home/asus/Timelapse_north")` (or skip if already standard-named).
2. `bootstrap_registry(...)` once, pointing at the north reference `.psx` + its own `registry_csv`.
3. `run_4dsfm_day(tlcam_dir="/home/asus/Timelapse_north", registry_csv=…, ref_cloud=…, glacier_mask=…, output_dir=…)` exactly as for the west — no code changes.

---

## 2026-06-02 — New `cntp/raster.py` module + `run_4dsfm_day_with_rasters` orchestrator

### Goal
Consolidate every raster-domain function (DEM/ortho/DoD/stable/M3C2 → 2-D GeoTIFF) into a single library module so the per-date raster pipeline can live behind one library entry point and the monthly batch notebook can be as minimal as `4d_sfm_pipeline.ipynb`.

### New file — `cntp/raster.py`
Houses eight functions, all moved from existing modules (no rewrites — byte-faithful relocation). Imports `load_las`, `read_las_bounds` from `cntp.io` and `_NDWI_A`, `_NDWI_B`, `run_m3c2` from `cntp.coreg`.

| Function | Moved from |
|---|---|
| `save_dem` | `cntp/io.py` |
| `save_ortho` | `cntp/io.py` |
| `interpolate_and_mask` | `cntp/coreg.py` |
| `build_dem_and_ortho` | `cntp/io.py` |
| `build_reference_dem_and_ortho` | `cntp/io.py` |
| `build_dod` | `cntp/io.py` |
| `extract_stable_terrain_from_dem` | `contributors/umayr/tools.py` |
| `m3c2_to_raster` | `contributors/umayr/tools.py` |

### Source modules trimmed (no re-export shims — clean break)
- **`cntp/io.py`** 484 → 136 lines. Now contains only LAS/LAZ I/O (`read_las_bounds`, `load_las`, `save_las`) + `apply_glacier_mask` (point-cloud filter). Dropped imports: `rasterio`, `RasterioIOError`, `cKDTree`.
- **`cntp/coreg.py`** 293 → 272 lines. `interpolate_and_mask` removed. Dropped imports: `griddata`, `cKDTree`.
- **`contributors/umayr/tools.py`** 466 → 149 lines. Only `point2dem` (ASP CLI wrapper) remains. Imports slimmed to `os`, `shutil`, `subprocess`, `Path`, `laspy`, `cntp`. `point2dem` itself still does a deferred `from cntp.io import read_las_bounds` for the `ref_las` projwin lookup.

### New entry point — `cntp/pipeline_4dsfm.py` `run_4dsfm_day_with_rasters`
Per-date orchestrator that bundles `run_4dsfm_day` (SfM Steps 1-7) with the full raster pipeline. Returns `{"date", "sfm", "dod_stats", "stable_stats", "m3c2_stats"}`. All raster knobs (`res`, `max_gap_pixels`, `ref_cloud_downsample`, `m3c2_ref_downsample`, `slope_threshold`, and six per-stage `overwrite_*` flags) and the SfM kwargs are top-level params. Writes histograms next to the rasters via `cntp.plot.plot_dod_histogram`. Reference rasters cache in `_ref_cache/` so the second+ date reuses them.

Imports done locally inside the function (rasterio / laspy / cntp.raster / cntp.plot) so callers of plain `run_4dsfm_day` aren't dragged through raster deps on the hot import path.

### Notebooks updated
- **`contributors/umayr/4d_sfm_dem.ipynb`** (single-date, interactive) — 6 import lines switched from `from cntp.io import …` / `from contributors.umayr.tools import …` to `from cntp.raster import …`. Cells and behaviour otherwise unchanged.
- **`contributors/umayr/4d_sfm_dem_monthly.ipynb`** rewritten from 8 cells (one of them a 161-line wall) to **6 minimal cells** mirroring `4d_sfm_pipeline.ipynb`: title / imports / `## Configuration` / paths+dates+params dict / `## Run` / loop calling `run_4dsfm_day_with_rasters(**params)` with try/except + summary print. The `add_to_registry=False` (and all overwrite flags) are exposed in the single `params` dict in cell 3.

### Verification
- `ast.parse` clean on all four touched library files.
- `import cntp.raster, cntp.io, cntp.coreg, cntp.pipeline_4dsfm` all succeed; old names absent from `cntp.io` / `cntp.coreg` / `contributors.umayr.tools`.
- All three test modules (`test_coreg_accuracy`, `test_stable_terrain`, `test_regression`) still import — none referenced the moving functions.

### To revert
Move the eight functions back from `cntp/raster.py` to their original homes (sections marked in the new file's comment headers map directly to the source files they came from). Drop the new `run_4dsfm_day_with_rasters` from `cntp/pipeline_4dsfm.py`. Restore the old `4d_sfm_dem_monthly.ipynb` from `git show 35d95ed -- contributors/umayr/4d_sfm_dem_monthly.ipynb` (the pre-rewrite 8-cell version is in that commit).

---

## 2026-06-02 — `add_to_registry` flag on `run_4dsfm_day`

### Problem
When a day's coregistration is shaky, the cameras still get appended to `reference_registry.csv` at Step 7. The next date's multi-temporal BA in Step 1 then bundle-adjusts against the polluted baseline, dragging down its alignment quality. The user observed that **freezing the registry at the original 2023-11-27 baseline** gave much more reliable results across the monthly batch than letting every date append to it.

### Fix — `cntp/pipeline_4dsfm.py` `run_4dsfm_day`
New parameter `add_to_registry: bool = True` (default preserves existing behaviour for all existing callers). When False, Step 7 prints `"[Step 7] Skipping — add_to_registry=False (registry kept frozen)"` and returns without touching `registry_csv`. All earlier steps (1–6 + 6b) run normally, so DEM / ortho / DoD / M3C2 outputs for the date are still produced.

The parameter name avoided `update_registry` to dodge shadowing the imported `update_registry` function at module scope (the call `update_registry(…)` inside Step 7 would otherwise resolve to the boolean).

### Caller updates
- **`cntp/pipeline_4dsfm.py` `run_4dsfm_day_with_rasters`** (added later in the same session) — exposes the flag at its own top-level signature and forwards it.
- **`contributors/umayr/4d_sfm_dem_monthly.ipynb`** — sets `add_to_registry=False` in the per-date `params` dict (will potentially be flipped back to True after supervisor confirms which method is better).

### To revert
Remove the parameter from both `run_4dsfm_day` and `run_4dsfm_day_with_rasters` signatures, restore the previous Step 7 block in `run_4dsfm_day` (no `if not add_to_registry: skip … else:` outer guard).

---

## 2026-06-02 — Flip DoD sign convention back to ``day − ref`` (standard glaciology)

### Change to `cntp/io.py` — `build_dod`
Formula reverted from ``ref_arr - day_arr`` to ``day_arr - ref_arr`` (line 477). New sign convention matches py4dgeo M3C2 on roughly horizontal terrain:

- **positive** = day surface higher than reference → **gain / accumulation**
- **negative** = day surface lower than reference → **loss / melt**

Earlier this session it had been flipped to ``ref - day`` (positive = melt), which produced a raster with **opposite sign to M3C2** outputs. This entry undoes that flip; the in-conversation rationale was "terrible mistake — should be day-ref so positive means gain".

Docstring at the top of the function was updated to match the new convention.

### Knock-on effects (need attention)
- DoD TIFFs already on disk (`DOD.tif`, `DOD_stable.tif`, `DOD_full.tif` under `output_new/<date>/single_day/` and `output_new/ASP_output/single_day/`) were produced with the old formula. Delete them or pass `overwrite=True` to regenerate with the new sign.
- The 2026-05-21 log entry "DEM + orthoimage + DoD raster pipeline" still describes the old ``ref − day`` convention. Left as-is for historical accuracy; this current entry is the authoritative state.

### To revert
Restore ``dod = ref_arr - day_arr`` and the previous docstring on lines 412-415 of `cntp/io.py`.

---

## 2026-05-21 — Metashape stale-state cleanup in 4D SfM project builders

### Problem
`run_4dsfm_day` raised `OSError: Document.save(): editing is disabled in read-only mode` from the second `doc.save()` (after `alignCameras`) when re-running after a prior crash. Root cause: stale `<basename>.psx` + `<basename>.files/` from the failed run on disk. `Metashape.Document()` + `doc.save(path)` to an existing path leaves the doc in a fragile state that flips to read-only on the next save.

### Fix
Mirror the cleanup `process_day` already had — nuke any stale `<basename>.psx` + `<basename>.files/` directory before `Metashape.Document()` in both 4D SfM builders.

- `cntp/metashape.py` `run_multitemporal_ba` (now line ~1124) — added 5-line cleanup before `doc = Metashape.Document()`.
- `cntp/metashape.py` `run_single_day_fixed_iop` (now line ~1312) — same.

Cleanup uses `psx_path.stem` so it works for both naming conventions (`<date>.psx` → `<date>.files`, `<date>_4DSfM.psx` → `<date>_4DSfM.files`). `shutil` already imported at module top.

### To revert
Remove the two `# Nuke any stale .psx + .files …` blocks in those two functions.

---

## 2026-05-21 — Fix cubic-griddata overshoot in DEM rasterisation

### Problem
DEMs built via `interpolate_and_mask` (`scipy.griddata(method='cubic')`) contained a tiny fraction (~0.2 %) of wildly out-of-range pixels — down to **−1,612,783 m** in one Changri DoD. Cause: Clough-Tocher 2-D cubic does not preserve monotonicity. At steep features (cone walls, cliff faces) the cubic polynomial swings far past either endpoint. The overshoot in the DEMs propagated 1:1 into `ref − day` DoDs.

Diagnostic (on the broken DOD.tif): median +0.055 m, p1/p99 ±5 m → 99.8 % of pixels were fine; only the tail was bad.

### Fix
Clip the interpolant to the input cloud's actual Z range.

`cntp/coreg.py` `interpolate_and_mask` — single line added immediately after `griddata`:
```python
zi = np.clip(zi, np.nanmin(z), np.nanmax(z))
```
Reasoning: any interpolated value outside `[min(z), max(z)]` cannot represent a real surface point in the cloud — it can only come from cubic overshoot, so capping it removes the hallucination without touching real signal.

### To revert
Remove the `np.clip(zi, np.nanmin(z), np.nanmax(z))` line in `interpolate_and_mask`. If overshoot recurs, prefer `method='linear'` (no overshoot by construction, ~3× faster on dense clouds).

---

## 2026-05-21 — `save_las` preserves CRS via header VLRs

### Problem
The cached downsampled reference (`_ref_cache/<ref_stem>_ds<f>.las`) had no CRS embedded — when `build_reference_dem_and_ortho` tried `laspy.open(...).header.parse_crs()` to auto-detect EPSG, it returned `None` and raised. Root cause: `save_las` built a fresh `LasHeader` with no CRS metadata.

### Fix
- `cntp/io.py` `save_las` — new optional `crs: int = None` parameter. When set, calls `header.add_crs(pyproj.CRS.from_epsg(crs))` which writes GeoTIFF VLRs into LAS 1.2 headers. Backward compatible: existing callers without `crs` still write without CRS.
- `cntp/pipeline_4dsfm.py:191` — downsampled ref cache now written with `crs=utm_epsg`.
- `cntp/pipeline_4dsfm.py:286` — `validated_stable.laz` also written with `crs=utm_epsg`.

Two other `save_las` callsites in `cntp/asp.py` (`extract_stable_reference`, `evaluate_coreg`) left unchanged — their outputs are used only for ICP/M3C2 where CRS isn't read.

**Note:** files written before this fix lack the CRS tag. Either delete and regenerate, or pass `utm_epsg` explicitly when calling downstream consumers (notebook does the latter already).

### To revert
Drop the `crs` parameter and `header.add_crs` block from `save_las`. Remove `crs=utm_epsg` from the two `cntp/pipeline_4dsfm.py` callsites.

---

## 2026-05-21 — DEM + orthoimage + DoD raster pipeline

### What was added

New raster-output stage downstream of the 4D SfM cloud pipeline:

`cntp/io.py`:
- `build_dem_and_ortho(cloud_las, ref_las, out_dir, name_stem, …)` — rasterise a coregistered cloud to a 1 m (configurable) DEM + ortho, both anchored to the reference cloud's XY bbox so every day lands on the same pixel grid. Reuses existing primitives `interpolate_and_mask` (Z) and `save_ortho` (RGB nearest-neighbour). Parameters: `res`, `max_gap_pixels`, `utm_epsg` (None ⇒ parse from LAS header), `cloud_downsample` (extra subsample at load time for OOM control), `overwrite` (skip if both TIFFs exist).
- `build_reference_dem_and_ortho(ref_cloud_path, cache_dir, …)` — thin wrapper that fixes `name_stem="reference"` so the cached outputs always land at `<cache_dir>/reference_dem.tif` and `reference_ortho.tif`. One-time cost, cached.
- `build_dod(ref_dem_path, day_dem_path, out_path, overwrite)` — pixel-wise `ref_dem − day_dem` on the common grid. Sanity-checks shape / transform / CRS match (which they do by construction when both DEMs come from `build_dem_and_ortho` on the same reference cloud). Writes to `<day_dem.parent>/DOD.tif` by default.

`cntp/plot.py`:
- `plot_dod_histogram(values, output_dir, title, …)` — single-distribution histogram styled like `plot_m3c2_distances`. Returns `{median, mean, std, n}` on the un-clipped data; x-axis clipped to ±3σ for visibility. Used after `build_dod` to QC the difference distribution.

### Naming decisions
- Reference rasters use the fixed stem `reference` (decision: simpler than encoding the downsample factor; the assumption is "one canonical reference per cache dir, recompute via `overwrite=True` if params change").
- Per-day rasters use the user-supplied `name_stem` (defaulting to the date string in the notebook).
- DoD output is `DOD.tif` (uppercase) — matches the user-stated naming convention.

### Sign convention for DoD
`ref − day`. Positive ⇒ reference surface higher than day ⇒ melt/scour if day is later than ref epoch. The sign convention was flipped from `day − ref` mid-session per user request.

### Notebook + scratch file
- `contributors/umayr/4d_sfm_dem.ipynb` — copy of `4d_sfm_pipeline.ipynb` with three new sections appended:
  - "## DEM + Orthoimage" → reference DEM/ortho cell + per-day DEM/ortho cell
  - "## DoD" → `build_dod` cell + `plot_dod_histogram` cell
- Each new cell is self-contained — derives all paths from cell 3 config (`output_dir`, `new_date`, `ref_cloud`, `params`). No path strings outside cell 3.
- `contributors/umayr/tools.py` — new scratch file mirroring marin's pattern (imports only); intended for prototyping before promotion into `cntp/`.

### To revert
Delete `build_dem_and_ortho`, `build_reference_dem_and_ortho`, `build_dod` from `cntp/io.py`. Delete `plot_dod_histogram` from `cntp/plot.py`. Remove the DEM/ortho/DoD cells from `4d_sfm_dem.ipynb` (cells 6–12 in current layout) or delete the notebook entirely. The new primitives don't touch any existing function — reverting them leaves the 4D SfM pipeline untouched.

---

## 2026-05-19 — Dedupe ECEF intermediates across stages and days

### Problem
Each pc_align stage wrote its own ECEF-converted copy of the reference and
TBA clouds, producing ~4.4 GB of duplicate intermediate data per day:
- `stage1/ecef/*_ecef.las` (1.9 GB ref + 204 MB TBA)
- `stage2/ecef/*_ecef.las` (same ref + TBA, byte-identical to stage1)
- `stage3/ecef/*_ecef.las` (38 MB stable ref + 204 MB TBA)

Stages 1 and 2 use the same `(ref, TBA)` pair, stage 3 swaps the ref for
the stable subset. The TBA ECEF is identical across all three. None of
this changed between days — same `las_utm_to_ecef` math on the same
shared-cache ref. ~2.3 GB/day wasted; reference ECEFs also regenerated
per-day even though they only depend on the (shared) ref cloud.

### Fix — convert each cloud once, in the orchestrator

`cntp/asp.py`:
- `pc_align_stage` no longer does any ECEF conversion or downsampling. It
  invokes pc_align on the paths it's given. Signature lost
  `ref_downsample_factor`, `tba_downsample_factor`, and `utm_epsg`.
- `pc_align_p2p_sp2p` does the ECEF conversion once up-front (before
  Stage 1), placing each output where it naturally belongs:
  - **Full ref ECEF** → `ref_icp.parent / "ecef" /<stem>_ecef.las`.
    Because the orchestrator now passes `ref_las = _ref_cache/.../.las`,
    this lands inside `_ref_cache/ecef/`, shared across all days.
  - **Stable ref ECEF** → `ref_stage3.parent / "ecef" /<stem>_ecef.las`.
    Same logic, ends up in `_ref_cache/ecef/`.
  - **TBA ECEF** → `coreg_dir / "ecef" /<stem>_ecef.las`. Per-day, but
    one copy shared across all three stages.

### Result

Old layout (per day):
```
coreg/stage1/ecef/{ref}_ecef.las  1.9 GB
coreg/stage1/ecef/{tba}_ecef.las  204 MB
coreg/stage2/ecef/{ref}_ecef.las  1.9 GB   ← duplicate of stage1
coreg/stage2/ecef/{tba}_ecef.las  204 MB   ← duplicate of stage1
coreg/stage3/ecef/{ref}_stable_ecef.las  38 MB
coreg/stage3/ecef/{tba}_ecef.las  204 MB   ← duplicate of stage1
                                  ─────
                                  ~4.4 GB / day
```

New layout:
```
output_new/_ref_cache/ecef/{ref}_ecef.las         1.9 GB  (shared across days)
output_new/_ref_cache/ecef/{ref}_stable_ecef.las  38 MB   (shared across days)
output_new/{date}/coreg/ecef/{tba}_ecef.las       204 MB  (per day)
```

Per-day footprint drops from ~4.4 GB to ~204 MB. The 1.94 GB of reference
ECEFs are also amortised across all days.

### To revert
- `cntp/asp.py`:
  - Restore `pc_align_stage`'s `ref_downsample_factor`,
    `tba_downsample_factor`, and `utm_epsg` parameters, plus the per-stage
    downsample + ECEF blocks (saves to `output_prefix.parent / "ecef"`).
  - Remove the up-front ECEF conversion block in `pc_align_p2p_sp2p` (the
    `if utm_epsg is not None:` block before Stage 1).
  - Restore the per-stage `utm_epsg=utm_epsg` kwarg on each `pc_align_stage`
    call and the `ref_icp`/`ref_stage3`/`tba_icp` arguments (no `_pc` suffix).

Old `coreg/stage{1,2,3}/ecef/` directories from prior runs are harmless;
delete them manually or wipe the day folder for a clean start.

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

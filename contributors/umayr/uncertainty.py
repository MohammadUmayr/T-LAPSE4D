"""Slope-binned uncertainty of M3C2 rasters over ice-free terrain.

The Step-3b ``<date>_stable_m3c2.tif`` rasters are post-co-registration M3C2
residuals on **ice-free terrain**, where the true change is zero, so their
dispersion measures the co-registration + reconstruction error.

Procedure
---------
1. Extract elevation offsets (dh) over ice-free areas.
2. Filter outliers using the **2–98 percentiles** of the data.
3. Bin the dh values in **5° slope intervals**.
4. Filter remaining outliers with a **3×NMAD** filter within each slope bin.

Each bin is then reported as its median (bias) and NMAD (random error), the
latter being the error bar in the figure.

Notes on interpretation
-----------------------
* M3C2 measures distance **along the local surface normal**, not vertically,
  so it already absorbs much of the ``tan(slope)`` amplification that a plain
  vertical DoD suffers. Expect a weaker slope dependence than in a vertical dh
  figure. Pass ``project="vertical"`` for a vertical-equivalent comparison.
* The 2–98 clip is global across all slopes, so it removes proportionally more
  from whichever bins carry the largest residuals. The in-bin 3×NMAD filter is
  the slope-relative one — a 2 m residual is a blunder on a smooth 30° slope
  but ordinary on a 70° rock wall.
* Because the NMAD is computed after its own ≥3-NMAD tail has been removed, it
  is biased slightly low by construction. Report it as a *filtered* NMAD.
* These are *precision* estimates for a single pixel, not the uncertainty on a
  spatially averaged quantity (e.g. glacier-wide volume change), which also
  requires the spatial correlation of the residuals.

Not part of the ``cntp`` package yet — this lives under ``contributors/``
until the approach settles. It depends only on numpy/pandas/matplotlib/
geoutils/xdem, nothing from ``cntp``, so promoting it later is a plain move.

Typical use
-----------
>>> import sys; sys.path.insert(0, "contributors/umayr")
>>> from uncertainty import run_slope_uncertainty
>>> res = run_slope_uncertainty(
...     output_dir   = "/mnt/e/umayr/Changri/Changri_North/output",
...     glacier_mask = "/mnt/e/umayr/Changri/Changri_North/shapefile/Shapefile_ChangriNorth.shp",
... )
>>> res["bins"].head()

or from the shell::

    python contributors/umayr/uncertainty.py \
        --output-dir .../Changri_North/output \
        --glacier-mask .../Shapefile_ChangriNorth.shp
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import geoutils as gu  # noqa: E402
import xdem  # noqa: E402

__all__ = [
    "SlopeBinning",
    "reference_slope",
    "collect_ice_free_dh",
    "bin_by_slope",
    "area_by_slope",
    "plot_slope_uncertainty",
    "plot_slope_comparison",
    "run_slope_uncertainty",
    "run_combined_slope_uncertainty",
]

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# Hard ceiling on slope. 90 deg is the geometric maximum, so this no longer
# discards anything — it only guards against a malformed slope raster.
SLOPE_CLIP_DEG = 90.0


# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class SlopeBinning:
    """Binning and outlier-filter settings.

    Parameters
    ----------
    width : float
        Slope bin width in degrees. Default 5.
    max_slope : float
        Upper edge of the last bin, in degrees. Steeper pixels are dropped.
    pct_clip : tuple of float, or None
        Percentile clip applied to the ice-free dh **before** binning.
        Default ``(2.0, 98.0)``. ``None`` disables it.
    nmad_factor : float, or None
        Within each slope bin, values further than this many NMADs from the bin
        median are dropped and the statistics recomputed. Default 3.0.
        ``None`` disables it.
    min_px : int
        A bin needs at least this many surviving pixel-observations before its
        statistics are trusted.
    min_area_m2 : float
        Secondary test on **ice-free ground area**. Pooling a year of
        acquisitions inflates the pixel count by the number of dates, so a bin
        covering a few hundred square metres can still show tens of thousands
        of pixels — repeat measurements of one sliver of terrain, not
        independent samples. Below 25° this is redundant with ``min_slope``,
        but it is what catches the thin 85–90° bin at the top of the range.
    min_slope : float
        Bins below this slope are not reported. At these sites the ice-free
        terrain below 25° totals 191 m² across both glaciers — 0.27% of all
        ice-free ground, scattered single pixels along the margin — so its
        NMAD describes those pixels rather than the slope class.
    """

    width: float = 5.0
    max_slope: float = 90.0
    pct_clip: tuple[float, float] | None = (2.0, 98.0)
    nmad_factor: float | None = 3.0
    min_px: int = 500
    min_area_m2: float = 1000.0
    min_slope: float = 25.0

    @property
    def edges(self) -> np.ndarray:
        return np.arange(0.0, self.max_slope + self.width / 2, self.width)


def _nmad(x: np.ndarray) -> float:
    """Normalised median absolute deviation — robust sigma."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _date_of(path: Path) -> str:
    m = _DATE_RE.search(path.name)
    return m.group(1) if m else path.stem


def coreg_nmad(output_dir: str | Path, date: str) -> float:
    """That acquisition's post-co-registration stable NMAD, or NaN.

    Reads the ``after`` row of ``<output>/<date>/coreg/<date>_m3c2_stats.csv`` —
    the number the pipeline already reported at Step 3b. Same source as
    ``cntp.postprocessing.load_coreg_nmad``, so the acquisition gate here
    matches the one behind the absolute/relative accuracy figures.
    """
    f = Path(output_dir) / date / "coreg" / f"{date}_m3c2_stats.csv"
    if not f.exists():
        return float("nan")
    try:
        return float(pd.read_csv(f, index_col="coreg").loc["after", "nmad"])
    except Exception:
        return float("nan")


def _gate_ok(nmad: float, date: str, max_nmad: float | None,
             max_nmad_from: str | None) -> bool:
    """Whether an acquisition survives the ``max_nmad`` gate.

    Mirrors ``cntp.postprocessing._nmad_keep_mask``: drop when ``NMAD >=
    max_nmad``, a NaN NMAD counts as a drop, and ``max_nmad_from`` restricts
    the gate to dates on/after it (earlier acquisitions are always kept).
    """
    if max_nmad is None:
        return True
    if max_nmad_from is not None and date < max_nmad_from:
        return True
    return bool(nmad < max_nmad)


# ──────────────────────────────────────────────────────────────────────────
# Slope from the fixed reference DEM
# ──────────────────────────────────────────────────────────────────────────
def reference_slope(
    ref_dem: str | Path,
    *,
    scale_m: float | None = None,
) -> gu.Raster:
    """Slope (degrees) of the cached reference DEM.

    Slope comes from the reference — not the per-date DEM — so the binning is
    identical for every date, and so the predictor is independent of the noise
    realisation being measured (slope from a noisy epoch DEM correlates the
    predictor with the error and inflates the apparent slope dependence).

    Parameters
    ----------
    ref_dem : path
        ``output/_ref_cache/reference_dem.tif``.
    scale_m : float, optional
        If coarser than the DEM resolution, the DEM is resampled to this pixel
        size first, so slope describes the *terrain* rather than metre-scale
        boulder roughness. On 1 m terrestrial SfM DEMs of rocky moraine, 3–5 m
        is often a more honest predictor than native resolution.
    """
    dem = xdem.DEM(str(ref_dem))
    if scale_m is not None and scale_m > max(dem.res):
        dem = dem.reproject(res=scale_m, resampling="bilinear", silent=True)
        print(f"  Slope scale: DEM resampled to {scale_m:g} m")
    return xdem.terrain.slope(dem)


def glacier_mask_array(glacier_mask: str | Path, ref: gu.Raster) -> np.ndarray:
    """Boolean glacier mask rasterised onto *ref*'s grid."""
    vect = gu.Vector(str(glacier_mask))
    if vect.crs != ref.crs:
        vect = vect.reproject(crs=ref.crs)
    m = vect.create_mask(ref).data
    m = m.filled(False) if isinstance(m, np.ma.MaskedArray) else m
    return np.squeeze(np.asarray(m, dtype=bool))


def _as_nan_array(raster: gu.Raster) -> np.ndarray:
    """Raster band 1 as float64 with masked/nodata cells as NaN."""
    data = raster.data
    if isinstance(data, np.ma.MaskedArray):
        arr = data.astype("float64").filled(np.nan)
    else:
        arr = np.asarray(data, dtype="float64")
    if raster.nodata is not None and np.isfinite(raster.nodata):
        arr = np.where(arr == raster.nodata, np.nan, arr)
    return np.squeeze(arr)


# ──────────────────────────────────────────────────────────────────────────
# Step 1 — extract ice-free dh, paired with reference slope
# ──────────────────────────────────────────────────────────────────────────
def collect_ice_free_dh(
    output_dir: str | Path,
    slope: gu.Raster,
    binning: SlopeBinning,
    *,
    dates: list[str] | None = None,
    project: str = "normal",
    max_nmad: float | None = None,
    max_nmad_from: str | None = None,
    source: str = "m3c2",
    glacier_mask: str | Path | None = None,
    ice_free_from: str = "m3c2",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pool every ice-free elevation offset with its reference slope.

    With ``source="m3c2"`` this walks ``<date>_stable_m3c2.tif``, which is
    already restricted to ice-free corepoints. With ``source="dod"`` it walks
    ``<date>_DOD.tif`` (``day_dem - reference_dem``, vertical differencing over
    the whole footprint) and applies an ice-free mask itself.

    The reference slope is reprojected onto each grid, cached per unique grid.
    DoD rasters share the reference DEM grid exactly, so that is a no-op there.

    Parameters
    ----------
    output_dir : path
        Site ``output`` directory.
    slope : gu.Raster
        From :func:`reference_slope`.
    binning : SlopeBinning
        Supplies the slope cut-off.
    dates : list of str, optional
        Restrict to these dates (``YYYY-MM-DD``). Default: all found.
    project : {"normal", "vertical"}
        ``"normal"`` keeps the native along-normal M3C2 distance;
        ``"vertical"`` converts to a vertical-equivalent (``d / cos(slope)``).
    max_nmad : float, optional
        Acquisition gate: drop whole dates whose post-coreg stable NMAD is
        ``>= max_nmad`` (failed co-registrations). Off by default. Set 0.5 to
        match the default of ``cntp.postprocessing.absolute_accuracy_boxplots``
        so the slope bins describe the same acquisitions as those figures.
    max_nmad_from : str, optional
        ``YYYY-MM-DD``; apply the gate only on/after this date.
    source : {"m3c2", "dod"}
        Which residual raster to read. ``"dod"`` is the vertical DEM difference
        — use it to test whether the along-normal M3C2 suppresses the
        ``tan(slope)`` amplification that vertical differencing suffers.
    glacier_mask : path, optional
        Required when ``source="dod"`` and ``ice_free_from="mask"``.
    ice_free_from : {"m3c2", "mask"}
        How ice-free is defined for ``source="dod"``. ``"m3c2"`` (default)
        keeps only pixels where that date's ``stable_m3c2.tif`` has data, so
        DoD and M3C2 are compared on **identical terrain and identical dates**.
        ``"mask"`` instead keeps everything outside the glacier outline, which
        covers more low-slope ground but is no longer like-for-like.

    Returns
    -------
    slopes, dh : np.ndarray
        Pooled 1-D float32 arrays, unfiltered.
    coverage : dict
        ``{"union_mask", "slope_grid", "n_dates", "n_gated"}`` — which ice-free
        pixels were sampled at least once, on the reference-DEM grid.
    """
    output_dir = Path(output_dir)
    if project not in ("normal", "vertical"):
        raise ValueError(f"project must be 'normal' or 'vertical', got {project!r}")
    if source not in ("m3c2", "dod"):
        raise ValueError(f"source must be 'm3c2' or 'dod', got {source!r}")
    if source == "dod" and ice_free_from not in ("m3c2", "mask"):
        raise ValueError(f"ice_free_from must be 'm3c2' or 'mask', got {ice_free_from!r}")
    if source == "dod" and ice_free_from == "mask" and glacier_mask is None:
        raise ValueError("source='dod' with ice_free_from='mask' needs glacier_mask")

    pattern = "*_stable_m3c2.tif" if source == "m3c2" else "*_DOD.tif"
    files = sorted(output_dir.glob(f"*/single_day/{pattern}"))
    if dates is not None:
        wanted = set(dates)
        files = [f for f in files if _date_of(f) in wanted]
    if source == "dod":
        # Only dates that also have a stable M3C2 raster, so the two sources
        # describe the same acquisitions even under ice_free_from='mask'.
        with_m3c2 = {_date_of(p) for p in output_dir.glob("*/single_day/*_stable_m3c2.tif")}
        n_before = len(files)
        files = [f for f in files if _date_of(f) in with_m3c2]
        if n_before != len(files):
            print(f"  Skipped {n_before - len(files)} DoD date(s) with no matching "
                  f"stable M3C2 raster")
    if not files:
        raise FileNotFoundError(f"no */single_day/{pattern} under {output_dir}")
    label = "ice-free M3C2" if source == "m3c2" else "DoD"
    print(f"  Found {len(files)} {label} rasters")

    slope_cache: dict[tuple, np.ndarray] = {}
    ice_free_cache: dict[tuple, np.ndarray] = {}
    ref_slope_arr = _as_nan_array(slope)
    union = np.zeros(ref_slope_arr.shape, dtype=bool)
    slopes_all: list[np.ndarray] = []
    dh_all: list[np.ndarray] = []
    n_used = 0

    # Acquisition gate — decided before any raster is read, from the per-date
    # NMAD the pipeline already reported.
    n_gated = 0
    if max_nmad is not None:
        kept = []
        for fp in files:
            date = _date_of(fp)
            if _gate_ok(coreg_nmad(output_dir, date), date, max_nmad, max_nmad_from):
                kept.append(fp)
        n_gated = len(files) - len(kept)
        win = f" (from {max_nmad_from})" if max_nmad_from else ""
        print(f"  Acquisition gate: dropped {n_gated} date(s) with post-coreg "
              f"NMAD >= {max_nmad:g} m{win} — {len(kept)} remain")
        if not kept:
            raise ValueError(f"no acquisition passes the max_nmad={max_nmad} gate")
        files = kept

    for i, fp in enumerate(files, 1):
        r = gu.Raster(str(fp))
        key = (r.shape, tuple(np.round(np.asarray(r.transform)[:6], 3)))
        if key not in slope_cache:
            slope_cache[key] = _as_nan_array(
                slope.reproject(ref=r, resampling="bilinear", silent=True)
            )
        s = slope_cache[key]
        d = _as_nan_array(r)

        # The M3C2 rasters are already ice-free; a DoD covers the whole
        # footprint and has to be masked here.
        if source == "dod":
            if ice_free_from == "mask":
                if key not in ice_free_cache:
                    ice_free_cache[key] = ~glacier_mask_array(glacier_mask, r)
                ice_free = ice_free_cache[key]
            else:
                stab = fp.with_name(f"{_date_of(fp)}_stable_m3c2.tif")
                if not stab.exists():
                    print(f"  [warn] {fp.name}: no stable M3C2 twin — skipped")
                    continue
                sm = gu.Raster(str(stab))
                sm.data = np.where(
                    np.isfinite(_as_nan_array(sm)), 1.0, np.nan
                ).astype("float32")[None, :, :]
                sm.set_nodata(np.nan)
                ice_free = np.isfinite(
                    _as_nan_array(sm.reproject(ref=r, resampling="nearest", silent=True))
                )
            d = np.where(ice_free, d, np.nan)

        ok = np.isfinite(s) & np.isfinite(d) & (s < min(binning.max_slope, SLOPE_CLIP_DEG))
        if not ok.any():
            print(f"  [warn] {fp.name}: no valid slope/residual pixels — skipped")
            continue

        # Fold this date's footprint into the reference-grid union coverage.
        cov = r.copy()
        cov.data = np.where(np.isfinite(d), 1.0, np.nan).astype("float32")[None, :, :]
        cov.set_nodata(np.nan)
        cov_ref = _as_nan_array(cov.reproject(ref=slope, resampling="nearest", silent=True))
        union |= np.isfinite(cov_ref) & (cov_ref > 0)

        s_ok, d_ok = s[ok], d[ok]
        if project == "vertical":
            d_ok = d_ok / np.cos(np.deg2rad(s_ok))
        slopes_all.append(s_ok.astype("float32"))
        dh_all.append(d_ok.astype("float32"))
        n_used += 1
        if i % 50 == 0 or i == len(files):
            print(f"    {i}/{len(files)} rasters read")

    slopes = np.concatenate(slopes_all)
    dh = np.concatenate(dh_all)
    print(f"  Pooled {dh.size:,} ice-free dh values from {n_used} dates "
          f"({len(slope_cache)} distinct grids)")
    return slopes, dh, {"union_mask": union, "slope_grid": ref_slope_arr,
                        "n_dates": n_used, "n_gated": n_gated}


# ──────────────────────────────────────────────────────────────────────────
# Steps 2–4 — percentile clip, bin, in-bin NMAD filter
# ──────────────────────────────────────────────────────────────────────────
def bin_by_slope(
    slopes: np.ndarray,
    dh: np.ndarray,
    binning: SlopeBinning,
) -> tuple[pd.DataFrame, dict]:
    """Filter, bin in 5° slope intervals, filter again, and summarise.

    Applies the 2–98 percentile clip to the pooled dh, bins by slope, then
    drops values beyond ``nmad_factor`` NMADs of each bin's median before
    recomputing that bin's statistics.

    Returns
    -------
    bins : pd.DataFrame
        One row per slope bin: ``n_px_raw`` (in-bin count before the NMAD
        filter), ``n_px`` (after), ``median``, ``mean``, ``nmad``, ``std``,
        ``p16``, ``p84``, and a ``sparse`` flag.
    filtering : dict
        Pixel counts removed at each stage, for reporting.
    """
    n_raw = int(dh.size)

    # Stage 1 — global percentile clip, before binning.
    if binning.pct_clip is not None:
        lo_v, hi_v = np.percentile(dh, binning.pct_clip)
        keep = (dh >= lo_v) & (dh <= hi_v)
        slopes, dh = slopes[keep], dh[keep]
    n_after_clip = int(dh.size)

    edges = binning.edges
    idx = np.digitize(slopes, edges) - 1
    rows = []
    n_nmad_removed = 0
    for b in range(len(edges) - 1):
        v = dh[idx == b]
        n_before = int(v.size)
        # Stage 2 — in-bin NMAD filter. This is the slope-relative one: what
        # counts as a blunder depends on the dispersion of its own slope class.
        if binning.nmad_factor is not None and n_before:
            m0, s0 = np.median(v), _nmad(v)
            if np.isfinite(s0) and s0 > 0:
                v = v[np.abs(v - m0) <= binning.nmad_factor * s0]
        n_nmad_removed += n_before - int(v.size)
        n = int(v.size)
        rows.append({
            "slope_lo": float(edges[b]),
            "slope_hi": float(edges[b + 1]),
            "slope_mid": float((edges[b] + edges[b + 1]) / 2),
            "n_px_raw": n_before,
            "n_px": n,
            "median": float(np.median(v)) if n else np.nan,
            "mean": float(np.mean(v)) if n else np.nan,
            "nmad": _nmad(v) if n else np.nan,
            "std": float(np.std(v)) if n else np.nan,
            "p16": float(np.percentile(v, 16)) if n else np.nan,
            "p84": float(np.percentile(v, 84)) if n else np.nan,
            "sparse": n < binning.min_px,
        })

    bins = pd.DataFrame(rows)
    n_kept = int(bins["n_px"].sum())
    pc = binning.pct_clip
    if pc:
        print(f"  Outlier filtering: {n_raw - n_after_clip:,} px removed by the "
              f"{pc[0]:g}–{pc[1]:g} percentile clip")
    if binning.nmad_factor is not None:
        print(f"                     {n_nmad_removed:,} px removed by the "
              f"{binning.nmad_factor:g}×NMAD per-bin filter")
    print(f"                     {n_kept:,} of {n_raw:,} ice-free px retained "
          f"({100 * n_kept / max(n_raw, 1):.1f}%)")

    filtering = {
        "n_px_raw": n_raw,
        "n_px_removed_pct_clip": n_raw - n_after_clip,
        "n_px_removed_nmad": n_nmad_removed,
        "n_px_kept": n_kept,
    }
    return bins, filtering


# ──────────────────────────────────────────────────────────────────────────
# Area distributions
# ──────────────────────────────────────────────────────────────────────────
def area_by_slope(
    slope: gu.Raster,
    binning: SlopeBinning,
    *,
    glacier_mask: str | Path | None = None,
    stable_union: np.ndarray | None = None,
) -> pd.DataFrame:
    """Terrain area per slope bin, in m^2, for glacier and ice-free terrain.

    ``glacier`` is reference-DEM area inside the outline; ``ice_free`` is the
    area actually sampled by the stable M3C2 rasters (``stable_union``) if
    given, otherwise everything outside the outline. Comparing the two shows
    whether the terrain the error model is *calibrated on* resembles the
    terrain it is *applied to*.
    """
    s = _as_nan_array(slope)
    px_area = float(abs(slope.res[0] * slope.res[1]))
    valid = np.isfinite(s)

    if glacier_mask is not None:
        vect = gu.Vector(str(glacier_mask))
        if vect.crs != slope.crs:
            vect = vect.reproject(crs=slope.crs)
        gmask = vect.create_mask(slope).data
        gmask = np.squeeze(
            gmask.filled(False) if isinstance(gmask, np.ma.MaskedArray) else gmask
        ).astype(bool)
    else:
        gmask = np.zeros(s.shape, dtype=bool)

    smask = np.squeeze(np.asarray(
        stable_union if stable_union is not None else ~gmask, dtype=bool
    ))

    edges = binning.edges
    idx = np.digitize(s, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        inb = valid & (idx == b)
        rows.append({
            "slope_lo": float(edges[b]),
            "slope_hi": float(edges[b + 1]),
            "slope_mid": float((edges[b] + edges[b + 1]) / 2),
            "glacier_area_m2": float(np.sum(inb & gmask) * px_area),
            "ice_free_area_m2": float(np.sum(inb & smask & ~gmask) * px_area),
        })
    return pd.DataFrame(rows)


def apply_area_threshold(
    bins: pd.DataFrame,
    areas: pd.DataFrame,
    binning: SlopeBinning,
) -> pd.DataFrame:
    """Attach ice-free ground area to *bins* and re-derive the ``sparse`` flag.

    ``bin_by_slope`` can only see pooled pixel counts, which are inflated by
    the number of acquisitions. This merges in the actual ice-free area per bin
    and marks a bin untrustworthy when **either** test fails:

    ``few_px``     fewer than ``binning.min_px`` surviving pixel-observations
    ``low_area``   less than ``binning.min_area_m2`` of ice-free ground

    Both component flags are kept so the reason is visible in the CSV.
    """
    a = areas.set_index("slope_lo")["ice_free_area_m2"]
    out = bins.copy()
    out["ice_free_area_m2"] = out["slope_lo"].map(a).to_numpy()
    out["px_per_m2"] = (out["n_px"] /
                        out["ice_free_area_m2"].replace(0, np.nan)).round(1)
    out["few_px"] = out["n_px"] < binning.min_px
    out["low_area"] = ((out["ice_free_area_m2"] < binning.min_area_m2)
                       if binning.min_area_m2 > 0 else False)
    out["sparse"] = out["few_px"] | out["low_area"]
    return out


C_GLAC, C_STAB = "#e8635a", "#b9d7e8"
# Blue/orange: the most reliably separable pair under colour-vision deficiency.
C_M3C2, C_DOD = "#1f4ed8", "#d95f02"


def _draw_area_panel(ax, areas: pd.DataFrame, width: float,
                     *, scale: float = 10.0, legend_loc: str = "upper right",
                     headroom: float | None = None) -> None:
    """Glacier and ice-free area bars, ice-free multiplied by *scale*.

    ``headroom`` multiplies the y-limit above the tallest bar. On the shared
    single-panel layout it keeps the bars in the lower part of the frame so the
    error bars, which sit on the other axis around zero, stay legible over them.
    """
    x = areas["slope_mid"].to_numpy()
    g_m2 = areas["glacier_area_m2"].to_numpy()
    s_m2 = areas["ice_free_area_m2"].to_numpy()
    use_km2 = max(g_m2.max(), s_m2.max()) >= 5e5
    conv, aunit = (1e-6, "km$^2$") if use_km2 else (1e-3, "10$^3$ m$^2$")
    g_a, s_a = g_m2 * conv, s_m2 * conv

    ax.bar(x - width / 6, g_a, width=width / 2.6, color=C_GLAC,
           label="Glacier area", zorder=2)
    ax.bar(x + width / 6, s_a * scale, width=width / 2.6, color=C_STAB,
           label="Ice-free area" + (f" ×{scale:g}" if scale != 1 else ""), zorder=2)
    ax.set_ylabel(f"Area ({aunit})")
    if headroom is not None:
        ax.set_ylim(0, max(g_a.max(), (s_a * scale).max()) * headroom)
    ax.legend(loc=legend_loc, fontsize=8, framealpha=0.9)


def _draw_series(ax, bins: pd.DataFrame, colour: str, label: str, *,
                 offset: float = 0.0, ylim=None) -> int:
    """One median ± NMAD series; returns how many bins the ylim would clip.

    Every reported bin is drawn the same way. The ``sparse`` / ``low_area``
    flags stay in the CSV for reference but are not rendered — the reporting
    range is set by ``min_slope``, so anything still on the plot is meant to
    be read as a measurement.
    """
    x = bins["slope_mid"].to_numpy() + offset
    med = bins["median"].to_numpy()
    nmad = bins["nmad"].to_numpy()

    ax.errorbar(x, med, yerr=nmad, fmt="o", ms=5,
                color=colour, ecolor=colour, elinewidth=1.4, capsize=3.5,
                zorder=3, label=label)
    if ylim is None:
        return 0
    lo, hi = ylim
    return int(np.nansum((med - nmad < lo) | (med + nmad > hi)))


def plot_slope_comparison(
    series: list[tuple[str, pd.DataFrame]],
    areas: pd.DataFrame,
    out_png: str | Path,
    *,
    colours: list[str] | None = None,
    title: str | None = None,
    value_label: str = "Elevation difference (m), ice-free",
    ylim: tuple[float, float] | None = (-1.5, 1.5),
    min_slope: float = 25.0,
    twin_axis: bool = False,
) -> Path:
    """Several dh series on one panel, over the shared terrain-area histograms.

    Built for the M3C2-vs-DoD comparison: the two methods are measured on the
    same pixels and dates, so putting them on one axis with one area panel
    makes the divergence with slope directly readable. Series are dodged
    slightly in x so overlapping whiskers stay distinguishable.

    Parameters
    ----------
    series : list of (label, bins DataFrame)
        Each from :func:`bin_by_slope`. Drawn in order.
    areas : pd.DataFrame
        From :func:`area_by_slope` — one shared area panel for all series.
    colours : list of str, optional
        One per series; defaults to blue/orange then matplotlib's cycle.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if not series:
        raise ValueError("need at least one series to plot")

    base = list(colours or [C_M3C2, C_DOD])
    while len(base) < len(series):
        base.append(f"C{len(base)}")
    width = float(series[0][1]["slope_hi"].iloc[0] - series[0][1]["slope_lo"].iloc[0])
    # Below min_slope the ice-free terrain is a few tens of square metres of
    # margin sliver, so its NMAD describes that sliver rather than the slope
    # class — drop those error bars. The area panel keeps every bin, since the
    # glacier's own distribution peaks well below 25 deg.
    series = [(lab, b[b["slope_lo"] >= min_slope]) for lab, b in series]
    # Spread the series symmetrically about each bin centre.
    n = len(series)
    offsets = (np.arange(n) - (n - 1) / 2) * (width * 0.30 if n > 1 else 0.0)

    if twin_axis:
        fig, ax = plt.subplots(figsize=(9, 6))
        axb = ax.twinx()
        # Bars on the right axis, behind; error bars on the left axis, in front.
        _draw_area_panel(axb, areas, width, legend_loc="upper left",
                         headroom=2.2)
        axb.set_zorder(1)
        ax.set_zorder(2)
        ax.patch.set_visible(False)
        ax.set_xlabel("Slope (degrees)")
        axes_bottom = ax
    else:
        fig, (ax, axb) = plt.subplots(
            2, 1, figsize=(9, 7.5), sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08},
        )
        _draw_area_panel(axb, areas, width)
        axb.set_xlabel("Slope (degrees)")
        axes_bottom = axb

    ax.axhline(0.0, color="0.35", ls="--", lw=1.2, zorder=1)
    n_clip = 0
    for (label, bins), colour, off in zip(series, base, offsets):
        n_clip += _draw_series(ax, bins, colour, label, offset=off, ylim=ylim)
    ax.set_ylabel(value_label)
    if ylim is not None:
        ax.set_ylim(*ylim)
        if n_clip:
            ax.text(0.015, 0.97, f"{n_clip} bin(s) clipped by the axis limits",
                    transform=ax.transAxes, fontsize=7.5, va="top", alpha=0.75)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    xs = areas["slope_mid"].to_numpy()
    for a in (ax, axb):
        a.grid(True, axis="y", alpha=0.18, lw=0.6)
        a.set_axisbelow(True)
    axes_bottom.set_xlim(-width / 2, xs.max() + width)
    axes_bottom.set_xticks(areas["slope_hi"].to_numpy()[::max(1, len(xs) // 14)])
    if title:
        fig.suptitle(title, fontsize=11, y=0.98 if not twin_axis else 0.96)
    if twin_axis:
        fig.tight_layout()

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison figure → {out_png}")
    return out_png


# ──────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────
def plot_slope_uncertainty(
    bins: pd.DataFrame,
    areas: pd.DataFrame,
    out_png: str | Path,
    *,
    title: str | None = None,
    twin_axis: bool = False,
    value_label: str = "Elevation difference (m), ice-free",
    ylim: tuple[float, float] | None = (-1.0, 1.0),
    min_slope: float = 25.0,
) -> Path:
    """Per-bin median dh with NMAD error bars, over the terrain-area histograms.

    Default is a two-panel layout sharing the slope axis — the error model on
    top, the area histograms below. ``twin_axis=True`` reproduces the compact
    single-panel published convention; the two y-scales are unrelated either
    way, so read the panels independently.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    x = areas["slope_mid"].to_numpy()
    width = float(areas["slope_hi"].iloc[0] - areas["slope_lo"].iloc[0])
    # Below min_slope the ice-free terrain is a few tens of square metres of
    # margin sliver, so its NMAD describes that sliver rather than the slope
    # class — drop those error bars. The area panel keeps every bin, since the
    # glacier's own distribution peaks well below 25 deg.
    shown = bins[bins["slope_lo"] >= min_slope]

    def _draw_error(ax):
        ax.axhline(0.0, color=C_M3C2, ls="--", lw=1.2, zorder=1)
        n_clip = _draw_series(ax, shown, C_M3C2, "Median $\\pm$ NMAD", ylim=ylim)
        ax.set_ylabel(value_label, color=C_M3C2)
        ax.tick_params(axis="y", colors=C_M3C2)
        if ylim is not None:
            ax.set_ylim(*ylim)
            # A whisker running past the fixed limits would read as a short one,
            # so say how many bins are clipped rather than silently truncating.
            if n_clip:
                ax.text(0.015, 0.97, f"{n_clip} bin(s) clipped by the axis limits",
                        transform=ax.transAxes, fontsize=7.5, va="top", alpha=0.75)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    def _draw_area(ax, *, twin=False):
        _draw_area_panel(ax, areas, width,
                         legend_loc="upper left" if twin else "upper right",
                         headroom=2.2 if twin else None)

    if twin_axis:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax2 = ax.twinx()
        _draw_area(ax2, twin=True)
        ax2.set_zorder(1)
        _draw_error(ax)
        ax.set_zorder(2)
        ax.patch.set_visible(False)
        ax.set_xlabel("Slope (degrees)")
        axes_bottom = ax
    else:
        fig, (ax, axb) = plt.subplots(
            2, 1, figsize=(9, 7.5), sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08},
        )
        _draw_error(ax)
        _draw_area(axb)
        axb.set_xlabel("Slope (degrees)")
        axes_bottom = axb

    for a in fig.get_axes():
        a.grid(True, axis="y", alpha=0.18, lw=0.6)
        a.set_axisbelow(True)
    axes_bottom.set_xlim(-width / 2, x.max() + width)
    axes_bottom.set_xticks(areas["slope_hi"].to_numpy()[::max(1, len(x) // 14)])
    if title:
        fig.suptitle(title, fontsize=11, y=0.98 if not twin_axis else 0.96)

    if twin_axis:
        fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {out_png}")
    return out_png


# ──────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────
def run_slope_uncertainty(
    output_dir: str | Path,
    *,
    glacier_mask: str | Path | None = None,
    ref_dem: str | Path | None = None,
    results_dir: str | Path | None = None,
    binning: SlopeBinning | None = None,
    slope_scale_m: float | None = None,
    project: str = "normal",
    dates: list[str] | None = None,
    max_nmad: float | None = None,
    max_nmad_from: str | None = None,
    source: str = "m3c2",
    ice_free_from: str = "m3c2",
    twin_axis: bool = False,
    ylim: tuple[float, float] | None = (-1.0, 1.0),
    title: str | None = None,
) -> dict:
    """End-to-end slope-binned ice-free uncertainty for a site.

    Writes ``slope_bins.csv``, ``slope_area.csv``, ``run_info.json`` and
    ``slope_uncertainty.png`` into *results_dir* (default
    ``<output_dir>/_uncertainty``).
    """
    output_dir = Path(output_dir)
    binning = binning or SlopeBinning()
    ref_dem = Path(ref_dem) if ref_dem else output_dir / "_ref_cache" / "reference_dem.tif"
    if not ref_dem.exists():
        raise FileNotFoundError(f"reference DEM not found: {ref_dem}")
    results_dir = Path(results_dir) if results_dir else output_dir / "_uncertainty"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[slope-uncertainty] {output_dir}")
    print(f"  Reference DEM : {ref_dem}")
    slope = reference_slope(ref_dem, scale_m=slope_scale_m)

    slopes, dh, coverage = collect_ice_free_dh(
        output_dir, slope, binning, dates=dates, project=project,
        max_nmad=max_nmad, max_nmad_from=max_nmad_from,
        source=source, glacier_mask=glacier_mask, ice_free_from=ice_free_from,
    )
    bins, filtering = bin_by_slope(slopes, dh, binning)
    areas = area_by_slope(
        slope, binning, glacier_mask=glacier_mask,
        stable_union=coverage["union_mask"],
    )
    bins = apply_area_threshold(bins, areas, binning)

    bins.to_csv(results_dir / "slope_bins.csv", index=False)
    areas.to_csv(results_dir / "slope_area.csv", index=False)
    meta = {
        "output_dir": str(output_dir),
        "ref_dem": str(ref_dem),
        "slope_scale_m": slope_scale_m,
        "projection": project,
        "bin_width_deg": binning.width,
        "max_slope_deg": binning.max_slope,
        "pct_clip": list(binning.pct_clip) if binning.pct_clip else None,
        "nmad_factor": binning.nmad_factor,
        "min_px": binning.min_px,
        "min_area_m2": binning.min_area_m2,
        "source": source,
        "ice_free_from": (ice_free_from if source == "dod" else "n/a"),
        "n_dates": coverage["n_dates"],
        "max_nmad": max_nmad,
        "max_nmad_from": max_nmad_from,
        "n_dates_gated_out": coverage["n_gated"],
        "filtering": filtering,
    }
    (results_dir / "run_info.json").write_text(json.dumps(meta, indent=2))

    if source == "dod":
        vlabel = "DoD elevation difference (m), ice-free"
    elif project == "vertical":
        vlabel = "Vertical-equivalent M3C2 difference (m), ice-free"
    else:
        vlabel = "M3C2 distance (m), ice-free"
    png = plot_slope_uncertainty(
        bins, areas, results_dir / "slope_uncertainty.png",
        title=title or f"Slope-binned ice-free uncertainty — {output_dir.parent.name}",
        twin_axis=twin_axis, value_label=vlabel, ylim=ylim,
    )

    ok = bins[~bins["sparse"]]
    print(f"  NMAD range: {ok['nmad'].min():.3f}–{ok['nmad'].max():.3f} m "
          f"across {len(ok)} usable bins")

    return {
        "bins": bins, "areas": areas, "meta": meta, "slope": slope,
        "figure": png, "results_dir": results_dir,
    }


def run_combined_slope_uncertainty(
    sites: list[dict],
    results_dir: str | Path,
    *,
    binning: SlopeBinning | None = None,
    slope_scale_m: float | None = None,
    project: str = "normal",
    source: str = "m3c2",
    ice_free_from: str = "m3c2",
    ylim: tuple[float, float] | None = (-1.0, 1.0),
    title: str | None = None,
) -> dict:
    """Same analysis, with the ice-free dh of several glaciers pooled together.

    Each site contributes dh paired against **its own** reference DEM's slope;
    the arrays are then concatenated, so the 2–98 percentile clip and the
    per-bin 3×NMAD filter act on the combined population rather than on each
    glacier separately. Area histograms are summed across sites.

    Per-site settings that differ (notably ``max_nmad``, which is applied at
    Changri North only) are honoured per site and recorded in ``run_info.json``.

    Parameters
    ----------
    sites : list of dict
        One per glacier, with keys ``name``, ``output_dir``, ``glacier_mask``
        and optionally ``ref_dem``, ``max_nmad``, ``max_nmad_from``, ``dates``.
    results_dir : path
        Where the combined CSVs, JSON and figure are written.

    Returns
    -------
    dict with ``bins``, ``areas``, ``per_site``, ``meta``, ``figure``.
    """
    binning = binning or SlopeBinning()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    slopes_all, dh_all, areas_all, per_site = [], [], [], []
    for site in sites:
        name = site.get("name") or Path(site["output_dir"]).parent.name
        out = Path(site["output_dir"])
        ref_dem = Path(site.get("ref_dem") or out / "_ref_cache" / "reference_dem.tif")
        if not ref_dem.exists():
            raise FileNotFoundError(f"[{name}] reference DEM not found: {ref_dem}")

        print(f"\n[combined] {name} — {out}")
        slope = reference_slope(ref_dem, scale_m=slope_scale_m)
        s, d, cov = collect_ice_free_dh(
            out, slope, binning,
            dates=site.get("dates"), project=project,
            max_nmad=site.get("max_nmad"), max_nmad_from=site.get("max_nmad_from"),
            source=source, glacier_mask=site.get("glacier_mask"),
            ice_free_from=ice_free_from,
        )
        slopes_all.append(s)
        dh_all.append(d)
        areas_all.append(area_by_slope(
            slope, binning, glacier_mask=site.get("glacier_mask"),
            stable_union=cov["union_mask"],
        ))
        per_site.append({
            "name": name, "output_dir": str(out), "ref_dem": str(ref_dem),
            "n_dates": cov["n_dates"], "n_dates_gated_out": cov["n_gated"],
            "max_nmad": site.get("max_nmad"),
            "max_nmad_from": site.get("max_nmad_from"),
            "n_px": int(d.size),
        })

    slopes = np.concatenate(slopes_all)
    dh = np.concatenate(dh_all)
    print(f"\n[combined] {dh.size:,} ice-free dh values from "
          f"{sum(p['n_dates'] for p in per_site)} acquisitions across "
          f"{len(per_site)} glaciers")

    bins, filtering = bin_by_slope(slopes, dh, binning)

    # Areas add: same bin edges everywhere, so summing the per-site tables gives
    # the combined terrain distribution.
    areas = areas_all[0][["slope_lo", "slope_hi", "slope_mid"]].copy()
    for col in ("glacier_area_m2", "ice_free_area_m2"):
        areas[col] = sum(a[col].to_numpy() for a in areas_all)
    bins = apply_area_threshold(bins, areas, binning)

    bins.to_csv(results_dir / "slope_bins.csv", index=False)
    areas.to_csv(results_dir / "slope_area.csv", index=False)
    for site_meta, a in zip(per_site, areas_all):
        a.to_csv(results_dir / f"slope_area_{site_meta['name']}.csv", index=False)

    meta = {
        "combined": True,
        "sites": per_site,
        "source": source,
        "ice_free_from": (ice_free_from if source == "dod" else "n/a"),
        "slope_scale_m": slope_scale_m,
        "projection": project,
        "bin_width_deg": binning.width,
        "max_slope_deg": binning.max_slope,
        "pct_clip": list(binning.pct_clip) if binning.pct_clip else None,
        "nmad_factor": binning.nmad_factor,
        "min_px": binning.min_px,
        "min_area_m2": binning.min_area_m2,
        "filtering": filtering,
    }
    (results_dir / "run_info.json").write_text(json.dumps(meta, indent=2))

    vlabel = ("DoD elevation difference (m), ice-free" if source == "dod"
              else "M3C2 distance (m), ice-free")
    png = plot_slope_uncertainty(
        bins, areas, results_dir / "slope_uncertainty.png",
        title=title or ("Combined glaciers — ice-free elevation difference by slope"),
        value_label=vlabel, ylim=ylim,
    )

    ok = bins[~bins["sparse"]]
    print(f"  NMAD range: {ok['nmad'].min():.3f}–{ok['nmad'].max():.3f} m "
          f"across {len(ok)} usable bins")

    return {"bins": bins, "areas": areas, "per_site": per_site,
            "meta": meta, "figure": png, "results_dir": results_dir}


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def _main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Slope-binned uncertainty from ice-free M3C2 rasters: "
                    "2-98 percentile clip, 5 deg slope bins, 3xNMAD in-bin filter.",
    )
    p.add_argument("--output-dir", required=True, help="site output/ directory")
    p.add_argument("--glacier-mask", default=None, help="glacier outline shapefile")
    p.add_argument("--ref-dem", default=None, help="override reference DEM path")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--bin-width", type=float, default=5.0)
    p.add_argument("--max-slope", type=float, default=90.0)
    p.add_argument("--pct-clip", type=float, nargs=2, metavar=("LO", "HI"),
                   default=[2.0, 98.0],
                   help="percentile clip on ice-free dh before binning; "
                        "pass '--pct-clip 0 100' to disable")
    p.add_argument("--nmad-factor", type=float, default=3.0,
                   help="in-bin outlier filter, in NMADs from the bin median; "
                        "0 disables")
    p.add_argument("--min-px", type=int, default=500,
                   help="min surviving pixel-observations before a bin is solid")
    p.add_argument("--min-area", type=float, default=1000.0, metavar="M2",
                   help="min ice-free ground area (m2) before a bin is solid; "
                        "the binding test, since pooled pixel counts are "
                        "inflated by the number of acquisitions")
    p.add_argument("--slope-scale", type=float, default=None,
                   help="resample DEM to this pixel size before slope (m)")
    p.add_argument("--project", choices=["normal", "vertical"], default="normal")
    p.add_argument("--source", choices=["m3c2", "dod"], default="m3c2",
                   help="m3c2 = along-normal stable rasters; dod = vertical DEM "
                        "difference (day_dem - reference_dem)")
    p.add_argument("--ice-free-from", choices=["m3c2", "mask"], default="m3c2",
                   help="for --source dod: 'm3c2' restricts to the same pixels the "
                        "stable M3C2 raster covers (like-for-like); 'mask' uses "
                        "everything outside the glacier outline")
    p.add_argument("--dates", nargs="*", default=None)
    p.add_argument("--max-nmad", type=float, default=None,
                   help="drop acquisitions whose post-coreg stable NMAD is >= this "
                        "(m); matches the accuracy-figure gate. Off by default")
    p.add_argument("--max-nmad-from", default=None, metavar="YYYY-MM-DD",
                   help="apply --max-nmad only on/after this date")
    p.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"),
                   default=[-1.0, 1.0],
                   help="dh axis limits (m); pass '--ylim 0 0' to autoscale")
    p.add_argument("--twin-axis", action="store_true",
                   help="single-panel published layout instead of two panels")
    p.add_argument("--title", default=None, help="figure title")
    a = p.parse_args(argv)

    run_slope_uncertainty(
        a.output_dir,
        glacier_mask=a.glacier_mask,
        ref_dem=a.ref_dem,
        results_dir=a.results_dir,
        binning=SlopeBinning(
            a.bin_width, a.max_slope,
            pct_clip=(None if tuple(a.pct_clip) == (0.0, 100.0) else tuple(a.pct_clip)),
            nmad_factor=(None if a.nmad_factor == 0 else a.nmad_factor),
            min_px=a.min_px,
            min_area_m2=a.min_area,
        ),
        slope_scale_m=a.slope_scale,
        project=a.project,
        dates=a.dates,
        max_nmad=a.max_nmad,
        max_nmad_from=a.max_nmad_from,
        source=a.source,
        ice_free_from=a.ice_free_from,
        twin_axis=a.twin_axis,
        ylim=(None if tuple(a.ylim) == (0.0, 0.0) else tuple(a.ylim)),
        title=a.title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

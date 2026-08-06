"""Slope-binned uncertainty of ice-free elevation differences.

The Step-3b ``<date>_stable_m3c2.tif`` rasters are post-co-registration M3C2
residuals on **ice-free terrain**, where the true change is zero, so their
dispersion measures the co-registration + reconstruction error. The same
analysis runs on ``<date>_DOD.tif`` (vertical DEM differencing) to compare the
two differencing geometries on identical pixels and dates.

Procedure
---------
1. Extract elevation offsets (dh) over ice-free areas.
2. Filter outliers using the **2-98 percentiles** of the data.
3. Bin the dh values in **5 degree slope intervals**.
4. Filter remaining outliers with a **3xNMAD** filter within each slope bin.

Each bin reports its median (bias) and NMAD (random error) — the dot and the
half-length of the error bar in the figure.

Notes on interpretation
-----------------------
* M3C2 measures distance **along the local surface normal**, a DoD measures it
  **vertically**. On a slope the vertical gap between two surfaces is the true
  separation divided by ``cos(slope)``, so DoD error grows as ``1/cos(slope)``
  while M3C2 error does not. That is the whole of the difference between them.
* The 2-98 clip is global across all slopes, so it removes proportionally more
  from whichever bins carry the largest residuals. The in-bin 3xNMAD filter is
  the slope-relative one.
* The NMAD is computed after its own >=3-NMAD tail was removed, so it is biased
  slightly low by construction. Report it as a *filtered* NMAD.
* These are precision estimates for a single pixel, not the uncertainty on a
  spatially averaged quantity, which also needs the spatial correlation.
* Pooling many acquisitions inflates pixel counts by the number of dates: a bin
  covering a few hundred square metres can still show tens of thousands of
  pixels. ``ice_free_area_m2`` in the output is the honest measure of how much
  ground a bin rests on.

Not part of the ``cntp`` package yet — this lives under ``contributors/``
until the approach settles. It imports nothing from ``cntp``, so promoting it
later is a plain move.

Typical use
-----------
>>> import sys; sys.path.insert(0, "contributors/umayr")
>>> from uncertainty import run
>>> res = run(
...     dict(name="Changri_North",
...          output_dir="/mnt/e/umayr/Changri/Changri_North/output",
...          glacier_mask="/mnt/e/umayr/Changri/Changri_North/shapefile/Shapefile_ChangriNorth.shp",
...          max_nmad=0.5),
...     results_dir="/mnt/e/umayr/Changri/Changri_North/output/_uncertainty",
... )
>>> res["bins"].head()

Pass a *list* of site dicts to pool several glaciers into one analysis. From
the shell::

    python contributors/umayr/uncertainty.py \
        --output-dir .../Changri_North/output \
        --glacier-mask .../Shapefile_ChangriNorth.shp --max-nmad 0.5
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import geoutils as gu  # noqa: E402
import xdem  # noqa: E402

__all__ = [
    "reference_slope",
    "coreg_nmad",
    "collect_ice_free_dh",
    "bin_by_slope",
    "area_by_slope",
    "apply_area_threshold",
    "plot_slope_bins",
    "run",
]

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# 90 deg is the geometric maximum; this only guards a malformed slope raster.
SLOPE_CEILING = 90.0

C_GLACIER, C_ICE_FREE = "#e8635a", "#b9d7e8"
# Blue/orange: the most reliably separable pair under colour-vision deficiency.
SERIES_COLOURS = ["#1f4ed8", "#d95f02"]


# ──────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────
def _nmad(x: np.ndarray) -> float:
    """Normalised median absolute deviation — robust sigma."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _date_of(path: Path) -> str:
    m = _DATE_RE.search(path.name)
    return m.group(1) if m else path.stem


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


def _edges(width: float, max_slope: float) -> np.ndarray:
    return np.arange(0.0, max_slope + width / 2, width)


def coreg_nmad(output_dir: str | Path, date: str) -> float:
    """That acquisition's post-co-registration stable NMAD, or NaN.

    Reads the ``after`` row of ``<output>/<date>/coreg/<date>_m3c2_stats.csv``,
    the number the pipeline already reported at Step 3b — the same source
    ``cntp.postprocessing.load_coreg_nmad`` uses, so the acquisition gate here
    matches the one behind the accuracy figures.
    """
    f = Path(output_dir) / date / "coreg" / f"{date}_m3c2_stats.csv"
    if not f.exists():
        return float("nan")
    try:
        return float(pd.read_csv(f, index_col="coreg").loc["after", "nmad"])
    except Exception:
        return float("nan")


# ──────────────────────────────────────────────────────────────────────────
# Step 1 — slope from the fixed reference DEM
# ──────────────────────────────────────────────────────────────────────────
def reference_slope(ref_dem: str | Path) -> gu.Raster:
    """Slope (degrees, Horn) of the cached reference DEM.

    Taken from the reference rather than each day's DEM, so the bins mean the
    same thing on every date and the predictor stays independent of the noise
    being measured — slope from a noisy epoch DEM would correlate with the
    error and inflate the apparent slope dependence.
    """
    return xdem.terrain.slope(xdem.DEM(str(ref_dem)))


# ──────────────────────────────────────────────────────────────────────────
# Step 2 — extract ice-free dh, paired with slope
# ──────────────────────────────────────────────────────────────────────────
def collect_ice_free_dh(
    output_dir: str | Path,
    slope: gu.Raster,
    *,
    source: str = "m3c2",
    dates: list[str] | None = None,
    max_nmad: float | None = None,
    max_slope: float = SLOPE_CEILING,
) -> dict:
    """Pool every ice-free elevation offset with its reference slope.

    ``source="m3c2"`` reads ``<date>_stable_m3c2.tif``, already restricted to
    ice-free corepoints. ``source="dod"`` reads ``<date>_DOD.tif`` (vertical,
    whole footprint) and keeps only the pixels its own stable M3C2 raster
    covers, so the two sources describe identical terrain on identical dates.

    ``max_nmad`` drops whole acquisitions whose post-coreg stable NMAD is
    ``>=`` that value (failed co-registrations), decided before any raster is
    opened.

    Returns ``{"slopes", "dh", "union", "n_dates", "n_gated"}`` — the pooled
    float32 arrays, plus which ice-free pixels were sampled at least once, on
    the reference-DEM grid.
    """
    output_dir = Path(output_dir)
    if source not in ("m3c2", "dod"):
        raise ValueError(f"source must be 'm3c2' or 'dod', got {source!r}")

    pattern = "*_stable_m3c2.tif" if source == "m3c2" else "*_DOD.tif"
    files = sorted(output_dir.glob(f"*/single_day/{pattern}"))
    if dates is not None:
        wanted = set(dates)
        files = [f for f in files if _date_of(f) in wanted]
    if source == "dod":
        # Only dates that also have a stable M3C2 raster to mask against.
        twins = {_date_of(p) for p in output_dir.glob("*/single_day/*_stable_m3c2.tif")}
        files = [f for f in files if _date_of(f) in twins]
    if not files:
        raise FileNotFoundError(f"no */single_day/{pattern} under {output_dir}")
    print(f"  Found {len(files)} {'ice-free M3C2' if source == 'm3c2' else 'DoD'} rasters")

    n_gated = 0
    if max_nmad is not None:
        kept = [f for f in files if coreg_nmad(output_dir, _date_of(f)) < max_nmad]
        n_gated = len(files) - len(kept)
        print(f"  Acquisition gate: dropped {n_gated} date(s) with post-coreg "
              f"NMAD >= {max_nmad:g} m — {len(kept)} remain")
        if not kept:
            raise ValueError(f"no acquisition passes the max_nmad={max_nmad} gate")
        files = kept

    ceiling = min(max_slope, SLOPE_CEILING)
    slope_cache: dict[tuple, np.ndarray] = {}
    union = np.zeros(_as_nan_array(slope).shape, dtype=bool)
    slopes_all: list[np.ndarray] = []
    dh_all: list[np.ndarray] = []
    n_used = 0

    for i, fp in enumerate(files, 1):
        r = gu.Raster(str(fp))
        key = (r.shape, tuple(np.round(np.asarray(r.transform)[:6], 3)))
        if key not in slope_cache:
            slope_cache[key] = _as_nan_array(
                slope.reproject(ref=r, resampling="bilinear", silent=True))
        s = slope_cache[key]
        d = _as_nan_array(r)

        if source == "dod":
            twin = fp.with_name(f"{_date_of(fp)}_stable_m3c2.tif")
            sm = gu.Raster(str(twin))
            sm.data = np.where(np.isfinite(_as_nan_array(sm)), 1.0,
                               np.nan).astype("float32")[None, :, :]
            sm.set_nodata(np.nan)
            ice_free = np.isfinite(_as_nan_array(
                sm.reproject(ref=r, resampling="nearest", silent=True)))
            d = np.where(ice_free, d, np.nan)

        ok = np.isfinite(s) & np.isfinite(d) & (s < ceiling)
        if not ok.any():
            print(f"  [warn] {fp.name}: no valid slope/residual pixels — skipped")
            continue

        # Fold this date's footprint into the reference-grid union coverage.
        cov = r.copy()
        cov.data = np.where(np.isfinite(d), 1.0, np.nan).astype("float32")[None, :, :]
        cov.set_nodata(np.nan)
        cov_ref = _as_nan_array(cov.reproject(ref=slope, resampling="nearest", silent=True))
        union |= np.isfinite(cov_ref) & (cov_ref > 0)

        slopes_all.append(s[ok].astype("float32"))
        dh_all.append(d[ok].astype("float32"))
        n_used += 1
        if i % 50 == 0 or i == len(files):
            print(f"    {i}/{len(files)} rasters read")

    dh = np.concatenate(dh_all)
    print(f"  Pooled {dh.size:,} ice-free dh values from {n_used} dates "
          f"({len(slope_cache)} distinct grids)")
    return {"slopes": np.concatenate(slopes_all), "dh": dh, "union": union,
            "n_dates": n_used, "n_gated": n_gated}


# ──────────────────────────────────────────────────────────────────────────
# Steps 2-4 — percentile clip, bin, in-bin NMAD filter
# ──────────────────────────────────────────────────────────────────────────
def bin_by_slope(
    slopes: np.ndarray,
    dh: np.ndarray,
    *,
    width: float = 5.0,
    max_slope: float = SLOPE_CEILING,
    pct_clip: tuple[float, float] | None = (2.0, 98.0),
    nmad_factor: float | None = 3.0,
) -> tuple[pd.DataFrame, dict]:
    """Clip, bin in slope intervals, filter again, and summarise.

    The percentile clip applies to the whole pooled array; the ``nmad_factor``
    filter applies within each bin, because what counts as a blunder depends on
    the dispersion of its own slope class. Statistics are recomputed after.

    Returns the per-bin table and the pixel counts removed at each stage.
    """
    n_raw = int(dh.size)
    if pct_clip is not None:
        lo, hi = np.percentile(dh, pct_clip)
        keep = (dh >= lo) & (dh <= hi)
        slopes, dh = slopes[keep], dh[keep]
    n_after_clip = int(dh.size)

    edges = _edges(width, max_slope)
    idx = np.digitize(slopes, edges) - 1
    rows, n_nmad_removed = [], 0
    for b in range(len(edges) - 1):
        v = dh[idx == b]
        n_before = int(v.size)
        if nmad_factor is not None and n_before:
            m0, s0 = np.median(v), _nmad(v)
            if np.isfinite(s0) and s0 > 0:
                v = v[np.abs(v - m0) <= nmad_factor * s0]
        n_nmad_removed += n_before - int(v.size)
        n = int(v.size)
        rows.append({
            "slope_lo": float(edges[b]),
            "slope_hi": float(edges[b + 1]),
            "slope_mid": float((edges[b] + edges[b + 1]) / 2),
            "n_px": n,
            "median": float(np.median(v)) if n else np.nan,
            "nmad": _nmad(v) if n else np.nan,
            "std": float(np.std(v)) if n else np.nan,
        })

    bins = pd.DataFrame(rows)
    n_kept = int(bins["n_px"].sum())
    if pct_clip:
        print(f"  Outlier filtering: {n_raw - n_after_clip:,} px removed by the "
              f"{pct_clip[0]:g}-{pct_clip[1]:g} percentile clip")
    if nmad_factor is not None:
        print(f"                     {n_nmad_removed:,} px removed by the "
              f"{nmad_factor:g}xNMAD per-bin filter")
    print(f"                     {n_kept:,} of {n_raw:,} ice-free px retained "
          f"({100 * n_kept / max(n_raw, 1):.1f}%)")

    return bins, {"n_px_raw": n_raw,
                  "n_px_removed_pct_clip": n_raw - n_after_clip,
                  "n_px_removed_nmad": n_nmad_removed,
                  "n_px_kept": n_kept}


# ──────────────────────────────────────────────────────────────────────────
# Terrain area per bin, and the data-quality flags
# ──────────────────────────────────────────────────────────────────────────
def area_by_slope(
    slope: gu.Raster,
    *,
    glacier_mask: str | Path | None = None,
    union: np.ndarray | None = None,
    width: float = 5.0,
    max_slope: float = SLOPE_CEILING,
) -> pd.DataFrame:
    """Glacier and ice-free terrain area per slope bin, in m^2.

    ``glacier`` is reference-DEM area inside the outline; ``ice_free`` is the
    area actually sampled (``union`` from :func:`collect_ice_free_dh`), or
    everything outside the outline if no union is given. Comparing the two
    shows whether the terrain the error is measured on resembles the terrain it
    will be applied to — at these sites it does not.
    """
    s = _as_nan_array(slope)
    px_area = float(abs(slope.res[0] * slope.res[1]))
    valid = np.isfinite(s)

    if glacier_mask is not None:
        vect = gu.Vector(str(glacier_mask))
        if vect.crs != slope.crs:
            vect = vect.reproject(crs=slope.crs)
        g = vect.create_mask(slope).data
        g = np.squeeze(np.asarray(
            g.filled(False) if isinstance(g, np.ma.MaskedArray) else g, dtype=bool))
    else:
        g = np.zeros(s.shape, dtype=bool)
    ice_free = np.squeeze(np.asarray(union if union is not None else ~g, dtype=bool))

    edges = _edges(width, max_slope)
    idx = np.digitize(s, edges) - 1
    return pd.DataFrame([{
        "slope_lo": float(edges[b]),
        "slope_hi": float(edges[b + 1]),
        "slope_mid": float((edges[b] + edges[b + 1]) / 2),
        "glacier_area_m2": float(np.sum(valid & (idx == b) & g) * px_area),
        "ice_free_area_m2": float(np.sum(valid & (idx == b) & ice_free & ~g) * px_area),
    } for b in range(len(edges) - 1)])


def apply_area_threshold(
    bins: pd.DataFrame,
    areas: pd.DataFrame,
    *,
    min_px: int = 500,
    min_area_m2: float = 1000.0,
) -> pd.DataFrame:
    """Attach ice-free ground area to *bins* and flag thin ones.

    Recorded for reference, not rendered — ``min_slope`` decides what the
    figure shows. ``few_px`` is fewer than *min_px* pixel-observations;
    ``low_area`` is less than *min_area_m2* of ice-free ground, which is the
    one that actually catches thin bins, since pooled pixel counts are inflated
    by the number of acquisitions. ``sparse`` is either.
    """
    out = bins.copy()
    out["ice_free_area_m2"] = out["slope_lo"].map(
        areas.set_index("slope_lo")["ice_free_area_m2"]).to_numpy()
    out["px_per_m2"] = (out["n_px"] /
                        out["ice_free_area_m2"].replace(0, np.nan)).round(1)
    out["few_px"] = out["n_px"] < min_px
    out["low_area"] = out["ice_free_area_m2"] < min_area_m2
    out["sparse"] = out["few_px"] | out["low_area"]
    return out


# ──────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────
def plot_slope_bins(
    series: list[tuple[str, pd.DataFrame]],
    areas: pd.DataFrame,
    out_png: str | Path,
    *,
    title: str | None = None,
    value_label: str = "Elevation difference (m), ice-free",
    ylim: tuple[float, float] | None = (-1.0, 1.0),
    min_slope: float = 25.0,
    area_scale: float = 10.0,
    twin_axis: bool = True,
) -> Path:
    """Median ± NMAD per slope bin, over the terrain-area histograms.

    One or more series (M3C2, DoD, …) share one area panel; with more than one
    they are dodged slightly in x so overlapping whiskers stay readable.

    ``twin_axis=True`` puts everything on one frame with area on a right-hand
    axis — the published convention. ``False`` splits it into two stacked
    panels sharing the slope axis, which avoids the two unrelated y-scales.

    Error bars start at *min_slope*: below that the ice-free terrain is a few
    tens of square metres of margin sliver, so its NMAD describes that sliver
    rather than the slope class. The area panel always spans every bin, since
    the glacier's own distribution peaks well below it.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if not series:
        raise ValueError("need at least one series to plot")

    width = float(areas["slope_hi"].iloc[0] - areas["slope_lo"].iloc[0])
    shown = [(lab, b[b["slope_lo"] >= min_slope]) for lab, b in series]
    n = len(shown)
    offsets = (np.arange(n) - (n - 1) / 2) * (width * 0.30 if n > 1 else 0.0)
    colours = (SERIES_COLOURS + [f"C{i}" for i in range(n)])[:n]

    if twin_axis:
        fig, ax = plt.subplots(figsize=(9, 6))
        axb = ax.twinx()
        axb.set_zorder(1)
        ax.set_zorder(2)
        ax.patch.set_visible(False)
        ax.set_xlabel("Slope (degrees)")
        bottom, legend_loc, headroom = ax, "upper left", 2.2
    else:
        fig, (ax, axb) = plt.subplots(
            2, 1, figsize=(9, 7.5), sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08})
        axb.set_xlabel("Slope (degrees)")
        bottom, legend_loc, headroom = axb, "upper right", None

    # Area bars, behind the error bars on the shared layout.
    x = areas["slope_mid"].to_numpy()
    g_m2, s_m2 = areas["glacier_area_m2"].to_numpy(), areas["ice_free_area_m2"].to_numpy()
    use_km2 = max(g_m2.max(), s_m2.max()) >= 5e5
    conv, unit = (1e-6, "km$^2$") if use_km2 else (1e-3, "10$^3$ m$^2$")
    g_a, s_a = g_m2 * conv, s_m2 * conv * area_scale
    axb.bar(x - width / 6, g_a, width=width / 2.6, color=C_GLACIER,
            label="Glacier area", zorder=2)
    axb.bar(x + width / 6, s_a, width=width / 2.6, color=C_ICE_FREE,
            label="Ice-free area" + (f" ×{area_scale:g}" if area_scale != 1 else ""),
            zorder=2)
    axb.set_ylabel(f"Area ({unit})")
    if headroom is not None:
        axb.set_ylim(0, max(g_a.max(), s_a.max()) * headroom)
    axb.legend(loc=legend_loc, fontsize=8, framealpha=0.9)

    # One series owns the axis, so the zero line matches it; with several a
    # neutral grey avoids implying the line belongs to one of them.
    ax.axhline(0.0, color=colours[0] if n == 1 else "0.35", ls="--", lw=1.2, zorder=1)
    n_clip = 0
    for (label, b), colour, off in zip(shown, colours, offsets):
        med, nmad = b["median"].to_numpy(), b["nmad"].to_numpy()
        ax.errorbar(b["slope_mid"].to_numpy() + off, med, yerr=nmad, fmt="o", ms=5,
                    color=colour, ecolor=colour, elinewidth=1.4, capsize=3.5,
                    zorder=3, label=label)
        if ylim is not None:
            n_clip += int(np.nansum((med - nmad < ylim[0]) | (med + nmad > ylim[1])))
    # With one series the left axis belongs to it, so colour them together —
    # the published convention. With several, a neutral label is unambiguous.
    ax.set_ylabel(value_label, color=colours[0] if n == 1 else "black")
    if n == 1:
        ax.tick_params(axis="y", colors=colours[0])
    if ylim is not None:
        ax.set_ylim(*ylim)
        # A whisker past the limits would read as a short one, so say so.
        if n_clip:
            ax.text(0.015, 0.97, f"{n_clip} bin(s) clipped by the axis limits",
                    transform=ax.transAxes, fontsize=7.5, va="top", alpha=0.75)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    for a in (ax, axb):
        a.grid(True, axis="y", alpha=0.18, lw=0.6)
        a.set_axisbelow(True)
    bottom.set_xlim(-width / 2, x.max() + width)
    bottom.set_xticks(areas["slope_hi"].to_numpy()[::max(1, len(x) // 14)])
    if title:
        fig.suptitle(title, fontsize=11, y=0.96 if twin_axis else 0.98)
    if twin_axis:
        fig.tight_layout()

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure -> {out_png}")
    return out_png


# ──────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────
def run(
    sites: dict | list[dict],
    results_dir: str | Path,
    *,
    source: str = "m3c2",
    dates: list[str] | None = None,
    width: float = 5.0,
    max_slope: float = SLOPE_CEILING,
    pct_clip: tuple[float, float] | None = (2.0, 98.0),
    nmad_factor: float | None = 3.0,
    min_px: int = 500,
    min_area_m2: float = 1000.0,
    min_slope: float = 25.0,
    ylim: tuple[float, float] | None = (-1.0, 1.0),
    twin_axis: bool = True,
    title: str | None = None,
) -> dict:
    """End-to-end analysis for one glacier, or several pooled together.

    Each site contributes dh paired against **its own** reference DEM's slope;
    the arrays are then concatenated, so the percentile clip and the per-bin
    filter act on the combined population. Area histograms are summed. Per-site
    settings (notably ``max_nmad``) are honoured individually and recorded.

    Parameters
    ----------
    sites : dict or list of dict
        Keys: ``name``, ``output_dir``, ``glacier_mask``, and optionally
        ``ref_dem``, ``max_nmad``.
    results_dir : path
        Where ``slope_bins.csv``, ``slope_area.csv``, ``run_info.json`` and
        ``slope_uncertainty.png`` are written.
    """
    if isinstance(sites, dict):
        sites = [sites]
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    slopes_all, dh_all, areas_all, per_site = [], [], [], []
    for site in sites:
        out = Path(site["output_dir"])
        name = site.get("name") or out.parent.name
        ref_dem = Path(site.get("ref_dem") or out / "_ref_cache" / "reference_dem.tif")
        if not ref_dem.exists():
            raise FileNotFoundError(f"[{name}] reference DEM not found: {ref_dem}")

        print(f"\n[{name}] {out}")
        slope = reference_slope(ref_dem)
        got = collect_ice_free_dh(out, slope, source=source, dates=dates,
                                  max_nmad=site.get("max_nmad"), max_slope=max_slope)
        slopes_all.append(got["slopes"])
        dh_all.append(got["dh"])
        areas_all.append(area_by_slope(slope, glacier_mask=site.get("glacier_mask"),
                                       union=got["union"], width=width,
                                       max_slope=max_slope))
        per_site.append({"name": name, "output_dir": str(out), "ref_dem": str(ref_dem),
                         "n_dates": got["n_dates"], "n_dates_gated_out": got["n_gated"],
                         "max_nmad": site.get("max_nmad"), "n_px": int(got["dh"].size)})

    slopes, dh = np.concatenate(slopes_all), np.concatenate(dh_all)
    if len(per_site) > 1:
        print(f"\n[combined] {dh.size:,} ice-free dh values from "
              f"{sum(p['n_dates'] for p in per_site)} acquisitions across "
              f"{len(per_site)} glaciers")

    bins, filtering = bin_by_slope(slopes, dh, width=width, max_slope=max_slope,
                                   pct_clip=pct_clip, nmad_factor=nmad_factor)
    # Same bin edges everywhere, so summing the per-site tables combines them.
    areas = areas_all[0][["slope_lo", "slope_hi", "slope_mid"]].copy()
    for col in ("glacier_area_m2", "ice_free_area_m2"):
        areas[col] = sum(a[col].to_numpy() for a in areas_all)
    bins = apply_area_threshold(bins, areas, min_px=min_px, min_area_m2=min_area_m2)

    bins.to_csv(results_dir / "slope_bins.csv", index=False)
    areas.to_csv(results_dir / "slope_area.csv", index=False)
    if len(per_site) > 1:
        for meta, a in zip(per_site, areas_all):
            a.to_csv(results_dir / f"slope_area_{meta['name']}.csv", index=False)

    info = {"sites": per_site, "source": source, "bin_width_deg": width,
            "max_slope_deg": max_slope,
            "pct_clip": list(pct_clip) if pct_clip else None,
            "nmad_factor": nmad_factor, "min_px": min_px,
            "min_area_m2": min_area_m2, "min_slope_reported": min_slope,
            "filtering": filtering}
    (results_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    label = ("DoD elevation difference (m), ice-free" if source == "dod"
             else "M3C2 distance (m), ice-free")
    png = plot_slope_bins([("Median $\\pm$ NMAD", bins)], areas,
                          results_dir / "slope_uncertainty.png", title=title,
                          value_label=label, ylim=ylim, min_slope=min_slope,
                          twin_axis=twin_axis)

    ok = bins[bins["slope_lo"] >= min_slope]
    print(f"  NMAD range: {ok['nmad'].min():.3f}-{ok['nmad'].max():.3f} m "
          f"over {len(ok)} reported bins")
    return {"bins": bins, "areas": areas, "per_site": per_site,
            "info": info, "figure": png, "results_dir": results_dir}


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def _main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Slope-binned uncertainty of ice-free elevation differences: "
                    "2-98 percentile clip, 5 deg slope bins, 3xNMAD in-bin filter.")
    p.add_argument("--output-dir", required=True, help="site output/ directory")
    p.add_argument("--glacier-mask", default=None, help="glacier outline shapefile")
    p.add_argument("--ref-dem", default=None, help="override reference DEM path")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--source", choices=["m3c2", "dod"], default="m3c2")
    p.add_argument("--dates", nargs="*", default=None)
    p.add_argument("--max-nmad", type=float, default=None,
                   help="drop acquisitions with post-coreg stable NMAD >= this (m)")
    p.add_argument("--bin-width", type=float, default=5.0)
    p.add_argument("--max-slope", type=float, default=SLOPE_CEILING)
    p.add_argument("--pct-clip", type=float, nargs=2, metavar=("LO", "HI"),
                   default=[2.0, 98.0], help="'--pct-clip 0 100' disables")
    p.add_argument("--nmad-factor", type=float, default=3.0, help="0 disables")
    p.add_argument("--min-px", type=int, default=500)
    p.add_argument("--min-area", type=float, default=1000.0, metavar="M2")
    p.add_argument("--min-slope", type=float, default=25.0,
                   help="lowest slope bin to report")
    p.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"),
                   default=[-1.0, 1.0], help="'--ylim 0 0' autoscales")
    p.add_argument("--two-panel", action="store_true",
                   help="stacked panels instead of the shared twin axis")
    p.add_argument("--title", default=None)
    a = p.parse_args(argv)

    run(dict(output_dir=a.output_dir, glacier_mask=a.glacier_mask,
             ref_dem=a.ref_dem, max_nmad=a.max_nmad),
        a.results_dir or Path(a.output_dir) / "_uncertainty",
        source=a.source, dates=a.dates, width=a.bin_width, max_slope=a.max_slope,
        pct_clip=(None if tuple(a.pct_clip) == (0.0, 100.0) else tuple(a.pct_clip)),
        nmad_factor=(None if a.nmad_factor == 0 else a.nmad_factor),
        min_px=a.min_px, min_area_m2=a.min_area, min_slope=a.min_slope,
        ylim=(None if tuple(a.ylim) == (0.0, 0.0) else tuple(a.ylim)),
        twin_axis=not a.two_panel, title=a.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

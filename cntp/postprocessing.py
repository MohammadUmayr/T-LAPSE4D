"""Post-hoc analysis of finished 4D SfM outputs.

Helpers that read products the pipeline already wrote (per-date co-registration
distances, M3C2 signal rasters, the cached stable reference) and assemble
derived figures / summaries. This is distinct from :mod:`cntp.pipeline_4dsfm`
(which *produces* the outputs) and :mod:`cntp.plot` (which *draws*): this module
*loads from disk and orchestrates*, calling the plot primitives.

Intended for notebook / post-run use over a date or a whole season.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from cntp.io import load_las
from cntp.plot import (
    plot_m3c2_coreg_and_signal,
    plot_maps_row,
    plot_relative_accuracy_boxplot,
    plot_absolute_accuracy_boxes,
    data_window,
    _robust_vmax,
)


def _read_raster(path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Read a single-band raster → (array with NaN nodata, (xmin, xmax, ymin, ymax))."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        b = src.bounds
    return arr, (b.left, b.right, b.bottom, b.top)


def coreg_and_signal_figure(
    date: str,
    output_dir,
    *,
    res: float = 1.0,
    vmax: float = 4.0,
    plot_dir=None,
    show: bool = False,
    save_pdf: bool = True,
) -> None:
    """Build the before/after-coreg + signal three-panel figure for one date.

    Loads the products the pipeline already wrote and calls
    :func:`cntp.plot.plot_m3c2_coreg_and_signal`:

    - coreg before/after stable-terrain M3C2 distances from
      ``output/<date>/coreg/<date>_m3c2_distances.npz``;
    - corepoint coords from the cached stable reference
      ``output/_ref_cache/*_stable.las`` (auto-discovered — there is one per
      glacier; same point order as the distances, so coords and distances line
      up, since both come from that frozen file loaded at full resolution);
    - the reference→day signal raster
      ``output/<date>/single_day/<date>_M3C2_raster.tif``.

    Parameters
    ----------
    date : str
        ``YYYY-MM-DD``.
    output_dir : str | Path
        Pipeline root (the parent of ``output/``), i.e. ``site.output_dir``.
    res, vmax :
        Forwarded to the plot. ``vmax`` is the shared ±limit (m) for all panels
        (default 4.0 = fixed scale across dates; ``None`` = per-figure auto).
    plot_dir : str | Path, optional
        Where to save (default ``output/<date>/coreg/m3c2_plots``).
    show : bool
        If True, display interactively instead of saving.
    save_pdf : bool
        Also write a vector PDF alongside the PNG (default True).
    """
    output_dir = Path(output_dir)
    date_dir = output_dir / "output" / date

    npz_path = date_dir / "coreg" / f"{date}_m3c2_distances.npz"
    signal_path = date_dir / "single_day" / f"{date}_M3C2_raster.tif"

    # The cached stable reference (one per glacier) supplies the corepoint
    # coords; its filename encodes the downsample factor, so just discover it.
    cache_dir = output_dir / "output" / "_ref_cache"
    stable_refs = sorted(cache_dir.glob("*_stable.las"))
    if not stable_refs:
        raise FileNotFoundError(f"no cached stable reference in {cache_dir} (expected *_stable.las)")
    if len(stable_refs) > 1:
        raise RuntimeError(
            f"multiple stable references in {cache_dir}: "
            f"{[p.name for p in stable_refs]} — remove the stale one(s)."
        )
    stable_ref = stable_refs[0]

    for p in (npz_path, signal_path):
        if not p.exists():
            raise FileNotFoundError(f"{date}: missing required input → {p}")

    d = np.load(npz_path)
    dist_before, dist_after = d["before"], d["after"]
    ref_stable = load_las(stable_ref)              # corepoint coords (same order)
    signal, signal_extent = _read_raster(signal_path)

    if ref_stable.shape[0] != dist_before.shape[0]:
        raise ValueError(
            f"{date}: corepoint count ({ref_stable.shape[0]}) != distance count "
            f"({dist_before.shape[0]}). The cached stable reference "
            f"({stable_ref.name}) was likely rebuilt since this date was "
            f"processed — re-run coreg for this date."
        )

    plot_m3c2_coreg_and_signal(
        ref_stable, dist_before, dist_after,
        signal, signal_extent,
        output_dir=None if show else (plot_dir or date_dir / "coreg" / "m3c2_plots"),
        title=date,
        filename=f"{date}_coreg_and_signal.png",
        res=res,
        vmax=vmax,
        save_pdf=save_pdf,
    )


# ---------------------------------------------------------------------------
# Signal-raster stack (per-date M3C2 rasters -> one cube)
# ---------------------------------------------------------------------------

def _is_iso_date(s: str) -> bool:
    import re
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", s))


@dataclass
class SignalStack:
    """A time stack of co-registered M3C2 signal rasters on one shared grid."""
    dates: list
    times: object
    cube: object
    transform: object
    crs: object
    paths: list

    def __len__(self):
        return len(self.dates)

    def raster(self, i):
        """Re-wrap slice *i* as a geoutils Raster (for interp_points sampling)."""
        import geoutils as gu
        return gu.Raster.from_array(
            np.ma.masked_invalid(self.cube[i]), self.transform, self.crs,
            nodata=-99999.0,
        )

    @property
    def extent(self):
        """(xmin, xmax, ymin, ymax) for matplotlib imshow of a slice."""
        H, W = self.cube.shape[1:]
        a = self.transform
        xmin, ymax = a.c, a.f
        return (xmin, xmin + a.a * W, ymax + a.e * H, ymax)


def load_signal_stack(output_dir, *, kind="M3C2_raster", date_from=None,
                      date_to=None, dtype="float32", cache=True, cache_dir=None):
    """Stack every ``output/<date>/single_day/<date>_<kind>.tif`` into a cube.

    Discovers the per-date rasters, checks all share one grid, and loads them
    into a ``(T, H, W)`` float32 cube with NaN nodata. ISO date strings compare
    chronologically, so ``date_from`` / ``date_to`` (inclusive) subset the record.

    Reading many GeoTIFFs off a slow mount takes minutes, so by default the
    assembled cube is cached to local disk (``cache_dir``, default
    ``~/.cache/cntp_signalstack``) as ``.npy`` + metadata and memory-mapped back
    on later calls; the cache auto-invalidates when the discovered date list
    changes. Pass ``cache=False`` to bypass. Returns a :class:`SignalStack`.
    """
    from affine import Affine

    base = Path(output_dir) / "output"
    pairs = []
    for p in sorted(base.glob(f"*/single_day/*_{kind}.tif")):
        date = p.parent.parent.name
        if not _is_iso_date(date):
            continue
        if date_from and date < date_from:
            continue
        if date_to and date > date_to:
            continue
        pairs.append((date, p))
    if not pairs:
        raise FileNotFoundError(
            f"no '*_{kind}.tif' under {base}/*/single_day/ "
            f"(date_from={date_from}, date_to={date_to})"
        )
    pairs.sort()
    want_dates = [d for d, _ in pairs]

    cdir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "cntp_signalstack"
    tag = f"{Path(output_dir).name}__{kind}__{date_from or 'all'}__{date_to or 'all'}"
    cube_path = cdir / f"{tag}.cube.npy"
    meta_path = cdir / f"{tag}.meta.npz"
    if cache and cube_path.exists() and meta_path.exists():
        meta = np.load(meta_path, allow_pickle=True)
        if list(meta["dates"]) == want_dates:
            cube = np.load(cube_path, mmap_mode="r")
            transform = Affine(*[float(v) for v in meta["transform"]])
            crs = rasterio.crs.CRS.from_epsg(int(meta["crs_epsg"]))
            times = np.array(want_dates, dtype="datetime64[D]")
            print(f"  Loaded {len(want_dates)} {kind} rasters from cache "
                  f"({cube.shape}, {cube.nbytes / 1e9:.2f} GB) -> {cube_path}")
            return SignalStack(want_dates, times, cube, transform, crs,
                               [p for _, p in pairs])

    with rasterio.open(pairs[0][1]) as src0:
        H, W = src0.height, src0.width
        transform, crs = src0.transform, src0.crs
    cube = np.empty((len(pairs), H, W), dtype=dtype)
    dates, paths = [], []
    for i, (date, p) in enumerate(pairs):
        with rasterio.open(p) as src:
            if (src.height, src.width) != (H, W):
                raise ValueError(
                    f"grid mismatch: {p.name} is {(src.height, src.width)}, "
                    f"expected {(H, W)}. All M3C2 rasters must share the "
                    f"corepoint grid — was the _ref_cache stable reference "
                    f"rebuilt mid-record?"
                )
            arr = src.read(1).astype(dtype)
            nod = src.nodata
        if nod is not None and not np.isnan(nod):
            arr = np.where(arr == nod, np.nan, arr)
        cube[i] = arr
        dates.append(date)
        paths.append(p)

    times = np.array(dates, dtype="datetime64[D]")
    print(f"  Loaded {len(dates)} {kind} rasters -> cube {cube.shape} "
          f"({cube.nbytes / 1e9:.2f} GB), {dates[0]} ... {dates[-1]}")

    if cache:
        cdir.mkdir(parents=True, exist_ok=True)
        np.save(cube_path, cube)
        np.savez(meta_path, dates=np.array(dates),
                 transform=np.array([transform.a, transform.b, transform.c,
                                     transform.d, transform.e, transform.f]),
                 crs_epsg=crs.to_epsg() or 0)
        print(f"  Cached cube -> {cube_path}")
    return SignalStack(dates, times, cube, transform, crs, paths)


# ---------------------------------------------------------------------------
# Per-pixel relative accuracy — compute + orchestrate
# ---------------------------------------------------------------------------
# The M3C2 stack is a valid precision estimate because every date is
# differenced against the same reference cloud: the reference elevation is a
# per-pixel constant in time, so it shifts the temporal mean but cancels out of
# the temporal SD/NMAD. The stable statistics stack the per-date .npz distances
# in corepoint space (1:1 aligned to the fixed reference corepoints), avoiding
# the per-date cropping of the stable-M3C2 rasters.

def per_pixel_obs_count(stack):
    """Per-pixel number of finite observations across the stack -> (H, W) int32."""
    return np.isfinite(stack.cube).sum(axis=0).astype(np.int32)


def _nanmad(mat):
    """Temporal (axis-0) NMAD and median of *mat*, NaN-aware -> (nmad, median).

    Works for a ``(T, H, W)`` cube or a ``(T, N)`` corepoint matrix.
    """
    with np.errstate(all="ignore"):
        med = np.nanmedian(mat, axis=0)
        nmad = 1.4826 * np.nanmedian(np.abs(mat - med), axis=0)
    return nmad, med


def per_pixel_nmad_map(stack, *, min_obs=3):
    """Whole-footprint per-pixel NMAD + observation count from the M3C2 stack.

    Returns ``{count, nmad, median, valid}`` (each (H, W)); ``nmad``/``median``
    are NaN where a pixel has ``< min_obs`` finite dates. The median map is the
    per-pixel temporal bias (carries the reference error), not precision.
    """
    count = per_pixel_obs_count(stack)
    nmad, med = _nanmad(stack.cube)
    valid = count >= min_obs
    return {"count": count, "nmad": np.where(valid, nmad, np.nan),
            "median": np.where(valid, med, np.nan), "valid": valid}


def _discover_stable_ref(output_dir):
    """Return the cached stable-terrain reference LAS path, or None."""
    cands = sorted((Path(output_dir) / "output" / "_ref_cache").glob("*_stable.las"))
    return cands[0] if cands else None


def load_stable_distance_stack(output_dir, *, which="after", dates=None,
                               date_from=None, date_to=None):
    """Stack per-date stable-terrain M3C2 distances from the coreg ``.npz`` files.

    Discovers ``output/<date>/coreg/<date>_m3c2_distances.npz`` and stacks the
    ``which`` array (``"after"`` = post-coregistration; ``"before"`` also
    available) into a ``(T, N)`` matrix. Every array is 1:1 aligned to the same
    cached reference corepoints, so they stack in corepoint space with no
    regridding. NaN marks a corepoint with no valid M3C2 that date. Returns
    ``(dates, times, matrix)``. A length mismatch (cached stable reference
    rebuilt mid-record) is raised explicitly.
    """
    base = Path(output_dir) / "output"
    pairs = []
    for p in sorted(base.glob("*/coreg/*_m3c2_distances.npz")):
        date = p.parent.parent.name
        if not _is_iso_date(date):
            continue
        if dates is not None and date not in dates:
            continue
        if date_from and date < date_from:
            continue
        if date_to and date > date_to:
            continue
        pairs.append((date, p))
    if not pairs:
        raise FileNotFoundError(
            f"no '*_m3c2_distances.npz' under {base}/*/coreg/ "
            f"(date_from={date_from}, date_to={date_to})"
        )
    pairs.sort()

    rows, ds, N = [], [], None
    for date, p in pairs:
        with np.load(p) as z:
            a = np.asarray(z[which], dtype="float32")
        if N is None:
            N = a.size
        elif a.size != N:
            raise ValueError(
                f"{p.name} has {a.size} corepoints, expected {N}. The cached "
                f"_ref_cache stable reference changed mid-record, so the stable "
                f"distances no longer share a corepoint set — re-run the affected "
                f"dates against one reference, or date-bound to a consistent span."
            )
        rows.append(a)
        ds.append(date)
    matrix = np.vstack(rows)
    times = np.array(ds, dtype="datetime64[D]")
    print(f"  Loaded {len(ds)} stable-M3C2 '{which}' distance arrays -> "
          f"matrix {matrix.shape} ({ds[0]} ... {ds[-1]})")
    return ds, times, matrix


def stable_precision_arrays(output_dir, *, which="after", min_obs=3,
                            date_from=None, date_to=None):
    """Per-corepoint stable-terrain SD and NMAD arrays (relative-accuracy inputs).

    Returns ``(sd, nmad)`` over corepoints with ``>= min_obs`` finite obs (others
    NaN). Stacks the ``.npz`` stable distances in corepoint space (no regrid).
    """
    _, _, mat = load_stable_distance_stack(
        output_dir, which=which, date_from=date_from, date_to=date_to)
    count = np.isfinite(mat).sum(axis=0).astype(np.int32)
    valid = count >= min_obs
    nmad, _ = _nanmad(mat)
    with np.errstate(all="ignore"):
        sd = np.nanstd(mat, axis=0)
    return np.where(valid, sd, np.nan), np.where(valid, nmad, np.nan)


def pixel_relative_accuracy(output_dir, *, kind="M3C2_raster", date_from=None,
                            date_to=None, min_obs=3, which="after", plot_dir=None,
                            site_label=None, count_cmap="magma", nmad_cmap="viridis",
                            nmad_vmax=None, nmad_vmax_pct=98.0, scalebar_m="auto",
                            box_ylim=None, show=False, save_pdf=True):
    """Relative accuracy from the M3C2 signal (coverage + NMAD maps + boxplot).

    Writes ``pixel_maps.png`` (coverage + whole-footprint per-pixel NMAD maps,
    side by side on one shared window; NMAD gated at ``min_obs``, ``vmax``
    auto-scaled to the ``nmad_vmax_pct``-th percentile unless ``nmad_vmax`` is
    given) and ``stable_relative_accuracy.png`` (SD/NMAD boxplot over
    stable terrain, from the ``.npz`` corepoint stack, same ``min_obs``) to
    *plot_dir* (default ``output/postprocessing/precision/``). Returns the maps
    + stable summary dict.
    """
    stack = load_signal_stack(output_dir, kind=kind,
                              date_from=date_from, date_to=date_to)
    out = None if show else Path(
        plot_dir or Path(output_dir) / "output" / "postprocessing" / "precision")
    suffix = f" — {site_label}" if site_label else ""

    count = per_pixel_obs_count(stack).astype("float64")
    count[count == 0] = float("nan")
    window = data_window(count, stack.extent)
    maps = per_pixel_nmad_map(stack, min_obs=min_obs)
    vmax = nmad_vmax if nmad_vmax is not None else _robust_vmax(
        maps["nmad"], pct=nmad_vmax_pct)
    plot_maps_row(
        [{"values": count, "cmap": count_cmap, "vmin": 0, "clabel": "Count",
          "title": f"M3C2 raster count{suffix}"},
         {"values": maps["nmad"], "cmap": nmad_cmap, "vmin": 0, "vmax": vmax,
          "extend": "max", "clabel": "NMAD (m)",
          "title": f"Per-pixel M3C2 raster NMAD{suffix}"}],
        stack.extent, window=window, scalebar_m=scalebar_m, output_dir=out,
        filename="pixel_maps.png", save_pdf=save_pdf)

    sd, nmad = stable_precision_arrays(output_dir, which=which, min_obs=min_obs,
                                       date_from=date_from, date_to=date_to)
    stable = plot_relative_accuracy_boxplot(
        sd, nmad, ylim=box_ylim, output_dir=out,
        title=f"Relative accuracy{suffix} (n>={min_obs})", save_pdf=save_pdf)
    print(f"\n  [{site_label or 'site'}] stable-terrain relative accuracy "
          f"({stable['nmad']['n']:,} corepoints, n>={min_obs}, '{which}'):")
    print(f"    NMAD — mean {stable['nmad']['mean']:.3f} m  "
          f"median {stable['nmad']['median']:.3f} m")
    print(f"    SD   — mean {stable['sd']['mean']:.3f} m  "
          f"median {stable['sd']['median']:.3f} m")

    return {"stack": stack, "count": maps["count"], "nmad": maps["nmad"],
            "nmad_vmax": vmax, "valid": maps["valid"], "stable_stats": stable}


def absolute_accuracy_boxplots(output_dir, *, which="after", bin_days=14,
                               select="max_area", date_from=None, date_to=None,
                               stable_ref_las=None, res=1.0, ylim=None,
                               cmap="Blues", median_color="red", site_label=None,
                               plot_dir=None, filename="stable_absolute_accuracy.png",
                               show=False, save_pdf=True):
    """Per-DEM stable-terrain absolute accuracy over time.

    Each box is one DEM's stable-terrain elevation difference vs the reference
    (post-coreg M3C2 ``which`` residuals), unpooled. To thin a daily record, one
    representative DEM is drawn per ``bin_days``-day window (``select='max_area'``
    = best-covered acquisition, ``select='nearest'`` = closest to window centre);
    ``bin_days=None`` draws every acquisition. Boxes sit at their real dates and
    are coloured by that DEM's stable surface area (km^2 from the corepoint
    coords; falls back to corepoint count when the cached stable reference LAS is
    missing). Returns the per-box records (date, area, median, iqr).
    """
    import pandas as pd

    _, times, mat = load_stable_distance_stack(
        output_dir, which=which, date_from=date_from, date_to=date_to)

    coords = None
    stable_ref_las = stable_ref_las or _discover_stable_ref(output_dir)
    if stable_ref_las is not None and Path(stable_ref_las).exists():
        xy = load_las(Path(stable_ref_las))[:, :2]
        if xy.shape[0] == mat.shape[1]:
            coords = xy
        else:
            print(f"  [warn] stable ref has {xy.shape[0]} pts but stack has "
                  f"{mat.shape[1]} corepoints — colouring by count, not km^2.")
    area_is_km2 = coords is not None
    if coords is not None:
        cell = (np.floor(coords[:, 0] / res).astype(np.int64),
                np.floor(coords[:, 1] / res).astype(np.int64))

    def _area(occupied):
        if coords is None:
            return int(occupied.sum())
        ids = cell[0][occupied] * (10 ** 9) + cell[1][occupied]
        return np.unique(ids).size * (res * res) / 1e6

    idx = pd.to_datetime(times.astype("datetime64[D]"))
    if bin_days:
        origin = idx.min().normalize()
        key = ((idx - origin).days // bin_days).to_numpy()
    else:
        key = np.arange(len(idx))
    finite_per_date = np.isfinite(mat).sum(axis=1)

    records = []
    for b in np.unique(key):
        rows = np.where(key == b)[0]
        if finite_per_date[rows].max() == 0:
            continue
        if bin_days and select == "nearest":
            c = origin + pd.Timedelta(days=(b + 0.5) * bin_days)
            r_sel = rows[np.argmin(np.abs((idx[rows] - c).days.to_numpy()))]
        else:
            r_sel = rows[np.argmax(finite_per_date[rows])]
        vals = mat[r_sel]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        occupied = np.isfinite(mat[r_sel])
        records.append({
            "date": idx[r_sel], "area": _area(occupied), "vals": vals,
            "median": float(np.median(vals)),
            "iqr": float(np.subtract(*np.percentile(vals, [75, 25]))),
        })
    if not records:
        raise ValueError("no non-empty windows to plot")

    out = None if show else Path(
        plot_dir or Path(output_dir) / "output" / "postprocessing" / "precision")
    plot_absolute_accuracy_boxes(
        records, area_is_km2=area_is_km2, bin_days=bin_days, site_label=site_label,
        cmap=cmap, median_color=median_color, ylim=ylim, output_dir=out,
        filename=filename, save_pdf=save_pdf)
    for r in records:
        r.pop("vals", None)
    return records

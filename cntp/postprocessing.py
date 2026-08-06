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


def load_stable_grid_stack(output_dir, *, which="after", res=1.0,
                           stable_ref_las=None, date_from=None, date_to=None):
    """Per-date stable-terrain M3C2 distances binned onto a ``res``-metre grid.

    Loads the point-space distances (:func:`load_stable_distance_stack`), places
    each corepoint on a ``res``-metre grid using the cached stable reference's
    coordinates, and takes the **median of the corepoints in each cell, per
    date**. Returns ``(times, grid)`` where ``grid`` is ``(T, n_cells)`` float32
    with NaN for cells that date did not observe.

    Both accuracy figures reduce this one matrix — the absolute accuracy across
    cells (per row, one box per acquisition), the relative accuracy along time
    (per column, SD/NMAD per cell) — so they share a unit and are comparable.

    Gridding is what makes the statistics reportable: corepoint density varies
    from one to several hundred points per square metre, so reducing in point
    space would weight the result by point density rather than by area (flattering
    it, since dense patches are the well-observed low-noise ones) and would not
    share its unit with the per-pixel M3C2 maps. On the grid every square metre of
    stable ground counts once, and the numbers describe the precision of the
    gridded M3C2 product itself.
    """
    import pandas as pd

    _, times, mat = load_stable_distance_stack(
        output_dir, which=which, date_from=date_from, date_to=date_to)

    stable_ref_las = stable_ref_las or _discover_stable_ref(output_dir)
    if stable_ref_las is None:
        raise FileNotFoundError(
            f"no cached stable reference (*_stable.las) under "
            f"{Path(output_dir) / 'output' / '_ref_cache'} — needed for the "
            f"corepoint coordinates that place each distance on the grid."
        )
    xy = load_las(Path(stable_ref_las))[:, :2]
    if xy.shape[0] != mat.shape[1]:
        raise ValueError(
            f"stable reference has {xy.shape[0]} points but the distance stack "
            f"has {mat.shape[1]} corepoints — the cached stable reference was "
            f"rebuilt mid-record; re-run the affected dates against one reference."
        )

    # The corepoints are fixed across dates, so the cell index is computed once.
    cells = np.stack([np.floor(xy[:, 0] / res).astype(np.int64),
                      np.floor(xy[:, 1] / res).astype(np.int64)], axis=1)
    _, inv = np.unique(cells, axis=0, return_inverse=True)
    n_cells = int(inv.max()) + 1

    grid = np.full((mat.shape[0], n_cells), np.nan, dtype="float32")
    for t in range(mat.shape[0]):
        v = mat[t]
        m = np.isfinite(v)
        if not m.any():
            continue
        s = pd.Series(v[m]).groupby(inv[m]).median()
        grid[t, s.index.to_numpy()] = s.to_numpy()

    print(f"  Gridded stable distances to {res:g} m -> {grid.shape[0]} dates "
          f"x {n_cells:,} cells ({n_cells * res * res:,.0f} m^2 stable)")
    return times, grid


def _nmad_keep_mask(times, date_nmad, max_nmad, max_nmad_from):
    """Keep-mask over acquisitions: drop those with ``NMAD >= max_nmad``.

    When ``max_nmad_from`` (``YYYY-MM-DD``) is given the threshold applies only
    on/after that date — acquisitions before it are always kept. Use that when a
    known event degrades part of the record (cameras lost, a cloudy season) and
    the earlier, unaffected period should be reported as-is. ``max_nmad=None``
    keeps everything.
    """
    import pandas as pd
    keep = np.ones(len(times), dtype=bool)
    if max_nmad is None:
        return keep
    t = pd.to_datetime(np.asarray(times, dtype="datetime64[D]"))
    in_window = (np.ones(len(t), dtype=bool) if max_nmad_from is None
                 else np.asarray(t >= pd.Timestamp(max_nmad_from), dtype=bool))
    bad = in_window & ~(date_nmad < max_nmad)      # NaN NMAD -> dropped in window
    return keep & ~bad


def per_acquisition_nmad(grid):
    """Per-acquisition post-coreg stable NMAD from a gridded stable stack."""
    with np.errstate(all="ignore"):
        med = np.nanmedian(grid, axis=1)
        return 1.4826 * np.nanmedian(np.abs(grid - med[:, None]), axis=1)


def stable_precision_arrays(output_dir, *, which="after", min_obs=3, res=1.0,
                            max_nmad=None, max_nmad_from=None,
                            stable_ref_las=None, date_from=None, date_to=None):
    """Per-pixel stable-terrain SD and NMAD on a ``res``-metre grid.

    Reduces the gridded stable stack (:func:`load_stable_grid_stack`) along time:
    the per-cell temporal SD and NMAD over cells with ``>= min_obs`` valid dates.
    Returns ``(sd, nmad)``, NaN where a cell has too few observations. These are
    the relative-accuracy (precision) inputs — same 1 m unit as the M3C2 maps and
    as the absolute-accuracy boxes.

    ``max_nmad`` / ``max_nmad_from`` drop poorly co-registered acquisitions before
    the reduction (see :func:`_nmad_keep_mask`); both default to off.
    """
    times, grid = load_stable_grid_stack(
        output_dir, which=which, res=res, stable_ref_las=stable_ref_las,
        date_from=date_from, date_to=date_to)
    if max_nmad is not None:
        dates = [str(d) for d in np.asarray(times, dtype="datetime64[D]")]
        keep = _nmad_keep_mask(times, load_coreg_nmad(output_dir, dates),
                               max_nmad, max_nmad_from)
        print(f"  NMAD gate (>= {max_nmad} m"
              f"{f', from {max_nmad_from}' if max_nmad_from else ''}): "
              f"kept {int(keep.sum())}/{keep.size} acquisitions")
        grid = grid[keep]
    count = np.isfinite(grid).sum(axis=0).astype(np.int32)
    valid = count >= min_obs
    nmad, _ = _nanmad(grid)
    with np.errstate(all="ignore"):
        sd = np.nanstd(grid, axis=0)
    return np.where(valid, sd, np.nan), np.where(valid, nmad, np.nan)


def load_coreg_nmad(output_dir, dates):
    """Post-coreg stable-terrain NMAD per acquisition, from the pipeline's stats.

    Reads the ``after`` row of ``output/<date>/coreg/<date>_m3c2_stats.csv`` — the
    number coregistration already reported — so nothing has to be re-loaded or
    re-computed. Returns a float array aligned with *dates*, NaN where the stats
    file is missing.

    This is the **gate**: one value per acquisition, describing that day's
    coregistration quality. It is not the relative accuracy, which is a per-pixel
    temporal statistic computed over the acquisitions this gate retains.
    """
    import pandas as pd
    base = Path(output_dir) / "output"
    out = np.full(len(dates), np.nan)
    for i, d in enumerate(dates):
        f = base / str(d) / "coreg" / f"{d}_m3c2_stats.csv"
        if f.exists():
            try:
                out[i] = float(pd.read_csv(f, index_col="coreg").loc["after", "nmad"])
            except Exception:
                pass
    return out


def _reference_ortho_panel(output_dir, stack, footprint, *, title=None):
    """RGBA ortho panel regridded to the stack grid and clipped to *footprint*.

    Reads ``output/_ref_cache/reference_ortho.tif``, resamples it onto the M3C2
    map grid (same CRS), and sets alpha=0 outside the boolean *footprint* mask so
    the ortho shows only the data shape shared with the other panels. Returns a
    panel dict for :func:`cntp.plot.plot_maps_row`, or None if the ortho is absent.
    """
    ref_ortho = Path(output_dir) / "output" / "_ref_cache" / "reference_ortho.tif"
    if not ref_ortho.exists():
        print(f"  [warn] no reference ortho at {ref_ortho} — skipping ortho panel")
        return None
    from rasterio.warp import reproject, Resampling
    H, W = stack.cube.shape[1:]
    with rasterio.open(ref_ortho) as src:
        src_arr = src.read()
        src_transform, src_crs = src.transform, src.crs
    dst = np.zeros((src_arr.shape[0], H, W), dtype=src_arr.dtype)
    reproject(source=src_arr, destination=dst,
              src_transform=src_transform, src_crs=src_crs,
              dst_transform=stack.transform, dst_crs=stack.crs,
              resampling=Resampling.bilinear)
    img = np.transpose(dst, (1, 2, 0))            # (H, W, bands)
    rgb = img[..., :3]
    alpha = img[..., 3] if img.shape[2] >= 4 else np.full((H, W), 255, np.uint8)
    alpha = np.where(np.asarray(footprint, bool), alpha, 0).astype("uint8")
    return {"values": np.dstack([rgb, alpha]), "rgb": True, "title": title}


def pixel_relative_accuracy(output_dir, *, kind="M3C2_raster", date_from=None,
                            date_to=None, min_obs=3, which="after", res=1.0,
                            max_nmad=None, max_nmad_from=None,
                            plot_dir=None, site_label=None, ortho=True,
                            panel_size=6.0, count_cmap="magma",
                            nmad_cmap="viridis", nmad_vmax=None, nmad_vmax_pct=98.0,
                            scalebar_m="auto", box_ylim=None, show=False,
                            save_pdf=True):
    """Relative accuracy from the M3C2 signal (ortho + coverage + NMAD + boxplot).

    Writes ``pixel_maps.png`` — the reference ortho (clipped to the coverage
    footprint), the coverage map, and the whole-footprint per-pixel NMAD map, side
    by side on one shared window (NMAD gated at ``min_obs``, ``vmax`` auto-scaled
    to the ``nmad_vmax_pct``-th percentile unless ``nmad_vmax`` is given) — and
    ``stable_relative_accuracy.png`` (SD/NMAD boxplot over stable terrain, same
    ``min_obs``) to *plot_dir* (default ``output/postprocessing/precision/``). Set
    ``ortho=False`` to drop the ortho panel. Figures are always written to
    *plot_dir*; set ``show=True`` to also display them inline. Returns the maps +
    stable summary dict.

    ``panel_size`` is the height (inches) each map panel is built from — raise it
    (e.g. 9-10) to render the rasters larger for visual inspection.

    ``max_nmad`` / ``max_nmad_from`` drop poorly co-registered acquisitions, using
    the per-date NMAD the pipeline reported (:func:`load_coreg_nmad`). The gate is
    applied to the coverage map, the NMAD map and the relative accuracy alike, so
    all three describe the same retained acquisitions and ``min_obs`` counts only
    days that were actually used. ``max_nmad_from`` restricts the gate to a date
    onward — for a record degraded partway through (cameras lost, a cloudy
    season), leaving the earlier, unaffected period reported as-is.
    """
    stack = load_signal_stack(output_dir, kind=kind,
                              date_from=date_from, date_to=date_to)
    if max_nmad is not None:
        keep = _nmad_keep_mask(stack.times, load_coreg_nmad(output_dir, stack.dates),
                               max_nmad, max_nmad_from)
        print(f"  NMAD gate (>= {max_nmad} m"
              f"{f', from {max_nmad_from}' if max_nmad_from else ''}): "
              f"kept {int(keep.sum())}/{keep.size} acquisitions for the maps")
        stack = SignalStack(
            [d for d, k in zip(stack.dates, keep) if k], stack.times[keep],
            stack.cube[keep], stack.transform, stack.crs,
            [p for p, k in zip(stack.paths, keep) if k])
    out = Path(plot_dir or Path(output_dir) / "output" / "postprocessing" / "precision")
    suffix = f" — {site_label}" if site_label else ""

    count = per_pixel_obs_count(stack).astype("float64")
    count[count == 0] = float("nan")
    window = data_window(count, stack.extent)
    maps = per_pixel_nmad_map(stack, min_obs=min_obs)
    vmax = nmad_vmax if nmad_vmax is not None else _robust_vmax(
        maps["nmad"], pct=nmad_vmax_pct)
    panels = []
    if ortho:
        op = _reference_ortho_panel(output_dir, stack, np.isfinite(count),
                                    title=f"UAV + TLC orthomosaic{suffix}")
        if op is not None:
            panels.append(op)
    panels += [
        {"values": count, "cmap": count_cmap, "vmin": 0, "clabel": "Count",
         "title": f"M3C2 raster count{suffix}"},
        {"values": maps["nmad"], "cmap": nmad_cmap, "vmin": 0, "vmax": vmax,
         "extend": "max", "clabel": "NMAD (m)",
         "title": f"Per-pixel M3C2 raster NMAD{suffix}"},
    ]
    plot_maps_row(panels, stack.extent, window=window, scalebar_m=scalebar_m,
                  panel_size=panel_size, output_dir=out, filename="pixel_maps.png",
                  save_pdf=save_pdf, show=show)

    sd, nmad = stable_precision_arrays(output_dir, which=which, min_obs=min_obs,
                                       res=res, max_nmad=max_nmad,
                                       max_nmad_from=max_nmad_from,
                                       date_from=date_from, date_to=date_to)
    stable = plot_relative_accuracy_boxplot(
        sd, nmad, ylim=box_ylim, output_dir=out,
        title=f"Relative accuracy{suffix} (n>={min_obs})", save_pdf=save_pdf,
        show=show)
    print(f"\n  [{site_label or 'site'}] stable-terrain relative accuracy "
          f"({stable['nmad']['n']:,} stable {res:g} m pixels, n>={min_obs}, '{which}'):")
    print(f"    NMAD — mean {stable['nmad']['mean']:.3f} m  "
          f"median {stable['nmad']['median']:.3f} m")
    print(f"    SD   — mean {stable['sd']['mean']:.3f} m  "
          f"median {stable['sd']['median']:.3f} m")

    return {"stack": stack, "count": maps["count"], "nmad": maps["nmad"],
            "nmad_vmax": vmax, "valid": maps["valid"], "stable_stats": stable}


def absolute_accuracy_boxplots(output_dir, *, which="after", bin_days=14,
                               max_nmad=0.5, max_nmad_from=None,
                               date_from=None, date_to=None,
                               stable_ref_las=None, res=1.0, ylim=None,
                               cmap="Blues", median_color="red", site_label=None,
                               plot_dir=None, filename="stable_absolute_accuracy.png",
                               show=False, save_pdf=True):
    """Absolute accuracy of each acquisition over time, on stable terrain.

    Each box is one acquisition's distribution of stable-terrain M3C2 ``which``
    distances against the reference cloud (``"after"`` = post-coregistration),
    binned onto the ``res``-metre grid (:func:`load_stable_grid_stack`) so one
    sample = one square metre of stable ground. A box centred on 0 means
    coregistration left no bias; the IQR/whiskers are that acquisition's
    precision. The relative-accuracy figure reduces the same gridded matrix along
    time, so the two are directly comparable.

    Acquisitions whose post-coreg stable NMAD is ``>= max_nmad`` (default 0.5 m)
    are excluded first — these are failed coregistrations whose metre-scale bias
    and spread would otherwise dominate the plot. Set ``max_nmad=None`` to keep
    every acquisition.

    Boxes are then drawn on a regular ``bin_days``-day cadence: each slot takes
    the acquisition **nearest** to that slot's date, in either direction, so gaps
    in the record are filled by the closest acquisition. If two are equidistant,
    the one with the **lower post-coreg stable NMAD** wins. ``bin_days=None``
    instead draws every (surviving) acquisition at its own date.

    Boxes are coloured by that acquisition's stable surface area (m^2 = the number
    of grid cells it observed). The figure is always written to *plot_dir*; set
    ``show=True`` to also display it inline. Returns the per-box records
    (``date`` = slot date, ``source_date`` = acquisition used, ``offset_days``,
    ``area``, ``median``, ``iqr``).
    """
    import pandas as pd

    times, grid = load_stable_grid_stack(
        output_dir, which=which, res=res, stable_ref_las=stable_ref_las,
        date_from=date_from, date_to=date_to)

    idx = pd.to_datetime(times.astype("datetime64[D]"))
    cells_per_date = np.isfinite(grid).sum(axis=1)        # occupied 1 m cells

    # Per-acquisition post-coreg stable NMAD, on the same grid: gates out failed
    # coregistrations, and breaks ties when two acquisitions are equally close.
    with np.errstate(all="ignore"):
        med = np.nanmedian(grid, axis=1)
        date_nmad = 1.4826 * np.nanmedian(np.abs(grid - med[:, None]), axis=1)

    ok = cells_per_date > 0
    if max_nmad is not None:
        dates = [str(d) for d in np.asarray(times, dtype="datetime64[D]")]
        gated = ok & _nmad_keep_mask(times, load_coreg_nmad(output_dir, dates),
                                     max_nmad, max_nmad_from)
        n_drop = int(ok.sum() - gated.sum())
        if gated.any():
            if n_drop:
                print(f"  Excluded {n_drop} acquisition(s): post-coreg NMAD "
                      f">= {max_nmad} m"
                      f"{f' (from {max_nmad_from})' if max_nmad_from else ''}")
            ok = gated
        else:
            print(f"  [warn] no acquisition passes the NMAD gate — gate ignored")
    usable = np.where(ok)[0]
    if usable.size == 0:
        raise ValueError("no acquisitions with valid stable distances")

    def _record(r_sel, slot_date, offset_days):
        vals = grid[r_sel]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        return {
            "date": slot_date, "source_date": idx[r_sel],
            "offset_days": int(offset_days),
            "area": float(vals.size * res * res),         # m^2 of stable ground
            "vals": vals,
            "median": float(np.median(vals)),
            "iqr": float(np.subtract(*np.percentile(vals, [75, 25]))),
        }

    records = []
    if bin_days:
        slots = pd.date_range(idx.min().normalize(), idx.max(), freq=f"{bin_days}D")
        for t in slots:
            d = np.abs((idx[usable] - t).days.to_numpy())
            near = usable[d == d.min()]                   # closest, either side
            r_sel = near[np.argmin(date_nmad[near])]      # tie -> lowest NMAD
            rec = _record(r_sel, t, (idx[r_sel] - t).days)
            if rec is not None:
                records.append(rec)
    else:
        for r_sel in usable:
            rec = _record(r_sel, idx[r_sel], 0)
            if rec is not None:
                records.append(rec)
    if not records:
        raise ValueError("no boxes to plot")

    out = Path(plot_dir or Path(output_dir) / "output" / "postprocessing" / "precision")
    plot_absolute_accuracy_boxes(
        records, area_is_m2=True, bin_days=bin_days, site_label=site_label,
        cmap=cmap, median_color=median_color, ylim=ylim, output_dir=out,
        filename=filename, save_pdf=save_pdf, show=show)
    for r in records:
        r.pop("vals", None)
    return records

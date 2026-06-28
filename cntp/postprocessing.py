"""Post-hoc analysis of finished 4D SfM outputs.

Helpers that read products the pipeline already wrote (per-date co-registration
distances, M3C2 signal rasters, the cached stable reference) and assemble
derived figures / summaries. This is distinct from :mod:`cntp.pipeline_4dsfm`
(which *produces* the outputs) and :mod:`cntp.plot` (which *draws*): this module
*loads from disk and orchestrates*, calling the plot primitives.

Intended for notebook / post-run use over a date or a whole season.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from cntp.io import load_las
from cntp.plot import plot_m3c2_coreg_and_signal


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

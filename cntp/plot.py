from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


def _plot_if_missing(overwrite: bool, path: Path, fn):
    """Call fn() only if overwrite=True or the file does not exist yet."""
    if overwrite or not path.exists():
        fn()
    else:
        tqdm.write(f"Skipping plot (already exists): {path}")


def plot_stable_terrain_geometry(stable_points, output_dir, filename="stable_terrain_geometry.png"):
    """3D scatter plot of geometrically extracted stable terrain."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(stable_points[:, 0], stable_points[:, 1], stable_points[:, 2], marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=30)
    plt.title('Geometrically extracted stable terrain')
    plt.savefig(Path(output_dir) / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ndwi_vs_intensity(ndwi, grayscale_intensity, colors, line_a, line_b, output_dir, filename="ndwi_vs_intensity.png"):
    """2D scatter plot of NDWI vs grayscale intensity with separation line.

    The separation line y = line_a * x + line_b is computed over the NDWI
    range of the data being plotted, so it always spans the visible scatter.
    """
    line_x_at_ymax = (255 - line_b) / line_a
    line_x_at_ymin = (0   - line_b) / line_a
    x_min = min(float(np.nanmin(ndwi)), line_x_at_ymax, line_x_at_ymin)
    x_max = max(float(np.nanmax(ndwi)), line_x_at_ymax, line_x_at_ymin)
    x_values = np.linspace(x_min, x_max, 100)
    y_values = line_a * x_values + line_b

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(ndwi, grayscale_intensity, c=colors / 255, marker='.')
    ax.plot(x_values, y_values, color='red')
    ax.set_xlabel('NDWI')
    ax.set_ylabel('INTENSITY')
    ax.set_ylim(0, 255)
    plt.savefig(Path(output_dir) / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_stable_terrain_rgb(stable_points, output_dir, title='Stable terrain',
                            filename="stable_terrain_rgb.png", elev=15, azim=-90):
    """3D scatter plot of stable terrain colored by RGB.

    ``elev``/``azim`` set the 3-D view. The default (elev=15, azim=-90) is a
    near-frontal view — looking along Northing so Easting × Elevation faces the
    viewer — so the terrain always appears from the front regardless of site.
    """
    from matplotlib.ticker import MaxNLocator

    # Subtract a local origin so the Easting/Northing ticks are short offsets
    # (0..extent) instead of full 6-7 digit UTM values — the long northings
    # otherwise clutter the tilted 3-D axis. The label records the origin.
    x0 = float(np.floor(np.nanmin(stable_points[:, 0])))
    y0 = float(np.floor(np.nanmin(stable_points[:, 1])))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(stable_points[:, 0] - x0, stable_points[:, 1] - y0, stable_points[:, 2],
               c=stable_points[:, 3:6] / 255, marker='.')
    # Axes show metres relative to the SW origin (short ticks); the origin
    # itself is noted once in the corner so the axis labels stay short and
    # don't run off the tilted 3-D frame.
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_zlabel('Elevation (m)')

    # De-clutter: few ticks per axis + small font.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=7)

    plt.title(title)
    fig.text(0.01, 0.01, f"origin: E {x0:,.0f}  N {y0:,.0f} (m)",
             fontsize=7, color='0.4')
    ax.view_init(elev=elev, azim=azim)
    plt.savefig(Path(output_dir) / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_m3c2_distances(dist_before: np.ndarray, dist_after: np.ndarray,
                        output_dir=None, title: str = '',
                        filename: str = "m3c2_distances.png",
                        xlim: float = 4.0) -> None:
    """Histogram of M3C2 distances before and after co-registration.

    Before is drawn green (left stat block), after blue (right stat block);
    each block reports med/nmad/std computed on the full un-clipped distance
    array. The x-axis is fixed to ±``xlim`` m with constant 0.1 m bins so the
    range and bin width don't drift day-to-day — figures stack cleanly across a
    time series. Distances beyond ±``xlim`` are dropped from the bars only.
    """
    d_before = dist_before[~np.isnan(dist_before)]
    d_after  = dist_after[~np.isnan(dist_after)]

    med_before  = float(np.median(d_before))
    med_after   = float(np.median(d_after))
    nmad_before = float(1.4826 * np.median(np.abs(d_before - med_before)))
    nmad_after  = float(1.4826 * np.median(np.abs(d_after  - med_after)))
    std_before  = float(np.std(d_before))
    std_after   = float(np.std(d_after))

    clip  = xlim                              # fixed window → same x-axis every day
    edges = np.arange(-clip, clip + 1e-9, 0.1)   # constant 0.1 m bins
    d_before_plot = d_before[np.abs(d_before) <= clip]
    d_after_plot  = d_after[np.abs(d_after)   <= clip]

    before_color, after_color = 'steelblue', 'tomato'

    fig, ax = plt.subplots()
    ax.hist(d_before_plot, bins=edges, alpha=0.5, color=before_color)
    ax.hist(d_after_plot,  bins=edges, alpha=0.5, color=after_color)
    ax.axvline(med_before, color=before_color, linestyle='--', linewidth=1.2)
    ax.axvline(med_after,  color=after_color,  linestyle='--', linewidth=1.2)
    ax.axvline(0, color='black', linewidth=0.8, linestyle=':')

    # Corner stat blocks coloured to match each histogram: before (green) on
    # the left, after (blue) on the right.
    ax.text(0.03, 0.97,
            f"Before\nmed:  {med_before:+.2f}\nnmad: {nmad_before:.2f}\nstd:  {std_before:.2f}",
            transform=ax.transAxes, va='top', ha='left',
            color=before_color, fontsize=10)
    ax.text(0.97, 0.97,
            f"After\nmed:  {med_after:+.2f}\nnmad: {nmad_after:.2f}\nstd:  {std_after:.2f}",
            transform=ax.transAxes, va='top', ha='right',
            color=after_color, fontsize=10)

    ax.set_xlim(-clip, clip)
    ax.set_xlabel('M3C2 distance (m)')
    ax.set_ylabel('Count')
    ax.set_title(f'M3C2 distances on stable terrain — {title}')
    plt.tight_layout()
    if output_dir is None:
        plt.show()
    else:
        plt.savefig(Path(output_dir) / filename, dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_dod_histogram(values: np.ndarray,
                       output_dir=None,
                       title: str = '',
                       filename: str = "dod_histogram.png",
                       bins: int = 60,
                       clip_sigma: float = 3.0) -> dict:
    """Histogram of DoD pixel values with median + std markers.

    NaN pixels (outside the day's footprint, or otherwise masked) are
    stripped before stats. The histogram x-axis is clipped to ±clip_sigma·σ
    of the distribution so a few outliers don't squash the visible bins;
    median/mean/std/n in the legend are from the full un-clipped array.

    Parameters
    ----------
    values : np.ndarray
        DoD values, any shape — flattened internally. NaNs are ignored.
    output_dir : str | Path, optional
        Directory to save the PNG. When None, the figure is shown
        interactively (in a notebook).
    title : str
        Trailing title text (e.g. the date).
    filename : str
        Output PNG filename.
    bins : int
        Number of histogram bins.
    clip_sigma : float
        Half-width of the x-axis clip in units of σ.

    Returns
    -------
    dict
        ``{'median', 'mean', 'std', 'n'}`` computed on the un-clipped data.
    """
    v = np.asarray(values).ravel()
    v = v[np.isfinite(v)]

    if v.size == 0:
        raise ValueError("No finite values in DoD array — nothing to plot.")

    med  = float(np.median(v))
    mean = float(np.mean(v))
    std  = float(np.std(v))

    clip   = clip_sigma * std if std > 0 else 1.0
    v_plot = np.clip(v, -clip, clip)

    fig, ax = plt.subplots()
    ax.hist(v_plot, bins=bins, color='steelblue', alpha=0.8)
    ax.axvline(med, color='tomato', linestyle='--', linewidth=1.5,
               label=f'median = {med:+.2f} m')
    ax.axvline(0, color='black', linewidth=0.8, linestyle=':')
    ax.plot([], [], ' ', label=f'mean   = {mean:+.2f} m')
    ax.plot([], [], ' ', label=f'std    = {std:.2f} m')
    ax.plot([], [], ' ', label=f'n      = {v.size:,}')
    ax.set_xlabel('DoD (m)')
    ax.set_ylabel('Count')
    ax.set_title(f'DoD distribution — {title}' if title else 'DoD distribution')
    ax.legend(fontsize=9)
    plt.tight_layout()

    if output_dir is None:
        plt.show()
    else:
        plt.savefig(Path(output_dir) / filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return {'median': med, 'mean': mean, 'std': std, 'n': int(v.size)}


def plot_stable_terrain_diagnostics(stable_slope: np.ndarray,
                                     stable_final: np.ndarray,
                                     ndwi: np.ndarray,
                                     grayscale_intensity: np.ndarray,
                                     line_a: float,
                                     line_b: float,
                                     output_dir,
                                     title: str,
                                     overwrite: bool = False) -> None:
    """Generate the two diagnostic plots for a point cloud.

    Produces:
      - ndwi_vs_intensity.png   (NDWI scatter with separation line, pre-NDWI-filter)
      - stable_terrain_rgb.png  (RGB-coloured points after both filters)
    """
    output_dir = Path(output_dir)

    _plot_if_missing(overwrite, output_dir / "ndwi_vs_intensity.png",
                     lambda: plot_ndwi_vs_intensity(
                         ndwi, grayscale_intensity, stable_slope[:, 3:6],
                         line_a, line_b, output_dir))

    _plot_if_missing(overwrite, output_dir / "stable_terrain_rgb.png",
                     lambda: plot_stable_terrain_rgb(
                         stable_final, output_dir,
                         title=f'Stable terrain — {title}',
                         filename='stable_terrain_rgb.png'))


def plot_m3c2_spatial(
    ref_stable: np.ndarray,
    dist_before: np.ndarray,
    dist_after: np.ndarray,
    output_dir=None,
    title: str = '',
    filename: str = "m3c2_spatial.png",
    res: float = 1.0,
    vmax: float = 3.0,
    cbar_label: str = 'M3C2 distance (m)',
    save_pdf: bool = True,
    layout: str = 'auto',
) -> None:
    """Publication-quality 2-panel map of stable-terrain M3C2 residuals.

    Bins the stable-terrain M3C2 distances onto a regular easting/northing grid
    (median per ``res``-metre cell) and shows before vs. after co-registration
    with a diverging colormap centred on zero — the point-cloud analogue of the
    demcoreg elevation-difference QC maps. A good co-registration shows a
    random, unstructured pattern near zero after coreg (no tilt/stripes).

    Parameters
    ----------
    ref_stable:
        (N, >=2) stable-terrain reference (corepoint) array — columns 0/1 are
        easting / northing in the cloud's UTM CRS.
    dist_before, dist_after:
        (N,) M3C2 distances (``run_m3c2`` corepoint distances), NaN where no
        valid distance was found.
    res:
        Ground cell size in metres (default 1.0). Bin counts follow the extent.
    vmax:
        Symmetric colour limit [m]; fixed at 3.0 by default so the scale is
        constant across dates. Pass ``None`` for the per-day 95th-percentile
        of ``|dist_before|``.
    layout:
        ``'auto'`` (default) stacks panels vertically for wide/short footprints
        and places them side-by-side otherwise; force with ``'vertical'`` /
        ``'horizontal'``.
    output_dir:
        Directory to save the PNG (+ PDF when ``save_pdf``). ``None`` → display.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.stats import binned_statistic_2d

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 12, 'axes.titlesize': 13,
        'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.linewidth': 0.8, 'mathtext.fontset': 'cm',
    })

    x, y = ref_stable[:, 0], ref_stable[:, 1]
    x0, y0 = np.floor(np.nanmin(x)), np.floor(np.nanmin(y))
    xr, yr = x - x0, y - y0
    rng_xy = [[float(xr.min()), float(xr.max())], [float(yr.min()), float(yr.max())]]

    # fixed ground resolution → square res-metre cells; bin counts from extent
    nx = max(int(np.ceil((rng_xy[0][1] - rng_xy[0][0]) / res)), 1)
    ny = max(int(np.ceil((rng_xy[1][1] - rng_xy[1][0]) / res)), 1)

    def _grid(dist):
        mask = ~np.isnan(dist)
        if mask.sum() < 4:
            return np.full((ny, nx), np.nan)
        stat, _, _, _ = binned_statistic_2d(
            xr[mask], yr[mask], dist[mask],
            statistic='median', bins=[nx, ny], range=rng_xy,
        )
        return stat.T          # rows = northing, cols = easting for imshow

    if vmax is None:
        valid = dist_before[~np.isnan(dist_before)]
        vmax = float(np.percentile(np.abs(valid), 95)) if len(valid) else 1.0
    vmax = max(vmax, 0.01)

    im_extent = [xr.min(), xr.max(), yr.min(), yr.max()]
    has = ~np.isnan(dist_before) | ~np.isnan(dist_after)
    xmin, xmax = float(xr[has].min()), float(xr[has].max())
    ymin, ymax = float(yr[has].min()), float(yr[has].max())
    span, yspan = xmax - xmin, ymax - ymin
    xlim = [xmin - 0.02 * span, xmax + 0.02 * span]
    ylim = [ymin - 0.02 * yspan, ymax + 0.02 * yspan]

    # round scale-bar length ~ 25 % of map width (m)
    sb = min([50, 100, 200, 250, 500, 1000, 2000], key=lambda s: abs(s - span * 0.25))

    # layout adapts to footprint: wide/short stacks; tall/square side-by-side
    data_aspect = span / yspan if yspan > 0 else 1.0
    stack = (layout == 'vertical') or (layout == 'auto' and data_aspect >= 1.5)

    if stack:
        ph = 3.2
        pw = min(max(ph * data_aspect, 4.0), 9.0)
        fig, axes = plt.subplots(2, 1, figsize=(pw + 1.6, 2 * ph + 1.0),
                                 sharex=True, sharey=True, constrained_layout=True)
    else:
        ph = 4.6
        pw = min(max(ph * data_aspect, 2.2), 6.5)
        fig, axes = plt.subplots(1, 2, figsize=(2 * pw + 1.8, ph + 1.0),
                                 sharex=True, sharey=True, constrained_layout=True)

    for k, (ax, dist, label) in enumerate(zip(
        axes, [dist_before, dist_after],
        ['Before co-registration', 'After co-registration'],
    )):
        im = ax.imshow(
            _grid(dist), origin='lower', extent=im_extent,
            cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='equal',
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(label, pad=8)
        if stack:
            ax.set_ylabel(f'Northing $-$ {y0:,.0f} (m)')
            if k == 1:
                ax.set_xlabel(f'Easting $-$ {x0:,.0f} (m)')
        else:
            ax.set_xlabel(f'Easting $-$ {x0:,.0f} (m)')
            if k == 0:
                ax.set_ylabel(f'Northing $-$ {y0:,.0f} (m)')

        # scale bar (lower-left)
        bx, by = xlim[0] + 0.05 * span, ylim[0] + 0.10 * yspan
        ax.add_patch(Rectangle((bx, by), sb, 0.018 * yspan, fc='k', ec='k', zorder=5))
        ax.text(bx + sb / 2, by + 0.05 * yspan, f'{sb:g} m',
                ha='center', va='bottom', fontsize=9, zorder=6)

    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, extend='both')
    cb.set_label(cbar_label)
    if title:
        fig.suptitle(title, y=1.02)

    if output_dir is None:
        plt.show()
    else:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / filename, dpi=300, bbox_inches='tight')
        if save_pdf:
            fig.savefig(out / f'{Path(filename).stem}.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out / filename}")


def plot_m3c2_coreg_and_signal(
    ref_stable: np.ndarray,
    dist_before: np.ndarray,
    dist_after: np.ndarray,
    signal: np.ndarray,
    signal_extent,
    output_dir=None,
    title: str = '',
    filename: str = "m3c2_coreg_and_signal.png",
    res: float = 1.0,
    vmax: float = 4.0,
    signal_origin: str = 'upper',
    cbar_label: str = 'M3C2 distance (m)',
    save_pdf: bool = True,
) -> None:
    """2×2 figure: before/after coreg maps + signal map + stable-terrain histogram.

    Top row = stable-terrain M3C2 residual maps **before** vs **after**
    co-registration (the QC / uncertainty, binned like :func:`plot_m3c2_spatial`).
    Bottom-left = the reference→day M3C2 **signal** raster (the surface change).
    The three maps share one diverging colour scale + colorbar, so the
    after-coreg residual reads as near-zero next to the signal. Bottom-right =
    the **distribution** of the stable-terrain M3C2 distances (before/after),
    with its x-axis fixed to ±``vmax`` so it lines up with the map colours.

    Parameters
    ----------
    ref_stable, dist_before, dist_after :
        Corepoint coords (cols 0/1 = easting/northing) and the before/after M3C2
        distances on stable terrain (from :func:`cntp.asp.evaluate_coreg`).
    signal :
        2-D M3C2 signal raster (NaN where no data), e.g. ``<date>_M3C2_raster.tif``.
    signal_extent :
        ``(xmin, xmax, ymin, ymax)`` of ``signal`` in the cloud's UTM CRS.
    signal_origin :
        ``'upper'`` (GeoTIFF/rasterio row 0 = north, default) or ``'lower'``.
    vmax :
        Symmetric colour limit [m] shared by all three panels (default ±4 m, so
        the scale is identical across dates and the coreg panels stay readable;
        signal beyond ±vmax saturates with extend arrows). Pass ``None`` to use
        the 95th percentile of ``|signal|`` instead.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import binned_statistic_2d

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 12, 'axes.titlesize': 13,
        'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.linewidth': 0.8, 'mathtext.fontset': 'cm',
    })

    x, y = ref_stable[:, 0], ref_stable[:, 1]
    # Common SW origin across the stable patches and the (larger) signal extent,
    # so all three panels share short, comparable offset tick labels.
    x0 = float(np.floor(min(np.nanmin(x), signal_extent[0])))
    y0 = float(np.floor(min(np.nanmin(y), signal_extent[2])))

    xr, yr = x - x0, y - y0
    rng = [[float(xr.min()), float(xr.max())], [float(yr.min()), float(yr.max())]]
    nx = max(int(np.ceil((rng[0][1] - rng[0][0]) / res)), 1)
    ny = max(int(np.ceil((rng[1][1] - rng[1][0]) / res)), 1)

    def _grid(dist):
        mask = ~np.isnan(dist)
        if mask.sum() < 4:
            return np.full((ny, nx), np.nan)
        stat, _, _, _ = binned_statistic_2d(
            xr[mask], yr[mask], dist[mask],
            statistic='median', bins=[nx, ny], range=rng,
        )
        return stat.T

    stable_extent = [rng[0][0], rng[0][1], rng[1][0], rng[1][1]]
    g_before, g_after = _grid(dist_before), _grid(dist_after)

    sx0, sx1, sy0, sy1 = signal_extent
    sig_extent = [sx0 - x0, sx1 - x0, sy0 - y0, sy1 - y0]

    # One shared colour scale for all three panels (default ±4 m); signal beyond
    # it saturates (extend arrows). Pass vmax=None for a per-figure auto range.
    if vmax is None:
        v = signal[np.isfinite(signal)]
        vmax = float(np.percentile(np.abs(v), 95)) if v.size else 1.0
    vmax = max(vmax, 0.01)

    # Crop each panel to ITS OWN data (robust 0.5–99.5th percentile) so every
    # raster fills its box with no empty right-hand margin: the coreg panels to
    # the stable-terrain extent, the signal panel to the glacier extent. (Stray
    # far points/cells are ignored by the percentile.)
    H, W = signal.shape
    xs = np.linspace(sig_extent[0], sig_extent[1], W)
    ys = (np.linspace(sig_extent[2], sig_extent[3], H) if signal_origin == 'lower'
          else np.linspace(sig_extent[3], sig_extent[2], H))
    fr, fc = np.where(np.isfinite(signal))

    def _lim(a, pad=0.08):
        if not a.size:
            return None
        lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
        m = pad * (hi - lo)
        return (lo - m, hi + m)

    # One shared "glacier frame" for all three maps = the signal (glacier)
    # extent, so the panels are spatially aligned. Stable terrain is drawn in
    # the same frame; where it's absent (e.g. the accumulation zone) the coreg
    # panels are legitimately blank — same binning as plot_m3c2_spatial, just a
    # wider frame than that per-date QC plot uses.
    common_xlim, common_ylim = _lim(xs[fc]), _lim(ys[fr])

    # ── 2×2 figure: 3 maps (before / after coreg + signal) + a stable-terrain
    # M3C2 histogram. Balanced grid → readable for any glacier shape.
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), constrained_layout=True)
    ax_before, ax_after = axes[0]
    ax_signal, ax_hist = axes[1]

    maps = [
        (ax_before, g_before, stable_extent, 'lower',       'Stable terrain — before co-registration'),
        (ax_after,  g_after,  stable_extent, 'lower',       'Stable terrain — after co-registration'),
        (ax_signal, signal,   sig_extent,    signal_origin, 'Glacier-wide surface change (reference → day)'),
    ]
    im = None
    for ax, data, extent, origin, label in maps:
        # aspect='auto' fills the box; the shared glacier frame keeps all three
        # panels on the same spatial extent.
        im = ax.imshow(data, origin=origin, extent=extent, cmap='RdBu',
                       vmin=-vmax, vmax=vmax, aspect='auto')
        if common_xlim:
            ax.set_xlim(common_xlim)
        if common_ylim:
            ax.set_ylim(common_ylim)
        ax.set_title(label)
        ax.set_xlabel(f'Easting $-$ {x0:,.0f} (m)')
        ax.set_ylabel(f'Northing $-$ {y0:,.0f} (m)')

    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02,
                      extend='both', location='right')
    cb.set_label(cbar_label)

    # 4th panel — distribution of the stable-terrain M3C2 distances (before vs
    # after); x-axis fixed to ±vmax so it reads on the same scale as the map
    # colours (steelblue = before, tomato = after).
    db = dist_before[~np.isnan(dist_before)]
    da = dist_after[~np.isnan(dist_after)]
    mb, ma = float(np.median(db)), float(np.median(da))
    nb = float(1.4826 * np.median(np.abs(db - mb)))
    na = float(1.4826 * np.median(np.abs(da - ma)))
    sb, sa = float(np.std(db)), float(np.std(da))
    edges = np.arange(-vmax, vmax + 1e-9, max(vmax / 40.0, 0.05))
    ax_hist.hist(db[np.abs(db) <= vmax], bins=edges, alpha=0.5, color='steelblue')
    ax_hist.hist(da[np.abs(da) <= vmax], bins=edges, alpha=0.5, color='tomato')
    ax_hist.axvline(mb, color='steelblue', linestyle='--', linewidth=1.2)
    ax_hist.axvline(ma, color='tomato', linestyle='--', linewidth=1.2)
    ax_hist.axvline(0, color='black', linewidth=0.8, linestyle=':')
    ax_hist.set_xlim(-vmax, vmax)
    ax_hist.set_xlabel(cbar_label)
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Stable-terrain M3C2 distances')
    ax_hist.text(0.03, 0.97, f"Before\nmed:  {mb:+.2f}\nnmad: {nb:.2f}\nstd:  {sb:.2f}",
                 transform=ax_hist.transAxes, va='top', ha='left',
                 color='steelblue', fontsize=9)
    ax_hist.text(0.97, 0.97, f"After\nmed:  {ma:+.2f}\nnmad: {na:.2f}\nstd:  {sa:.2f}",
                 transform=ax_hist.transAxes, va='top', ha='right',
                 color='tomato', fontsize=9)
    if title:
        fig.suptitle(title, y=1.04)

    if output_dir is None:
        plt.show()
    else:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / filename, dpi=300, bbox_inches='tight')
        if save_pdf:
            fig.savefig(out / f'{Path(filename).stem}.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out / filename}")


# ---------------------------------------------------------------------------
# Relative / absolute accuracy figures — drawing primitives
# ---------------------------------------------------------------------------
# Pure drawing helpers for the per-pixel relative-accuracy maps + boxplot and
# the per-acquisition absolute-accuracy boxes. Loaders/orchestrators live in
# cntp.postprocessing, which calls these.

_PLOT_STYLE = {
    'font.family': 'serif', 'font.size': 12, 'axes.titlesize': 13,
    'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.linewidth': 0.8, 'mathtext.fontset': 'cm',
}


def _save_or_show(fig, output_dir, filename, save_pdf, show=False):
    """Save PNG (+ PDF when *save_pdf*) to *output_dir*, and/or display the figure.

    Saves whenever *output_dir* is given; displays when *show* is true (or when
    there is nowhere to save). The two are independent, so a figure can be both
    written to disk and rendered inline.
    """
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / filename, dpi=300, bbox_inches='tight')
        if save_pdf:
            fig.savefig(out / f'{Path(filename).stem}.pdf', bbox_inches='tight')
        print(f"  Saved: {out / filename}")
    if show or output_dir is None:
        plt.show()
    else:
        plt.close(fig)


def _robust_vmax(matrix, pct=98.0):
    """Colour limit from the ``pct``-th percentile of ``|matrix|`` (rounded up)."""
    import math
    a = np.abs(np.asarray(matrix, dtype="float64"))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 1.0
    v = float(np.percentile(a, pct))
    step = 0.5 if v < 5 else 1.0
    return max(math.ceil(v / step) * step, step)


def _summary(values):
    """Mean/median/count of the finite entries of an array."""
    a = np.asarray(values, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "median": float(np.median(a)), "n": int(a.size)}


def _nice_scalebar_len(width_m, *, frac=0.25):
    """A round scale-bar length ~ *frac* of the map width (1/2/2.5/5 x 10^n)."""
    nice = [10, 20, 25, 50, 100, 150, 200, 250, 300, 500, 750,
            1000, 2000, 2500, 5000, 10000]
    target = max(width_m * frac, nice[0])
    below = [n for n in nice if n <= target]
    return below[-1] if below else nice[0]


def _add_scalebar(ax, length_m, *, label=None, color="white", loc="lower right",
                  thickness_m=None):
    """Anchored scale bar *length_m* metres long (axes extent is in metres)."""
    import matplotlib.font_manager as fm
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    if label is None:
        label = f"{length_m/1000:g} km" if length_m >= 1000 else f"{length_m:g} m"
    if thickness_m is None:
        thickness_m = length_m * 0.03
    bar = AnchoredSizeBar(ax.transData, length_m, label, loc, pad=0.4, sep=5,
                          color=color, frameon=False, size_vertical=thickness_m,
                          fontproperties=fm.FontProperties(size=10))
    ax.add_artist(bar)


def data_window(arr, extent, *, pad_frac=0.06, q=0.5):
    """World-coordinate viewing window ``(xmin, xmax, ymin, ymax)`` for *arr*.

    Framed by the ``q``..``100-q`` percentiles of the finite pixels'
    easting/northing (not their min/max) and grown by ``pad_frac`` on each side,
    so a small blob of pixels detached from the main body doesn't stretch the
    frame. Pass the window to :func:`plot_maps_row` so several rasters share one
    placement. ``q=0`` gives the full min/max box.
    """
    a = np.asarray(arr, dtype="float64")
    rows, cols = np.where(np.isfinite(a))
    if rows.size == 0:
        return extent
    H, W = a.shape
    xmin, xmax, ymin, ymax = extent
    pw, ph = (xmax - xmin) / W, (ymax - ymin) / H
    xs = xmin + (cols + 0.5) * pw
    ys = ymax - (rows + 0.5) * ph
    x0, x1 = np.percentile(xs, [q, 100 - q])
    y0, y1 = np.percentile(ys, [q, 100 - q])
    s = max(x1 - x0, y1 - y0) * pad_frac
    return (x0 - s, x1 + s, y0 - s, y1 + s)


def plot_maps_row(panels, extent, *, window=None, scalebar_m="auto",
                  bad_color="white", panel_size=6.0, output_dir=None,
                  filename="pixel_maps.png", save_pdf=True, show=False):
    """Draw several rasters side by side on one shared grid + window.

    ``panels`` is a list of dicts sharing the same *extent* (grid). A single-band
    panel has ``values`` (H, W) plus optional ``cmap``, ``vmin``, ``vmax``,
    ``extend``, ``clabel``; an RGB(A) panel sets ``rgb=True`` with ``values``
    (H, W, 3 or 4) and gets no colourbar (its alpha channel carries any
    transparency, e.g. an ortho pre-clipped to the data footprint). ``window``
    (from :func:`data_window`) sets a common view so the panels align; NaN cells /
    transparent pixels / the canvas use ``bad_color``. Colourbar space is reserved
    on every panel (hidden on RGB ones) so all panels keep the same width.
    """
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    view = window if window is not None else extent
    r, g, b = mcolors.to_rgb(bad_color)
    sb_color = "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.5 else "white"
    plt.rcParams.update(_PLOT_STYLE)
    fig, axes = plt.subplots(1, len(panels), squeeze=False,
                             figsize=(panel_size * len(panels), panel_size))
    for ax, p in zip(axes[0], panels):
        ax.set_facecolor(bad_color)
        # The colourbar axes is tied to the image axes' *drawn* box, so it tracks
        # the height aspect='equal' gives it. A plain fig.colorbar(ax=...) sizes
        # itself to the whole cell, and runs far past a wide, short map.
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
        if p.get("rgb"):
            ax.imshow(np.asarray(p["values"]), extent=extent, origin="upper")
            cax.set_axis_off()                # reserve the width, draw nothing
        else:
            cmap = plt.get_cmap(p.get("cmap", "viridis")).copy()
            cmap.set_bad(bad_color)
            im = ax.imshow(np.ma.masked_invalid(np.asarray(p["values"], float)),
                           extent=extent, origin="upper", cmap=cmap,
                           vmin=p.get("vmin"), vmax=p.get("vmax"))
            fig.colorbar(im, cax=cax, extend=p.get("extend", "neither")).set_label(
                p.get("clabel", ""))
        ax.set_xlim(view[0], view[1])
        ax.set_ylim(view[2], view[3])
        ax.set_aspect("equal")
        # aspect='equal' shrinks each axes inside its cell; centre-anchoring then
        # leaves their tops at different heights, so the titles don't line up.
        ax.set_anchor("N")
        ax.set_xlabel("Easting (m)")
        if p.get("title"):
            # Matplotlib only raises a title clear of the y-axis offset text
            # ("1e6") when the two would overlap — so a short title stays low
            # while a long one gets bumped, and the row's titles end up ragged.
            # A pad that always clears the offset keeps them on one line.
            ax.set_title(p["title"], pad=16)
        if scalebar_m:
            length = (_nice_scalebar_len(view[1] - view[0])
                      if scalebar_m == "auto" else scalebar_m)
            _add_scalebar(ax, length, color=sb_color)
    axes[0][0].set_ylabel("Northing (m)")
    plt.tight_layout()
    _save_or_show(fig, output_dir, filename, save_pdf, show=show)


def plot_relative_accuracy_boxplot(sd, nmad, *,
                                   title="Relative accuracy of stable-terrain "
                                   "M3C2 stack", ylabel="Per-pixel deviation (m)",
                                   ylim=None, output_dir=None,
                                   filename="stable_relative_accuracy.png",
                                   save_pdf=True, show=False):
    """Relative-accuracy boxplot of per-corepoint stable SD and NMAD.

    Two boxes (SD, NMAD): box = IQR, red median line, whiskers 1.5xIQR, fliers
    hidden, black-triangle mean (legend "mean"/"median"). Returns the mean/median
    of each so the printed number and the figure agree. Pass the finite-or-NaN
    arrays from :func:`cntp.postprocessing.stable_precision_arrays`; NaNs are
    dropped here.
    """
    from matplotlib.lines import Line2D

    sd = np.asarray(sd, dtype="float64"); sd = sd[np.isfinite(sd)]
    nmad = np.asarray(nmad, dtype="float64"); nmad = nmad[np.isfinite(nmad)]

    plt.rcParams.update(_PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(3.2, 4.2))
    ax.boxplot(
        [sd, nmad], positions=[1, 2], widths=0.5,
        whis=1.5, showfliers=False, showmeans=True,
        medianprops=dict(color="red", lw=1.2),
        meanprops=dict(marker="^", markerfacecolor="black",
                       markeredgecolor="black", markersize=7),
        boxprops=dict(color="black"), whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["SD", "NMAD"])
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, None)
    ax.legend(handles=[Line2D([0], [0], marker="^", color="none",
                              markerfacecolor="black", markeredgecolor="black",
                              markersize=7, label="mean"),
                       Line2D([0], [0], color="red", lw=1.2, label="median")],
              loc="upper right", frameon=False, handletextpad=0.2)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    summary = {"sd": _summary(sd), "nmad": _summary(nmad)}
    _save_or_show(fig, output_dir, filename, save_pdf, show=show)
    return summary


def plot_absolute_accuracy_boxes(records, *, area_is_m2=True, bin_days=14,
                                 site_label=None, cmap="Blues", median_color="red",
                                 ylim=None, width=None, output_dir=None,
                                 filename="stable_absolute_accuracy.png",
                                 save_pdf=True, show=False):
    """Absolute-accuracy boxes: per-acquisition stable residuals over time.

    ``records`` is a list of ``{date, area, vals}`` where ``vals`` are that
    acquisition's finite stable-terrain M3C2 distances. One box per record at its
    ``date``, coloured by ``area`` (``Blues`` + colourbar; m^2 when ``area_is_m2``,
    else valid-corepoint count). Red median, whiskers 1.5xIQR, fliers hidden,
    dashed zero line; y auto-fits the whiskers symmetric about 0 unless ``ylim``
    is given. ``bin_days`` only sets the cadence word in the title.
    """
    import matplotlib.colors as mcolors
    import matplotlib.dates as mdates
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D

    areas = np.array([r["area"] for r in records], dtype="float64")
    norm = mcolors.Normalize(vmin=0.0, vmax=float(areas.max()))
    cmap_obj = plt.get_cmap(cmap)
    if width is None:
        width = bin_days * 0.6 if bin_days else 5.0

    plt.rcParams.update(_PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axhline(0, color="0.4", lw=0.9, ls="--", zorder=1)
    whi = 0.0
    for r in records:
        pos = mdates.date2num(r["date"])
        bp = ax.boxplot([r["vals"]], positions=[pos], widths=width,
                        whis=1.5, showfliers=False, patch_artist=True,
                        medianprops=dict(color=median_color, lw=1.2),
                        boxprops=dict(color="black"),
                        whiskerprops=dict(color="black"),
                        capprops=dict(color="black"), manage_ticks=False)
        bp["boxes"][0].set_facecolor(cmap_obj(norm(r["area"])))
        whi = max(whi, abs(bp["caps"][0].get_ydata()[0]),
                  abs(bp["caps"][1].get_ydata()[0]))

    ax.set_ylabel("Elevation difference (m)")
    ax.set_xlabel("Date")
    ax.set_ylim(*(ylim if ylim is not None else (-1.1 * whi, 1.1 * whi)))
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    suffix = f" — {site_label}" if site_label else ""
    cadence = {7: "weekly", 14: "biweekly", 30: "monthly", 31: "monthly"}
    if bin_days:
        ax.set_title(f"Absolute accuracy "
                     f"({cadence.get(bin_days, f'{bin_days}-day')}){suffix}")
    else:
        ax.set_title(f"Absolute accuracy{suffix}")
    ax.legend(handles=[Line2D([0], [0], color=median_color, lw=1.2,
                              label="median")],
              loc="upper right", frameon=False, handletextpad=0.4)
    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label("Stable surface area (m$^2$)" if area_is_m2
                 else "Valid stable corepoints")
    plt.tight_layout()
    _save_or_show(fig, output_dir, filename, save_pdf, show=show)

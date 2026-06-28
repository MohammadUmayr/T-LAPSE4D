import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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

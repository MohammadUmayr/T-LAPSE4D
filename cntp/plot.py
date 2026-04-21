import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
    plt.title('Zones de terrain stable extraites géométriquement')
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


def plot_stable_terrain_rgb(stable_points, output_dir, title='Zone de terrain stable', filename="stable_terrain_rgb.png"):
    """3D scatter plot of stable terrain colored by RGB."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(stable_points[:, 0], stable_points[:, 1], stable_points[:, 2], c=stable_points[:, 3:6] / 255, marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    ax.view_init(elev=30, azim=30)
    plt.savefig(Path(output_dir) / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_stable_terrain_diagnostics(stable_slope: np.ndarray,
                                     stable_final: np.ndarray,
                                     ndwi: np.ndarray,
                                     grayscale_intensity: np.ndarray,
                                     line_a: float,
                                     line_b: float,
                                     output_dir,
                                     title: str,
                                     overwrite: bool = False) -> None:
    """Generate all three diagnostic plots for a point cloud.

    Produces:
      - stable_terrain_geometry.png  (slope-filtered points, before NDWI filter)
      - ndwi_vs_intensity.png        (NDWI scatter with separation line, before NDWI filter)
      - stable_terrain_rgb.png       (RGB-coloured points after both filters)

    Parameters
    ----------
    stable_slope : np.ndarray
        Points after slope filter only (Nx9).
    stable_final : np.ndarray
        Points after slope + NDWI filter (Nx9).
    ndwi : np.ndarray
        NDWI values computed from stable_slope.
    grayscale_intensity : np.ndarray
        Mean RGB intensity computed from stable_slope.
    line_a : float
        Slope of the NDWI vs intensity separation line.
    line_b : float
        Intercept of the NDWI vs intensity separation line.
    output_dir : str | Path
        Directory where plots are saved.
    title : str
        Title suffix used in plot titles and the RGB plot title.
    overwrite : bool
        If False (default), skip plots that already exist.
    """
    output_dir = Path(output_dir)

    _plot_if_missing(overwrite, output_dir / "stable_terrain_geometry.png",
                     lambda: plot_stable_terrain_geometry(stable_slope, output_dir))

    _plot_if_missing(overwrite, output_dir / "ndwi_vs_intensity.png",
                     lambda: plot_ndwi_vs_intensity(
                         ndwi, grayscale_intensity, stable_slope[:, 3:6],
                         line_a, line_b, output_dir))

    _plot_if_missing(overwrite, output_dir / "stable_terrain_rgb.png",
                     lambda: plot_stable_terrain_rgb(
                         stable_final, output_dir,
                         title=f'Zone de terrain stable - {title}',
                         filename='stable_terrain_rgb.png'))

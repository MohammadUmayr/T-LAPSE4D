import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


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


def plot_ndwi_vs_intensity(ndwi, grayscale_intensity, colors, x_values, y_values, output_dir, filename="ndwi_vs_intensity.png"):
    """2D scatter plot of NDWI vs grayscale intensity with separation line."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(ndwi, grayscale_intensity, c=colors / 255, marker='.')
    ax.plot(x_values, y_values, color='red', label='Line: y = ax + b')
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



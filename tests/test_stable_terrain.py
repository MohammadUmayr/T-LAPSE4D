import numpy as np
import pytest
from cntp.coreg import (
    calculate_aspect_slope,
    extract_stable_terrain,
    otsu_thresholding,
    filter_points_inside_box,
    filter_points_outside_box,
)


# ---------------------------------------------------------------------------
# calculate_aspect_slope
# ---------------------------------------------------------------------------

def test_slope_upward_normal_is_zero():
    """Normal pointing straight up → flat terrain → slope = 0°."""
    normal = np.array([[0.0, 0.0, 1.0]])
    _, slope = calculate_aspect_slope(normal)
    np.testing.assert_allclose(slope, 0.0, atol=1e-6)


def test_slope_horizontal_normal_is_90():
    """Normal pointing horizontally → vertical wall → slope = 90°.

    This is the upper bound of the slope filter — anything at 90° must pass.
    """
    normal = np.array([[1.0, 0.0, 0.0]])
    _, slope = calculate_aspect_slope(normal)
    np.testing.assert_allclose(slope, 90.0, atol=1e-6)


def test_slope_45_degree_normal():
    """Normal at 45° tilt → slope = 45°."""
    v = 1.0 / np.sqrt(2)
    normal = np.array([[v, 0.0, v]])
    _, slope = calculate_aspect_slope(normal)
    np.testing.assert_allclose(slope, 45.0, atol=1e-5)


# ---------------------------------------------------------------------------
# extract_stable_terrain
# ---------------------------------------------------------------------------

def _make_cloud(normals, rgb=None):
    """Build an Nx9 cloud. RGB defaults to grey-brown (non-water, passes NDWI filter)."""
    xyz = np.random.default_rng(0).uniform(0, 10, (len(normals), 3))
    if rgb is None:
        rgb = np.full((len(normals), 3), [100.0, 80.0, 60.0])
    return np.column_stack([xyz, rgb, normals])


def test_extract_stable_terrain_keeps_steep_points():
    """Steep points (slope=90°) survive; flat points (slope=0°) are removed."""
    steep_normals = np.tile([1.0, 0.0, 0.0], (200, 1))
    flat_normals  = np.tile([0.0, 0.0, 1.0], (200, 1))
    cloud = _make_cloud(np.vstack([steep_normals, flat_normals]))

    stable_slope, _ = extract_stable_terrain(cloud, slope_threshold=60)

    assert len(stable_slope) == 200, (
        f"Expected 200 steep points, got {len(stable_slope)}"
    )


def test_extract_stable_terrain_ndwi_removes_water():
    """Water-like points (B >> R) are removed by the NDWI filter after slope filter."""
    steep_normals = np.tile([1.0, 0.0, 0.0], (100, 1))
    water_rgb = np.tile([10.0, 10.0, 200.0], (100, 1))
    rock_rgb  = np.tile([120.0, 100.0, 80.0], (100, 1))

    cloud = np.vstack([_make_cloud(steep_normals, rgb=water_rgb),
                       _make_cloud(steep_normals, rgb=rock_rgb)])

    _, stable_final = extract_stable_terrain(cloud, slope_threshold=60)

    assert len(stable_final) > 0, "No stable points survived"
    assert len(stable_final) <= 100, (
        f"Water points were not filtered: {len(stable_final)} points survived"
    )


# ---------------------------------------------------------------------------
# otsu_thresholding
# ---------------------------------------------------------------------------

def test_otsu_classification_accuracy():
    """Otsu threshold must correctly classify >99% of points in a clearly bimodal case."""
    rng = np.random.default_rng(7)
    low  = rng.normal(20, 2, 500)
    high = rng.normal(80, 2, 500)
    labels = np.array([0] * 500 + [1] * 500)
    data = np.concatenate([low, high])

    threshold = otsu_thresholding(data, bins_num=256)
    accuracy = np.mean((data >= threshold).astype(int) == labels)

    assert accuracy > 0.99, (
        f"Otsu accuracy {accuracy:.3f} below 99% — threshold {threshold:.1f}"
    )


def test_otsu_moderately_overlapping_bimodal():
    """Otsu must separate two overlapping peaks (std=8) with >90% accuracy.

    This is the realistic case: NDWI or grayscale distributions in real point
    clouds will have partial overlap between terrain classes.
    """
    rng = np.random.default_rng(42)
    low  = rng.normal(40, 8, 1000)
    high = rng.normal(80, 8, 1000)
    labels = np.array([0] * 1000 + [1] * 1000)
    data = np.concatenate([low, high])

    threshold = otsu_thresholding(data, bins_num=128)
    accuracy = np.mean((data >= threshold).astype(int) == labels)

    assert accuracy > 0.90, (
        f"Otsu accuracy {accuracy:.3f} below 90% — threshold: {threshold:.1f}"
    )


# ---------------------------------------------------------------------------
# filter_points_inside_box / filter_points_outside_box
# ---------------------------------------------------------------------------

def _make_points_grid():
    """25 points on a 5×5 grid in XY, Z=0. XY in [0,4]."""
    x, y = np.meshgrid(np.arange(5), np.arange(5))
    extra = np.zeros((25, 6))
    return np.column_stack([x.ravel(), y.ravel(), np.zeros(25), extra])


def test_filter_inside_box_keeps_correct_points():
    """Box [1,1,-1]→[3,3,1] covers a 3×3 subgrid of the 5×5 grid → exactly 9 points."""
    points = _make_points_grid()
    inside = filter_points_inside_box(points, np.array([1, 1, -1]), np.array([3, 3, 1]))
    assert len(inside) == 9, f"Expected 9 points inside box, got {len(inside)}"


def test_filter_outside_box_is_complement():
    """Inside and outside filters must partition the full point set with no gaps or overlaps."""
    points = _make_points_grid()
    min_b, max_b = np.array([1, 1, -1]), np.array([3, 3, 1])
    inside  = filter_points_inside_box(points, min_b, max_b)
    outside = filter_points_outside_box(points, min_b, max_b)
    assert len(inside) + len(outside) == len(points)

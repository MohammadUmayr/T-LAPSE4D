"""
Functions to test the point-cloud co-registration tools.

``extract_stable_terrain``, ``calculate_aspect_slope`` and ``run_m3c2`` are the live parts of this module:
``asp.extract_stable_reference`` and ``asp.evaluate_coreg`` call them on every co-registration, and
``raster.m3c2_to_raster`` and ``pipeline_4dsfm`` call them too.

The py4dgeo ICP path in this module (``coreg_pc``) has been superseded by ASP ``pc_align`` and has no
callers outside ``batch.py``, which is itself unused. It is deliberately not covered — testing it would
lock in code that should be deleted.
"""

from __future__ import annotations

import numpy as np
import py4dgeo
import pytest

from cntp.coreg import (
    _NDWI_A,
    _NDWI_B,
    calculate_aspect_slope,
    downsample_point_cloud,
    extract_stable_terrain,
    filter_points,
    filter_points_inside_box,
    filter_points_outside_box,
    otsu_thresholding,
    run_m3c2,
)
from conftest import make_cloud


def _plane(n: int = 1500, z: float = 0.0, seed: int = 42) -> np.ndarray:
    """Return a gently rough horizontal plane — M3C2 resolves normals cleanly on it."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-10, 10, (n, 2))
    zz = np.full(n, z) + rng.normal(0, 0.01, n)
    return np.column_stack([xy, zz])


class TestAspectSlope:
    """Aspect and slope derived from point normals — the geometric half of the stable-terrain filter."""

    def test_calculate_aspect_slope__upward_normal(self) -> None:
        # Normal straight up -> flat terrain -> slope 0 degrees.
        _, slope = calculate_aspect_slope(np.array([[0.0, 0.0, 1.0]]))
        np.testing.assert_allclose(slope, 0.0, atol=1e-6)

    def test_calculate_aspect_slope__horizontal_normal(self) -> None:
        # Normal horizontal -> vertical wall -> slope 90 degrees, the filter's upper bound.
        _, slope = calculate_aspect_slope(np.array([[1.0, 0.0, 0.0]]))
        np.testing.assert_allclose(slope, 90.0, atol=1e-6)

    @pytest.mark.parametrize("deg", [15.0, 30.0, 45.0, 60.0, 75.0])
    def test_calculate_aspect_slope__recovers_the_angle(self, deg: float) -> None:
        # Build a normal at a known tilt and check the slope comes back as that tilt.
        theta = np.deg2rad(deg)
        normal = np.array([[np.sin(theta), 0.0, np.cos(theta)]])

        _, slope = calculate_aspect_slope(normal)

        assert slope[0] == pytest.approx(deg, abs=1e-6)

    def test_calculate_aspect_slope__aspect_points_downslope(self) -> None:
        # Aspect is measured from the negated horizontal components, so a normal tilted towards +X
        # describes a surface facing -X, i.e. aspect 180 degrees.
        normal = np.array([[np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)]])

        aspect, _ = calculate_aspect_slope(normal)

        assert np.abs(aspect[0]) == pytest.approx(180.0, abs=1e-6)

    def test_calculate_aspect_slope__vectorised(self) -> None:
        aspect, slope = calculate_aspect_slope(np.tile([[0.0, 0.0, 1.0]], (50, 1)))

        assert aspect.shape == (50,)
        assert slope.shape == (50,)


class TestExtractStableTerrain:
    """
    The slope + NDWI filter that selects ice-free, non-water terrain for co-registration. Everything
    downstream of it — the ASP transform, the reported NMAD, the precision maps — depends on this
    selecting the same points it selected yesterday.
    """

    def test_extract_stable_terrain__steep_points_kept(self) -> None:
        cloud = make_cloud(n=500, slope_deg=75.0)

        stable_slope, _ = extract_stable_terrain(cloud, slope_threshold=60)

        assert len(stable_slope) == len(cloud)

    def test_extract_stable_terrain__shallow_points_dropped(self) -> None:
        cloud = make_cloud(n=500, slope_deg=45.0)

        stable_slope, _ = extract_stable_terrain(cloud, slope_threshold=60)

        assert len(stable_slope) == 0

    def test_extract_stable_terrain__threshold_is_strict(self) -> None:
        # The comparison is `slope > threshold`, so points sitting exactly on it are dropped.
        cloud = make_cloud(n=200, slope_deg=60.0)

        stable_slope, _ = extract_stable_terrain(cloud, slope_threshold=60)

        assert len(stable_slope) == 0

    def test_extract_stable_terrain__splits_a_mixed_cloud(self) -> None:
        steep = make_cloud(n=300, slope_deg=70.0, seed=1)
        shallow = make_cloud(n=200, slope_deg=20.0, seed=2)

        stable_slope, _ = extract_stable_terrain(np.vstack([steep, shallow]), slope_threshold=60)

        assert len(stable_slope) == 300

    def test_extract_stable_terrain__rock_survives_ndwi(self) -> None:
        # Grey-brown rock: R > B gives a negative NDWI, below the separation line -> kept.
        cloud = make_cloud(n=400, grey=120.0, blue=60.0)

        _, stable_final = extract_stable_terrain(cloud, slope_threshold=60)

        assert len(stable_final) == len(cloud)

    def test_extract_stable_terrain__water_removed_by_ndwi(self) -> None:
        # Blue-dominant and bright: positive NDWI, above the separation line -> dropped.
        cloud = make_cloud(n=400, grey=120.0, blue=200.0)

        _, stable_final = extract_stable_terrain(cloud, slope_threshold=60)

        assert len(stable_final) == 0

    def test_extract_stable_terrain__ndwi_constants(self) -> None:
        # These constants define the separation line and are imported by asp.py and raster.py too, so
        # changing them silently changes what counts as stable terrain everywhere.
        assert _NDWI_A == -600.0
        assert _NDWI_B == 150.0

    def test_extract_stable_terrain__final_is_subset_of_slope(self) -> None:
        cloud = np.vstack(
            [
                make_cloud(n=200, slope_deg=75.0, grey=120.0, blue=60.0, seed=1),  # steep rock
                make_cloud(n=200, slope_deg=75.0, grey=120.0, blue=200.0, seed=2),  # steep water
                make_cloud(n=200, slope_deg=20.0, seed=3),  # shallow ground
            ]
        )

        stable_slope, stable_final = extract_stable_terrain(cloud, slope_threshold=60)

        # The slope filter keeps both steep groups; NDWI then removes the water half.
        assert len(stable_slope) == 400
        assert len(stable_final) == 200

    def test_extract_stable_terrain__preserves_nine_columns(self) -> None:
        stable_slope, stable_final = extract_stable_terrain(make_cloud(n=100))

        assert stable_slope.shape[1] == 9
        assert stable_final.shape[1] == 9


class TestBoxFilters:
    """Bounding-box inclusion and exclusion, used to crop clouds to the co-registration region."""

    def test_filter_points_inside_box(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [20.0, 20.0, 20.0]])

        kept = filter_points_inside_box(pts, np.array([1, 1, 1]), np.array([10, 10, 10]))

        np.testing.assert_allclose(kept, [[5.0, 5.0, 5.0]])

    def test_filter_points_outside_box__is_the_complement(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [20.0, 20.0, 20.0]])
        lo, hi = np.array([1, 1, 1]), np.array([10, 10, 10])

        inside = filter_points_inside_box(pts, lo, hi)
        outside = filter_points_outside_box(pts, lo, hi)

        # Every point lands in exactly one of the two halves.
        assert len(inside) + len(outside) == len(pts)

    def test_filter_points_inside_box__bounds_are_inclusive(self) -> None:
        pts = np.array([[1.0, 1.0, 1.0], [10.0, 10.0, 10.0]])

        kept = filter_points_inside_box(pts, np.array([1, 1, 1]), np.array([10, 10, 10]))

        assert len(kept) == 2

    def test_filter_points_inside_box__only_looks_at_xyz(self) -> None:
        # Nx9 clouds must filter on XYZ and carry the colour and normal columns through untouched.
        cloud = make_cloud(n=50)

        kept = filter_points_inside_box(cloud, np.array([0, 0, 0]), np.array([20, 20, 10]))

        assert kept.shape == (50, 9)


class TestOtsuThresholding:
    """Otsu's method, used to pick the intensity split between snow and rock."""

    def test_otsu_thresholding__bimodal(self) -> None:
        rng = np.random.default_rng(42)
        data = np.concatenate([rng.normal(10, 1, 5000), rng.normal(50, 1, 5000)])

        threshold = otsu_thresholding(data, bins_num=256)

        # The threshold must land in the valley between the two modes.
        assert 10 < threshold < 50

    def test_otsu_thresholding__inside_the_data_range(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 2000)

        threshold = otsu_thresholding(data, bins_num=128)

        assert data.min() <= threshold <= data.max()


class TestRunM3c2:
    """
    M3C2 distances between two epochs, and the median/NMAD/SD triple derived from them. This is the
    number the pipeline reports as co-registration quality, so the reductions are checked against
    offsets known by construction.
    """

    def test_run_m3c2__identical_clouds(self) -> None:
        cloud = _plane()

        med, nmad, _, distances = run_m3c2(py4dgeo.Epoch(cloud), py4dgeo.Epoch(cloud.copy()))

        # A cloud differenced against itself has no signal.
        assert med == pytest.approx(0.0, abs=0.01)
        assert nmad < 0.05
        assert len(distances) == len(cloud)

    def test_run_m3c2__recovers_a_known_offset(self) -> None:
        ref = _plane()
        tba = ref.copy()
        tba[:, 2] += 0.5

        med, nmad, _, _ = run_m3c2(py4dgeo.Epoch(ref), py4dgeo.Epoch(tba))

        # A 0.5 m lift must come back as a 0.5 m median, and a pure translation adds no spread.
        assert med == pytest.approx(0.5, abs=0.05)
        assert nmad < 0.05

    def test_run_m3c2__statistics_match_the_returned_distances(self) -> None:
        ref = _plane()
        tba = ref.copy()
        tba[:, 2] += 0.2

        med, nmad, std, distances = run_m3c2(py4dgeo.Epoch(ref), py4dgeo.Epoch(tba))

        # The reported triple must be the NaN-aware reduction of the distances actually returned —
        # in particular NMAD is 1.4826 * median(|d - median(d)|), not a standard deviation.
        expected_med = float(np.nanmedian(distances))
        assert med == pytest.approx(expected_med)
        assert nmad == pytest.approx(1.4826 * np.nanmedian(np.abs(distances - expected_med)))
        assert std == pytest.approx(float(np.nanstd(distances)))

    def test_run_m3c2__statistics_ignore_unmatched_corepoints(self) -> None:
        # Corepoints with no counterpart within max_distance come back as NaN; the reductions must
        # skip them rather than propagating NaN into the reported co-registration quality.
        ref = _plane()
        tba = ref[ref[:, 0] > 0.0].copy()  # the day's cloud covers only half the reference footprint
        tba[:, 2] += 0.2

        med, nmad, std, distances = run_m3c2(py4dgeo.Epoch(ref), py4dgeo.Epoch(tba))

        assert np.isnan(distances).any()
        assert np.isfinite([med, nmad, std]).all()


class TestDownsamplePointCloud:
    """Random thinning, used to cap the memory M3C2's KD-trees need on full-resolution clouds."""

    def test_downsample_point_cloud__keeps_the_requested_fraction(self) -> None:
        cloud = make_cloud(n=1000)

        out = downsample_point_cloud(cloud, 0.25)

        assert len(out) == 250

    def test_downsample_point_cloud__keeps_all_columns(self) -> None:
        out = downsample_point_cloud(make_cloud(n=100), 0.5)

        assert out.shape[1] == 9

    def test_downsample_point_cloud__factor_one_keeps_everything(self) -> None:
        cloud = make_cloud(n=200)

        assert len(downsample_point_cloud(cloud, 1.0)) == 200

    def test_downsample_point_cloud__returns_original_points(self) -> None:
        # Thinning selects points, it never interpolates new ones.
        cloud = make_cloud(n=200)

        out = downsample_point_cloud(cloud, 0.5)

        assert all(row in cloud[:, :3].tolist() for row in out[:, :3].tolist())


class TestFilterPoints:
    """Threshold filter assembling XYZ, colour and normals back into the Nx9 layout."""

    def test_filter_points(self) -> None:
        points = np.arange(30, dtype="float64").reshape(10, 3)
        colors = np.full((10, 3), 120.0)
        normals = np.tile([0.0, 0.0, 1.0], (10, 1))
        criterium = np.arange(10, dtype="float64")

        out = filter_points(points, colors, normals, criterium, threshold=4.0)

        # Strictly less than the threshold, so 0..3 survive.
        assert len(out) == 4
        assert out.shape[1] == 9

    def test_filter_points__threshold_above_everything_keeps_all(self) -> None:
        points = np.zeros((5, 3))
        out = filter_points(points, np.zeros((5, 3)), np.zeros((5, 3)), np.arange(5.0), threshold=99.0)

        assert len(out) == 5

    def test_filter_points__threshold_below_everything_keeps_none(self) -> None:
        points = np.zeros((5, 3))
        out = filter_points(points, np.zeros((5, 3)), np.zeros((5, 3)), np.arange(5.0), threshold=-1.0)

        assert len(out) == 0

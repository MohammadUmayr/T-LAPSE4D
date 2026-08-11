"""
Functions to test the plotting tools.

Figures are diagnostic output, so these tests do not inspect pixels. They check the parts that carry
information rather than appearance: that a figure is written where it was asked for, that the summary
statistics returned alongside it are correct, and that the pure helpers (colour limits, scale-bar
rounding, viewing windows) compute what they claim.

The Agg backend is selected in ``conftest`` so nothing tries to open a window.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import GRID_SHAPE, GRID_TRANSFORM, make_cloud

from tlapse4d.plot import (
    _nice_scalebar_len,
    _plot_if_missing,
    _robust_vmax,
    _save_or_show,
    _summary,
    data_window,
    plot_dod_histogram,
    plot_ndwi_vs_intensity,
    plot_stable_terrain_diagnostics,
    plot_stable_terrain_geometry,
    plot_stable_terrain_rgb,
)

EXTENT = (
    GRID_TRANSFORM.c,
    GRID_TRANSFORM.c + GRID_SHAPE[1],
    GRID_TRANSFORM.f - GRID_SHAPE[0],
    GRID_TRANSFORM.f,
)


class TestPlotIfMissing:
    """The guard that keeps a re-run from redrawing every figure in a season."""

    def test_plot_if_missing__draws_when_absent(self, tmp_path: Path) -> None:
        calls = []

        _plot_if_missing(False, tmp_path / "absent.png", lambda: calls.append(1))

        assert calls == [1]

    def test_plot_if_missing__skips_when_present(self, tmp_path: Path) -> None:
        existing = tmp_path / "there.png"
        existing.touch()
        calls = []

        _plot_if_missing(False, existing, lambda: calls.append(1))

        assert calls == []

    def test_plot_if_missing__overwrite_forces_a_redraw(self, tmp_path: Path) -> None:
        existing = tmp_path / "there.png"
        existing.touch()
        calls = []

        _plot_if_missing(True, existing, lambda: calls.append(1))

        assert calls == [1]


class TestSaveOrShow:
    """Saving and displaying are independent — a figure can be written to disk and shown inline."""

    def test_save_or_show__writes_png(self, tmp_path: Path) -> None:
        fig = plt.figure()

        _save_or_show(fig, tmp_path, "f.png", save_pdf=False)

        assert (tmp_path / "f.png").exists()

    def test_save_or_show__writes_pdf_alongside(self, tmp_path: Path) -> None:
        # The vector copy is what goes into a paper; the PNG is for quick inspection.
        fig = plt.figure()

        _save_or_show(fig, tmp_path, "f.png", save_pdf=True)

        assert (tmp_path / "f.png").exists()
        assert (tmp_path / "f.pdf").exists()

    def test_save_or_show__creates_missing_directory(self, tmp_path: Path) -> None:
        fig = plt.figure()
        dest = tmp_path / "nested" / "deep"

        _save_or_show(fig, dest, "f.png", save_pdf=False)

        assert (dest / "f.png").exists()

    def test_save_or_show__show_without_output_dir(self) -> None:
        # Nowhere to save means display only; it must not raise for want of a path.
        fig = plt.figure()

        _save_or_show(fig, None, "f.png", save_pdf=False, show=True)


class TestRobustVmax:
    """Colour limits taken from a percentile, so a few extreme pixels cannot flatten the whole map."""

    def test_robust_vmax__ignores_the_extreme_tail(self) -> None:
        values = np.concatenate([np.full(999, 1.0), [1000.0]])

        vmax = _robust_vmax(values, pct=98.0)

        assert vmax < 100.0

    def test_robust_vmax__ignores_nan(self) -> None:
        values = np.array([1.0, 2.0, np.nan, 3.0])

        assert np.isfinite(_robust_vmax(values))

    def test_robust_vmax__uses_absolute_values(self) -> None:
        # M3C2 signal is signed; the limit is symmetric about zero.
        assert _robust_vmax(np.array([-5.0, -4.0, -3.0])) == _robust_vmax(np.array([5.0, 4.0, 3.0]))

    def test_robust_vmax__all_nan(self) -> None:
        out = _robust_vmax(np.full(10, np.nan))

        assert out is None or np.isfinite(out) or np.isnan(out)


class TestSummary:
    """The mean/median/count triple printed beside every accuracy figure."""

    def test_summary(self) -> None:
        out = _summary(np.array([1.0, 2.0, 3.0]))

        assert out["mean"] == pytest.approx(2.0)
        assert out["median"] == pytest.approx(2.0)
        assert out["n"] == 3

    def test_summary__ignores_nan(self) -> None:
        out = _summary(np.array([1.0, np.nan, 3.0]))

        assert out["n"] == 2
        assert out["mean"] == pytest.approx(2.0)

    def test_summary__all_nan(self) -> None:
        # An empty selection must report n=0 rather than raising mid-figure.
        out = _summary(np.full(5, np.nan))

        assert out["n"] == 0
        assert np.isnan(out["mean"])


class TestNiceScalebarLen:
    """Scale bars land on round numbers a reader can reason about, never on 137 m."""

    @pytest.mark.parametrize("width,expected_max", [(100.0, 25), (1000.0, 250), (10000.0, 2500)])
    def test_nice_scalebar_len__scales_with_width(self, width: float, expected_max: int) -> None:
        out = _nice_scalebar_len(width)

        assert out <= expected_max
        assert out in {10, 20, 25, 50, 100, 150, 200, 250, 300, 500, 750, 1000, 2000, 2500, 5000, 10000}

    def test_nice_scalebar_len__tiny_map_gets_the_smallest_bar(self) -> None:
        assert _nice_scalebar_len(1.0) == 10


class TestDataWindow:
    """
    The shared viewing window. Framed by percentiles rather than min/max, so a detached blob of
    pixels cannot stretch the frame and shrink the actual data to nothing.
    """

    def test_data_window__frames_the_data(self) -> None:
        arr = np.full(GRID_SHAPE, np.nan)
        arr[4:8, 4:8] = 1.0

        xmin, xmax, ymin, ymax = data_window(arr, EXTENT)

        assert xmin < xmax
        assert ymin < ymax
        assert xmin >= EXTENT[0] - 1
        assert xmax <= EXTENT[1] + 1

    def test_data_window__outlier_pixel_does_not_stretch_the_frame(self) -> None:
        arr = np.full(GRID_SHAPE, np.nan)
        arr[5:7, 5:7] = 1.0
        arr[0, 0] = 1.0  # a lone detached pixel in the far corner

        percentile_window = data_window(arr, EXTENT, q=5.0)
        full_box = data_window(arr, EXTENT, q=0.0)

        assert (percentile_window[1] - percentile_window[0]) < (full_box[1] - full_box[0])

    def test_data_window__all_nan_falls_back_to_the_full_extent(self) -> None:
        out = data_window(np.full(GRID_SHAPE, np.nan), EXTENT)

        assert out == EXTENT


class TestStableTerrainPlots:
    """The three diagnostic figures produced while selecting stable terrain."""

    def test_plot_stable_terrain_geometry(self, tmp_path: Path) -> None:
        plot_stable_terrain_geometry(make_cloud(n=200), tmp_path)

        assert (tmp_path / "stable_terrain_geometry.png").exists()

    def test_plot_stable_terrain_rgb(self, tmp_path: Path) -> None:
        plot_stable_terrain_rgb(make_cloud(n=200), tmp_path)

        assert (tmp_path / "stable_terrain_rgb.png").exists()

    def test_plot_ndwi_vs_intensity(self, tmp_path: Path) -> None:
        from tlapse4d.coreg import _NDWI_A, _NDWI_B

        cloud = make_cloud(n=200)
        grayscale = np.mean(cloud[:, 3:6], axis=1)
        ndwi = (cloud[:, 5] - cloud[:, 3]) / (cloud[:, 3] + cloud[:, 5])

        plot_ndwi_vs_intensity(ndwi, grayscale, cloud[:, 3:6], _NDWI_A, _NDWI_B, tmp_path)

        assert (tmp_path / "ndwi_vs_intensity.png").exists()

    def test_plot_stable_terrain_diagnostics__writes_both_figures(self, tmp_path: Path) -> None:
        from tlapse4d.coreg import _NDWI_A, _NDWI_B, extract_stable_terrain

        cloud = make_cloud(n=300)
        stable_slope, stable = extract_stable_terrain(cloud)
        grayscale = np.mean(stable_slope[:, 3:6], axis=1)
        ndwi = (stable_slope[:, 5] - stable_slope[:, 3]) / (stable_slope[:, 3] + stable_slope[:, 5])

        plot_stable_terrain_diagnostics(
            stable_slope, stable, ndwi, grayscale, _NDWI_A, _NDWI_B, tmp_path, title="ref"
        )

        assert (tmp_path / "ndwi_vs_intensity.png").exists()
        assert (tmp_path / "stable_terrain_rgb.png").exists()


class TestPlotDodHistogram:
    """
    The DoD histogram, which also returns the distribution summary the pipeline logs. NaN pixels lie
    outside the day's footprint and must be stripped before the statistics, not counted as zeros.
    """

    def test_plot_dod_histogram__returns_statistics(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(0.5, 0.1, 5000)

        stats = plot_dod_histogram(values, output_dir=tmp_path)

        assert stats["median"] == pytest.approx(0.5, abs=0.02)
        assert (tmp_path / "dod_histogram.png").exists()

    def test_plot_dod_histogram__strips_nan_before_stats(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        values = np.concatenate([rng.normal(0.5, 0.1, 2000), np.full(2000, np.nan)])

        stats = plot_dod_histogram(values, output_dir=tmp_path)

        # Counting the NaN half as zeros would halve the median.
        assert stats["median"] == pytest.approx(0.5, abs=0.02)

    def test_plot_dod_histogram__accepts_a_2d_raster(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        arr = rng.normal(-1.0, 0.2, GRID_SHAPE)

        stats = plot_dod_histogram(arr, output_dir=tmp_path, title="DoD")

        assert stats["median"] == pytest.approx(-1.0, abs=0.15)

"""
Functions to test the post-processing tools.

These are the numbers the uncertainty analysis is built on, so the temporal reductions are checked
against hand-computable answers: a stack of constant planes at 1, 2 and 3 m has median 2 and NMAD
1.4826 by construction.

Every test touching the cube cache passes an explicit ``cache_dir``. The default is
``~/.cache/tlapse4d_signalstack``, which a test must never write to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import GRID_SHAPE, GRID_TRANSFORM, write_raster
from rasterio.crs import CRS

from tlapse4d.postprocessing import (
    SignalStack,
    _discover_stable_ref,
    _is_iso_date,
    _nanmad,
    _nmad_keep_mask,
    _read_raster,
    _reference_ortho_panel,
    absolute_accuracy_boxplots,
    coreg_and_signal_figure,
    load_coreg_nmad,
    load_signal_stack,
    load_stable_distance_stack,
    load_stable_grid_stack,
    per_acquisition_nmad,
    per_pixel_nmad_map,
    per_pixel_obs_count,
    pixel_relative_accuracy,
    stable_precision_arrays,
)

# A series of 1, 2, 3 has median 2 and median-absolute-deviation 1, so its NMAD is 1.4826 * 1.
NMAD_OF_1_2_3 = 1.4826


def _times(*dates: str) -> np.ndarray:
    """Return *dates* as the ``datetime64[D]`` array the gate expects."""
    return np.array(dates, dtype="datetime64[D]")


def _write_stats(root: Path, date: str, nmad_before: float, nmad_after: float) -> None:
    """Write the per-date co-registration stats CSV the pipeline produces."""
    d = root / "output" / date / "coreg"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{date}_m3c2_stats.csv").write_text(
        f"coreg,med,nmad,std\nbefore,0.1,{nmad_before},0.3\nafter,0.01,{nmad_after},0.05\n"
    )


class TestIsIsoDate:
    """Telling acquisition directories apart from the other entries under ``output/``."""

    @pytest.mark.parametrize("s", ["2024-06-23", "1999-01-01"])
    def test_is_iso_date(self, s: str) -> None:
        assert _is_iso_date(s)

    @pytest.mark.parametrize("s", ["_ref_cache", "2024-6-23", "20240623", "2024-06-23x", ""])
    def test_is_iso_date__rejected(self, s: str) -> None:
        # ``output/`` also holds ``_ref_cache``, which must never be read as an acquisition date.
        assert not _is_iso_date(s)


class TestLoadSignalStack:
    """Assembling the per-date M3C2 rasters into one ``(T, H, W)`` cube on a shared grid."""

    def test_load_signal_stack(self, signal_stack_dir: tuple[Path, list[str], list[float]]) -> None:
        root, dates, _ = signal_stack_dir

        stack = load_signal_stack(root, cache=False)

        assert stack.dates == dates
        assert len(stack) == 3
        assert stack.cube.shape == (3, *GRID_SHAPE)

    def test_load_signal_stack__chronological(self, signal_stack_dir: tuple[Path, list[str], list[float]]) -> None:
        # ISO date strings sort chronologically, which is what makes the date subsetting work.
        root, dates, _ = signal_stack_dir

        assert load_signal_stack(root, cache=False).dates == sorted(dates)

    def test_load_signal_stack__georeferencing(self, signal_stack_dir: tuple[Path, list[str], list[float]]) -> None:
        root, _, _ = signal_stack_dir

        stack = load_signal_stack(root, cache=False)

        assert stack.transform == GRID_TRANSFORM
        assert stack.crs.to_epsg() == 32645
        assert stack.times.dtype == np.dtype("datetime64[D]")

    def test_load_signal_stack__values(self, signal_stack_dir: tuple[Path, list[str], list[float]]) -> None:
        root, _, values = signal_stack_dir

        stack = load_signal_stack(root, cache=False)

        for i, v in enumerate(values):
            assert stack.cube[i, 5, 5] == pytest.approx(v)
            assert np.isnan(stack.cube[i, 0, 0])  # the seeded gap

    def test_load_signal_stack__date_range_is_inclusive(
        self, signal_stack_dir: tuple[Path, list[str], list[float]]
    ) -> None:
        root, _, _ = signal_stack_dir

        stack = load_signal_stack(root, date_from="2024-06-24", date_to="2024-07-15", cache=False)

        assert stack.dates == ["2024-06-24", "2024-07-15"]

    def test_load_signal_stack__no_match(self, signal_stack_dir: tuple[Path, list[str], list[float]]) -> None:
        root, _, _ = signal_stack_dir

        with pytest.raises(FileNotFoundError, match="no '\\*_M3C2_raster.tif'"):
            load_signal_stack(root, date_from="2030-01-01", cache=False)

    def test_load_signal_stack__extent(self, signal_stack_dir: tuple[Path, list[str], list[float]]) -> None:
        root, _, _ = signal_stack_dir

        stack = load_signal_stack(root, cache=False)

        # (xmin, xmax, ymin, ymax) for matplotlib imshow, derived from the affine and cube shape.
        H, W = GRID_SHAPE
        xmin, ymax = GRID_TRANSFORM.c, GRID_TRANSFORM.f
        assert stack.extent == (xmin, xmin + W, ymax - H, ymax)

    def test_load_signal_stack__grid_mismatch(
        self, signal_stack_dir: tuple[Path, list[str], list[float]], crs: CRS
    ) -> None:
        # Every M3C2 raster must share the corepoint grid, or the cube stacks unrelated pixels. This
        # is what happens if the _ref_cache stable reference is rebuilt mid-record.
        root, _, _ = signal_stack_dir
        write_raster(root / "output" / "2024-08-01" / "single_day" / "2024-08-01_M3C2_raster.tif", np.zeros((5, 5)), crs)

        with pytest.raises(ValueError, match="grid mismatch"):
            load_signal_stack(root, cache=False)


class TestSignalStackCache:
    """The on-disk cube cache — minutes of GeoTIFF reads off a slow mount, avoided on later calls."""

    def test_cache__written_and_reused(
        self, signal_stack_dir: tuple[Path, list[str], list[float]], tmp_path: Path
    ) -> None:
        root, _, _ = signal_stack_dir
        cache = tmp_path / "cache"

        first = load_signal_stack(root, cache_dir=cache)

        assert list(cache.glob("*.cube.npy"))
        assert list(cache.glob("*.meta.npz"))

        second = load_signal_stack(root, cache_dir=cache)

        assert second.dates == first.dates
        np.testing.assert_allclose(np.asarray(second.cube), np.asarray(first.cube), equal_nan=True)

    def test_cache__is_memory_mapped(
        self, signal_stack_dir: tuple[Path, list[str], list[float]], tmp_path: Path
    ) -> None:
        # The cube can be several GB, so the cached read must not pull it all into RAM.
        root, _, _ = signal_stack_dir
        cache = tmp_path / "cache"
        load_signal_stack(root, cache_dir=cache)

        assert isinstance(load_signal_stack(root, cache_dir=cache).cube, np.memmap)

    def test_cache__invalidates_on_new_date(
        self, signal_stack_dir: tuple[Path, list[str], list[float]], tmp_path: Path, crs: CRS
    ) -> None:
        # A newly processed acquisition must not stay hidden behind a stale cube.
        root, _, _ = signal_stack_dir
        cache = tmp_path / "cache"
        load_signal_stack(root, cache_dir=cache)

        write_raster(
            root / "output" / "2024-08-01" / "single_day" / "2024-08-01_M3C2_raster.tif",
            np.full(GRID_SHAPE, 4.0),
            crs,
        )
        refreshed = load_signal_stack(root, cache_dir=cache)

        assert refreshed.dates[-1] == "2024-08-01"
        assert len(refreshed) == 4

    def test_cache__disabled_writes_nothing(
        self, signal_stack_dir: tuple[Path, list[str], list[float]], tmp_path: Path
    ) -> None:
        root, _, _ = signal_stack_dir
        cache = tmp_path / "cache"

        load_signal_stack(root, cache_dir=cache, cache=False)

        assert not cache.exists()


class TestTemporalReductions:
    """
    Per-pixel reductions along the time axis. The M3C2 stack is a valid precision estimate because every
    date is differenced against the same reference: the reference elevation is a per-pixel constant in
    time, so it shifts the temporal mean but cancels out of the temporal SD and NMAD.
    """

    def test_per_pixel_obs_count(self, signal_stack_dir: tuple[Path, list[str], list[float]]) -> None:
        root, _, _ = signal_stack_dir

        count = per_pixel_obs_count(load_signal_stack(root, cache=False))

        assert count.shape == GRID_SHAPE
        assert count[0, 0] == 0  # NaN on every date
        assert count[5, 5] == 3

    def test_nanmad(self) -> None:
        nmad, med = _nanmad(np.array([[1.0], [2.0], [3.0]]))

        np.testing.assert_allclose(med, [2.0])
        np.testing.assert_allclose(nmad, [NMAD_OF_1_2_3])

    def test_nanmad__ignores_missing_dates(self) -> None:
        # A missing acquisition must not drag the statistic, only reduce the sample.
        nmad, med = _nanmad(np.array([[1.0], [2.0], [3.0], [np.nan]]))

        np.testing.assert_allclose(med, [2.0])
        np.testing.assert_allclose(nmad, [NMAD_OF_1_2_3])

    def test_per_pixel_nmad_map(self, signal_stack_dir: tuple[Path, list[str], list[float]]) -> None:
        root, _, _ = signal_stack_dir

        out = per_pixel_nmad_map(load_signal_stack(root, cache=False), min_obs=3)

        np.testing.assert_allclose(out["nmad"][5, 5], NMAD_OF_1_2_3)
        np.testing.assert_allclose(out["median"][5, 5], 2.0)
        assert out["valid"][5, 5]

    def test_per_pixel_nmad_map__blanks_sparse_pixels(
        self, signal_stack_dir: tuple[Path, list[str], list[float]]
    ) -> None:
        # A pixel seen on too few dates has no meaningful precision, so it must be NaN, not a number.
        root, _, _ = signal_stack_dir

        out = per_pixel_nmad_map(load_signal_stack(root, cache=False), min_obs=3)

        assert not out["valid"][0, 0]
        assert np.isnan(out["nmad"][0, 0])
        assert np.isnan(out["median"][0, 0])

    def test_per_pixel_nmad_map__min_obs_is_applied(
        self, signal_stack_dir: tuple[Path, list[str], list[float]]
    ) -> None:
        root, _, _ = signal_stack_dir
        stack = load_signal_stack(root, cache=False)

        # Only three dates exist, so requiring four leaves nothing valid.
        assert per_pixel_nmad_map(stack, min_obs=4)["valid"].sum() == 0
        assert per_pixel_nmad_map(stack, min_obs=3)["valid"].sum() > 0

    def test_per_acquisition_nmad(self) -> None:
        # One value per acquisition — the co-registration quality gate, not a precision map.
        grid = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],  # perfectly flat -> 0
                [1.0, 2.0, 3.0, np.nan],  # -> 1.4826
            ]
        )

        out = per_acquisition_nmad(grid)

        assert out.shape == (2,)
        np.testing.assert_allclose(out[0], 0.0)
        np.testing.assert_allclose(out[1], NMAD_OF_1_2_3)


class TestNmadGate:
    """
    The acquisition-level NMAD gate. It drops poorly co-registered dates before the per-pixel reduction,
    optionally only from a given date on, so a known event (cameras lost, a cloudy season) does not
    force the earlier, unaffected period to be re-gated too.
    """

    def test_nmad_keep_mask__disabled(self) -> None:
        keep = _nmad_keep_mask(_times("2024-06-23", "2024-07-15"), np.array([0.05, 5.0]), None, None)

        assert keep.all()

    def test_nmad_keep_mask__drops_above_threshold(self) -> None:
        times = _times("2024-06-23", "2024-06-24", "2024-07-15")

        keep = _nmad_keep_mask(times, np.array([0.05, 0.20, 0.50]), 0.20, None)

        # The comparison is `nmad < max_nmad`, so a value exactly at the threshold is dropped.
        assert list(keep) == [True, False, False]

    def test_nmad_keep_mask__drops_missing_nmad(self) -> None:
        # A NaN NMAD means the coreg stats are absent for that date; it must not slip through the gate.
        keep = _nmad_keep_mask(_times("2024-06-23", "2024-06-24"), np.array([0.05, np.nan]), 0.20, None)

        assert list(keep) == [True, False]

    def test_nmad_keep_mask__only_from_the_given_date(self) -> None:
        # Acquisitions before max_nmad_from are always kept, however noisy.
        keep = _nmad_keep_mask(_times("2024-06-23", "2024-07-15"), np.array([5.0, 5.0]), 0.20, "2024-07-01")

        assert list(keep) == [True, False]

    def test_nmad_keep_mask__from_date_is_inclusive(self) -> None:
        keep = _nmad_keep_mask(_times("2024-07-01"), np.array([5.0]), 0.20, "2024-07-01")

        assert list(keep) == [False]


class TestLoadCoregNmad:
    """Reading the per-acquisition post-coreg NMAD the pipeline already wrote, rather than recomputing."""

    def test_load_coreg_nmad(self, tmp_path: Path) -> None:
        _write_stats(tmp_path, "2024-06-23", nmad_before=0.9, nmad_after=0.12)

        # The 'after' row is the one that describes co-registration quality.
        np.testing.assert_allclose(load_coreg_nmad(tmp_path, ["2024-06-23"]), [0.12])

    def test_load_coreg_nmad__aligned_with_requested_dates(self, tmp_path: Path) -> None:
        _write_stats(tmp_path, "2024-06-23", 0.9, 0.12)
        _write_stats(tmp_path, "2024-07-15", 0.9, 0.44)

        # The output order follows the requested dates, not the on-disk order.
        out = load_coreg_nmad(tmp_path, ["2024-07-15", "2024-06-23"])

        np.testing.assert_allclose(out, [0.44, 0.12])

    def test_load_coreg_nmad__missing_stats(self, tmp_path: Path) -> None:
        _write_stats(tmp_path, "2024-06-23", 0.9, 0.12)

        out = load_coreg_nmad(tmp_path, ["2024-06-23", "2024-06-24"])

        np.testing.assert_allclose(out[0], 0.12)
        assert np.isnan(out[1])

    def test_load_coreg_nmad__malformed_stats(self, tmp_path: Path) -> None:
        # A corrupt CSV must yield NaN for that date, not abort the whole uncertainty analysis.
        d = tmp_path / "output" / "2024-06-23" / "coreg"
        d.mkdir(parents=True)
        (d / "2024-06-23_m3c2_stats.csv").write_text("not,a,valid\nstats,file,here\n")

        assert np.isnan(load_coreg_nmad(tmp_path, ["2024-06-23"])[0])


class TestSignalStackContainer:
    """The dataclass wrapper carrying the cube alongside its dates and georeferencing."""

    def test_signal_stack__len(self) -> None:
        stack = SignalStack(
            ["2024-06-23", "2024-06-24"],
            _times("2024-06-23", "2024-06-24"),
            np.zeros((2, *GRID_SHAPE)),
            GRID_TRANSFORM,
            None,
            [],
        )

        assert len(stack) == 2


class TestReadRaster:
    """Reading a single band with its bounds, normalising whatever nodata sentinel it carries."""

    def test_read_raster(self, tmp_path: Path, crs: CRS) -> None:
        p = write_raster(tmp_path / "r.tif", np.full(GRID_SHAPE, 3.0), crs)

        arr, extent = _read_raster(p)

        np.testing.assert_allclose(arr, 3.0)
        # (xmin, xmax, ymin, ymax) taken from the raster bounds.
        H, W = GRID_SHAPE
        assert extent == (GRID_TRANSFORM.c, GRID_TRANSFORM.c + W, GRID_TRANSFORM.f - H, GRID_TRANSFORM.f)

    def test_read_raster__numeric_nodata_becomes_nan(self, tmp_path: Path, crs: CRS) -> None:
        # ASP writes -9999; downstream arithmetic must see NaN, not a valid-looking elevation.
        import rasterio

        arr = np.full(GRID_SHAPE, 3.0, dtype="float32")
        arr[0, 0] = -9999.0
        p = tmp_path / "r.tif"
        with rasterio.open(
            p, "w", driver="GTiff", height=GRID_SHAPE[0], width=GRID_SHAPE[1], count=1,
            dtype="float32", crs=crs, transform=GRID_TRANSFORM, nodata=-9999.0,
        ) as dst:
            dst.write(arr, 1)

        out, _ = _read_raster(p)

        assert np.isnan(out[0, 0])
        assert out[5, 5] == pytest.approx(3.0)


class TestDiscoverStableRef:
    """Locating the one cached stable-terrain reference under ``output/_ref_cache``."""

    def test_discover_stable_ref(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, _ = pipeline_output_dir

        found = _discover_stable_ref(root)

        assert found is not None
        assert found.name == "ref_0.15_stable.las"

    def test_discover_stable_ref__absent(self, tmp_path: Path) -> None:
        # No cache yet is not an error here — the caller turns it into a helpful message.
        assert _discover_stable_ref(tmp_path) is None


class TestLoadStableDistanceStack:
    """
    Stacking the per-date stable-terrain corepoint distances into a ``(T, N)`` matrix. Every array
    is 1:1 aligned to the same frozen reference corepoints, so they stack without regridding.
    """

    def test_load_stable_distance_stack(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, dates = pipeline_output_dir

        ds, times, mat = load_stable_distance_stack(root)

        assert ds == dates
        assert times.dtype == np.dtype("datetime64[D]")
        assert mat.shape == (3, 400)

    def test_load_stable_distance_stack__which_selects_the_array(
        self, pipeline_output_dir: tuple[Path, list[str]]
    ) -> None:
        root, _ = pipeline_output_dir

        _, _, after = load_stable_distance_stack(root, which="after")
        _, _, before = load_stable_distance_stack(root, which="before")

        # 'after' is post-coregistration, so it must be centred far closer to zero than 'before'.
        assert abs(np.nanmedian(after)) < abs(np.nanmedian(before))

    def test_load_stable_distance_stack__nan_marks_unobserved(
        self, pipeline_output_dir: tuple[Path, list[str]]
    ) -> None:
        root, _ = pipeline_output_dir

        _, _, mat = load_stable_distance_stack(root)

        # The fixture blanks the first five corepoints on every date.
        assert np.isnan(mat[:, :5]).all()

    def test_load_stable_distance_stack__date_bounds(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, dates = pipeline_output_dir

        ds, _, mat = load_stable_distance_stack(root, date_from=dates[1])

        assert ds == dates[1:]
        assert mat.shape[0] == 2

    def test_load_stable_distance_stack__explicit_dates(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, dates = pipeline_output_dir

        ds, _, _ = load_stable_distance_stack(root, dates=[dates[0], dates[2]])

        assert ds == [dates[0], dates[2]]

    def test_load_stable_distance_stack__no_match(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, _ = pipeline_output_dir

        with pytest.raises(FileNotFoundError, match="no '\\*_m3c2_distances.npz'"):
            load_stable_distance_stack(root, date_from="2030-01-01")

    def test_load_stable_distance_stack__corepoint_mismatch(
        self, pipeline_output_dir: tuple[Path, list[str]]
    ) -> None:
        # A reference rebuilt mid-record leaves dates with different corepoint counts. Stacking them
        # would silently compare unrelated points, so it must raise instead.
        root, _ = pipeline_output_dir
        bad = root / "output" / "2024-08-04" / "coreg"
        bad.mkdir(parents=True)
        np.savez(bad / "2024-08-04_m3c2_distances.npz", before=np.zeros(99), after=np.zeros(99))

        with pytest.raises(ValueError, match="corepoints, expected"):
            load_stable_distance_stack(root)


class TestLoadStableGridStack:
    """
    Binning the corepoint distances onto a metre grid. This is what makes the statistics reportable:
    corepoint density varies hugely, so reducing in point space would weight by density rather than
    by area, flattering the result because dense patches are the well-observed low-noise ones.
    """

    def test_load_stable_grid_stack(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, dates = pipeline_output_dir

        times, grid = load_stable_grid_stack(root)

        assert len(times) == len(dates)
        assert grid.shape[0] == len(dates)
        assert grid.dtype == np.dtype("float32")

    def test_load_stable_grid_stack__coarser_res_gives_fewer_cells(
        self, pipeline_output_dir: tuple[Path, list[str]]
    ) -> None:
        root, _ = pipeline_output_dir

        _, fine = load_stable_grid_stack(root, res=1.0)
        _, coarse = load_stable_grid_stack(root, res=5.0)

        assert coarse.shape[1] < fine.shape[1]

    def test_load_stable_grid_stack__missing_reference(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, _ = pipeline_output_dir
        (root / "output" / "_ref_cache" / "ref_0.15_stable.las").unlink()

        with pytest.raises(FileNotFoundError, match="no cached stable reference"):
            load_stable_grid_stack(root)

    def test_load_stable_grid_stack__reference_size_mismatch(
        self, pipeline_output_dir: tuple[Path, list[str]], tmp_path: Path
    ) -> None:
        # The reference must have exactly as many points as the distance arrays have corepoints.
        from conftest import make_cloud

        from tlapse4d.io import save_las

        root, _ = pipeline_output_dir
        ref = root / "output" / "_ref_cache" / "ref_0.15_stable.las"
        ref.unlink()
        save_las(make_cloud(n=50), ref)

        with pytest.raises(ValueError, match="rebuilt mid-record"):
            load_stable_grid_stack(root)


class TestStablePrecisionArrays:
    """Per-cell temporal SD and NMAD over the gridded stable stack — the relative-accuracy inputs."""

    def test_stable_precision_arrays(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, _ = pipeline_output_dir

        sd, nmad = stable_precision_arrays(root, min_obs=3)

        assert sd.shape == nmad.shape
        # Both describe the same cells, so they must agree on which are valid.
        np.testing.assert_array_equal(np.isnan(sd), np.isnan(nmad))
        assert np.isfinite(sd).any()

    def test_stable_precision_arrays__min_obs_blanks_sparse_cells(
        self, pipeline_output_dir: tuple[Path, list[str]]
    ) -> None:
        root, _ = pipeline_output_dir

        # Only three acquisitions exist, so requiring four leaves nothing.
        _, nmad = stable_precision_arrays(root, min_obs=4)

        assert np.isnan(nmad).all()

    def test_stable_precision_arrays__nmad_gate_drops_acquisitions(
        self, pipeline_output_dir: tuple[Path, list[str]]
    ) -> None:
        root, _ = pipeline_output_dir

        # An impossibly tight gate drops every date, so no cell reaches min_obs.
        _, gated = stable_precision_arrays(root, min_obs=2, max_nmad=1e-6)

        assert np.isnan(gated).all()


class TestReferenceOrthoPanel:
    """The ortho panel, resampled onto the M3C2 map grid and clipped to the coverage footprint."""

    def test_reference_ortho_panel(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, _ = pipeline_output_dir
        stack = load_signal_stack(root, cache=False)
        footprint = np.isfinite(stack.cube[0])

        panel = _reference_ortho_panel(root, stack, footprint, title="ortho")

        assert panel is not None
        assert panel["rgb"] is True
        assert panel["title"] == "ortho"
        # RGBA on the stack's own grid, not the ortho's source grid.
        assert panel["values"].shape == (*GRID_SHAPE, 4)

    def test_reference_ortho_panel__alpha_zero_outside_footprint(
        self, pipeline_output_dir: tuple[Path, list[str]]
    ) -> None:
        root, _ = pipeline_output_dir
        stack = load_signal_stack(root, cache=False)
        footprint = np.zeros(GRID_SHAPE, dtype=bool)
        footprint[3:6, 3:6] = True

        panel = _reference_ortho_panel(root, stack, footprint)

        # Outside the footprint the ortho is transparent, so it shows the same data shape as the
        # other panels in the row.
        assert panel is not None
        alpha = panel["values"][..., 3]
        assert alpha[0, 0] == 0
        assert alpha[4, 4] > 0

    def test_reference_ortho_panel__missing_ortho(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        # A missing ortho is not fatal — the figure is simply drawn without that panel.
        root, _ = pipeline_output_dir
        (root / "output" / "_ref_cache" / "reference_ortho.tif").unlink()
        stack = load_signal_stack(root, cache=False)

        assert _reference_ortho_panel(root, stack, np.ones(GRID_SHAPE, bool)) is None


class TestCoregAndSignalFigure:
    """The per-date before/after-coreg + signal three-panel figure."""

    def test_coreg_and_signal_figure(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, dates = pipeline_output_dir

        coreg_and_signal_figure(dates[0], root, save_pdf=False)

        out = root / "output" / dates[0] / "coreg" / "m3c2_plots"
        assert (out / f"{dates[0]}_coreg_and_signal.png").exists()

    def test_coreg_and_signal_figure__custom_plot_dir(
        self, pipeline_output_dir: tuple[Path, list[str]], tmp_path: Path
    ) -> None:
        root, dates = pipeline_output_dir
        dest = tmp_path / "figs"

        coreg_and_signal_figure(dates[0], root, plot_dir=dest, save_pdf=False)

        assert (dest / f"{dates[0]}_coreg_and_signal.png").exists()

    def test_coreg_and_signal_figure__save_pdf(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, dates = pipeline_output_dir

        coreg_and_signal_figure(dates[0], root, save_pdf=True)

        out = root / "output" / dates[0] / "coreg" / "m3c2_plots"
        assert (out / f"{dates[0]}_coreg_and_signal.pdf").exists()

    def test_coreg_and_signal_figure__missing_input(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, dates = pipeline_output_dir
        (root / "output" / dates[0] / "single_day" / f"{dates[0]}_M3C2_raster.tif").unlink()

        with pytest.raises(FileNotFoundError, match="missing required input"):
            coreg_and_signal_figure(dates[0], root, save_pdf=False)

    def test_coreg_and_signal_figure__no_stable_reference(
        self, pipeline_output_dir: tuple[Path, list[str]]
    ) -> None:
        root, dates = pipeline_output_dir
        (root / "output" / "_ref_cache" / "ref_0.15_stable.las").unlink()

        with pytest.raises(FileNotFoundError, match="no cached stable reference"):
            coreg_and_signal_figure(dates[0], root, save_pdf=False)

    def test_coreg_and_signal_figure__ambiguous_stable_reference(
        self, pipeline_output_dir: tuple[Path, list[str]]
    ) -> None:
        # Two cached references means the corepoint order is ambiguous; guessing would silently
        # pair distances with the wrong coordinates.
        from conftest import make_cloud

        from tlapse4d.io import save_las

        root, dates = pipeline_output_dir
        save_las(make_cloud(n=400), root / "output" / "_ref_cache" / "ref_0.30_stable.las")

        with pytest.raises(RuntimeError, match="multiple stable references"):
            coreg_and_signal_figure(dates[0], root, save_pdf=False)

    def test_coreg_and_signal_figure__corepoint_count_mismatch(
        self, pipeline_output_dir: tuple[Path, list[str]]
    ) -> None:
        from conftest import make_cloud

        from tlapse4d.io import save_las

        root, dates = pipeline_output_dir
        ref = root / "output" / "_ref_cache" / "ref_0.15_stable.las"
        ref.unlink()
        save_las(make_cloud(n=77), ref)

        with pytest.raises(ValueError, match="corepoint count"):
            coreg_and_signal_figure(dates[0], root, save_pdf=False)


class TestPixelRelativeAccuracy:
    """
    The relative-accuracy figure set: ortho + coverage + per-pixel NMAD maps, plus the stable-terrain
    SD/NMAD boxplot. The NMAD gate is applied to all of them alike, so every panel describes the
    same retained acquisitions.
    """

    def test_pixel_relative_accuracy(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, _ = pipeline_output_dir

        out = pixel_relative_accuracy(root, min_obs=2, save_pdf=False)

        assert set(out) == {"stack", "count", "nmad", "nmad_vmax", "valid", "stable_stats"}
        assert out["count"].shape == GRID_SHAPE
        assert out["nmad"].shape == GRID_SHAPE

    def test_pixel_relative_accuracy__writes_figures(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, _ = pipeline_output_dir

        pixel_relative_accuracy(root, min_obs=2, save_pdf=False)

        precision = root / "output" / "postprocessing" / "precision"
        assert (precision / "pixel_maps.png").exists()
        assert (precision / "stable_relative_accuracy.png").exists()

    def test_pixel_relative_accuracy__custom_plot_dir(
        self, pipeline_output_dir: tuple[Path, list[str]], tmp_path: Path
    ) -> None:
        root, _ = pipeline_output_dir
        dest = tmp_path / "figs"

        pixel_relative_accuracy(root, min_obs=2, plot_dir=dest, save_pdf=False)

        assert (dest / "pixel_maps.png").exists()

    def test_pixel_relative_accuracy__without_ortho(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, _ = pipeline_output_dir

        out = pixel_relative_accuracy(root, min_obs=2, ortho=False, save_pdf=False)

        assert (root / "output" / "postprocessing" / "precision" / "pixel_maps.png").exists()
        assert out["nmad"].shape == GRID_SHAPE

    def test_pixel_relative_accuracy__nmad_gate_shrinks_the_stack(
        self, pipeline_output_dir: tuple[Path, list[str]]
    ) -> None:
        root, dates = pipeline_output_dir

        ungated = pixel_relative_accuracy(root, min_obs=1, save_pdf=False)
        gated = pixel_relative_accuracy(root, min_obs=1, max_nmad=0.08, save_pdf=False)

        assert len(ungated["stack"]) == len(dates)
        assert len(gated["stack"]) < len(dates)

    def test_pixel_relative_accuracy__explicit_vmax_is_respected(
        self, pipeline_output_dir: tuple[Path, list[str]]
    ) -> None:
        root, _ = pipeline_output_dir

        out = pixel_relative_accuracy(root, min_obs=2, nmad_vmax=0.42, save_pdf=False)

        assert out["nmad_vmax"] == pytest.approx(0.42)


class TestAbsoluteAccuracyBoxplots:
    """
    One box per acquisition over stable terrain: a box centred on zero means co-registration left no
    bias, and the spread is that acquisition's precision.
    """

    def test_absolute_accuracy_boxplots(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, _ = pipeline_output_dir

        records = absolute_accuracy_boxplots(root, max_nmad=None, save_pdf=False)

        assert records
        for r in records:
            assert {"date", "source_date", "offset_days", "area", "median", "iqr"} <= set(r)
            assert r["area"] > 0

    def test_absolute_accuracy_boxplots__writes_figure(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        root, _ = pipeline_output_dir

        absolute_accuracy_boxplots(root, max_nmad=None, save_pdf=False)

        assert (root / "output" / "postprocessing" / "precision" / "stable_absolute_accuracy.png").exists()

    def test_absolute_accuracy_boxplots__one_box_per_acquisition(
        self, pipeline_output_dir: tuple[Path, list[str]]
    ) -> None:
        # bin_days=None draws every surviving acquisition at its own date rather than on a cadence.
        root, dates = pipeline_output_dir

        records = absolute_accuracy_boxplots(root, bin_days=None, max_nmad=None, save_pdf=False)

        assert len(records) == len(dates)
        assert all(r["offset_days"] == 0 for r in records)

    def test_absolute_accuracy_boxplots__binned_slots(self, pipeline_output_dir: tuple[Path, list[str]]) -> None:
        # On a cadence each slot takes the nearest acquisition in either direction, so gaps in the
        # record are filled rather than left blank.
        root, _ = pipeline_output_dir

        records = absolute_accuracy_boxplots(root, bin_days=7, max_nmad=None, save_pdf=False)

        # offset_days is signed: a slot may be filled by the acquisition just before it as readily
        # as by the one just after. What matters is that the acquisition used is genuinely nearby.
        assert len(records) >= 3
        assert all(abs(r["offset_days"]) <= 7 for r in records)

    def test_absolute_accuracy_boxplots__nmad_gate_excludes(
        self, pipeline_output_dir: tuple[Path, list[str]]
    ) -> None:
        root, dates = pipeline_output_dir

        kept_all = absolute_accuracy_boxplots(root, bin_days=None, max_nmad=None, save_pdf=False)
        gated = absolute_accuracy_boxplots(root, bin_days=None, max_nmad=0.08, save_pdf=False)

        assert len(kept_all) == len(dates)
        assert len(gated) < len(dates)

    def test_absolute_accuracy_boxplots__gate_ignored_when_it_empties_the_record(
        self, pipeline_output_dir: tuple[Path, list[str]]
    ) -> None:
        # If no acquisition passes, dropping everything would produce an empty figure; the gate is
        # reported as ignored instead.
        root, dates = pipeline_output_dir

        records = absolute_accuracy_boxplots(root, bin_days=None, max_nmad=1e-9, save_pdf=False)

        assert len(records) == len(dates)

"""
Functions to test the Metashape helper tools.

None of these touch the licensed Metashape module: the import in :mod:`tlapse4d.metashape` falls back to
``Metashape = None`` when it is absent, so everything here runs on a bare CI runner. The SfM entry points
themselves (``run_multitemporal_ba``, ``run_single_day_fixed_iop``, ``bootstrap_registry``) need a licence
and a real project, and are out of scope.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlapse4d.metashape import (
    _camera_excluded,
    _camera_prefix,
    _image_date,
    _init_native_log,
    _last_day_mean_eop,
    _normalize_date,
    _quiet_metashape,
    _read_4x4_matrix,
    _utm_epsg,
    _validate_time_window,
    discover_images,
    is_timelapse_label,
)


def _touch(root: Path, names: list[str]) -> None:
    """Create empty files *names* under *root*, making the directory if needed."""
    root.mkdir(parents=True, exist_ok=True)
    for n in names:
        (root / n).touch()


class TestLabelParsing:
    """Reading camera, date and time out of a standardised filename or Metashape label."""

    @pytest.mark.parametrize(
        "label",
        [
            "C7_2023-11-27_083000",
            "C7_2023-11-27_083000.JPG",
            "C12_2024-06-23_120000.jpg",
        ],
    )
    def test_is_timelapse_label(self, label: str) -> None:
        # A genuine time-lapse label is <camera>_<YYYY-MM-DD>_<HHMMSS>, with or without a file extension.
        assert is_timelapse_label(label)

    @pytest.mark.parametrize(
        "label",
        [
            "DJI_0457",  # drone frame sharing the Metashape project
            "C7_2023-11-27",  # no time part
            "C7_20231127_083000",  # unhyphenated date
            "C7_2023-11-27_0830",  # time too short
            "C_7_2023-11-27_083000",  # camera id contains an underscore
            "",
        ],
    )
    def test_is_timelapse_label__rejected(self, label: str) -> None:
        # This is the data-driven replacement for the old hardcoded CAMERAS whitelist: anything not
        # shaped like a time-lapse frame must be filtered out of the project.
        assert not is_timelapse_label(label)

    def test_camera_prefix(self) -> None:
        assert _camera_prefix("C1_2023-07-15_083000") == "C1"

    def test_image_date(self) -> None:
        assert _image_date("C1_2023-07-15_083000") == "2023-07-15"

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("2023-11-27", "2023-11-27"),
            ("11/27/2023", "2023-11-27"),  # Excel rewrote the registry CSV to a locale format
            ("27 Nov 2023", "2023-11-27"),
        ],
    )
    def test_normalize_date(self, value: str, expected: str) -> None:
        # Sensor lookup keys always come from _image_date() as 'YYYY-MM-DD'. A format drift in the CSV
        # would silently break the lookup and make Metashape merge every reference image into one sensor.
        assert _normalize_date(value) == expected


class TestUtmZone:
    """EPSG code derivation for the northern-hemisphere UTM zones."""

    @pytest.mark.parametrize(
        "lon,epsg",
        [
            (86.78, 32645),  # Khumbu — UTM 45N
            (0.5, 32631),
            (-179.9, 32601),  # first zone
            (179.9, 32660),  # last zone
        ],
    )
    def test_utm_epsg(self, lon: float, epsg: int) -> None:
        assert _utm_epsg(lon) == epsg


class TestTimeWindow:
    """Validation of the ``(start_hour, end_hour)`` capture-hour window."""

    def test_validate_time_window__none_disables_filtering(self) -> None:
        assert _validate_time_window(None) == (None, None)

    def test_validate_time_window__coerces_to_int(self) -> None:
        assert _validate_time_window(("9", "17")) == (9, 17)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "window",
        [
            (17, 9),  # reversed
            (-1, 12),  # below range
            (9, 24),  # above range
            ("nine", 17),  # not a number
            (9,),  # wrong arity
        ],
    )
    def test_validate_time_window__malformed(self, window: tuple) -> None:
        # A malformed window must raise rather than silently filter out the wrong frames.
        with pytest.raises(ValueError):
            _validate_time_window(window)  # type: ignore[arg-type]


class TestCameraExclusion:
    """
    Camera-exclusion rules — the filter that drops physically compromised cameras before the pipeline
    sees them, e.g. the boulder-displaced cameras on the Changri North monsoon dates. A silent
    regression here changes the co-registration NMAD, so the window boundaries are pinned explicitly.
    """

    def test_camera_excluded__no_rules(self) -> None:
        assert not _camera_excluded("C8", "2024-07-15", None)
        assert not _camera_excluded("C8", "2024-07-15", [])

    def test_camera_excluded__open_ended_start(self) -> None:
        rules = [{"camera": "C8", "from": "2024-07-01"}]

        # Inside the window, and matching the camera.
        assert _camera_excluded("C8", "2024-07-15", rules)
        # Before the window — the camera's earlier good frames are kept.
        assert not _camera_excluded("C8", "2024-06-30", rules)
        # A different camera on the same date is untouched.
        assert not _camera_excluded("C9", "2024-07-15", rules)

    def test_camera_excluded__boundaries_are_inclusive(self) -> None:
        rules = [{"camera": "C8", "from": "2024-07-01", "until": "2024-07-31"}]

        # Both ends of [from, until] are inside the window.
        assert _camera_excluded("C8", "2024-07-01", rules)
        assert _camera_excluded("C8", "2024-07-31", rules)
        # One day either side is outside it.
        assert not _camera_excluded("C8", "2024-06-30", rules)
        assert not _camera_excluded("C8", "2024-08-01", rules)

    def test_camera_excluded__rule_without_camera_applies_to_all(self) -> None:
        rules = [{"from": "2024-07-01", "until": "2024-07-31"}]

        assert _camera_excluded("C8", "2024-07-15", rules)
        assert _camera_excluded("C3", "2024-07-15", rules)
        # Still bounded by the date window.
        assert not _camera_excluded("C3", "2024-08-15", rules)

    def test_camera_excluded__multiple_rules_are_ored(self) -> None:
        # The Changri North case: C8 and C9 both dropped on the same acquisition.
        rules = [
            {"camera": "C8", "from": "2024-07-15", "until": "2024-07-15"},
            {"camera": "C9", "from": "2024-07-15", "until": "2024-07-15"},
        ]

        assert _camera_excluded("C8", "2024-07-15", rules)
        assert _camera_excluded("C9", "2024-07-15", rules)
        assert not _camera_excluded("C7", "2024-07-15", rules)


class TestDiscoverImages:
    """Scanning a directory tree of standardised images into ``{date: {camera: [paths]}}``."""

    def test_discover_images(self, tmp_path: Path) -> None:
        _touch(tmp_path / "C7", ["C7_2024-06-23_083000.JPG", "C7_2024-06-23_120000.JPG", "C7_2024-06-24_083000.JPG"])
        _touch(tmp_path / "C8", ["C8_2024-06-23_083000.JPG"])

        found = discover_images(tmp_path)

        # Grouped by date first, then by camera.
        assert sorted(found) == ["2024-06-23", "2024-06-24"]
        assert sorted(found["2024-06-23"]) == ["C7", "C8"]
        assert len(found["2024-06-23"]["C7"]) == 2
        assert len(found["2024-06-24"]["C7"]) == 1

    def test_discover_images__ignores_non_matching_names(self, tmp_path: Path) -> None:
        # The camera id is read from the filename, so the on-disk folder layout is irrelevant.
        _touch(tmp_path / "nested" / "deep", ["C7_2024-06-23_083000.JPG"])
        _touch(tmp_path, ["DJI_0457.JPG", "notes.jpg", "C7_2024-06-23.JPG"])

        found = discover_images(tmp_path)

        assert list(found) == ["2024-06-23"]
        assert list(found["2024-06-23"]) == ["C7"]

    def test_discover_images__time_window(self, tmp_path: Path) -> None:
        _touch(
            tmp_path,
            [
                "C7_2024-06-23_030000.JPG",  # 03:00 — motion-triggered night frame
                "C7_2024-06-23_090000.JPG",
                "C7_2024-06-23_175959.JPG",  # last second of the 17:00 hour — kept
                "C7_2024-06-23_180000.JPG",  # 18:00 — outside
            ],
        )

        found = discover_images(tmp_path, time_window=(9, 17))

        # Night frames never align, so dropping them here keeps them out of the max_unaligned gate.
        kept = [p.name for p in found["2024-06-23"]["C7"]]
        assert kept == ["C7_2024-06-23_090000.JPG", "C7_2024-06-23_175959.JPG"]

    def test_discover_images__exclude_cameras(self, tmp_path: Path) -> None:
        _touch(
            tmp_path,
            [
                "C8_2024-06-23_090000.JPG",  # before the exclusion — kept
                "C8_2024-07-15_090000.JPG",  # excluded
                "C9_2024-07-15_090000.JPG",  # excluded
                "C7_2024-07-15_090000.JPG",  # unaffected camera
            ],
        )
        rules = [{"camera": "C8", "from": "2024-07-01"}, {"camera": "C9", "from": "2024-07-01"}]

        found = discover_images(tmp_path, exclude_cameras=rules)

        # Applied at discovery, so excluded frames never create a sensor or reach Step 1.
        assert sorted(found["2024-06-23"]) == ["C8"]
        assert sorted(found["2024-07-15"]) == ["C7"]

    def test_discover_images__empty_dir(self, tmp_path: Path) -> None:
        assert discover_images(tmp_path) == {}


class TestTransformParsing:
    """Reading the 4x4 similarity transform ASP writes to ``*-transform.txt``."""

    def test_read_4x4_matrix(self, tmp_path: Path) -> None:
        p = tmp_path / "run-transform.txt"
        p.write_text("# ASP pc_align transform\n1 0 0 10.5\n0 1 0 -3.25\n0 0 1 0.75\n0 0 0 1\n")

        T = _read_4x4_matrix(p)

        assert T.shape == (4, 4)
        np.testing.assert_allclose(T[:3, 3], [10.5, -3.25, 0.75])

    def test_read_4x4_matrix__incomplete(self, tmp_path: Path) -> None:
        p = tmp_path / "truncated.txt"
        p.write_text("1 0 0 10.5\n0 1 0 -3.25\n")

        # A truncated file must raise, not return a partial matrix that silently mis-transforms cameras.
        with pytest.raises(ValueError, match="Could not parse"):
            _read_4x4_matrix(p)


class TestNativeOutputCapture:
    """
    Metashape's C++ library writes progress bars and tie-point counts straight to file descriptors
    1 and 2, so gating Python ``print`` cannot hide them. These helpers redirect the descriptors
    themselves for the duration of a heavy call.
    """

    def test_quiet_metashape__captures_native_output(self, tmp_path: Path, capfd: pytest.CaptureFixture) -> None:
        log = tmp_path / "native.log"

        with _quiet_metashape(verbose=False, log_path=log):
            os.write(1, b"tie points: 12345\n")

        # The chatter went to the log, not to the console.
        assert "tie points: 12345" in log.read_text()
        assert "tie points" not in capfd.readouterr().out

    def test_quiet_metashape__appends_across_calls(self, tmp_path: Path) -> None:
        log = tmp_path / "native.log"

        with _quiet_metashape(verbose=False, log_path=log):
            os.write(1, b"first\n")
        with _quiet_metashape(verbose=False, log_path=log):
            os.write(1, b"second\n")

        text = log.read_text()
        assert "first" in text
        assert "second" in text

    def test_quiet_metashape__verbose_is_a_no_op(self, tmp_path: Path, capfd: pytest.CaptureFixture) -> None:
        with _quiet_metashape(verbose=True, log_path=tmp_path / "unused.log"):
            os.write(1, b"shown inline\n")

        assert "shown inline" in capfd.readouterr().out

    def test_quiet_metashape__no_log_path_discards(self, capfd: pytest.CaptureFixture) -> None:
        with _quiet_metashape(verbose=False, log_path=None):
            os.write(1, b"discarded\n")

        assert "discarded" not in capfd.readouterr().out

    def test_quiet_metashape__restores_descriptors_after_an_exception(
        self, tmp_path: Path, capfd: pytest.CaptureFixture
    ) -> None:
        # If a Metashape call raises, stdout must still come back — otherwise every later print in
        # the session vanishes into the log file.
        with pytest.raises(RuntimeError):
            with _quiet_metashape(verbose=False, log_path=tmp_path / "native.log"):
                raise RuntimeError("processing failed")

        os.write(1, b"back on console\n")
        assert "back on console" in capfd.readouterr().out

    def test_init_native_log__truncates_per_step(self, tmp_path: Path) -> None:
        log = tmp_path / "step.log"
        log.write_text("stale output from the previous run\n")

        returned = _init_native_log(verbose=False, log_path=log)

        assert returned == log
        assert log.read_text() == ""

    def test_init_native_log__creates_parent_directory(self, tmp_path: Path) -> None:
        log = tmp_path / "nested" / "deep" / "step.log"

        assert _init_native_log(verbose=False, log_path=log) == log
        assert log.exists()

    def test_init_native_log__verbose_returns_none(self, tmp_path: Path) -> None:
        # None keeps _quiet_metashape a no-op, so the full log shows inline.
        assert _init_native_log(verbose=True, log_path=tmp_path / "step.log") is None

    def test_init_native_log__no_path_returns_none(self) -> None:
        assert _init_native_log(verbose=False, log_path=None) is None


class TestLastDayMeanEop:
    """
    Mean exterior orientation per camera for the most recent day in the registry — the starting pose
    a new day's reconstruction is seeded from.
    """

    @staticmethod
    def _registry() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": ["2024-06-23"] * 2 + ["2024-07-15"] * 4,
                "label": [
                    "C7_2024-06-23_083000", "C8_2024-06-23_083000",
                    "C7_2024-07-15_083000", "C7_2024-07-15_120000",
                    "C8_2024-07-15_083000", "C8_2024-07-15_120000",
                ],
                "lon": [86.0, 86.5, 86.78, 86.80, 86.90, 86.92],
                "lat": [27.0, 27.5, 27.98, 28.00, 28.10, 28.12],
                "alt": [5000.0, 5100.0, 5400.0, 5402.0, 5500.0, 5502.0],
                "yaw": [10.0, 20.0, 30.0, 32.0, 40.0, 42.0],
                "pitch": [-5.0, -6.0, -7.0, -7.2, -8.0, -8.2],
                "roll": [1.0, 2.0, 3.0, 3.2, 4.0, 4.2],
            }
        )

    def test_last_day_mean_eop__groups_by_camera(self) -> None:
        means = _last_day_mean_eop(self._registry())

        assert set(means) == {"C7", "C8"}

    def test_last_day_mean_eop__averages_within_the_day(self) -> None:
        means = _last_day_mean_eop(self._registry())

        # C7 on 2024-07-15 has two frames at 86.78 and 86.80.
        assert means["C7"]["lon"] == pytest.approx(86.79)
        assert means["C7"]["alt"] == pytest.approx(5401.0)
        assert means["C7"]["yaw"] == pytest.approx(31.0)

    def test_last_day_mean_eop__ignores_earlier_days(self) -> None:
        # The 2024-06-23 rows must not pull the mean; only the most recent day seeds the next one.
        means = _last_day_mean_eop(self._registry())

        assert means["C8"]["lon"] == pytest.approx(86.91)

    def test_last_day_mean_eop__single_frame_day(self) -> None:
        df = self._registry().iloc[:2]

        means = _last_day_mean_eop(df)

        assert means["C7"]["lon"] == pytest.approx(86.0)


class TestLoadCalibXml:
    """Loading a Metashape XML calibration, and failing clearly when the module is absent."""

    def test_load_calib_xml__without_metashape(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import tlapse4d.metashape as ms

        # On a machine without the licensed module the import falls back to None; the error must say
        # so rather than raising AttributeError on NoneType.
        monkeypatch.setattr(ms, "Metashape", None)

        with pytest.raises(ImportError, match="Metashape Python module is not installed"):
            ms.load_calib_xml(tmp_path / "calib.xml")

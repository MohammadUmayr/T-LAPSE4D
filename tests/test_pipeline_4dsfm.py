"""
Functions to test the multi-date orchestration tools.

``run_4dsfm_day`` and ``run_4dsfm_day_with_rasters`` drive Metashape and ASP end to end and cannot run
here. What is covered is the layer above them: choosing which dates to process, and running a list of
dates without letting one failure abandon the rest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tlapse4d import pipeline_4dsfm
from tlapse4d.pipeline_4dsfm import run_batch, select_dates


def _imagery(root: Path, names: list[str]) -> Path:
    """Create standardised time-lapse filenames under *root* and return it."""
    root.mkdir(parents=True, exist_ok=True)
    for n in names:
        (root / n).touch()
    return root


def _registry(path: Path, dates: list[str]) -> Path:
    """Write a minimal reference registry listing the given reference days."""
    pd.DataFrame({"date": dates, "lon": [86.78] * len(dates)}).to_csv(path, index=False)
    return path


class TestSelectDates:
    """
    Choosing which dates to process. Reference days are the baseline the multi-temporal bundle
    adjustment aligns *against*, so reprocessing them as new dates would be circular.
    """

    def test_select_dates(self, tmp_path: Path) -> None:
        imgs = _imagery(
            tmp_path / "imgs",
            [
                "C7_2024-06-23_090000.JPG",
                "C7_2024-07-15_090000.JPG",
                "C7_2023-11-27_090000.JPG",  # the reference day
            ],
        )
        reg = _registry(tmp_path / "reg.csv", ["2023-11-27"])

        assert select_dates(imgs, reg) == ["2024-06-23", "2024-07-15"]

    def test_select_dates__excludes_every_reference_day(self, tmp_path: Path) -> None:
        imgs = _imagery(
            tmp_path / "imgs",
            ["C7_2023-11-27_090000.JPG", "C7_2023-12-05_090000.JPG", "C7_2024-06-23_090000.JPG"],
        )
        reg = _registry(tmp_path / "reg.csv", ["2023-11-27", "2023-12-05"])

        assert select_dates(imgs, reg) == ["2024-06-23"]

    def test_select_dates__registry_dates_in_a_locale_format(self, tmp_path: Path) -> None:
        # A registry opened in Excel comes back as 11/27/2023; the exclusion must still match.
        imgs = _imagery(tmp_path / "imgs", ["C7_2023-11-27_090000.JPG", "C7_2024-06-23_090000.JPG"])
        reg = tmp_path / "reg.csv"
        pd.DataFrame({"date": ["11/27/2023"], "lon": [86.78]}).to_csv(reg, index=False)

        assert select_dates(imgs, reg) == ["2024-06-23"]

    @pytest.mark.parametrize(
        "date_from,date_to,expected",
        [
            (None, None, ["2024-06-23", "2024-07-15", "2024-08-01"]),
            ("2024-07-15", None, ["2024-07-15", "2024-08-01"]),  # lower bound inclusive
            (None, "2024-07-15", ["2024-06-23", "2024-07-15"]),  # upper bound inclusive
            ("2024-07-15", "2024-07-15", ["2024-07-15"]),
            ("2030-01-01", None, []),
        ],
    )
    def test_select_dates__bounds_are_inclusive(
        self, tmp_path: Path, date_from: str | None, date_to: str | None, expected: list[str]
    ) -> None:
        imgs = _imagery(
            tmp_path / "imgs",
            ["C7_2024-06-23_090000.JPG", "C7_2024-07-15_090000.JPG", "C7_2024-08-01_090000.JPG"],
        )
        reg = _registry(tmp_path / "reg.csv", ["2023-11-27"])

        assert select_dates(imgs, reg, date_from=date_from, date_to=date_to) == expected

    def test_select_dates__time_window_drops_night_only_dates(self, tmp_path: Path) -> None:
        # A date whose only frame is a 03:00 motion trigger has nothing left once the daytime
        # window is applied. Selecting it anyway means the pipeline later raises "No images found".
        imgs = _imagery(
            tmp_path / "imgs",
            ["C7_2024-06-23_090000.JPG", "C7_2024-08-01_030000.JPG"],
        )
        reg = _registry(tmp_path / "reg.csv", ["2023-11-27"])

        assert select_dates(imgs, reg) == ["2024-06-23", "2024-08-01"]
        assert select_dates(imgs, reg, time_window=(9, 17)) == ["2024-06-23"]

    def test_select_dates__exclude_cameras_drops_emptied_dates(self, tmp_path: Path) -> None:
        # C8 is the only camera on 2024-08-01; excluding it leaves that date with no imagery.
        imgs = _imagery(
            tmp_path / "imgs",
            ["C7_2024-06-23_090000.JPG", "C8_2024-08-01_090000.JPG"],
        )
        reg = _registry(tmp_path / "reg.csv", ["2023-11-27"])

        assert select_dates(imgs, reg, exclude_cameras=[{"camera": "C8", "from": "2024-07-01"}]) == [
            "2024-06-23"
        ]

    def test_select_dates__no_imagery(self, tmp_path: Path) -> None:
        reg = _registry(tmp_path / "reg.csv", ["2023-11-27"])

        assert select_dates(_imagery(tmp_path / "imgs", []), reg) == []


class TestRunBatch:
    """
    Running a list of dates. A season is hours of processing, so one bad date -- a failed alignment
    gate, a cloudy day -- must not cost the rest. The same contract
    :func:`tlapse4d.preprocess.homogenize_images` keeps for one unreadable image in a folder.
    """

    @staticmethod
    def _result(date: str) -> dict:
        """A stand-in for what run_4dsfm_day_with_rasters returns for one date."""
        return {
            "date": date,
            "sfm": {},
            "dod_stats": {"median": -0.5, "std": 0.2},
            "stable_stats": {"median": 0.01, "std": 0.05},
            "m3c2_stats": {"median": -0.4, "std": 0.3},
        }

    def _patch(self, monkeypatch: pytest.MonkeyPatch, failing: set[str] | None = None) -> list[str]:
        """Replace the per-date pipeline with a stub; return the list it records calls in."""
        called: list[str] = []
        failing = failing or set()

        def _fake(new_date: str, **kwargs: object) -> dict:
            called.append(new_date)
            if new_date in failing:
                raise RuntimeError(f"{new_date}: alignment gate")
            return self._result(new_date)

        monkeypatch.setattr(pipeline_4dsfm, "run_4dsfm_day_with_rasters", _fake)
        return called

    def _paths(self, tmp_path: Path) -> dict:
        return dict(
            tlcam_dir=tmp_path / "imgs",
            ref_cloud=tmp_path / "ref.laz",
            glacier_mask=tmp_path / "mask.shp",
            registry_csv=tmp_path / "reg.csv",
            output_dir=tmp_path,
        )

    def test_run_batch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        called = self._patch(monkeypatch)
        dates = ["2024-06-23", "2024-07-15"]

        df = run_batch(dates, **self._paths(tmp_path))

        assert called == dates
        assert list(df["date"]) == dates
        assert set(df.columns) >= {"date", "dod_median", "stable_median", "m3c2_median"}

    def test_run_batch__one_failure_does_not_stop_the_rest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dates = ["2024-06-23", "2024-07-15", "2024-08-01"]
        called = self._patch(monkeypatch, failing={"2024-07-15"})

        df = run_batch(dates, **self._paths(tmp_path))

        # Every date was attempted, and the failure is a row rather than an exception.
        assert called == dates
        assert len(df) == 3
        failed = df[df["date"] == "2024-07-15"].iloc[0]
        assert "alignment gate" in failed["error"]

    def test_run_batch__writes_a_summary_csv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch)

        run_batch(["2024-06-23"], **self._paths(tmp_path))

        out = tmp_path / "output" / "batch_summary.csv"
        assert out.exists()
        assert list(pd.read_csv(out)["date"]) == ["2024-06-23"]

    def test_run_batch__summary_csv_path_can_be_chosen(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch)
        dest = tmp_path / "elsewhere" / "summary.csv"

        run_batch(["2024-06-23"], summary_csv=dest, **self._paths(tmp_path))

        assert dest.exists()

    def test_run_batch__params_are_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every knob the caller passes must reach the per-date pipeline untouched.
        seen: dict = {}

        def _fake(new_date: str, **kwargs: object) -> dict:
            seen.update(kwargs)
            return self._result(new_date)

        monkeypatch.setattr(pipeline_4dsfm, "run_4dsfm_day_with_rasters", _fake)

        run_batch(["2024-06-23"], **self._paths(tmp_path), res=2.0, max_unaligned=6, verbose=True)

        assert seen["res"] == 2.0
        assert seen["max_unaligned"] == 6
        assert seen["verbose"] is True

    def test_run_batch__no_dates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        called = self._patch(monkeypatch)

        df = run_batch([], **self._paths(tmp_path))

        assert called == []
        assert df.empty

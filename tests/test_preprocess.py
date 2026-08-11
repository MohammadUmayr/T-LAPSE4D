"""
Functions to test the raw-image homogenising tools.

``_exif_datetime`` shells out to ImageMagick's ``identify``, so it is stubbed: committing binary JPEGs
with crafted EXIF would test ImageMagick, not this module. Everything around it — the timestamp
conversion, unique-name assignment, skip handling and the manifest — runs for real against files on disk.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from tlapse4d import preprocess
from tlapse4d.preprocess import _read_stamp, ensure_standardized, homogenize_images


@pytest.fixture
def fake_exif(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """
    Stub ``_exif_datetime`` with a per-filename lookup, and return the dict backing it.

    A filename absent from the dict raises, which is how the "unreadable image" path is exercised.
    ``homogenize_images`` also refuses to run without ``identify`` on PATH, so that check is satisfied
    here too.
    """
    stamps: dict[str, str] = {}

    def _lookup(img_path: Path) -> str:
        try:
            return stamps[img_path.name]
        except KeyError:
            raise ValueError("no EXIF DateTimeOriginal") from None

    monkeypatch.setattr(preprocess, "_exif_datetime", _lookup)
    monkeypatch.setattr(preprocess.shutil, "which", lambda _: "/usr/bin/identify")
    return stamps


def _raw_tree(root: Path, files: list[tuple[str, str]]) -> None:
    """Create ``root/<camera>/<name>`` for each ``(camera, name)`` pair."""
    for cam, name in files:
        d = root / cam
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(b"jpeg-bytes")


def _manifest_rows(path: Any) -> list[dict[str, str]]:
    """Read the manifest CSV back as a list of row dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


class TestReadStamp:
    """Turning an EXIF capture time into the date and time parts of a standard filename."""

    def test_read_stamp(self, fake_exif: dict[str, str], tmp_path: Path) -> None:
        fake_exif["a.JPG"] = "2023:11:27 08:30:00"

        date, time, err = _read_stamp(tmp_path / "a.JPG")

        # EXIF uses colons throughout; the filename wants hyphens in the date and nothing in the time.
        assert (date, time) == ("2023-11-27", "083000")
        assert err is None

    def test_read_stamp__failure_is_returned_not_raised(self, fake_exif: dict[str, str], tmp_path: Path) -> None:
        # Parallel-safe and non-fatal: one unreadable image must not abort a 16k-frame run.
        date, time, err = _read_stamp(tmp_path / "missing.JPG")

        assert date is None
        assert time is None
        assert isinstance(err, Exception)


class TestHomogenizeImages:
    """
    Copying a raw camera tree into the standard ``<cam>_<date>_<time>.JPG`` layout. This is the single
    adapter between whatever a field team's cards look like and the one filename shape the rest of the
    library understands.
    """

    def test_homogenize_images(self, fake_exif: dict[str, str], tmp_path: Path) -> None:
        _raw_tree(tmp_path / "raw", [("C7", "DSC_0001.JPG"), ("C7", "DSC_0002.JPG")])
        fake_exif["DSC_0001.JPG"] = "2023:11:27 08:30:00"
        fake_exif["DSC_0002.JPG"] = "2023:11:27 12:00:00"
        out = tmp_path / "std"

        homogenize_images(tmp_path / "raw", output_dir=out, verbose=False)

        names = sorted(p.name for p in (out / "C7").glob("*.JPG"))
        assert names == ["C7_2023-11-27_083000.JPG", "C7_2023-11-27_120000.JPG"]

    def test_homogenize_images__originals_untouched(self, fake_exif: dict[str, str], tmp_path: Path) -> None:
        # The raw tree is copied, never moved — the field data is the only irreplaceable thing here.
        _raw_tree(tmp_path / "raw", [("C7", "DSC_0001.JPG")])
        fake_exif["DSC_0001.JPG"] = "2023:11:27 08:30:00"

        homogenize_images(tmp_path / "raw", output_dir=tmp_path / "std", verbose=False)

        assert (tmp_path / "raw" / "C7" / "DSC_0001.JPG").exists()

    def test_homogenize_images__finds_nested_card_folders(self, fake_exif: dict[str, str], tmp_path: Path) -> None:
        # EK-card subfolders nest arbitrarily deep, so the search has to be recursive.
        deep = tmp_path / "raw" / "C7" / "100EK113" / "sub"
        deep.mkdir(parents=True)
        (deep / "DSC_0001.JPG").write_bytes(b"jpeg-bytes")
        fake_exif["DSC_0001.JPG"] = "2023:11:27 08:30:00"
        out = tmp_path / "std"

        homogenize_images(tmp_path / "raw", output_dir=out, verbose=False)

        assert (out / "C7" / "C7_2023-11-27_083000.JPG").exists()

    def test_homogenize_images__same_second_collisions(self, fake_exif: dict[str, str], tmp_path: Path) -> None:
        # Three frames captured in the same second: the old script overwrote from the third clash on.
        _raw_tree(tmp_path / "raw", [("C7", "a.JPG"), ("C7", "b.JPG"), ("C7", "c.JPG")])
        for n in ("a.JPG", "b.JPG", "c.JPG"):
            fake_exif[n] = "2023:11:27 08:30:00"
        out = tmp_path / "std"

        summary = homogenize_images(tmp_path / "raw", output_dir=out, verbose=False)

        names = sorted(p.name for p in (out / "C7").glob("*.JPG"))
        assert names == [
            "C7_2023-11-27_083000.JPG",
            "C7_2023-11-27_083000_1.JPG",
            "C7_2023-11-27_083000_2.JPG",
        ]
        assert summary["C7"]["copied"] == 3

    def test_homogenize_images__unreadable_images_are_skipped(self, fake_exif: dict[str, str], tmp_path: Path) -> None:
        _raw_tree(tmp_path / "raw", [("C7", "good.JPG"), ("C7", "corrupt.JPG")])
        fake_exif["good.JPG"] = "2023:11:27 08:30:00"  # corrupt.JPG has no stamp

        summary = homogenize_images(tmp_path / "raw", output_dir=tmp_path / "std", verbose=False)

        assert summary["C7"] == {"copied": 1, "skipped": 1}

    def test_homogenize_images__manifest(self, fake_exif: dict[str, str], tmp_path: Path) -> None:
        _raw_tree(tmp_path / "raw", [("C7", "good.JPG"), ("C7", "corrupt.JPG")])
        fake_exif["good.JPG"] = "2023:11:27 08:30:00"

        summary = homogenize_images(tmp_path / "raw", output_dir=tmp_path / "std", verbose=False)

        # Every image is recorded, copied or not, so the mapping back to the original is never lost.
        rows = _manifest_rows(summary["manifest"])
        assert len(rows) == 2
        assert sorted(r["status"].split(":")[0] for r in rows) == ["copied", "skipped"]

        copied = next(r for r in rows if r["status"] == "copied")
        assert copied["standard_name"] == "C7_2023-11-27_083000.JPG"
        assert copied["camera"] == "C7"
        assert copied["datetime"] == "2023-11-27 083000"

    def test_homogenize_images__cameras_auto_detected(self, fake_exif: dict[str, str], tmp_path: Path) -> None:
        # Each immediate sub-directory is a camera, and its folder name becomes the camera id.
        _raw_tree(tmp_path / "raw", [("C7", "a.JPG"), ("C8", "b.JPG")])
        fake_exif["a.JPG"] = "2023:11:27 08:30:00"
        fake_exif["b.JPG"] = "2023:11:27 08:30:00"
        out = tmp_path / "std"

        summary = homogenize_images(tmp_path / "raw", output_dir=out, verbose=False)

        assert set(summary) == {"C7", "C8", "manifest"}
        assert (out / "C7" / "C7_2023-11-27_083000.JPG").exists()
        assert (out / "C8" / "C8_2023-11-27_083000.JPG").exists()

    def test_homogenize_images__cameras_argument(self, fake_exif: dict[str, str], tmp_path: Path) -> None:
        _raw_tree(tmp_path / "raw", [("C7", "a.JPG"), ("C8", "b.JPG")])
        fake_exif["a.JPG"] = "2023:11:27 08:30:00"
        fake_exif["b.JPG"] = "2023:11:27 08:30:00"

        summary = homogenize_images(tmp_path / "raw", output_dir=tmp_path / "std", cameras=["C7"], verbose=False)

        assert set(summary) == {"C7", "manifest"}

    def test_homogenize_images__flat_output(self, fake_exif: dict[str, str], tmp_path: Path) -> None:
        # The camera id lives in the filename, so the per-camera folder split is only for tidiness.
        _raw_tree(tmp_path / "raw", [("C7", "a.JPG")])
        fake_exif["a.JPG"] = "2023:11:27 08:30:00"
        out = tmp_path / "std"

        homogenize_images(tmp_path / "raw", output_dir=out, per_camera_subdir=False, verbose=False)

        assert (out / "C7_2023-11-27_083000.JPG").exists()
        assert not (out / "C7").exists()

    def test_homogenize_images__default_output_is_a_sibling(self, fake_exif: dict[str, str], tmp_path: Path) -> None:
        # The default sits beside the raw tree, never inside it, so it is not rescanned as a camera.
        _raw_tree(tmp_path / "raw", [("C7", "a.JPG")])
        fake_exif["a.JPG"] = "2023:11:27 08:30:00"

        homogenize_images(tmp_path / "raw", verbose=False)

        assert (tmp_path / "raw_renamed" / "C7" / "C7_2023-11-27_083000.JPG").exists()

    def test_homogenize_images__independent_of_worker_count(self, fake_exif: dict[str, str], tmp_path: Path) -> None:
        # Name assignment is deliberately serial, so n_jobs must not change the output at all.
        for n in ("a.JPG", "b.JPG", "c.JPG"):
            fake_exif[n] = "2023:11:27 08:30:00"

        results = []
        for jobs, tag in ((1, "serial"), (8, "parallel")):
            raw = tmp_path / tag / "raw"
            _raw_tree(raw, [("C7", "a.JPG"), ("C7", "b.JPG"), ("C7", "c.JPG")])
            out = tmp_path / tag / "std"
            homogenize_images(raw, output_dir=out, n_jobs=jobs, verbose=False)
            results.append(sorted(p.name for p in (out / "C7").glob("*.JPG")))

        assert results[0] == results[1]

    def test_homogenize_images__missing_identify(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Without ImageMagick there are no timestamps at all, so fail early with a usable message.
        monkeypatch.setattr(preprocess.shutil, "which", lambda _: None)

        with pytest.raises(RuntimeError, match="identify"):
            homogenize_images(tmp_path, verbose=False)


class TestEnsureStandardized:
    """
    The idempotent front door for a notebook setup cell: it decides what to do by looking at filenames,
    so re-running the cell is cheap and never redoes a completed homogenise.
    """

    def test_ensure_standardized__already_standard(self, tmp_path: Path) -> None:
        d = tmp_path / "images"
        d.mkdir()
        (d / "C7_2023-11-27_083000.JPG").write_bytes(b"jpeg-bytes")

        # Nothing to do — the directory is returned unchanged.
        assert ensure_standardized(d, verbose=False) == d

    def test_ensure_standardized__reuses_completed_run(self, tmp_path: Path) -> None:
        raw = tmp_path / "images"
        _raw_tree(raw, [("C7", "DSC_0001.JPG")])

        # A previous run left standard images and a manifest in the sibling folder.
        renamed = tmp_path / "images_renamed"
        (renamed / "C7").mkdir(parents=True)
        (renamed / "C7" / "C7_2023-11-27_083000.JPG").write_bytes(b"jpeg-bytes")
        (renamed / "manifest.csv").write_text("camera\n")

        assert ensure_standardized(raw, verbose=False) == renamed

    def test_ensure_standardized__runs_homogenise(self, fake_exif: dict[str, str], tmp_path: Path) -> None:
        raw = tmp_path / "images"
        _raw_tree(raw, [("C7", "DSC_0001.JPG")])
        fake_exif["DSC_0001.JPG"] = "2023:11:27 08:30:00"

        out = ensure_standardized(raw, verbose=False)

        assert out == tmp_path / "images_renamed"
        assert (out / "C7" / "C7_2023-11-27_083000.JPG").exists()


class TestExifDatetime:
    """
    The ImageMagick call itself. ``subprocess.run`` is stubbed rather than shipping binary JPEGs with
    crafted EXIF, which would test ImageMagick instead of this wrapper.
    """

    def test_exif_datetime(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from tlapse4d.preprocess import _exif_datetime

        class _Result:
            stdout = "2023:11:27 08:30:00\n"

        monkeypatch.setattr(preprocess.subprocess, "run", lambda *a, **k: _Result())

        assert _exif_datetime(tmp_path / "a.JPG") == "2023:11:27 08:30:00"

    def test_exif_datetime__empty_output_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from tlapse4d.preprocess import _exif_datetime

        class _Result:
            stdout = "   \n"

        monkeypatch.setattr(preprocess.subprocess, "run", lambda *a, **k: _Result())

        # An image with no capture time cannot be named; the caller turns this into a skip.
        with pytest.raises(ValueError, match="no EXIF DateTimeOriginal"):
            _exif_datetime(tmp_path / "a.JPG")


class TestCopyTask:
    """The per-file copy, which reports failures rather than raising so one bad file is survivable."""

    def test_copy_task(self, tmp_path: Path) -> None:
        from tlapse4d.preprocess import _copy_task

        src = tmp_path / "src.JPG"
        src.write_bytes(b"jpeg-bytes")
        dst = tmp_path / "dst.JPG"

        task, err = _copy_task((src, dst, "2023-11-27 083000"))

        assert err is None
        assert dst.read_bytes() == b"jpeg-bytes"

    def test_copy_task__failure_is_returned(self, tmp_path: Path) -> None:
        from tlapse4d.preprocess import _copy_task

        # Copying into a directory that does not exist fails; the error comes back in the tuple.
        src = tmp_path / "src.JPG"
        src.write_bytes(b"jpeg-bytes")

        _, err = _copy_task((src, tmp_path / "missing" / "dst.JPG", ""))

        assert isinstance(err, Exception)


class TestHomogenizeEdgeCases:
    """Failure and reporting paths that a normal run never reaches."""

    def test_homogenize_images__copy_failure_is_recorded(
        self, fake_exif: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _raw_tree(tmp_path / "raw", [("C7", "a.JPG")])
        fake_exif["a.JPG"] = "2023:11:27 08:30:00"

        def _boom(src, dst):
            raise PermissionError("read-only filesystem")

        monkeypatch.setattr(preprocess.shutil, "copyfile", _boom)

        summary = homogenize_images(tmp_path / "raw", output_dir=tmp_path / "std", verbose=False)

        assert summary["C7"] == {"copied": 0, "skipped": 1}
        rows = _manifest_rows(summary["manifest"])
        assert "copy failed" in rows[0]["status"]

    def test_homogenize_images__non_directory_entry_is_skipped(
        self, fake_exif: dict[str, str], tmp_path: Path
    ) -> None:
        # A stray file sitting beside the camera folders must not be treated as a camera.
        raw = tmp_path / "raw"
        _raw_tree(raw, [("C7", "a.JPG")])
        (raw / "notes.txt").write_text("field notes")
        fake_exif["a.JPG"] = "2023:11:27 08:30:00"

        summary = homogenize_images(raw, output_dir=tmp_path / "std", cameras=["C7", "notes.txt"], verbose=False)

        assert set(summary) == {"C7", "manifest"}

    def test_homogenize_images__verbose_reports_progress(
        self, fake_exif: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _raw_tree(tmp_path / "raw", [("C7", "good.JPG"), ("C7", "corrupt.JPG")])
        fake_exif["good.JPG"] = "2023:11:27 08:30:00"

        homogenize_images(tmp_path / "raw", output_dir=tmp_path / "std", verbose=True)

        out = capsys.readouterr().out
        assert "copied 1, skipped 1" in out
        assert "Manifest" in out

    def test_ensure_standardized__warns_about_an_incomplete_previous_run(
        self, fake_exif: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        # A _renamed folder with no manifest is a half-finished run; silently reusing it would
        # process an incomplete image set.
        raw = tmp_path / "images"
        _raw_tree(raw, [("C7", "DSC_0001.JPG")])
        fake_exif["DSC_0001.JPG"] = "2023:11:27 08:30:00"
        (tmp_path / "images_renamed").mkdir()

        ensure_standardized(raw, verbose=True)

        assert "looks incomplete" in capsys.readouterr().out

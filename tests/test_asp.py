"""
Functions to test the Ames Stereo Pipeline wrapper tools.

``pc_align_stage``, ``pc_align_p2p_sp2p`` and ``point2dem`` shell out to ASP binaries and cannot run
without them installed, so they are out of scope. Everything else in the module is pure Python and is
covered here: parsing the 4x4 transform ASP writes, the coordinate conversions, applying a transform to
clouds and camera positions, and the before/after co-registration evaluation.

Sign and axis errors in this module are the dangerous kind — they displace every camera or every point
in the project while producing output that looks entirely plausible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import UTM45N

from cntp.asp import (
    _apply_transform_to_las,
    _check_asp,
    _read_asp_transform,
    _run_command,
    apply_coreg_to_cameras,
    apply_coreg_to_cameras_ecef,
    evaluate_coreg,
    extract_stable_reference,
    las_utm_to_ecef,
    utm_to_wgs84,
    wgs84_to_utm,
)

# Changri Nup, Khumbu — inside UTM zone 45N.
LON, LAT, ALT = 86.78, 27.98, 5400.0


def _write_transform(path: Path, matrix: np.ndarray, header: str = "# ASP pc_align transform\n") -> Path:
    """Write *matrix* in the layout ASP uses: one header line, then four rows of four floats."""
    body = "\n".join(" ".join(f"{float(v):.12g}" for v in row) for row in matrix)
    path.write_text(header + body + "\n")
    return path


def _to_utm(df: pd.DataFrame) -> np.ndarray:
    """Project a camera DataFrame's Lon/Lat/Alt columns into UTM metres as an Nx3 array."""
    return np.column_stack(
        wgs84_to_utm(df["Lon"].to_numpy(), df["Lat"].to_numpy(), df["Alt"].to_numpy(), UTM45N)
    )


@pytest.fixture
def cameras_csv(tmp_path: Path) -> Path:
    """Write the camera-positions CSV the Metashape pipeline produces, and return its path."""
    path = tmp_path / "cameras.csv"
    pd.DataFrame(
        {
            "Label": ["C7_2024-06-23_083000", "C8_2024-06-23_083000"],
            "Lon": [LON, LON + 0.001],
            "Lat": [LAT, LAT + 0.001],
            "Alt": [ALT, ALT + 10.0],
            "Yaw": [12.0, 34.0],
            "Pitch": [-5.0, -6.0],
            "Roll": [1.0, 2.0],
        }
    ).to_csv(path, index=False)
    return path


class TestReadAspTransform:
    """
    Parsing ``*-transform.txt``. Each stage passes its output to the next as ``--initial-transform``,
    so ASP compounds them internally and Stage 3's file already holds the full composition T3 . T2 . T1.
    """

    def test_read_asp_transform(self, tmp_path: Path) -> None:
        T = np.eye(4)
        T[:3, 3] = [10.5, -3.25, 0.75]

        parsed = _read_asp_transform(_write_transform(tmp_path / "t.txt", T))

        assert parsed.shape == (4, 4)
        np.testing.assert_allclose(parsed, T)

    def test_read_asp_transform__ignores_non_numeric_rows(self, tmp_path: Path) -> None:
        # An ASP header can itself carry four words; those must not be read as a matrix row.
        p = tmp_path / "t.txt"
        p.write_text("this header has four\n1 0 0 10.5\n0 1 0 -3.25\n0 0 1 0.75\n0 0 0 1\n")

        np.testing.assert_allclose(_read_asp_transform(p)[:3, 3], [10.5, -3.25, 0.75])

    def test_read_asp_transform__stops_at_first_matrix(self, tmp_path: Path) -> None:
        # Stage files can carry trailing diagnostics; only the first four numeric rows count.
        p = tmp_path / "t.txt"
        p.write_text("1 0 0 1\n0 1 0 2\n0 0 1 3\n0 0 0 1\n9 9 9 9\n")

        np.testing.assert_allclose(_read_asp_transform(p)[:3, 3], [1, 2, 3])

    def test_read_asp_transform__truncated(self, tmp_path: Path) -> None:
        p = tmp_path / "t.txt"
        p.write_text("1 0 0 10.5\n0 1 0 -3.25\n")

        with pytest.raises(ValueError, match="Could not parse"):
            _read_asp_transform(p)


class TestCoordinateConversions:
    """WGS84 to UTM and back — the projection step every camera position passes through."""

    def test_wgs84_to_utm__roundtrip(self) -> None:
        e, n, z = wgs84_to_utm(LON, LAT, ALT, UTM45N)

        lon, lat, alt = utm_to_wgs84(e, n, z, UTM45N)

        assert lon == pytest.approx(LON, abs=1e-9)
        assert lat == pytest.approx(LAT, abs=1e-9)
        assert alt == pytest.approx(ALT)

    def test_wgs84_to_utm__plausible_for_zone_45n(self) -> None:
        e, n, _ = wgs84_to_utm(LON, LAT, ALT, UTM45N)

        assert 100_000 < e < 900_000  # valid easting band
        assert 0 < n < 10_000_000  # northern hemisphere

    def test_wgs84_to_utm__altitude_passes_through(self) -> None:
        # Only the horizontal components are projected; ellipsoidal height is carried unchanged.
        _, _, z = wgs84_to_utm(LON, LAT, ALT, UTM45N)

        assert z == pytest.approx(ALT)

    def test_wgs84_to_utm__vectorised(self) -> None:
        e, n, z = wgs84_to_utm(np.full(5, LON), np.full(5, LAT), np.full(5, ALT), UTM45N)

        assert e.shape == n.shape == z.shape == (5,)
        np.testing.assert_allclose(e, e[0])

    def test_utm_to_wgs84__easting_increases_longitude(self) -> None:
        # Sign check: a positive easting step must move east, not west.
        e, n, z = wgs84_to_utm(LON, LAT, ALT, UTM45N)

        lon_shifted, _, _ = utm_to_wgs84(e + 100.0, n, z, UTM45N)

        assert lon_shifted > LON


class TestApplyCoregToCameras:
    """
    Applying the ASP co-registration transform to camera positions. Positions are converted to UTM,
    transformed by the 4x4 similarity, and converted back; orientation is passed through untouched.
    """

    def test_apply_coreg_to_cameras__identity(self, tmp_path: Path, cameras_csv: Path) -> None:
        t = _write_transform(tmp_path / "t.txt", np.eye(4))
        original = pd.read_csv(cameras_csv)

        out = apply_coreg_to_cameras(cameras_csv, t, UTM45N)

        np.testing.assert_allclose(out["Lon"], original["Lon"], atol=1e-9)
        np.testing.assert_allclose(out["Lat"], original["Lat"], atol=1e-9)
        np.testing.assert_allclose(out["Alt"], original["Alt"], atol=1e-6)

    def test_apply_coreg_to_cameras__translation(self, tmp_path: Path, cameras_csv: Path) -> None:
        shift = np.array([10.0, -5.0, 2.0])
        T = np.eye(4)
        T[:3, 3] = shift
        t = _write_transform(tmp_path / "t.txt", T)
        original = pd.read_csv(cameras_csv)

        out = apply_coreg_to_cameras(cameras_csv, t, UTM45N)

        # The shift is in UTM metres, so it has to be measured back in UTM, not in degrees.
        np.testing.assert_allclose(_to_utm(out) - _to_utm(original), np.tile(shift, (2, 1)), atol=1e-6)

    def test_apply_coreg_to_cameras__scale_is_applied(self, tmp_path: Path, cameras_csv: Path) -> None:
        # pc_align solves a similarity transform, so the rotation block can carry scale. Dropping it
        # would leave the cloud and the cameras on subtly different scales.
        T = np.eye(4)
        T[:3, :3] *= 1.0001
        t = _write_transform(tmp_path / "t.txt", T)
        original = pd.read_csv(cameras_csv)

        out = apply_coreg_to_cameras(cameras_csv, t, UTM45N)

        np.testing.assert_allclose(_to_utm(out), _to_utm(original) * 1.0001, rtol=1e-9)

    def test_apply_coreg_to_cameras__orientation_passes_through(self, tmp_path: Path, cameras_csv: Path) -> None:
        T = np.eye(4)
        T[:3, 3] = [10.0, -5.0, 2.0]
        t = _write_transform(tmp_path / "t.txt", T)
        original = pd.read_csv(cameras_csv)

        out = apply_coreg_to_cameras(cameras_csv, t, UTM45N)

        # Only positions are transformed — yaw, pitch, roll and the labels are untouched.
        for col in ("Yaw", "Pitch", "Roll"):
            np.testing.assert_allclose(out[col], original[col])
        assert list(out["Label"]) == list(original["Label"])

    def test_apply_coreg_to_cameras__source_csv_unmodified(self, tmp_path: Path, cameras_csv: Path) -> None:
        T = np.eye(4)
        T[:3, 3] = [10.0, -5.0, 2.0]
        t = _write_transform(tmp_path / "t.txt", T)

        apply_coreg_to_cameras(cameras_csv, t, UTM45N)

        # The function returns a copy; re-running it must not compound the transform on disk.
        assert pd.read_csv(cameras_csv)["Lon"].iloc[0] == pytest.approx(LON)

    def test_apply_coreg_to_cameras__writes_out_csv(self, tmp_path: Path, cameras_csv: Path) -> None:
        t = _write_transform(tmp_path / "t.txt", np.eye(4))
        dest = tmp_path / "cameras_coreg.csv"

        out = apply_coreg_to_cameras(cameras_csv, t, UTM45N, out_csv=dest)

        assert dest.exists()
        np.testing.assert_allclose(pd.read_csv(dest)["Lon"], out["Lon"])


class TestAspAvailability:
    """The guard that turns a missing ASP install into a message rather than a confusing failure."""

    def test_check_asp__raises_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cntp.asp as asp_mod

        monkeypatch.setattr(asp_mod.shutil, "which", lambda _: None)

        with pytest.raises(RuntimeError, match="pc_align not found on PATH"):
            _check_asp()

    def test_check_asp__passes_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cntp.asp as asp_mod

        monkeypatch.setattr(asp_mod.shutil, "which", lambda _: "/opt/asp/bin/pc_align")

        _check_asp()  # must not raise


class TestRunCommand:
    """The subprocess wrapper every ASP call goes through."""

    def test_run_command__returns_stdout(self) -> None:
        out = _run_command(["echo", "hello"])

        assert "hello" in out

    def test_run_command__raises_on_nonzero_exit(self) -> None:
        # A failed ASP stage must abort loudly, with the tool's own output attached.
        with pytest.raises(RuntimeError, match="Command failed"):
            _run_command(["python", "-c", "import sys; sys.exit(3)"])

    def test_run_command__verbose_filters_progress_bars(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        # ASP's "--> [****....]" progress lines flood the log and are dropped. The emitting script
        # lives in a file rather than -c, so the marker is not echoed as part of the command line.
        script = tmp_path / "noisy.py"
        script.write_text("print('--> [****....]')\nprint('real output')\n")

        _run_command(["python", script], verbose=True)

        captured = capsys.readouterr().out
        assert "real output" in captured
        assert "--> [" not in captured

    def test_run_command__accepts_non_string_arguments(self, tmp_path: Path) -> None:
        # Callers pass Paths and numbers directly; they are coerced before exec.
        out = _run_command(["python", "-c", "print(1)"])

        assert "1" in out


class TestLasUtmToEcef:
    """
    Reprojecting a UTM cloud into ECEF so pc_align runs ICP in true Earth-centred space rather than
    on a projected plane, where distances are distorted.
    """

    def test_las_utm_to_ecef(self, tmp_path: Path, write_cloud_las) -> None:
        import laspy

        src = write_cloud_las("utm.las")
        out = las_utm_to_ecef(src, UTM45N, tmp_path / "ecef.las")

        assert out.exists()
        with laspy.open(src) as a, laspy.open(out) as b:
            assert a.header.point_count == b.header.point_count

    def test_las_utm_to_ecef__coordinates_are_geocentric(self, tmp_path: Path, write_cloud_las) -> None:
        from pyproj import Transformer

        src = write_cloud_las("utm.las")
        out = las_utm_to_ecef(src, UTM45N, tmp_path / "ecef.las")

        from cntp.io import load_las

        utm_pts = load_las(src)
        ecef_pts = load_las(out)
        t = Transformer.from_crs(f"EPSG:{UTM45N}", "EPSG:4978", always_xy=True)
        ex, ey, ez = t.transform(utm_pts[0, 0], utm_pts[0, 1], utm_pts[0, 2])

        # Fixed 0.01 m scale, so agreement to a couple of centimetres is the storable precision.
        assert np.min(np.abs(ecef_pts[:, 0] - ex)) < 0.02
        assert np.min(np.abs(ecef_pts[:, 1] - ey)) < 0.02
        assert np.min(np.abs(ecef_pts[:, 2] - ez)) < 0.02

    def test_las_utm_to_ecef__creates_parent_directory(self, tmp_path: Path, write_cloud_las) -> None:
        src = write_cloud_las("utm.las")

        out = las_utm_to_ecef(src, UTM45N, tmp_path / "nested" / "deep" / "ecef.las")

        assert out.exists()


class TestApplyTransformToLas:
    """
    Applying the solved transform to a whole cloud. Streams in chunks so full-resolution clouds do not
    exhaust memory, and rotates the normals by the rotation part so slope stays meaningful afterwards.
    """

    def test_apply_transform_to_las__translation(self, tmp_path: Path, write_cloud_las) -> None:
        from cntp.io import load_las

        src = write_cloud_las("cloud.las")
        T = np.eye(4)
        T[:3, 3] = [10.0, -5.0, 2.0]
        out = tmp_path / "moved.las"

        _apply_transform_to_las(src, T, out)

        before = load_las(src)
        after = load_las(out)
        np.testing.assert_allclose(after[:, :3].mean(axis=0) - before[:, :3].mean(axis=0),
                                   [10.0, -5.0, 2.0], atol=1e-2)

    def test_apply_transform_to_las__identity_is_a_no_op(self, tmp_path: Path, write_cloud_las) -> None:
        from cntp.io import load_las

        src = write_cloud_las("cloud.las")
        out = tmp_path / "same.las"

        _apply_transform_to_las(src, np.eye(4), out)

        np.testing.assert_allclose(load_las(out)[:, :3], load_las(src)[:, :3], atol=1e-3)

    def test_apply_transform_to_las__normals_are_rotated_not_scaled(
        self, tmp_path: Path, write_cloud_las
    ) -> None:
        from cntp.io import load_las

        # A similarity transform carries scale, but normals must stay unit length or every downstream
        # slope calculation shifts.
        src = write_cloud_las("cloud.las")
        T = np.eye(4)
        T[:3, :3] *= 2.0
        out = tmp_path / "scaled.las"

        _apply_transform_to_las(src, T, out)

        norms = np.linalg.norm(load_las(out)[:, 6:9], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_apply_transform_to_las__rotation_changes_normal_direction(
        self, tmp_path: Path, write_cloud_las
    ) -> None:
        from cntp.io import load_las

        src = write_cloud_las("cloud.las")
        # 90 degrees about Z.
        T = np.eye(4)
        T[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        out = tmp_path / "rot.las"

        _apply_transform_to_las(src, T, out)

        before = load_las(src)
        after = load_las(out)
        np.testing.assert_allclose(after[:, 6], -before[:, 7], atol=1e-5)
        np.testing.assert_allclose(after[:, 7], before[:, 6], atol=1e-5)

    def test_apply_transform_to_las__ecef_mode_roundtrips(self, tmp_path: Path, write_cloud_las) -> None:
        from cntp.io import load_las

        # In ECEF mode the cloud goes UTM -> ECEF -> T -> UTM, so identity must land back where it
        # started; the output stays in UTM so M3C2 and the rasterisers work unchanged.
        src = write_cloud_las("cloud.las")
        out = tmp_path / "ecef_identity.las"

        _apply_transform_to_las(src, np.eye(4), out, utm_epsg=UTM45N)

        np.testing.assert_allclose(load_las(out)[:, :3], load_las(src)[:, :3], atol=1e-2)


class TestApplyCoregToCamerasEcef:
    """The ECEF-mode camera update, used when pc_align was run on ECEF clouds."""

    def test_apply_coreg_to_cameras_ecef__identity(self, tmp_path: Path, cameras_csv: Path) -> None:
        t = _write_transform(tmp_path / "t.txt", np.eye(4))
        original = pd.read_csv(cameras_csv)

        out = apply_coreg_to_cameras_ecef(cameras_csv, t)

        np.testing.assert_allclose(out["Lon"], original["Lon"], atol=1e-9)
        np.testing.assert_allclose(out["Lat"], original["Lat"], atol=1e-9)
        np.testing.assert_allclose(out["Alt"], original["Alt"], atol=1e-6)

    def test_apply_coreg_to_cameras_ecef__translation_moves_positions(
        self, tmp_path: Path, cameras_csv: Path
    ) -> None:
        from pyproj import Transformer

        shift = np.array([50.0, 50.0, 50.0])  # metres in ECEF
        T = np.eye(4)
        T[:3, 3] = shift
        t = _write_transform(tmp_path / "t.txt", T)
        original = pd.read_csv(cameras_csv)

        out = apply_coreg_to_cameras_ecef(cameras_csv, t)

        # Measured back in ECEF metres, not in degrees: a 50 m shift is under the default rtol of
        # np.allclose at longitude 86, so comparing degrees would call an unmoved camera "moved".
        to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
        before = np.column_stack(
            to_ecef.transform(original["Lon"], original["Lat"], original["Alt"])
        )
        after = np.column_stack(to_ecef.transform(out["Lon"], out["Lat"], out["Alt"]))

        np.testing.assert_allclose(after - before, np.tile(shift, (2, 1)), atol=1e-4)

    def test_apply_coreg_to_cameras_ecef__utm_mode_matches_the_utm_function(
        self, tmp_path: Path, cameras_csv: Path
    ) -> None:
        # Passing utm_epsg switches this function into flat UTM mode, which must agree exactly with
        # the dedicated UTM entry point.
        T = np.eye(4)
        T[:3, 3] = [10.0, -5.0, 2.0]
        t = _write_transform(tmp_path / "t.txt", T)

        via_ecef_fn = apply_coreg_to_cameras_ecef(cameras_csv, t, utm_epsg=UTM45N)
        via_utm_fn = apply_coreg_to_cameras(cameras_csv, t, UTM45N)

        np.testing.assert_allclose(via_ecef_fn["Lon"], via_utm_fn["Lon"])
        np.testing.assert_allclose(via_ecef_fn["Lat"], via_utm_fn["Lat"])

    def test_apply_coreg_to_cameras_ecef__orientation_passes_through(
        self, tmp_path: Path, cameras_csv: Path
    ) -> None:
        T = np.eye(4)
        T[:3, 3] = [50.0, 50.0, 50.0]
        t = _write_transform(tmp_path / "t.txt", T)
        original = pd.read_csv(cameras_csv)

        out = apply_coreg_to_cameras_ecef(cameras_csv, t)

        for col in ("Yaw", "Pitch", "Roll"):
            np.testing.assert_allclose(out[col], original[col])

    def test_apply_coreg_to_cameras_ecef__writes_out_csv(self, tmp_path: Path, cameras_csv: Path) -> None:
        t = _write_transform(tmp_path / "t.txt", np.eye(4))
        dest = tmp_path / "out.csv"

        apply_coreg_to_cameras_ecef(cameras_csv, t, out_csv=dest)

        assert dest.exists()


class TestExtractStableReference:
    """
    Building the glacier-masked, slope-filtered reference cloud that Stage 3 ICP aligns against —
    the equivalent of HSfM's ``mask_dem``, removing unstable surfaces before the final pass.
    """

    def test_extract_stable_reference(self, tmp_path: Path, write_cloud_las) -> None:
        from cntp.io import load_las

        src = write_cloud_las("ref.las")

        out = extract_stable_reference(src, tmp_path / "cache")

        assert out.name == "ref_stable.las"
        # The cloud fixture is built at 75 degrees on rock, so every point survives both filters.
        assert len(load_las(out)) > 0

    def test_extract_stable_reference__slope_threshold_filters(self, tmp_path: Path, cloud: np.ndarray) -> None:
        from cntp.io import save_las

        # The shared cloud fixture sits at 75 degrees, so an 80 degree threshold must empty it.
        src = tmp_path / "shallow.las"
        save_las(cloud, src)

        with pytest.raises(ValueError):
            # save_las cannot write an empty cloud — the filter removed everything, which is the
            # signal that the threshold is wrong for this terrain.
            extract_stable_reference(src, tmp_path / "cache", slope_threshold=80.0)

    def test_extract_stable_reference__glacier_mask_applied(self, tmp_path: Path, write_cloud_las) -> None:
        import geopandas as gpd
        from conftest import GRID_SHAPE, GRID_TRANSFORM
        from shapely.geometry import box

        from cntp.io import load_las

        src = write_cloud_las("ref.las")
        # Mask the western half of the footprint.
        x0 = GRID_TRANSFORM.c
        y0 = GRID_TRANSFORM.f - GRID_SHAPE[0]
        mask = tmp_path / "glacier.geojson"
        gpd.GeoDataFrame(
            geometry=[box(x0 - 1, y0 - 1, x0 + GRID_SHAPE[1] / 2, y0 + GRID_SHAPE[0] + 1)],
            crs=f"EPSG:{UTM45N}",
        ).to_file(mask)

        out = extract_stable_reference(src, tmp_path / "cache", glacier_mask_path=mask)

        assert np.all(load_las(out)[:, 0] >= x0 + GRID_SHAPE[1] / 2 - 1e-6)

    def test_extract_stable_reference__is_cached(self, tmp_path: Path, write_cloud_las) -> None:
        src = write_cloud_las("ref.las")
        cache = tmp_path / "cache"
        out = extract_stable_reference(src, cache)
        mtime = out.stat().st_mtime_ns

        extract_stable_reference(src, cache)

        assert out.stat().st_mtime_ns == mtime

    def test_extract_stable_reference__writes_diagnostic_plots(self, tmp_path: Path, write_cloud_las) -> None:
        src = write_cloud_las("ref.las")
        plots = tmp_path / "plots"

        extract_stable_reference(src, tmp_path / "cache", plot_dir=plots)

        assert (plots / "ndwi_vs_intensity.png").exists()
        assert (plots / "stable_terrain_rgb.png").exists()


class TestEvaluateCoreg:
    """
    The before/after co-registration report. This is where the NMAD the whole pipeline is judged on
    comes from, so the direction of improvement is asserted explicitly.
    """

    def test_evaluate_coreg(self, tmp_path: Path, write_cloud_las) -> None:
        ref = write_cloud_las("ref.las")
        before = write_cloud_las("before.las", dz=1.0, seed=7)
        after = write_cloud_las("after.las", dz=0.0, seed=7)

        out = evaluate_coreg(ref, before, after, ref_downsample_factor=1.0)

        assert set(out) >= {
            "med_before", "nmad_before", "std_before", "dist_before",
            "med_after", "nmad_after", "std_after", "dist_after",
            "stable_tba_after", "ref_stable",
        }

    def test_evaluate_coreg__reports_the_improvement(self, tmp_path: Path, write_cloud_las) -> None:
        # 'before' sits a metre above the reference; 'after' is aligned. The median must reflect that.
        ref = write_cloud_las("ref.las")
        before = write_cloud_las("before.las", dz=1.0, seed=7)
        after = write_cloud_las("after.las", dz=0.0, seed=7)

        out = evaluate_coreg(ref, before, after, ref_downsample_factor=1.0)

        assert abs(out["med_after"]) < abs(out["med_before"])

    def test_evaluate_coreg__uses_a_prebuilt_stable_reference(self, tmp_path: Path, write_cloud_las) -> None:
        # Passing the cached stable reference skips the reference load and KDTree filter entirely.
        ref = write_cloud_las("ref.las")
        day = write_cloud_las("day.las", seed=7)
        stable_ref = extract_stable_reference(ref, tmp_path / "cache")

        out = evaluate_coreg(None, day, day, stable_ref_las=stable_ref)

        assert np.isfinite(out["med_after"])

    def test_evaluate_coreg__writes_stable_cloud(self, tmp_path: Path, write_cloud_las) -> None:
        ref = write_cloud_las("ref.las")
        day = write_cloud_las("day.las", seed=7)
        stable_dir = tmp_path / "stable"

        evaluate_coreg(ref, day, day, ref_downsample_factor=1.0, stable_dir=stable_dir)

        assert (stable_dir / "day_stable.laz").exists()

    def test_evaluate_coreg__writes_stats_and_distances(self, tmp_path: Path, write_cloud_las) -> None:
        # These two files are what the whole of postprocessing later reads back.
        ref = write_cloud_las("ref.las")
        day = write_cloud_las("2024-06-23_cloud_coreg.las", seed=7)
        coreg_dir = tmp_path / "coreg"
        plot_dir = coreg_dir / "m3c2_plots"

        evaluate_coreg(ref, day, day, ref_downsample_factor=1.0, plot_dir=plot_dir)

        assert (coreg_dir / "2024-06-23_m3c2_stats.csv").exists()
        assert (coreg_dir / "2024-06-23_m3c2_distances.npz").exists()

    def test_evaluate_coreg__tba_downsample_defaults_to_ref(self, tmp_path: Path, write_cloud_las) -> None:
        ref = write_cloud_las("ref.las")
        day = write_cloud_las("day.las", seed=7)

        out = evaluate_coreg(ref, day, day, ref_downsample_factor=0.5)

        assert out["dist_after"].size > 0

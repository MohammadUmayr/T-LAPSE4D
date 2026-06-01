"""
Metashape time-lapse multi-camera processing pipeline.

All processing logic is exposed as plain Python functions so the pipeline
can be driven from a Jupyter notebook by passing paths as variables:

    from cntp.metashape import run_pipeline

    run_pipeline(
        tlcam_dir  = "TLCAM/",
        calib_dir  = "Calib_IOP/",
        reference  = "ref.xlsx",
        output_dir = "output/",
        dates      = None,          # None → all dates found
    )

Expected on-disk layout
-----------------------
  TLCAM/
    C1_renamed/   C1_YYYY-MM-DD_HHMMSS.JPG  ...
    C2_renamed/   ...
    C3_renamed/
    C4_renamed/
    C5_renamed/
  Calib_IOP/
    C1.xml  C2.xml  C3.xml  C4.xml  C5.xml   (Metashape XML format)
  ref.xlsx          columns: Label, Lon, Lat, Alt,
                              xAcc, yAcc, zAcc,
                              yaw, pitch, roll,
                              yawAcc, pitchAcc, rollAcc

Outputs (per day, inside  output_dir/YYYY-MM-DD/)
-------------------------------------------------
  YYYY-MM-DD.psx           Metashape project
  YYYY-MM-DD_cloud.laz     point cloud in UTM (EPSG auto-detected)
  YYYY-MM-DD_cameras.csv   estimated camera positions in WGS84
  adjusted_calib/C1.xml …  optimised intrinsics per camera (Metashape XML)
"""

from __future__ import annotations

import io
import os
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Metashape is an optional runtime dependency; imported lazily inside
# functions so the module can be imported for unit-testing without it.
try:
    import Metashape
except ImportError:
    Metashape = None  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAMERAS = ["C1", "C2", "C3", "C4", "C5"]

_IMG_RE = re.compile(
    r"^(C[1-5])_(\d{4}-\d{2}-\d{2})_\d{6}\.(jpg|jpeg|JPG|JPEG)$"
)


# ---------------------------------------------------------------------------
# Calibration I/O
# ---------------------------------------------------------------------------

def load_calib_xml(path: str | Path) -> Any:
    """Load a Metashape XML calibration file into a Calibration object.

    The returned object is in Metashape pixel-normalised units — no conversion
    required.  Assign directly to ``sensor.user_calib``.
    """
    if Metashape is None:
        raise ImportError("Metashape Python module is not installed.")
    calib = Metashape.Calibration()
    calib.load(str(Path(path)), format=Metashape.CalibrationFormatXML)
    return calib




# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _utm_epsg(lon: float) -> int:
    """Return the EPSG code for the UTM zone containing *lon* (northern hemisphere)."""
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def discover_images(tlcam_dir: str | Path) -> dict[str, dict[str, list[Path]]]:
    """Scan camera sub-directories and return ``{date: {camera: [paths]}}``.

    Parameters
    ----------
    tlcam_dir : str | Path
        Root directory containing ``C1_renamed/``, ``C2_renamed/``, …
    """
    tlcam_dir = Path(tlcam_dir)
    by_date: dict = defaultdict(lambda: defaultdict(list))
    for cam in CAMERAS:
        cam_dir = tlcam_dir / f"{cam}_renamed"
        if not cam_dir.is_dir():
            continue
        for img in sorted(cam_dir.iterdir()):
            m = _IMG_RE.match(img.name)
            if m:
                cam_label, date = m.group(1), m.group(2)
                by_date[date][cam_label].append(img)

    return {d: dict(cams) for d, cams in sorted(by_date.items())}


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------

def load_reference(ref_path: str | Path) -> pd.DataFrame:
    """Load the per-camera reference Excel file.

    Expected columns (case-insensitive, spaces stripped):
      Label, Lon, Lat, Alt, xAcc, yAcc, zAcc,
      yaw, pitch, roll, yawAcc, pitchAcc, rollAcc
    """
    df = pd.read_excel(Path(ref_path))
    df.columns = df.columns.str.strip()

    # Build a case-insensitive → canonical name mapping
    _std = ["label", "lon", "lat", "alt",
            "xacc", "yacc", "zacc",
            "yaw", "pitch", "roll",
            "yawacc", "pitchacc", "rollacc"]
    col_map = {}
    for col in df.columns:
        low = col.lower().replace(" ", "").replace("_", "")
        if low in _std:
            col_map[col] = low

    df = df.rename(columns=col_map)
    df["label"] = df["label"].astype(str).str.strip()
    return df


def _build_image_ref_csv(
    date: str,
    date_images: dict[str, list[Path]],
    ref_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Expand the 5-row camera reference into one row per image.

    Every image from camera Cx inherits Cx's position and orientation.
    The label column uses the image filename stem so Metashape can match it.
    """
    rows = []
    for cam, paths in date_images.items():
        cam_row = ref_df[ref_df["label"].str.upper() == cam.upper()]
        if cam_row.empty:
            raise ValueError(
                f"Camera '{cam}' not found in reference file. "
                f"Available labels: {list(ref_df['label'])}"
            )
        r = cam_row.iloc[0]
        for img_path in paths:
            rows.append({
                "label":    img_path.stem,
                "lon":      r["lon"],
                "lat":      r["lat"],
                "alt":      r["alt"],
                "xacc":     r["xacc"],
                "yacc":     r["yacc"],
                "zacc":     r["zacc"],
                "yaw":      r["yaw"],
                "pitch":    r["pitch"],
                "roll":     r["roll"],
                "yawacc":   r["yawacc"],
                "pitchacc": r["pitchacc"],
                "rollacc":  r["rollacc"],
            })

    csv_path = out_dir / f"ref_{date}.csv"
    # No header — importReference uses the columns string to interpret columns
    pd.DataFrame(rows).to_csv(csv_path, index=False, header=False)
    return csv_path


# ---------------------------------------------------------------------------
# Sensor setup
# ---------------------------------------------------------------------------

def _camera_prefix(label: str) -> str:
    """Extract camera prefix from image label: 'C1_2023-07-15_083000' → 'C1'."""
    return label.split("_")[0]


def _image_date(label: str) -> str:
    """Extract date from image label: 'C1_2023-07-15_083000' → '2023-07-15'."""
    return label.split("_")[1]


def _normalize_date(value) -> str:
    """Coerce any pandas-readable date to canonical 'YYYY-MM-DD' string.

    Registry CSVs are sometimes opened in Excel/LibreOffice, which rewrites
    the date column to a locale format (e.g. '11/27/2023'). Sensor lookup
    keys come from `_image_date()` which always yields 'YYYY-MM-DD' from the
    image label, so a format drift in the CSV silently breaks the lookup
    and Metashape auto-merges all reference images into one sensor.
    """
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _setup_sensors(
    chunk: "Metashape.Chunk",
    calib_xmls: dict[str, Path],
    fixed_calibration: bool = False,
) -> dict[str, Any]:
    """Create one sensor per camera and load its XML calibration as prior.

    Parameters
    ----------
    fixed_calibration : bool
        When True, lock the sensor IOP during optimizeCameras (equivalent to
        sensor.fixed_calibration = True in the Metashape GUI).
    """
    sensors: dict = {}
    for cam in CAMERAS:
        if cam not in calib_xmls:
            continue
        calib = load_calib_xml(calib_xmls[cam])
        sensor = chunk.addSensor()
        sensor.label             = cam
        sensor.type              = Metashape.Sensor.Type.Frame
        sensor.width             = calib.width
        sensor.height            = calib.height
        sensor.user_calib        = calib
        sensor.fixed_calibration = fixed_calibration
        sensors[cam] = sensor
    return sensors


def _set_camera_references(
    chunk: "Metashape.Chunk",
    date_images: dict[str, list[Path]],
    ref_df: pd.DataFrame,
    wgs84_crs: "Metashape.CoordinateSystem",
) -> None:
    """Set camera reference positions/orientations directly on each camera object.

    Avoids importReference label-matching by building a stem→reference lookup
    and assigning directly via the Metashape Python API.
    """
    # Build {filename_stem: reference_row} lookup
    ref_lookup: dict = {}
    for cam, paths in date_images.items():
        cam_row = ref_df[ref_df["label"].str.upper() == cam.upper()]
        if cam_row.empty:
            continue
        r = cam_row.iloc[0]
        for img_path in paths:
            ref_lookup[img_path.stem] = r

    chunk.crs = wgs84_crs
    matched = 0
    for camera in chunk.cameras:
        r = ref_lookup.get(camera.label)
        if r is not None:
            camera.reference.location = Metashape.Vector(
                [float(r["lon"]), float(r["lat"]), float(r["alt"])]
            )
            camera.reference.location_accuracy = Metashape.Vector(
                [float(r["xacc"]), float(r["yacc"]), float(r["zacc"])]
            )
            camera.reference.rotation = Metashape.Vector(
                [float(r["yaw"]), float(r["pitch"]), float(r["roll"])]
            )
            camera.reference.rotation_accuracy = Metashape.Vector(
                [float(r["yawacc"]), float(r["pitchacc"]), float(r["rollacc"])]
            )
            camera.reference.enabled          = True
            camera.reference.rotation_enabled = True
            matched += 1

    print(f"  Set reference       : {matched}/{len(chunk.cameras)} cameras matched", flush=True)
    if matched == 0:
        sample = [c.label for c in chunk.cameras[:3]]
        print(f"  DEBUG camera labels : {sample}", flush=True)
        print(f"  DEBUG lookup keys   : {list(ref_lookup.keys())[:3]}", flush=True)


def _assign_sensors(
    chunk: "Metashape.Chunk",
    sensors: dict[str, "Metashape.Sensor"],
) -> None:
    """Assign each Metashape camera to the sensor matching its filename prefix."""
    for camera in chunk.cameras:
        for cam_label, sensor in sensors.items():
            if camera.label.startswith(cam_label + "_"):
                camera.sensor = sensor
                break


def _setup_camera_groups(
    chunk: "Metashape.Chunk",
    date_images: dict[str, list[Path]],
) -> dict[str, Any]:
    """Create one camera group per camera label for workspace organisation."""
    groups: dict = {}
    for cam in date_images.keys():
        group = chunk.addCameraGroup()
        group.label = cam
        groups[cam] = group
    return groups


def _assign_camera_groups(
    chunk: "Metashape.Chunk",
    groups: dict[str, Any],
) -> None:
    """Assign each camera to the group matching its filename prefix."""
    for camera in chunk.cameras:
        for cam_label, group in groups.items():
            if camera.label.startswith(cam_label + "_"):
                camera.group = group
                break


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _export_camera_csv(chunk: "Metashape.Chunk", out_path: Path) -> None:
    """Export estimated camera positions and orientations to a WGS84 CSV.

    Follows the HSFM approach: position from chunk.crs.project(T.mulp(camera.center)),
    Yaw/Pitch/Roll from Metashape.utils.mat2ypr with the local geographic frame.
    """
    T = chunk.transform.matrix
    rows = []

    for cam in chunk.cameras:
        if cam.transform is None:
            continue
        try:
            lon, lat, alt = chunk.crs.project(T.mulp(cam.center))
            m = chunk.crs.localframe(T.mulp(cam.center))
            R = m * T * cam.transform * Metashape.Matrix().Diag([1, -1, -1, 1])
            rows_R = []
            for j in range(3):
                r = R.row(j)
                r.size = 3
                r.normalize()
                rows_R.append(r)
            R = Metashape.Matrix([rows_R[0], rows_R[1], rows_R[2]])
            yaw, pitch, roll = Metashape.utils.mat2ypr(R)
        except Exception:
            lon = lat = alt = yaw = pitch = roll = float("nan")

        rows.append({
            "Label": cam.label,
            "Lon":   lon,
            "Lat":   lat,
            "Alt":   alt,
            "Yaw":   yaw,
            "Pitch": pitch,
            "Roll":  roll,
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Exported camera CSV : {out_path.name}", flush=True)


def _export_calib_xml(
    sensors: dict[str, Any],
    iop_dir: Path,
) -> None:
    """Write one updated Metashape XML calibration file per camera after BA."""
    iop_dir.mkdir(parents=True, exist_ok=True)
    for cam_label, sensor in sensors.items():
        out_path = iop_dir / f"{cam_label}.xml"
        sensor.calibration.save(
            str(out_path),
            format=Metashape.CalibrationFormatXML,
        )
        print(f"  Exported calib XML  : {out_path.name}", flush=True)


# ---------------------------------------------------------------------------
# Per-day processing
# ---------------------------------------------------------------------------

def process_day(
    date: str,
    date_images: dict[str, list[Path]],
    ref_df: pd.DataFrame,
    calib_xmls: dict[str, Path],
    output_dir: Path,
    utm_epsg: int,
    match_downscale: int = 1,
    depth_downscale: int = 2,
    stop_after_alignment: bool = False,
    resume: bool = False,
) -> None:
    """Run the full Metashape pipeline for a single calendar day.

    Parameters
    ----------
    date : str
        Date string ``YYYY-MM-DD``.
    date_images : dict
        ``{camera_label: [image_paths]}`` for this day.
    ref_df : pd.DataFrame
        Per-camera reference table from :func:`load_reference`.
    calib_xmls : dict
        ``{camera_label: Path}`` mapping to Metashape XML calibration files.
    output_dir : Path
        Root output directory.  A ``date/`` sub-directory is created here.
    utm_epsg : int
        EPSG code for the UTM zone used when exporting the LAZ.
    match_downscale : int
        Controls the image resolution used during key-point detection and
        matching (``matchPhotos`` ``downscale`` argument).

        === ========= ==============================================
        0   Highest   2× upscaled  — most tie points, very slow
        1   High      Original resolution (default)
        2   Medium    ½ resolution
        4   Low       ¼ resolution
        8   Lowest    ⅛ resolution — fewest tie points, fastest
        === ========= ==============================================

    depth_downscale : int
        Controls the image resolution used when building depth maps
        (``buildDepthMaps`` ``downscale`` argument).

        === =========== ===========================================
        1   Ultra High  Full resolution — very slow
        2   High        ½ resolution    — good balance (default)
        4   Medium      ¼ resolution
        8   Low         ⅛ resolution
        16  Lowest      1/16 resolution — fastest
        === =========== ===========================================

    resume : bool
        When ``True``, open the existing ``.psx`` (which must already contain
        aligned cameras) and skip straight to ``buildDepthMaps`` →
        ``buildPointCloud`` → exports.  The project is **not** deleted or
        re-created.  Raises ``RuntimeError`` if the ``.psx`` is missing or no
        cameras are aligned.
    """
    if Metashape is None:
        raise ImportError("Metashape Python module is not installed.")

    n_imgs = sum(len(v) for v in date_images.values())
    print(f"\n{'='*60}", flush=True)
    print(f"  Date : {date}   ({n_imgs} images across {len(date_images)} cameras)", flush=True)
    print(f"{'='*60}", flush=True)

    day_dir  = output_dir / "output" / date
    day_dir.mkdir(parents=True, exist_ok=True)
    psx_path = day_dir / f"{date}.psx"

    doc = Metashape.Document()

    if resume:
        # ── Resume: open existing aligned project ─────────────────────────
        if not psx_path.exists():
            raise RuntimeError(
                f"resume=True but no project found at {psx_path}. "
                "Run without resume=True first to complete alignment."
            )
        doc.open(str(psx_path), read_only=False, ignore_lock=True)
        chunk   = doc.chunk
        sensors = {s.label: s for s in chunk.sensors}
        aligned = sum(1 for c in chunk.cameras if c.transform)
        if aligned == 0:
            raise RuntimeError(
                f"resume=True but no cameras are aligned in {psx_path.name}. "
                "Re-run without resume=True to redo alignment."
            )
        print(f"  Resumed project     : {psx_path.name}  ({aligned} cameras already aligned)", flush=True)
    else:
        # ── Fresh run: delete old project and start from scratch ──────────
        psx_files_dir = day_dir / f"{date}.files"
        if psx_files_dir.exists():
            shutil.rmtree(psx_files_dir, ignore_errors=True)
        if psx_path.exists():
            psx_path.unlink(missing_ok=True)

        doc.save(str(psx_path))
        chunk = doc.addChunk()
        chunk.label = date

        # ── Photos ────────────────────────────────────────────────────────
        all_images = [str(p) for imgs in date_images.values() for p in imgs]
        chunk.addPhotos(all_images)
        print(f"  Added {len(all_images)} photos", flush=True)

        # ── Sensors + camera groups ───────────────────────────────────────
        sensors = _setup_sensors(chunk, calib_xmls)
        _assign_sensors(chunk, sensors)
        groups = _setup_camera_groups(chunk, date_images)
        _assign_camera_groups(chunk, groups)
        print(f"  Configured {len(sensors)} sensors / {len(groups)} camera groups", flush=True)

        # ── Reference ─────────────────────────────────────────────────────
        wgs84_crs = Metashape.CoordinateSystem("EPSG::4326")
        _set_camera_references(chunk, date_images, ref_df, wgs84_crs)

        # ── Bundle adjustment ─────────────────────────────────────────────
        print("  Matching photos ...", flush=True)
        chunk.matchPhotos(
            downscale=match_downscale,
            keypoint_limit=80000,
            tiepoint_limit=8000,
            generic_preselection=True,
            reference_preselection=False,
        )

        print("  Aligning cameras (bundle adjustment) ...", flush=True)
        chunk.alignCameras(adaptive_fitting=False)
        doc.save()

        aligned = sum(1 for c in chunk.cameras if c.transform)
        print(f"  Aligned {aligned}/{len(chunk.cameras)} cameras", flush=True)
        if aligned == 0:
            print(f"  WARNING: no cameras aligned for {date}, skipping exports.", flush=True)
            return

        print("  Optimising camera alignment ...", flush=True)
        chunk.optimizeCameras(
            fit_f=True,
            fit_cx=True,
            fit_cy=True,
            fit_k1=True,
            fit_k2=True,
            fit_k3=True,
            fit_k4=False,
            fit_p1=True,
            fit_p2=True,
            fit_b1=False,
            fit_b2=False,
            adaptive_fitting=False,
            tiepoint_covariance=True,
        )
        doc.save()

        if stop_after_alignment:
            print(f"  Stopped after alignment (stop_after_alignment=True)", flush=True)
            print(f"  Project saved : {psx_path.name}", flush=True)
            return

    # ── Dense point cloud ─────────────────────────────────────────────────
    print("  Building depth maps ...", flush=True)
    chunk.buildDepthMaps(
        downscale=depth_downscale,
        filter_mode=Metashape.MildFiltering,
    )
    print("  Building point cloud ...", flush=True)
    chunk.buildPointCloud()
    doc.save()

    # ── Exports ───────────────────────────────────────────────────────────
    laz_path = day_dir / f"{date}_cloud.laz"
    utm_crs  = Metashape.CoordinateSystem(f"EPSG::{utm_epsg}")
    chunk.exportPointCloud(
        str(laz_path),
        source_data=Metashape.PointCloudData,
        format=Metashape.PointCloudFormatLAS,
        crs=utm_crs,
        save_point_color=True,
    )
    print(f"  Exported LAZ        : {laz_path.name}  (EPSG:{utm_epsg})", flush=True)

    _export_camera_csv(chunk, day_dir / f"{date}_cameras.csv")
    _export_calib_xml(sensors, day_dir / "adjusted_calib")

    doc.save()
    print(f"  Project saved       : {psx_path.name}", flush=True)


# ---------------------------------------------------------------------------
# Pipeline entry point (call from Jupyter notebook)
# ---------------------------------------------------------------------------

def run_pipeline(
    tlcam_dir:  str | Path,
    calib_dir:  str | Path,
    reference:  str | Path,
    output_dir: str | Path,
    dates:      list[str] | None = None,
    match_downscale: int = 1,
    depth_downscale: int = 2,
    stop_after_alignment: bool = False,
    resume: bool = False,
) -> None:
    """Process all (or selected) days in the time-lapse dataset.

    Parameters
    ----------
    tlcam_dir : str | Path
        Root containing ``C1_renamed/``, ``C2_renamed/``, …
    calib_dir : str | Path
        Directory containing ``C1.txt`` … ``C5.txt``.
    reference : str | Path
        Excel file with per-camera WGS84 positions and orientations.
    output_dir : str | Path
        Root output directory.
    dates : list[str] | None
        If given, process only these dates (``YYYY-MM-DD``).
        If None, all dates found in *tlcam_dir* are processed.
    match_downscale : int
        Controls the image resolution used during key-point detection and
        matching (``matchPhotos`` ``downscale`` argument).

        === ========= ==============================================
        0   Highest   2× upscaled  — most tie points, very slow
        1   High      Original resolution (default)
        2   Medium    ½ resolution
        4   Low       ¼ resolution
        8   Lowest    ⅛ resolution — fewest tie points, fastest
        === ========= ==============================================

    depth_downscale : int
        Controls the image resolution used when building depth maps
        (``buildDepthMaps`` ``downscale`` argument).

        === =========== ===========================================
        1   Ultra High  Full resolution — very slow
        2   High        ½ resolution    — good balance (default)
        4   Medium      ¼ resolution
        8   Low         ⅛ resolution
        16  Lowest      1/16 resolution — fastest
        === =========== ===========================================
    """
    tlcam_dir  = Path(tlcam_dir)
    calib_dir  = Path(calib_dir)
    reference  = Path(reference)
    output_dir = Path(output_dir)

    # ── Load calibrations ─────────────────────────────────────────────────
    calib_xmls: dict[str, Path] = {}
    for cam in CAMERAS:
        xml_file = calib_dir / f"{cam}.xml"
        if xml_file.exists():
            calib_xmls[cam] = xml_file
            print(f"Loaded calibration : {xml_file.name}", flush=True)
        else:
            print(f"WARNING: {xml_file} not found — {cam} will use Metashape defaults", flush=True)

    # ── Load reference ────────────────────────────────────────────────────
    ref_df = load_reference(reference)
    print(f"Loaded reference   : {len(ref_df)} camera entries from {reference.name}", flush=True)

    # Auto-detect UTM zone from mean longitude
    mean_lon = ref_df["lon"].mean()
    utm_epsg = _utm_epsg(mean_lon)
    print(f"UTM EPSG           : EPSG:{utm_epsg}  (mean lon = {mean_lon:.3f}°)", flush=True)

    # ── Discover images ───────────────────────────────────────────────────
    by_date = discover_images(tlcam_dir)
    if not by_date:
        raise RuntimeError(f"No images matched the expected naming pattern under {tlcam_dir}")

    if dates is not None:
        by_date = {d: v for d, v in by_date.items() if d in dates}
        if not by_date:
            raise ValueError(f"None of the requested dates {dates} were found.")

    print(f"Dates to process   : {list(by_date.keys())}\n", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Process each day ──────────────────────────────────────────────────
    for date, date_images in by_date.items():
        process_day(
            date=date,
            date_images=date_images,
            ref_df=ref_df,
            calib_xmls=calib_xmls,
            output_dir=output_dir,
            utm_epsg=utm_epsg,
            match_downscale=match_downscale,
            depth_downscale=depth_downscale,
            stop_after_alignment=stop_after_alignment,
            resume=resume,
        )

    print("\nAll dates processed.", flush=True)


# ---------------------------------------------------------------------------
# Co-registration camera import  (run after ASP pc_align)
# ---------------------------------------------------------------------------

def update_metashape_cameras_after_transform(
    psx_path: Path,
    transform_path: Path,
    chunk_label: str = None,
) -> None:
    """Apply an ASP pc_align ECEF transform directly to the Metashape chunk transform.

    Workflow: run_pipeline → ASP pc_align (use_ecef=True) → this function.

    Reads the 4×4 rigid-body transform from *transform_path* (Stage 3
    ``run-transform.txt``, which contains the full composed T3∘T2∘T1) and
    composes it with the current ``chunk.transform.matrix``::

        M_new = T_ecef @ M_old

    Both matrices are in WGS84 geocentric ECEF (meters), so no CRS conversion
    is needed.  ``chunk.transform.matrix`` maps local chunk space → ECEF; left-
    multiplying by ``T_ecef`` shifts the entire chunk to the co-registered
    position without any GPS-prior fitting.

    Parameters
    ----------
    psx_path : Path
        Metashape project file (.psx).
    transform_path : Path
        ASP ``stage3/run-transform.txt`` (full composed ECEF transform).
    chunk_label : str, optional
        Chunk to update.  Defaults to the first chunk when *None*.
    """
    if Metashape is None:
        raise ImportError("Metashape Python module is not installed.")

    import math
    import numpy as np

    # Parse ASP 4×4 ECEF transform: one header line then four rows of four floats.
    rows: list[list[float]] = []
    for line in Path(transform_path).read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 4:
            try:
                rows.append(list(map(float, parts)))
            except ValueError:
                pass
        if len(rows) == 4:
            break
    if len(rows) != 4:
        raise ValueError(f"Could not parse a 4×4 matrix from {transform_path}")
    T = np.array(rows)
    print(f"  ASP transform loaded from {Path(transform_path).name}", flush=True)

    doc = Metashape.Document()
    doc.clear()
    doc.open(str(psx_path), read_only=False, ignore_lock=True)

    if chunk_label is None:
        chunk = doc.chunk
    else:
        chunk = next((c for c in doc.chunks if c.label == chunk_label), None)
        if chunk is None:
            raise ValueError(f"Chunk '{chunk_label}' not found in {Path(psx_path).name}")

    diag_cam = next((c for c in chunk.cameras if c.transform), None)
    if diag_cam:
        p0 = chunk.crs.project(chunk.transform.matrix.mulp(diag_cam.center))
        print(f"  [{diag_cam.label}] BEFORE : lon={p0[0]:.7f}  lat={p0[1]:.7f}  alt={p0[2]:.3f} m", flush=True)

    # Direct composition: T_ecef @ M_old.  Both in WGS84 geocentric ECEF (metres).
    M_old = chunk.transform.matrix
    M_np  = np.array([[M_old[r, c] for c in range(4)] for r in range(4)])
    M_new = T @ M_np
    chunk.transform.matrix = Metashape.Matrix(
        [[float(M_new[r, c]) for c in range(4)] for r in range(4)]
    )

    if diag_cam:
        p1 = chunk.crs.project(chunk.transform.matrix.mulp(diag_cam.center))
        print(f"  [{diag_cam.label}] AFTER  : lon={p1[0]:.7f}  lat={p1[1]:.7f}  alt={p1[2]:.3f} m", flush=True)
        dlat = (p1[1] - p0[1]) * 111_320.0
        dlon = (p1[0] - p0[0]) * 111_320.0 * math.cos(math.radians((p0[1] + p1[1]) / 2))
        dalt = p1[2] - p0[2]
        shift = (dlat**2 + dlon**2 + dalt**2) ** 0.5
        print(f"  Position shift   : {shift:.4f} m", flush=True)

    doc.save()
    print(f"  Saved → {psx_path}", flush=True)


# ---------------------------------------------------------------------------
# 4D SfM helpers
# ---------------------------------------------------------------------------

def _set_camera_references_from_csv(
    chunk: "Metashape.Chunk",
    cameras_csv: Path,
    loc_acc: tuple = (0.5, 0.5, 0.5),
    rot_acc: tuple = (5.0, 5.0, 5.0),
) -> int:
    """Set per-image camera references from a cameras CSV with configurable accuracy.

    The CSV must have columns: Label, Lon, Lat, Alt, Yaw, Pitch, Roll.
    Accuracies are passed as parameters — not stored in the CSV — so the same
    CSV can be reused with different accuracy settings (tight for validation,
    loose for new-day processing).
    """
    df = pd.read_csv(Path(cameras_csv))
    lookup = {str(row["Label"]): row for _, row in df.iterrows()}

    matched = 0
    for camera in chunk.cameras:
        r = lookup.get(camera.label)
        if r is None:
            continue
        camera.reference.location          = Metashape.Vector([float(r["Lon"]), float(r["Lat"]), float(r["Alt"])])
        camera.reference.location_accuracy = Metashape.Vector(list(loc_acc))
        camera.reference.rotation          = Metashape.Vector([float(r["Yaw"]), float(r["Pitch"]), float(r["Roll"])])
        camera.reference.rotation_accuracy = Metashape.Vector(list(rot_acc))
        camera.reference.enabled           = True
        camera.reference.rotation_enabled  = True
        matched += 1

    print(f"  Set references from CSV : {matched}/{len(chunk.cameras)} cameras matched", flush=True)
    return matched


def _last_day_mean_eop(registry_df: "pd.DataFrame") -> dict[str, dict]:
    """Return mean EOP per camera group for the most recently added day in the registry.

    Groups rows by camera prefix (C1, C2, …) extracted from the label column.
    Returns ``{cam_prefix: {lon, lat, alt, yaw, pitch, roll}}``.
    """
    last_date = registry_df["date"].iloc[-1]
    last_df   = registry_df[registry_df["date"] == last_date].copy()
    last_df["cam"] = last_df["label"].apply(_camera_prefix)

    means: dict = {}
    for cam, grp in last_df.groupby("cam"):
        means[cam] = {
            "lon":   grp["lon"].mean(),
            "lat":   grp["lat"].mean(),
            "alt":   grp["alt"].mean(),
            "yaw":   grp["yaw"].mean(),
            "pitch": grp["pitch"].mean(),
            "roll":  grp["roll"].mean(),
        }
    return means


def _export_camera_csv_filtered(
    chunk: "Metashape.Chunk",
    out_path: Path,
    date_filter: str,
) -> None:
    """Export camera positions only for cameras whose label contains date_filter."""
    T = chunk.transform.matrix
    rows = []

    for cam in chunk.cameras:
        if date_filter not in cam.label:
            continue
        if cam.transform is None:
            continue
        try:
            lon, lat, alt = chunk.crs.project(T.mulp(cam.center))
            m = chunk.crs.localframe(T.mulp(cam.center))
            R = m * T * cam.transform * Metashape.Matrix().Diag([1, -1, -1, 1])
            rows_R = []
            for j in range(3):
                r = R.row(j)
                r.size = 3
                r.normalize()
                rows_R.append(r)
            R = Metashape.Matrix([rows_R[0], rows_R[1], rows_R[2]])
            yaw, pitch, roll = Metashape.utils.mat2ypr(R)
        except Exception:
            lon = lat = alt = yaw = pitch = roll = float("nan")

        rows.append({"Label": cam.label, "Lon": lon, "Lat": lat, "Alt": alt,
                     "Yaw": yaw, "Pitch": pitch, "Roll": roll})

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Exported 4DSfM cameras CSV : {out_path.name}  ({len(rows)} cameras)", flush=True)


def _setup_sensors_multitemporal(
    chunk: "Metashape.Chunk",
    reg_df: "pd.DataFrame",
    new_calib_xmls: dict[str, Path],
) -> tuple[dict, dict]:
    """Create one fixed sensor per camera per reference day, plus floating new-day sensors.

    Each reference day gets its own sensor per camera (e.g. ``C1_2023-07-15``)
    loaded from that day's validated ``calib_dir``, with ``fixed_calibration = True``.
    This preserves the per-day IOP refinement rather than forcing all reference
    images to share a single calibration.

    New-day sensors (``C1_new``, …) are loaded from *new_calib_xmls* with
    ``fixed_calibration = False``.

    Returns
    -------
    ref_sensors : dict  {(cam_prefix, date) → Sensor}
    new_sensors : dict  {cam_prefix → Sensor}
    """
    ref_sensors: dict = {}
    new_sensors: dict = {}

    for date, grp in reg_df.groupby("date"):
        calib_dir = Path(grp["calib_dir"].iloc[0])
        for cam in CAMERAS:
            ref_xml = calib_dir / f"{cam}.xml"
            if not ref_xml.exists():
                continue
            calib = load_calib_xml(ref_xml)
            s = chunk.addSensor()
            s.label             = f"{cam}_{date}"
            s.type              = Metashape.Sensor.Type.Frame
            s.width             = calib.width
            s.height            = calib.height
            s.user_calib        = calib
            s.fixed_calibration = True
            ref_sensors[(cam, date)] = s

    for cam in CAMERAS:
        new_xml = new_calib_xmls.get(cam)
        if new_xml and Path(new_xml).exists():
            calib = load_calib_xml(new_xml)
            s = chunk.addSensor()
            s.label             = f"{cam}_new"
            s.type              = Metashape.Sensor.Type.Frame
            s.width             = calib.width
            s.height            = calib.height
            s.user_calib        = calib
            s.fixed_calibration = False
            new_sensors[cam]    = s

    return ref_sensors, new_sensors


def _assign_sensors_multitemporal(
    chunk: "Metashape.Chunk",
    ref_sensors: dict,
    new_sensors: dict,
    new_date: str,
) -> None:
    """Assign each camera to its per-day ref sensor or to the new-day sensor."""
    for camera in chunk.cameras:
        cam_prefix = _camera_prefix(camera.label)
        if new_date in camera.label:
            sensor = new_sensors.get(cam_prefix)
        else:
            img_date = _image_date(camera.label)
            sensor   = ref_sensors.get((cam_prefix, img_date))
        if sensor is not None:
            camera.sensor = sensor


# ---------------------------------------------------------------------------
# Reference registry
# ---------------------------------------------------------------------------

_REGISTRY_COLS = ["date", "label", "image_path", "lon", "lat", "alt",
                  "yaw", "pitch", "roll", "calib_dir"]


def update_registry(
    registry_csv: Path,
    date: str,
    date_images: dict[str, list[Path]],
    cameras_coreg_csv: Path,
    calib_dir: Path,
) -> None:
    """Append a validated day to the reference registry.

    Reads the co-registered camera positions (Label, Lon, Lat, Alt, Yaw,
    Pitch, Roll) and writes one row per image into *registry_csv*, adding the
    original image path and calibration directory so future multi-temporal runs
    can reload everything from the registry alone.

    Parameters
    ----------
    registry_csv : Path
        Registry file.  Created with a header row if it does not yet exist.
    date : str
        Date string YYYY-MM-DD of the validated day.
    date_images : dict
        ``{camera_label: [image_paths]}`` for this day — used to resolve the
        image path that matches each camera label.
    cameras_coreg_csv : Path
        Co-registered camera positions CSV produced by
        ``apply_coreg_to_cameras*``.
    calib_dir : Path
        ``adjusted_calib_4DSfM/`` directory for this day.
    """
    registry_csv = Path(registry_csv)
    df_coreg     = pd.read_csv(Path(cameras_coreg_csv))

    # Build stem → absolute path lookup from date_images
    stem_to_path: dict = {}
    for paths in date_images.values():
        for p in paths:
            stem_to_path[Path(p).stem] = str(Path(p).resolve())

    rows = []
    for _, row in df_coreg.iterrows():
        label     = str(row["Label"])
        img_path  = stem_to_path.get(label, "")
        rows.append({
            "date":      date,
            "label":     label,
            "image_path": img_path,
            "lon":       row["Lon"],
            "lat":       row["Lat"],
            "alt":       row["Alt"],
            "yaw":       row["Yaw"],
            "pitch":     row["Pitch"],
            "roll":      row["Roll"],
            "calib_dir": str(Path(calib_dir).resolve()),
        })

    new_df = pd.DataFrame(rows, columns=_REGISTRY_COLS)

    if registry_csv.exists():
        existing = pd.read_csv(registry_csv)
        existing["date"] = existing["date"].map(_normalize_date)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined["date"] = combined["date"].map(_normalize_date)
    combined.to_csv(registry_csv, index=False)
    print(f"  Registry updated : {registry_csv.name}  ({len(rows)} rows added, "
          f"{len(combined)} total)")


# ---------------------------------------------------------------------------
# 4D SfM pipeline — multi-temporal bundle adjustment
# ---------------------------------------------------------------------------

def run_multitemporal_ba(
    date: str,
    date_images: dict[str, list[Path]],
    reference_registry_csv: Path,
    output_dir: Path,
    utm_epsg: int,
    match_downscale: int = 1,
    loc_acc_new: tuple = (0.5, 0.5, 0.5),
    rot_acc_new: tuple = (5.0, 5.0, 5.0),
) -> tuple[Path, Path]:
    """Run multi-temporal bundle adjustment combining all reference + new day images.

    Reference cameras (from registry) are held fixed with accuracy 0.001.
    New-day cameras are initialised with the mean EOP of the most recently
    added registry day, per camera group, with loose accuracy.
    New-day sensor IOP is loaded from the last registry day's adjusted_calib_4DSfM/.

    Parameters
    ----------
    date : str
        New date to process (YYYY-MM-DD).
    date_images : dict
        ``{camera_label: [image_paths]}`` for the new day.
    reference_registry_csv : Path
        Registry CSV produced by :func:`update_registry`.
    output_dir : Path
        Root output directory.
    utm_epsg : int
        UTM EPSG code.
    match_downscale : int
        Downscale for ``matchPhotos``.
    loc_acc_new : tuple
        Position accuracy (m) for new-day cameras.
    rot_acc_new : tuple
        Rotation accuracy (°) for new-day cameras.

    Returns
    -------
    cameras_csv : Path
        ``4D_SfM/YYYY-MM-DD_cameras_4DSfM.csv`` — new day refined EOP.
    calib_dir : Path
        ``4D_SfM/adjusted_calib_4DSfM/`` — new day refined IOP.
    """
    if Metashape is None:
        raise ImportError("Metashape Python module is not installed.")

    reference_registry_csv = Path(reference_registry_csv)
    output_dir             = Path(output_dir)

    sfm_dir  = output_dir / "output_new" / date / "4D_SfM"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    psx_path = sfm_dir / f"{date}_4DSfM.psx"

    reg_df = pd.read_csv(reference_registry_csv)
    reg_df["date"] = reg_df["date"].map(_normalize_date)
    n_ref  = len(reg_df)
    n_new  = sum(len(v) for v in date_images.values())

    print(f"\n{'='*60}", flush=True)
    print(f"  4D SfM BA : {date}   ({n_new} new images, {n_ref} reference images)", flush=True)
    print(f"{'='*60}", flush=True)

    mean_eop = _last_day_mean_eop(reg_df)

    # Load new-day sensor IOP from last registry day's validated calibration.
    last_calib_dir = Path(reg_df["calib_dir"].iloc[-1])
    new_calib_xmls: dict = {}
    for cam in CAMERAS:
        refined_xml = last_calib_dir / f"{cam}.xml"
        if not refined_xml.exists():
            raise FileNotFoundError(
                f"Calibration XML for {cam} not found in last registry calib dir: "
                f"{last_calib_dir}. Re-run bootstrap_registry or update_registry."
            )
        new_calib_xmls[cam] = refined_xml

    # ── Build project ─────────────────────────────────────────────────────
    # Nuke any stale .psx + .files from a prior crashed run. Without this,
    # the new doc.save() can latch onto leftover state and Metashape flips
    # the document to read-only on the next save (see run_4dsfm_day Step 2
    # / Step 3 errors).
    psx_files_dir = psx_path.parent / f"{psx_path.stem}.files"
    if psx_files_dir.exists():
        shutil.rmtree(psx_files_dir, ignore_errors=True)
    if psx_path.exists():
        psx_path.unlink(missing_ok=True)

    doc = Metashape.Document()
    doc.save(str(psx_path))
    chunk       = doc.addChunk()
    chunk.label = f"{date}_4DSfM"

    chunk.addPhotos(reg_df["image_path"].tolist())
    chunk.addPhotos([str(p) for imgs in date_images.values() for p in imgs])
    print(f"  Added {n_ref} reference + {n_new} new-day images", flush=True)

    # ── Sensors ───────────────────────────────────────────────────────────
    # One fixed sensor per camera per reference day preserves each day's
    # validated IOP; new-day gets one floating sensor per camera.
    ref_sensors, new_sensors = _setup_sensors_multitemporal(
        chunk, reg_df, new_calib_xmls
    )
    _assign_sensors_multitemporal(chunk, ref_sensors, new_sensors, date)
    n_ref_days = reg_df["date"].nunique()
    print(f"  Sensors : {n_ref_days} ref days × {len(CAMERAS)} cameras (fixed IOP) "
          f"+ {len(new_sensors)} new (floating IOP)")

    # ── Camera groups (by camera prefix) ──────────────────────────────────
    groups = _setup_camera_groups(chunk, date_images)
    _assign_camera_groups(chunk, groups)

    chunk.crs = Metashape.CoordinateSystem("EPSG::4326")

    # ── Reference EOP — reference cameras (tight) ─────────────────────────
    ref_lookup = {str(r["label"]): r for _, r in reg_df.iterrows()}
    ref_matched = 0
    for camera in chunk.cameras:
        if date in camera.label:
            continue
        r = ref_lookup.get(camera.label)
        if r is None:
            continue
        camera.reference.location          = Metashape.Vector([float(r["lon"]), float(r["lat"]), float(r["alt"])])
        camera.reference.location_accuracy = Metashape.Vector([0.001, 0.001, 0.001])
        camera.reference.rotation          = Metashape.Vector([float(r["yaw"]), float(r["pitch"]), float(r["roll"])])
        camera.reference.rotation_accuracy = Metashape.Vector([0.001, 0.001, 0.001])
        camera.reference.enabled           = True
        camera.reference.rotation_enabled  = True
        ref_matched += 1
    print(f"  Reference cameras (tight 0.001) : {ref_matched}/{n_ref}", flush=True)

    # ── Reference EOP — new-day cameras (mean of last registry day, loose) ─
    new_matched = 0
    for camera in chunk.cameras:
        if date not in camera.label:
            continue
        m = mean_eop.get(_camera_prefix(camera.label))
        if m is None:
            continue
        camera.reference.location          = Metashape.Vector([m["lon"], m["lat"], m["alt"]])
        camera.reference.location_accuracy = Metashape.Vector(list(loc_acc_new))
        camera.reference.rotation          = Metashape.Vector([m["yaw"], m["pitch"], m["roll"]])
        camera.reference.rotation_accuracy = Metashape.Vector(list(rot_acc_new))
        camera.reference.enabled           = True
        camera.reference.rotation_enabled  = True
        new_matched += 1
    print(f"  New-day cameras  (loose {loc_acc_new[0]} m / {rot_acc_new[0]}°) : {new_matched}/{n_new}", flush=True)

    # ── Bundle adjustment ─────────────────────────────────────────────────
    print("  Matching photos ...", flush=True)
    chunk.matchPhotos(
        downscale=match_downscale,
        keypoint_limit=80000,
        tiepoint_limit=8000,
        generic_preselection=True,
        reference_preselection=False,
    )

    print("  Aligning cameras ...", flush=True)
    chunk.alignCameras(adaptive_fitting=False)
    doc.save()

    aligned = sum(1 for c in chunk.cameras if c.transform)
    print(f"  Aligned {aligned}/{len(chunk.cameras)} cameras", flush=True)
    if aligned == 0:
        raise RuntimeError(f"No cameras aligned in 4D SfM BA for {date}.")

    print("  Optimising cameras ...", flush=True)
    chunk.optimizeCameras(
        fit_f=True, fit_cx=True, fit_cy=True,
        fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=False,
        fit_p1=True, fit_p2=True,
        fit_b1=False, fit_b2=False,
        adaptive_fitting=False,
        tiepoint_covariance=True,
    )
    doc.save()

    # ── Exports ───────────────────────────────────────────────────────────
    cameras_csv = sfm_dir / f"{date}_cameras_4DSfM.csv"
    calib_dir   = sfm_dir / "adjusted_calib_4DSfM"

    _export_camera_csv_filtered(chunk, cameras_csv, date)
    _export_calib_xml(new_sensors, calib_dir)

    doc.save()
    print(f"  Project saved : {psx_path.name}", flush=True)
    return cameras_csv, calib_dir


# ---------------------------------------------------------------------------
# 4D SfM pipeline — single-day re-run with fixed IOP
# ---------------------------------------------------------------------------

def run_single_day_fixed_iop(
    date: str,
    date_images: dict[str, list[Path]],
    calib_dir: Path,
    cameras_csv: Path,
    output_dir: Path,
    utm_epsg: int,
    match_downscale: int = 1,
    depth_downscale: int = 2,
    loc_acc: tuple = (0.5, 0.5, 0.5),
    rot_acc: tuple = (5.0, 5.0, 5.0),
) -> tuple[Path, Path]:
    """Single-day re-run with IOP fixed and EOP loose.

    Uses the calibration from :func:`run_multitemporal_ba` (locked) and the
    refined camera positions as EOP priors.  Generates the point cloud that
    will be co-registered against the initial reference cloud.

    Parameters
    ----------
    date : str
        Date to process (YYYY-MM-DD).
    date_images : dict
        ``{camera_label: [image_paths]}`` for this day.
    calib_dir : Path
        ``adjusted_calib_4DSfM/`` from :func:`run_multitemporal_ba`.
    cameras_csv : Path
        ``YYYY-MM-DD_cameras_4DSfM.csv`` from :func:`run_multitemporal_ba`.
    output_dir : Path
        Root output directory.
    utm_epsg : int
        UTM EPSG code.
    match_downscale : int
        Downscale for ``matchPhotos``.
    depth_downscale : int
        Downscale for ``buildDepthMaps``.
    loc_acc : tuple
        Position accuracy (m) — same loose value as in multi-temporal BA.
    rot_acc : tuple
        Rotation accuracy (°) — same loose value as in multi-temporal BA.

    Returns
    -------
    laz_path : Path
        Exported point cloud (``single_day/YYYY-MM-DD_cloud.laz``).
    cameras_csv_out : Path
        Exported camera positions after single-day BA
        (``single_day/YYYY-MM-DD_cameras.csv``).
    """
    if Metashape is None:
        raise ImportError("Metashape Python module is not installed.")

    calib_dir   = Path(calib_dir)
    cameras_csv = Path(cameras_csv)
    output_dir  = Path(output_dir)

    day_dir  = output_dir / "output_new" / date / "single_day"
    day_dir.mkdir(parents=True, exist_ok=True)
    psx_path = day_dir / f"{date}.psx"

    n_imgs = sum(len(v) for v in date_images.values())
    print(f"\n{'='*60}", flush=True)
    print(f"  Single-day BA (fixed IOP) : {date}   ({n_imgs} images)", flush=True)
    print(f"{'='*60}", flush=True)

    calib_xmls = {
        cam: calib_dir / f"{cam}.xml"
        for cam in CAMERAS
        if (calib_dir / f"{cam}.xml").exists()
    }

    # ── Build project ─────────────────────────────────────────────────────
    # Nuke any stale .psx + .files from a prior crashed run. Without this,
    # Metashape can flip the document to read-only on the second doc.save()
    # (after alignCameras) — see the "editing is disabled in read-only mode"
    # error that surfaces when re-running after a crash.
    psx_files_dir = psx_path.parent / f"{psx_path.stem}.files"
    if psx_files_dir.exists():
        shutil.rmtree(psx_files_dir, ignore_errors=True)
    if psx_path.exists():
        psx_path.unlink(missing_ok=True)

    doc = Metashape.Document()
    doc.save(str(psx_path))
    chunk       = doc.addChunk()
    chunk.label = date

    all_images = [str(p) for imgs in date_images.values() for p in imgs]
    chunk.addPhotos(all_images)
    print(f"  Added {len(all_images)} photos", flush=True)

    # ── Sensors (IOP fixed) ───────────────────────────────────────────────
    sensors = _setup_sensors(chunk, calib_xmls, fixed_calibration=True)
    _assign_sensors(chunk, sensors)
    groups = _setup_camera_groups(chunk, date_images)
    _assign_camera_groups(chunk, groups)
    print(f"  Configured {len(sensors)} sensors (IOP fixed)", flush=True)

    # ── EOP priors from 4DSfM cameras CSV (loose) ─────────────────────────
    chunk.crs = Metashape.CoordinateSystem("EPSG::4326")
    matched   = _set_camera_references_from_csv(chunk, cameras_csv, loc_acc, rot_acc)
    print(f"  EOP priors (loose {loc_acc[0]} m / {rot_acc[0]}°) : {matched}/{len(chunk.cameras)}", flush=True)

    # ── Bundle adjustment ─────────────────────────────────────────────────
    print("  Matching photos ...", flush=True)
    chunk.matchPhotos(
        downscale=match_downscale,
        keypoint_limit=80000,
        tiepoint_limit=8000,
        generic_preselection=True,
        reference_preselection=False,
    )

    print("  Aligning cameras ...", flush=True)
    chunk.alignCameras(adaptive_fitting=False)
    doc.save()

    aligned = sum(1 for c in chunk.cameras if c.transform)
    print(f"  Aligned {aligned}/{len(chunk.cameras)} cameras", flush=True)
    if aligned == 0:
        raise RuntimeError(f"No cameras aligned in single-day BA for {date}.")

    print("  Optimising cameras (IOP fixed) ...", flush=True)
    chunk.optimizeCameras(
        fit_f=True, fit_cx=True, fit_cy=True,
        fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=False,
        fit_p1=True, fit_p2=True,
        fit_b1=False, fit_b2=False,
        adaptive_fitting=False,
        tiepoint_covariance=True,
    )
    doc.save()

    # ── Dense point cloud ─────────────────────────────────────────────────
    print("  Building depth maps ...", flush=True)
    chunk.buildDepthMaps(downscale=depth_downscale, filter_mode=Metashape.MildFiltering)
    print("  Building point cloud ...", flush=True)
    chunk.buildPointCloud()
    doc.save()

    # ── Exports ───────────────────────────────────────────────────────────
    # Export as uncompressed .las so ASP pc_align can read it directly without
    # decompressing on every stage (saves a ~250 MB duplicate `_full.las`
    # write inside the coreg step).
    cloud_path = day_dir / f"{date}_cloud.las"
    utm_crs    = Metashape.CoordinateSystem(f"EPSG::{utm_epsg}")
    chunk.exportPointCloud(
        str(cloud_path),
        source_data=Metashape.PointCloudData,
        format=Metashape.PointCloudFormatLAS,
        crs=utm_crs,
        save_point_color=True,
    )
    print(f"  Exported LAS : {cloud_path.name}  (EPSG:{utm_epsg})", flush=True)

    cameras_csv_out = day_dir / f"{date}_cameras.csv"
    _export_camera_csv(chunk, cameras_csv_out)

    doc.save()
    print(f"  Project saved : {psx_path.name}", flush=True)
    return cloud_path, cameras_csv_out


# ---------------------------------------------------------------------------
# 4D SfM pipeline — rebuild co-registered cloud from a Metashape project
# ---------------------------------------------------------------------------

def _read_4x4_matrix(path: Path) -> np.ndarray:
    """Parse the first 4-row × 4-col float matrix found in *path* (ASP format)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 4:
            try:
                rows.append(list(map(float, parts)))
            except ValueError:
                pass
        if len(rows) == 4:
            break
    if len(rows) != 4:
        raise ValueError(f"Could not parse a 4×4 matrix from {path}")
    return np.array(rows)


def rebuild_coreg_cloud(
    psx_path: Path,
    transform_path: Path,
    output_laz: Path,
    depth_downscale: int,
    utm_epsg: int,
) -> Path:
    """Apply the ASP ECEF transform to a Metashape project and rebuild its cloud.

    Composes ``M_new = T_ecef @ M_old`` on ``chunk.transform.matrix`` (both
    live in WGS84 geocentric ECEF — same space as the ASP transform, so no
    CRS conversion needed), then runs ``buildDepthMaps``, ``buildPointCloud``,
    and ``exportPointCloud`` in the same session. Metashape recomputes
    ``chunk.transform.matrix`` from GPS priors on every ``doc.open()``, so any
    matrix assignment must be followed by the cloud rebuild + export in the
    same session — otherwise the corrected transform is silently lost.

    Parameters
    ----------
    psx_path : Path
        Single-day Metashape project (``*.psx``) from
        :func:`run_single_day_fixed_iop`.
    transform_path : Path
        ASP ``run-transform.txt`` from the final pc_align stage (Stage 3's
        file contains the full composed T3 ∘ T2 ∘ T1).
    output_laz : Path
        Validated cloud destination (``.laz``).
    depth_downscale : int
        ``buildDepthMaps`` downscale (1 = full, 2 = half, 4 = quarter …).
    utm_epsg : int
        UTM EPSG for the exported cloud's CRS.

    Returns
    -------
    Path
        ``output_laz``.
    """
    if Metashape is None:
        raise ImportError("Metashape Python module is not installed.")

    psx_path       = Path(psx_path)
    transform_path = Path(transform_path)
    output_laz     = Path(output_laz)
    output_laz.parent.mkdir(parents=True, exist_ok=True)

    T_ecef = _read_4x4_matrix(transform_path)

    doc = Metashape.Document()
    doc.clear()
    doc.open(str(psx_path), read_only=False, ignore_lock=True)
    chunk = doc.chunk

    def _cam_wgs84() -> tuple:
        cam = next((c for c in chunk.cameras if c.transform), None)
        if cam is None:
            return None, None
        p = chunk.crs.project(chunk.transform.matrix.mulp(cam.center))
        return cam.label, (p[0], p[1], p[2])

    def _shift_m(p0: tuple, p1: tuple) -> float:
        import math
        dlat = (p1[1] - p0[1]) * 111_320
        dlon = (p1[0] - p0[0]) * 111_320 * math.cos(math.radians((p0[1] + p1[1]) / 2))
        return (dlat ** 2 + dlon ** 2 + (p1[2] - p0[2]) ** 2) ** 0.5

    lbl, p0 = _cam_wgs84()
    print(f"  [{lbl}] ON LOAD : alt={p0[2]:.3f} m", flush=True)

    M_old = chunk.transform.matrix
    M_np  = np.array([[M_old[r, c] for c in range(4)] for r in range(4)])
    M_new = T_ecef @ M_np
    chunk.transform.matrix = Metashape.Matrix(
        [[float(M_new[r, c]) for c in range(4)] for r in range(4)]
    )

    _, p1 = _cam_wgs84()
    print(f"  [{lbl}] AFTER T : alt={p1[2]:.3f} m  shift={_shift_m(p0, p1):.4f} m", flush=True)

    print("  Building depth maps ...", flush=True)
    chunk.buildDepthMaps(downscale=depth_downscale, filter_mode=Metashape.MildFiltering)

    print("  Building point cloud ...", flush=True)
    chunk.buildPointCloud()

    _, p2 = _cam_wgs84()
    print(f"  [{lbl}] AFTER buildPointCloud : alt={p2[2]:.3f} m  shift from T={_shift_m(p1, p2):.4f} m", flush=True)

    utm_crs = Metashape.CoordinateSystem(f"EPSG::{utm_epsg}")
    chunk.exportPointCloud(
        str(output_laz),
        source_data      = Metashape.PointCloudData,
        format           = Metashape.PointCloudFormatLAS,
        crs              = utm_crs,
        save_point_color = True,
    )
    doc.save()
    print(f"  Validation cloud → {output_laz.name}", flush=True)
    return output_laz

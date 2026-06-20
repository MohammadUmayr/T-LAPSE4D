"""
Metashape SfM engine for the 4D time-lapse pipeline.

Thin Python wrappers around the Agisoft Metashape API that drive the
multi-temporal bundle adjustment, single-day fixed-IOP reconstruction,
sensor/calibration setup, and post-coregistration camera updates. They are
orchestrated per date by :mod:`cntp.pipeline_4dsfm` (``run_4dsfm_day``); the
reference registry that feeds them is built by :func:`bootstrap_registry`.

Metashape is an optional runtime dependency, imported lazily inside the
functions that need it, so this module (and non-Metashape helpers like
:func:`discover_images`) imports cleanly without a license.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

# Metashape is an optional runtime dependency, so the module can be imported
# (e.g. for the non-Metashape helpers, or unit tests) without it installed.
# The TYPE_CHECKING branch lets the type checker resolve `Metashape.*` type
# annotations; without it the runtime `Metashape = None` fallback makes
# `Metashape` look like a variable ("Variable not allowed in type expression").
if TYPE_CHECKING:
    import Metashape
else:
    try:
        import Metashape
    except ImportError:
        Metashape = None

# ---------------------------------------------------------------------------
# Native-output control
# ---------------------------------------------------------------------------

@contextmanager
def _quiet_metashape(verbose: bool, log_path: "str | Path | None" = None):
    """Route Metashape's native processing chatter to a log file unless ``verbose``.

    Metashape's progress bars, tie-point counts and timing lines are written by
    the C++ library straight to the process's stdout/stderr file descriptors, so
    gating Python ``print`` calls can't hide them. When ``verbose`` is False this
    redirects fd 1 & 2 to ``log_path`` (appended) for the duration of the wrapped
    call — the library output is captured to the file while our own ``print``
    summaries (emitted outside the wrapped block) still reach the console. When
    True it is a no-op, so the full Metashape log is shown inline. With no
    ``log_path`` the captured output is discarded (``os.devnull``).
    """
    if verbose:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    if log_path is None:
        target, flags = os.devnull, os.O_WRONLY
    else:
        target = os.fspath(log_path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(target, flags, 0o644)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(fd)
        os.close(saved_out)
        os.close(saved_err)


def _init_native_log(verbose: bool, log_path: "Path | None") -> "Path | None":
    """Truncate/create the native-output log once at the start of a step.

    Returns the path to pass to :func:`_quiet_metashape` for each heavy call
    (``None`` when ``verbose`` so the context manager stays a no-op).
    """
    if verbose or log_path is None:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")  # fresh file per step run
    return log_path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A standardised time-lapse image filename is
#   <camera>_<YYYY-MM-DD>_<HHMMSS>.<ext>     e.g.  C7_2023-11-27_083000.JPG
# The camera id is everything before the first underscore, so *any* naming
# works (C1..C10, CAM1, EastRidge, …) as long as the id has no underscore.
# Camera ids are read from the data on disk — never hardcoded — so a new site
# needs no code change, only its own renamer (or already-standard filenames)
# producing this pattern. See cntp.preprocess.homogenize_images.
_IMG_RE = re.compile(
    r"^(?P<cam>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})_\d{6}\.(?:jpg|jpeg|JPG|JPEG)$"
)

# Same shape without the file extension, for matching Metashape camera /
# sensor *labels* (which usually carry no extension). Used to tell genuine
# time-lapse photos apart from drone/UAV images sharing a project — the
# drone's labels (e.g. "DJI_0457") don't match this shape.
_LABEL_RE = re.compile(r"^[^_]+_\d{4}-\d{2}-\d{2}_\d{6}$")


def is_timelapse_label(label: object) -> bool:
    """Return True if *label* looks like a standardised time-lapse photo.

    A genuine time-lapse image/sensor label has the shape
    ``<camera>_<YYYY-MM-DD>_<HHMMSS>`` (optionally with a file extension).
    Drone/UAV images mixed into the same Metashape project — e.g.
    ``DJI_0457`` — don't match, so this is the data-driven replacement for the
    old hardcoded ``CAMERAS`` whitelist when filtering them out.
    """
    return bool(_LABEL_RE.match(Path(str(label)).stem))


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
    """Scan *tlcam_dir* recursively and return ``{date: {camera: [paths]}}``.

    The camera id and date are read from each standardised filename
    (``<camera>_<YYYY-MM-DD>_<HHMMSS>.<ext>``), so the on-disk folder layout is
    irrelevant — ``C6_renamed/``, ``C6/``, or all images in one folder all
    work. Files that don't match the pattern (drone shots, ``notes.jpg``, …)
    are ignored.

    Parameters
    ----------
    tlcam_dir : str | Path
        Any directory (searched recursively) holding standardised time-lapse
        images. See :func:`cntp.preprocess.homogenize_images` to produce them.
    """
    tlcam_dir = Path(tlcam_dir)
    by_date: dict = defaultdict(lambda: defaultdict(list))
    # Match on the filename only — no per-entry is_file() stat. The strict
    # `<cam>_<date>_<time>.<ext>` pattern already excludes directories, and on
    # network/drvfs mounts (e.g. /mnt/g) a stat per file is ~1000x slower than
    # the name match (200 s vs 0.2 s for ~16k files).
    for img in sorted(tlcam_dir.rglob("*")):
        m = _IMG_RE.match(img.name)
        if m:
            by_date[m.group("date")][m.group("cam")].append(img)

    return {d: dict(cams) for d, cams in sorted(by_date.items())}


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
    for cam in sorted(calib_xmls):
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

def _export_camera_csv(chunk: "Metashape.Chunk", out_path: Path, keep=None) -> None:
    """Export estimated camera positions and orientations to a WGS84 CSV.

    Follows the HSFM approach: position from chunk.crs.project(T.mulp(camera.center)),
    Yaw/Pitch/Roll from Metashape.utils.mat2ypr with the local geographic frame.

    Parameters
    ----------
    keep : callable, optional
        Predicate ``label -> bool``. When given, only cameras whose label
        satisfies it are exported — used to drop drone/UAV images via
        :func:`is_timelapse_label`. Default exports every aligned camera.
    """
    T = chunk.transform.matrix
    rows = []

    for cam in chunk.cameras:
        if cam.transform is None:
            continue
        if keep is not None and not keep(cam.label):
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
# Co-registration camera import  (run after ASP pc_align)
# ---------------------------------------------------------------------------

def update_metashape_cameras_after_transform(
    psx_path: Path,
    transform_path: Path,
    chunk_label: str = None,
) -> None:
    """Apply an ASP pc_align ECEF transform directly to the Metashape chunk transform.

    Workflow: single-day reconstruction → ASP pc_align (use_ecef=True) → this
    function (driven per date by :func:`cntp.pipeline_4dsfm.run_4dsfm_day`).

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
        for ref_xml in sorted(calib_dir.glob("*.xml")):
            cam = ref_xml.stem
            calib = load_calib_xml(ref_xml)
            s = chunk.addSensor()
            s.label             = f"{cam}_{date}"
            s.type              = Metashape.Sensor.Type.Frame
            s.width             = calib.width
            s.height            = calib.height
            s.user_calib        = calib
            s.fixed_calibration = True
            ref_sensors[(cam, date)] = s

    for cam in sorted(new_calib_xmls):
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
        existing = existing[existing["date"] != _normalize_date(date)]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined["date"] = combined["date"].map(_normalize_date)
    combined.to_csv(registry_csv, index=False)
    print(f"  Registry updated : {registry_csv.name}  ({len(rows)} rows added, "
          f"{len(combined)} total)")


def bootstrap_registry(
    ref_psx: str | Path,
    chunk_label: str,
    ref_date: str,
    output_dir: str | Path,
    registry_csv: str | Path,
    cameras_csv_out: str | Path = None,
    calib_dir_out: str | Path = None,
    ref_cloud_out: str | Path = None,
    export_ref_cloud: bool = True,
    overwrite: bool = False,
) -> dict:
    """Initialise the reference registry from an existing reference project.

    One-time, per-glacier setup. Opens a finished reference Metashape project,
    pulls out the time-lapse cameras' interior orientation (calibration XMLs)
    and exterior orientation (position + yaw/pitch/roll), exports the dense
    **reference point cloud** (UTM ``.laz``), and writes the registry so the
    per-day 4D SfM pipeline can run afterwards. Replaces the old standalone
    ``bootstrap_registry.ipynb`` — call this once from the main notebook instead.

    Drone/UAV images sharing the project are skipped automatically via the
    standardised-filename test :func:`is_timelapse_label` (no camera list to
    maintain).

    Parameters
    ----------
    ref_psx : str | Path
        Path to the reference ``.psx`` project.
    chunk_label : str
        Exact label of the chunk to read (as shown in Metashape's workspace).
    ref_date : str
        Date (``YYYY-MM-DD``) of the reference day.
    output_dir : str | Path
        Root output directory (parent of ``output/``), matching the value
        used by the per-day pipeline.
    registry_csv : str | Path
        Registry CSV to create.
    cameras_csv_out, calib_dir_out : str | Path, optional
        Where to write the reference day's camera CSV and calibration XMLs.
        Default ``output_dir/output/<ref_date>/4D_SfM/…`` — the location
        the pipeline expects.
    ref_cloud_out : str | Path, optional
        Where to write the reference point cloud (``.laz``, exported from the
        chunk's dense cloud in UTM). Default ``output_dir/output/reference.laz``.
        This is the ``ref_cloud`` the per-day pipeline co-registers against.
    export_ref_cloud : bool
        When True (default), also export the dense reference point cloud. Set
        False to only (re)build the registry + calibrations.
    overwrite : bool
        When False (default) and *registry_csv* already exists, do nothing and
        return — so re-running the setup cell is harmless. True rebuilds.

    Returns
    -------
    dict
        ``registry_csv, cameras_csv, calib_dir, ref_cloud, n_cameras, n_sensors,
        cameras``.
    """
    if Metashape is None:
        raise ImportError(
            "Metashape Python module is not installed (or AGISOFT_LICENSE_PATH "
            "was not set before `import cntp`)."
        )

    ref_psx      = Path(ref_psx)
    output_dir   = Path(output_dir)
    registry_csv = Path(registry_csv)

    if registry_csv.exists() and not overwrite:
        print(f"Registry already exists — skipping bootstrap: {registry_csv}")
        print("  (pass overwrite=True to rebuild)")
        return {
            "registry_csv": registry_csv, "cameras_csv": cameras_csv_out,
            "calib_dir": calib_dir_out, "ref_cloud": None, "n_cameras": None,
            "n_sensors": None, "cameras": None,
        }

    ref_export_dir = output_dir / "output" / ref_date / "4D_SfM"
    if cameras_csv_out is None:
        cameras_csv_out = ref_export_dir / f"{ref_date}_cameras_4DSfM.csv"
    if calib_dir_out is None:
        calib_dir_out = ref_export_dir / "adjusted_calib_4DSfM"
    cameras_csv_out = Path(cameras_csv_out)
    calib_dir_out   = Path(calib_dir_out)
    cameras_csv_out.parent.mkdir(parents=True, exist_ok=True)
    calib_dir_out.mkdir(parents=True, exist_ok=True)
    registry_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── Open project + select chunk ──────────────────────────────────────
    doc = Metashape.Document()
    doc.open(str(ref_psx), read_only=True, ignore_lock=True)
    chunk = next((c for c in doc.chunks if c.label == chunk_label), None)
    if chunk is None:
        available = [c.label for c in doc.chunks]
        raise ValueError(f"Chunk '{chunk_label}' not found. Available: {available}")
    aligned = sum(1 for c in chunk.cameras if c.transform)
    print(f"Chunk   : '{chunk.label}'  ({aligned}/{len(chunk.cameras)} aligned)")

    # ── Exterior orientation (EOP) — time-lapse cameras only ─────────────
    _export_camera_csv(chunk, cameras_csv_out, keep=is_timelapse_label)

    # Valid camera ids = prefixes of the standardised photos. Used to keep the
    # matching sensors and drop the drone sensor (whose prefix never appears).
    valid_cams = {
        _camera_prefix(c.label)
        for c in chunk.cameras
        if is_timelapse_label(c.label)
    }

    # ── Interior orientation (IOP) — matching sensors only ───────────────
    exported = []
    for sensor in chunk.sensors:
        cam = sensor.label.split("_")[0]
        if cam not in valid_cams:
            print(f"  Skipping sensor : {sensor.label}")
            continue
        out_path = calib_dir_out / f"{cam}.xml"
        sensor.calibration.save(str(out_path), format=Metashape.CalibrationFormatXML)
        exported.append(cam)
        print(f"  Exported calibration : {out_path.name}")

    # ── Reconstruct date_images from photo paths ─────────────────────────
    date_images: dict = defaultdict(list)
    missing = 0
    for cam in chunk.cameras:
        if not is_timelapse_label(cam.label):
            continue
        if cam.photo is None or not cam.photo.path:
            missing += 1
            continue
        date_images[_camera_prefix(cam.label)].append(Path(cam.photo.path))
    date_images = dict(date_images)
    if missing:
        print(f"  WARNING: {missing} time-lapse cameras had no photo path")

    # ── Write registry ───────────────────────────────────────────────────
    if overwrite and registry_csv.exists():
        registry_csv.unlink()
    update_registry(
        registry_csv      = registry_csv,
        date              = ref_date,
        date_images       = date_images,
        cameras_coreg_csv = cameras_csv_out,
        calib_dir         = calib_dir_out,
    )

    # ── Reference point cloud (chunk dense cloud → UTM .laz) ──────────────
    # Exported as .laz (compressed) to match the existing reference cloud, so
    # the per-day `ref_cloud` path stays unchanged.
    ref_cloud_path = None
    if export_ref_cloud:
        cloud_out = (Path(ref_cloud_out) if ref_cloud_out is not None
                     else output_dir / "output" / "reference.laz")
        cloud_out.parent.mkdir(parents=True, exist_ok=True)
        if cloud_out.exists() and not overwrite:
            print(f"  Reference cloud exists — skipping export: {cloud_out.name}")
        else:
            # UTM zone from the time-lapse cameras' mean longitude — the same
            # zone run_4dsfm_day derives from the registry, so the per-day clouds
            # co-register against this cloud in a matching CRS.
            mean_lon = pd.read_csv(cameras_csv_out)["Lon"].mean()
            utm_epsg = _utm_epsg(mean_lon)
            chunk.exportPointCloud(
                str(cloud_out),
                source_data      = Metashape.PointCloudData,
                format           = Metashape.PointCloudFormatLAZ,
                crs              = Metashape.CoordinateSystem(f"EPSG::{utm_epsg}"),
                save_point_color = True,
            )
            print(f"  Exported reference cloud : {cloud_out.name}  (EPSG:{utm_epsg}, .laz)")
        ref_cloud_path = cloud_out

    return {
        "registry_csv": registry_csv,
        "cameras_csv":  cameras_csv_out,
        "calib_dir":    calib_dir_out,
        "ref_cloud":    ref_cloud_path,
        "n_cameras":    len(date_images),
        "n_sensors":    len(set(exported)),
        "cameras":      sorted(valid_cams),
    }


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
    verbose: bool = False,
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

    sfm_dir  = output_dir / "output" / date / "4D_SfM"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    psx_path = sfm_dir / f"{date}_4DSfM.psx"

    # Native Metashape chatter → 4D_SfM/<date>_metashape.log unless verbose.
    native_log = _init_native_log(verbose, sfm_dir / f"{date}_metashape.log")

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
    new_calib_xmls: dict = {
        xml.stem: xml for xml in sorted(last_calib_dir.glob("*.xml"))
    }
    if not new_calib_xmls:
        raise FileNotFoundError(
            f"No calibration XMLs found in last registry calib dir: "
            f"{last_calib_dir}. Re-run bootstrap_registry or update_registry."
        )

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
    print(f"  Sensors : {n_ref_days} ref days × {len(new_calib_xmls)} cameras (fixed IOP) "
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
    with _quiet_metashape(verbose, native_log):
        chunk.matchPhotos(
            downscale=match_downscale,
            keypoint_limit=80000,
            tiepoint_limit=8000,
            generic_preselection=True,
            reference_preselection=False,
        )

    print("  Aligning cameras ...", flush=True)
    with _quiet_metashape(verbose, native_log):
        chunk.alignCameras(adaptive_fitting=False)
    doc.save()

    aligned = sum(1 for c in chunk.cameras if c.transform)
    print(f"  Aligned {aligned}/{len(chunk.cameras)} cameras", flush=True)
    if aligned == 0:
        raise RuntimeError(f"No cameras aligned in 4D SfM BA for {date}.")

    # Step 1 never aborts on partial alignment (Step 2 may still recover the
    # cameras and enforces the cloud-cover gate). But if any new-day camera
    # failed to align, record which ones to a text file in the 4D_SfM folder —
    # written only when there is something to report.
    unaligned = [c.label for c in chunk.cameras if date in c.label and not c.transform]
    if unaligned:
        report = sfm_dir / f"{date}_unaligned_step1.txt"
        report.write_text(
            f"{date} multi-temporal BA (Step 1): "
            f"{len(unaligned)}/{n_new} new-day cameras unaligned\n"
            + "\n".join(unaligned) + "\n"
        )
        print(f"  NOTE: {len(unaligned)}/{n_new} new-day cameras unaligned in "
              f"Step 1 → {report.name}", flush=True)

    print("  Optimising cameras ...", flush=True)
    with _quiet_metashape(verbose, native_log):
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
    filter_mode: str = "Mild",
    loc_acc: tuple = (0.5, 0.5, 0.5),
    rot_acc: tuple = (5.0, 5.0, 5.0),
    verbose: bool = False,
    max_unaligned: int = 10,
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

    day_dir  = output_dir / "output" / date / "single_day"
    day_dir.mkdir(parents=True, exist_ok=True)
    psx_path = day_dir / f"{date}.psx"

    # Native Metashape chatter → single_day/<date>_metashape.log unless verbose.
    native_log = _init_native_log(verbose, day_dir / f"{date}_metashape.log")

    n_imgs = sum(len(v) for v in date_images.values())
    print(f"\n{'='*60}", flush=True)
    print(f"  Single-day BA (fixed IOP) : {date}   ({n_imgs} images)", flush=True)
    print(f"{'='*60}", flush=True)

    calib_xmls = {
        xml.stem: xml for xml in sorted(calib_dir.glob("*.xml"))
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
    with _quiet_metashape(verbose, native_log):
        chunk.matchPhotos(
            downscale=match_downscale,
            keypoint_limit=80000,
            tiepoint_limit=8000,
            generic_preselection=True,
            reference_preselection=False,
        )

    print("  Aligning cameras ...", flush=True)
    with _quiet_metashape(verbose, native_log):
        chunk.alignCameras(adaptive_fitting=False)
    doc.save()

    aligned   = sum(1 for c in chunk.cameras if c.transform)
    unaligned = len(chunk.cameras) - aligned
    print(f"  Aligned {aligned}/{len(chunk.cameras)} cameras", flush=True)
    if aligned == 0:
        raise RuntimeError(f"No cameras aligned in single-day BA for {date}.")

    # Cloud-cover gate: abort before the costly dense build when too many of the
    # day's cameras failed to align (heavy cloud → many unaligned frames).
    if unaligned >= max_unaligned:
        raise RuntimeError(
            f"{date}: {unaligned}/{len(chunk.cameras)} cameras unaligned "
            f"(>= max_unaligned={max_unaligned}) — likely cloud cover; skipping date."
        )

    # `matched` = this day's cameras Step 1 aligned & exported to the CSV. If
    # Step 2 aligned more, it recovered cameras Step 1 missed — surface that.
    if aligned > matched:
        print(f"  NOTE: Step 2 aligned {aligned - matched} camera(s) that Step 1 "
              f"did not (Step 1 aligned {matched}/{len(chunk.cameras)}).", flush=True)

    print("  Optimising cameras (IOP fixed) ...", flush=True)
    with _quiet_metashape(verbose, native_log):
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
    _filter = {"Mild": Metashape.MildFiltering, "Moderate": Metashape.ModerateFiltering,
               "Aggressive": Metashape.AggressiveFiltering}
    print(f"  Building depth maps (filter={filter_mode}) ...", flush=True)
    with _quiet_metashape(verbose, native_log):
        chunk.buildDepthMaps(downscale=depth_downscale, filter_mode=_filter[filter_mode])
    print("  Building point cloud ...", flush=True)
    with _quiet_metashape(verbose, native_log):
        chunk.buildPointCloud()
    doc.save()

    # ── Exports ───────────────────────────────────────────────────────────
    # Export as uncompressed .las so ASP pc_align can read it directly without
    # decompressing on every stage (saves a ~250 MB duplicate `_full.las`
    # write inside the coreg step).
    cloud_path = day_dir / f"{date}_cloud.las"
    utm_crs    = Metashape.CoordinateSystem(f"EPSG::{utm_epsg}")
    with _quiet_metashape(verbose, native_log):
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
    filter_mode: str = "Mild",
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

    _filter = {"Mild": Metashape.MildFiltering, "Moderate": Metashape.ModerateFiltering,
               "Aggressive": Metashape.AggressiveFiltering}
    print(f"  Building depth maps (filter={filter_mode}) ...", flush=True)
    chunk.buildDepthMaps(downscale=depth_downscale, filter_mode=_filter[filter_mode])

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

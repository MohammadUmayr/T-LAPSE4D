"""
HSfM-style three-stage ICP co-registration for point clouds.

Mirrors the ``pc_align_p2p_sp2p`` pipeline from Knuth et al. (2023) §4.3.3,
wrapping NASA's Ames Stereo Pipeline (ASP) ``pc_align`` CLI tool.

Stages (paper thresholds for km-scale NAGAP errors / UAV defaults):
  Stage 1 — point-to-plane ICP          2000 m / 2.0 m  rigid body
  Stage 2 — similarity-point-to-point   1000 m / 1.0 m  + scale correction
  Stage 3 — similarity-pt-to-pt,         100 m / 0.1 m  stable terrain only
             stable terrain only

ASP installation
----------------
  https://stereopipeline.readthedocs.io/en/latest/installation.html
"""

import os
import shutil
import subprocess
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import py4dgeo
from pyproj import Transformer
from cntp.coreg import extract_stable_terrain, run_m3c2, _NDWI_A, _NDWI_B
from cntp.io import load_las, apply_glacier_mask, save_las, read_las_bounds
from cntp.plot import plot_stable_terrain_diagnostics, plot_m3c2_distances


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_asp() -> None:
    """Raise a RuntimeError when pc_align is not on PATH."""
    if shutil.which("pc_align") is None:
        raise RuntimeError(
            "pc_align not found on PATH.\n"
            "Install NASA Ames Stereo Pipeline:\n"
            "  https://stereopipeline.readthedocs.io/en/latest/installation.html"
        )


def _run_command(cmd: list, verbose: bool = False) -> str:
    """Run *cmd* as a subprocess; print output if verbose; raise on non-zero exit."""
    str_cmd = [str(c) for c in cmd]
    if verbose:
        print(" ".join(str_cmd))
    result = subprocess.run(
        str_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if verbose:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n{result.stdout}"
        )
    return result.stdout


def _read_asp_transform(transform_path: Path) -> np.ndarray:
    """Parse an ASP pc_align ``*-transform.txt`` and return the 4×4 numpy array.

    ASP writes one header line then four rows of four space-separated floats.
    Because each stage passes its output as ``--initial-transform`` to the next,
    ASP compounds the transforms internally: Stage 3's file already contains
    the full composition T3 ∘ T2 ∘ T1.
    """
    rows = []
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
    return np.array(rows)


# ---------------------------------------------------------------------------
# Camera position transform helpers
# ---------------------------------------------------------------------------

def wgs84_to_utm(
    lon: np.ndarray,
    lat: np.ndarray,
    alt: np.ndarray,
    utm_epsg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert WGS84 geographic coordinates to UTM.

    Parameters
    ----------
    lon, lat : array-like
        Longitude and latitude in decimal degrees.
    alt : array-like
        Ellipsoidal height in metres.
    utm_epsg : int
        EPSG code of the target UTM zone (e.g. 32645 for UTM 45N).

    Returns
    -------
    easting, northing, elevation : np.ndarray
        UTM coordinates in metres.
    """
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    easting, northing = t.transform(np.asarray(lon), np.asarray(lat))
    return easting, northing, np.asarray(alt, dtype=np.float64)


def utm_to_wgs84(
    easting: np.ndarray,
    northing: np.ndarray,
    elevation: np.ndarray,
    utm_epsg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert UTM coordinates back to WGS84 geographic coordinates.

    Parameters
    ----------
    easting, northing : array-like
        UTM coordinates in metres.
    elevation : array-like
        Ellipsoidal height in metres.
    utm_epsg : int
        EPSG code of the source UTM zone.

    Returns
    -------
    lon, lat, alt : np.ndarray
        WGS84 longitude, latitude (decimal degrees) and height (metres).
    """
    t = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    lon, lat = t.transform(np.asarray(easting), np.asarray(northing))
    return lon, lat, np.asarray(elevation, dtype=np.float64)


def apply_coreg_to_cameras(
    cameras_csv: Path,
    transform_path: Path,
    utm_epsg: int,
    out_csv: Path = None,
) -> pd.DataFrame:
    """Apply an ASP co-registration transform to camera positions.

    Reads the CSV produced by the Metashape pipeline (columns: Label, Lon,
    Lat, Alt, Yaw, Pitch, Roll), converts positions to UTM, applies the 4×4
    similarity transform, and converts back to WGS84.  Yaw/Pitch/Roll are
    passed through unchanged — only positions are transformed.

    Parameters
    ----------
    cameras_csv : Path
        Camera positions CSV (Label, Lon, Lat, Alt, Yaw, Pitch, Roll).
    transform_path : Path
        ASP ``*-transform.txt`` — use ``stage3/run-transform.txt`` for the
        full composed transform T3 ∘ T2 ∘ T1.
    utm_epsg : int
        EPSG code of the UTM zone that matches the point cloud CRS.
    out_csv : Path, optional
        If given, write the updated DataFrame to this path.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with Lon, Lat, Alt replaced by the
        co-registered values.
    """
    df = pd.read_csv(Path(cameras_csv))
    T  = _read_asp_transform(Path(transform_path))

    east, north, elev = wgs84_to_utm(
        df["Lon"].to_numpy(), df["Lat"].to_numpy(), df["Alt"].to_numpy(),
        utm_epsg,
    )

    pts   = np.column_stack([east, north, elev])
    pts_t = (T[:3, :3] @ pts.T).T + T[:3, 3]

    lon, lat, alt = utm_to_wgs84(pts_t[:, 0], pts_t[:, 1], pts_t[:, 2], utm_epsg)

    out = df.copy()
    out["Lon"] = lon
    out["Lat"] = lat
    out["Alt"] = alt

    if out_csv is not None:
        out.to_csv(Path(out_csv), index=False)
        print(f"Saved transformed cameras → {out_csv}")

    return out


from cntp.metashape import update_metashape_cameras_after_transform  # noqa: F401


def las_utm_to_ecef(
    utm_las: Path,
    utm_epsg: int,
    out_path: Path,
    chunk_size: int = 2_000_000,
) -> Path:
    """Reproject a UTM LAZ/LAS cloud to ECEF (EPSG:4978) using laspy + pyproj.

    Converts XYZ coordinates via pyproj, computes new int32 raw values using
    ECEF-derived offsets and a fixed scale of 0.01 m (covers ±21 M m), and
    writes all other point attributes unchanged.  When the output is passed to
    ``pc_align``, ASP runs ICP in true ECEF space.

    Parameters
    ----------
    utm_las : Path
        Source LAZ/LAS in UTM projected coordinates.
    utm_epsg : int
        EPSG code of the source UTM zone (e.g. 32645).
    out_path : Path
        Destination LAS path (must end in .las — laspy writes uncompressed).
    chunk_size : int
        Points per chunk for memory-efficient processing.
    """
    utm_las  = Path(utm_las)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4978", always_xy=True)

    # Derive ECEF offsets from the UTM header offsets so int32 values never overflow
    with laspy.open(utm_las) as reader:
        utm_off = np.asarray(reader.header.offsets, dtype=np.float64)
        pf  = reader.header.point_format
        ver = reader.header.version

    ox, oy, oz = t.transform(utm_off[0], utm_off[1], utm_off[2])

    new_header = laspy.LasHeader(point_format=pf, version=ver)
    new_header.offsets = np.array([ox, oy, oz], dtype=np.float64)
    # scale=0.01 → max int32 = 2.1 B → covers ±21 M m (ECEF < 6.4 M m from offset)
    new_header.scales  = np.array([0.01, 0.01, 0.01], dtype=np.float64)

    with laspy.open(utm_las) as reader:
        with laspy.LasWriter(open(out_path, "wb"), header=new_header) as writer:
            for chunk in reader.chunk_iterator(chunk_size):
                xe, ye, ze = t.transform(
                    np.asarray(chunk.x, dtype=np.float64),
                    np.asarray(chunk.y, dtype=np.float64),
                    np.asarray(chunk.z, dtype=np.float64),
                )
                # chunk.array["X"][:] = ... writes to a copy (laspy returns views that
                # don't propagate back).  Copy the raw array first, then assign.
                arr = chunk.array.copy()
                arr["X"] = np.round((xe - ox) / 0.01).astype(np.int32)
                arr["Y"] = np.round((ye - oy) / 0.01).astype(np.int32)
                arr["Z"] = np.round((ze - oz) / 0.01).astype(np.int32)
                writer.write_points(laspy.PackedPointRecord(arr, new_header.point_format))

    return out_path


def apply_coreg_to_cameras_ecef(
    cameras_csv: Path,
    transform_path: Path,
    out_csv: Path = None,
    utm_epsg: int = None,
) -> "pd.DataFrame":
    """Apply an ASP co-registration transform to camera positions.

    When ``utm_epsg`` is *None* (default), the transform is applied in ECEF
    space: WGS84 → ECEF → T → ECEF → WGS84.  Use this when ``pc_align`` was
    run on ECEF point clouds (i.e. ``utm_epsg`` was passed to
    :func:`pc_align_p2p_sp2p`).

    When ``utm_epsg`` is provided, the transform is applied in flat UTM space:
    WGS84 → UTM → T → UTM → WGS84.  Use this for the standard UTM-mode
    pipeline.

    Parameters
    ----------
    cameras_csv : Path
        Camera positions CSV (Label, Lon, Lat, Alt, Yaw, Pitch, Roll).
    transform_path : Path
        ASP ``*-transform.txt`` — use ``stage3/run-transform.txt`` for the
        full composed transform.
    out_csv : Path, optional
        If given, write the updated DataFrame to this path.
    utm_epsg : int, optional
        EPSG code of the UTM zone (e.g. 32645).  When set, UTM mode is used;
        when ``None``, ECEF mode is used.
    """
    df = pd.read_csv(Path(cameras_csv))
    T  = _read_asp_transform(Path(transform_path))

    if utm_epsg is not None:
        east, north, elev = wgs84_to_utm(
            df["Lon"].to_numpy(), df["Lat"].to_numpy(), df["Alt"].to_numpy(),
            utm_epsg,
        )
        pts   = np.column_stack([east, north, elev])
        pts_t = (T[:3, :3] @ pts.T).T + T[:3, 3]
        lon, lat, alt = utm_to_wgs84(pts_t[:, 0], pts_t[:, 1], pts_t[:, 2], utm_epsg)
    else:
        to_ecef   = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
        from_ecef = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
        x, y, z = to_ecef.transform(
            df["Lon"].to_numpy(), df["Lat"].to_numpy(), df["Alt"].to_numpy()
        )
        pts   = np.column_stack([x, y, z])
        pts_t = (T[:3, :3] @ pts.T).T + T[:3, 3]
        lon, lat, alt = from_ecef.transform(pts_t[:, 0], pts_t[:, 1], pts_t[:, 2])

    out = df.copy()
    out["Lon"] = lon
    out["Lat"] = lat
    out["Alt"] = alt

    if out_csv is not None:
        out.to_csv(Path(out_csv), index=False)
        print(f"Saved transformed cameras → {out_csv}")

    return out


def _apply_transform_to_las(las_path: Path, T: np.ndarray, out_path: Path,
                             utm_epsg: int = None,
                             chunk_size: int = 2_000_000) -> None:
    """Apply 4×4 homogeneous transform *T* to a LAZ/LAS cloud and write *out_path*.

    UTM mode (``utm_epsg=None``): T is applied directly in UTM.
    ECEF mode (``utm_epsg`` set): each chunk is converted UTM → ECEF,
    T is applied in ECEF, then converted back to UTM.  The output LAS
    stays in UTM so M3C2 and downstream tools work unchanged.

    Normals are rotated by the pure rotation part of T.

    Streams in chunks so full-resolution clouds (100M+ pts) don't OOM.
    """
    R = T[:3, :3]
    scale = np.cbrt(np.linalg.det(R))
    R_rot = R / scale

    if utm_epsg is not None:
        to_ecef   = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4978", always_xy=True)
        from_ecef = Transformer.from_crs("EPSG:4978", f"EPSG:{utm_epsg}", always_xy=True)

    # Detect normal field storage dtype
    normal_dtype = None
    with laspy.open(las_path) as reader:
        chunk0 = next(reader.chunk_iterator(1))
        if 'normal x' in chunk0.array.dtype.names:
            normal_dtype = chunk0.array.dtype['normal x']

    with laspy.open(las_path) as reader:
        scales  = np.asarray(reader.header.scales,  dtype=np.float64)
        offsets = np.asarray(reader.header.offsets, dtype=np.float64)
        with open(Path(out_path), "wb") as f:
            with laspy.LasWriter(f, header=reader.header) as writer:
                for chunk in reader.chunk_iterator(chunk_size):
                    xyz = np.column_stack([np.asarray(chunk.x, dtype=np.float64),
                                           np.asarray(chunk.y, dtype=np.float64),
                                           np.asarray(chunk.z, dtype=np.float64)])

                    if utm_epsg is not None:
                        ex, ey, ez = to_ecef.transform(xyz[:, 0], xyz[:, 1], xyz[:, 2])
                        ecef = np.column_stack([ex, ey, ez])
                        ecef_t = (R @ ecef.T).T + T[:3, 3]
                        ux, uy, uz = from_ecef.transform(ecef_t[:, 0], ecef_t[:, 1], ecef_t[:, 2])
                        xyz_t = np.column_stack([ux, uy, uz])
                    else:
                        xyz_t = (R @ xyz.T).T + T[:3, 3]

                    chunk.array["X"] = np.round((xyz_t[:, 0] - offsets[0]) / scales[0]).astype(np.int32)
                    chunk.array["Y"] = np.round((xyz_t[:, 1] - offsets[1]) / scales[1]).astype(np.int32)
                    chunk.array["Z"] = np.round((xyz_t[:, 2] - offsets[2]) / scales[2]).astype(np.int32)

                    if normal_dtype is not None:
                        nx = np.asarray(chunk['normal x'], dtype=np.float64)
                        ny = np.asarray(chunk['normal y'], dtype=np.float64)
                        nz = np.asarray(chunk['normal z'], dtype=np.float64)
                        normals_t = (R_rot @ np.column_stack([nx, ny, nz]).T).T
                        if np.issubdtype(normal_dtype, np.integer):
                            imax = np.iinfo(normal_dtype).max
                            imin = np.iinfo(normal_dtype).min
                            chunk.array['normal x'] = np.clip(np.round(normals_t[:, 0] * imax), imin, imax).astype(normal_dtype)
                            chunk.array['normal y'] = np.clip(np.round(normals_t[:, 1] * imax), imin, imax).astype(normal_dtype)
                            chunk.array['normal z'] = np.clip(np.round(normals_t[:, 2] * imax), imin, imax).astype(normal_dtype)
                        else:
                            chunk.array['normal x'] = normals_t[:, 0].astype(normal_dtype)
                            chunk.array['normal y'] = normals_t[:, 1].astype(normal_dtype)
                            chunk.array['normal z'] = normals_t[:, 2].astype(normal_dtype)

                    writer.write_points(chunk)


# ---------------------------------------------------------------------------
# Single-stage pc_align wrapper
# ---------------------------------------------------------------------------

def pc_align_stage(
    tba_las: Path,
    ref_las: Path,
    output_prefix: Path,
    alignment_method: str,
    max_displacement: float,
    initial_transform: Path = None,
    verbose: bool = False,
) -> Path:
    """Run one stage of ASP ``pc_align`` on two LAZ/LAS point clouds.

    Caller is responsible for preparing *tba_las* and *ref_las* in the
    coordinate space pc_align should read (UTM or ECEF) and at the desired
    sampling density. This function just invokes pc_align with the given
    paths — no on-the-fly downsampling or CRS conversion. See
    :func:`pc_align_p2p_sp2p` for the full three-stage orchestrator that
    handles ECEF conversion and TBA/ref caching once for all stages.

    Parameters
    ----------
    tba_las, ref_las:
        Source and reference LAZ/LAS clouds — already in the target CRS and
        at the target density.
    output_prefix:
        File-path prefix for pc_align outputs (parent directory is created).
    alignment_method:
        ``'point-to-plane'`` for Stage 1 or
        ``'similarity-point-to-point'`` for Stages 2 & 3.
    max_displacement:
        Maximum allowed correspondence distance [m].
    initial_transform:
        Path to the ``*-transform.txt`` from the previous stage. ASP
        pre-applies this and compounds it with the new ICP result, so the
        output file contains the full accumulated composition.
    verbose:
        Print the pc_align call and its stdout.

    Returns
    -------
    Path to the output ``*-transform.txt`` file.
    """
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "pc_align",
        "--threads", str(os.cpu_count() or 1),
        "--max-displacement", str(max_displacement),
        "--alignment-method", alignment_method,
        "-o", str(output_prefix),
    ]
    if initial_transform is not None:
        cmd += ["--initial-transform", str(initial_transform)]

    # ASP convention: reference first, then source
    cmd += [str(ref_las), str(tba_las)]

    _run_command(cmd, verbose=verbose)
    return output_prefix.parent / (output_prefix.name + "-transform.txt")


# ---------------------------------------------------------------------------
# Stable-terrain reference builder  (replaces HSfM's mask_dem)
# ---------------------------------------------------------------------------

def extract_stable_reference(
    ref_cloud_path: Path,
    output_dir: Path,
    glacier_mask_path: Path = None,
    slope_threshold: float = 60.0,
    plot_dir: Path = None,
) -> Path:
    """Build the glacier-masked, slope-filtered reference cloud for Stage 3.

    Equivalent to ``hsfm.utils.mask_dem`` which removes unstable surfaces
    from the reference DEM before the final ICP pass:

    * **Glacier removal** (HSfM: ``dem_mask.py --glaciers`` using RGI polygons)
      → applied here via :func:`cntp.io.apply_glacier_mask` when a shapefile
      is provided.
    * **Slope + NDWI filter** (HSfM: ``dem_mask.py --nlcd`` for forest/water)
      → applied here via :func:`cntp.coreg.extract_stable_terrain`.

    Parameters
    ----------
    ref_cloud_path:
        Path to the reference LAZ/LAS cloud. Pass a pre-downsampled copy
        (e.g. from ``output_dir/downsampled/``) to control memory and runtime;
        the function loads it as-is without further subsampling.
    output_dir:
        Directory to write the stable-terrain cloud.
    glacier_mask_path:
        Shapefile with glacier polygon(s) in the same CRS as the cloud.
        Points inside the polygon are removed before the slope/NDWI filter.
        If *None*, only the slope+NDWI filter is applied.
    slope_threshold:
        Minimum slope angle [°] for stable-terrain detection.

    Returns
    -------
    Path
        ``<stem>_stable.las`` — glacier-masked + slope/NDWI-filtered cloud.
        When ``plot_dir`` is set, the reference NDWI-vs-intensity and stable
        terrain RGB plots are written there inline so the pre-NDWI-filter
        cloud doesn't need to be saved to disk.
    """
    ref_cloud_path = Path(ref_cloud_path)
    output_dir     = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / (ref_cloud_path.stem + "_stable.las")

    # Full cache: if both the stable cloud and the requested plots already
    # exist, skip the expensive load + KDTree pass entirely.
    plots_exist = True
    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        plots_exist = (
            (plot_dir / "ndwi_vs_intensity.png").exists()
            and (plot_dir / "stable_terrain_rgb.png").exists()
        )
    if out_path.exists() and plots_exist:
        print(f"  Stable reference cached → {out_path.name}")
        return out_path

    cloud = load_las(ref_cloud_path)
    if glacier_mask_path is not None:
        cloud = apply_glacier_mask(cloud, glacier_mask_path)

    stable_slope, stable = extract_stable_terrain(cloud, slope_threshold=slope_threshold)

    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        grayscale = np.mean(stable_slope[:, 3:6], axis=1)
        ndwi = (stable_slope[:, 5] - stable_slope[:, 3]) / (stable_slope[:, 3] + stable_slope[:, 5])
        plot_stable_terrain_diagnostics(
            stable_slope, stable, ndwi, grayscale,
            _NDWI_A, _NDWI_B, plot_dir, title="reference",
        )

    save_las(stable, out_path)
    print(f"  Stable reference → {out_path}  ({len(stable):,} pts)")
    return out_path


# ---------------------------------------------------------------------------
# Three-stage pipeline  (HSfM pc_align_p2p_sp2p)
# ---------------------------------------------------------------------------

def pc_align_p2p_sp2p(
    tba_las: Path,
    ref_las: Path,
    output_dir: Path,
    stable_ref_las: Path = None,
    p2p_max_displacement: float = 2.0,
    sp2p_max_displacement: float = 1.0,
    m_sp2p_max_displacement: float = 0.1,
    ref_downsample_factor: float = 1.0,
    tba_downsample_factor: float = None,
    utm_epsg: int = None,
    verbose: bool = False,
) -> tuple:
    """Three-stage ICP co-registration following Knuth et al. (2023) §4.3.3.

    Faithful port of the HSfM library's ``pc_align_p2p_sp2p``, adapted for
    LAZ/LAS point clouds instead of raster DEMs.
    LAZ files are passed directly to ASP — no intermediate CSV conversion.

    +---------+--------------------------------------+------------------------+
    | Stage   | pc_align method                      | Reference cloud        |
    +=========+======================================+========================+
    | 1       | ``point-to-plane`` (rigid body)      | full reference         |
    +---------+--------------------------------------+------------------------+
    | 2       | ``similarity-point-to-point``        | full reference         |
    |         | rotation + translation + scale       |                        |
    +---------+--------------------------------------+------------------------+
    | 3       | ``similarity-point-to-point``        | stable terrain only    |
    |         | rotation + translation + scale       | (glacier-masked +      |
    |         |                                      |  slope/NDWI-filtered)  |
    +---------+--------------------------------------+------------------------+

    Each stage passes its ``*-transform.txt`` as ``--initial-transform`` to the
    next stage; ASP compounds the transforms so Stage 3's output file contains
    the full composition T3 ∘ T2 ∘ T1.  That matrix is applied to the original
    full-resolution *tba_las* to produce the co-registered output.

    Downsampled copies of *ref_las* and *tba_las* are saved once to
    *output_dir/downsampled/* and reused across all three stages.
    The final transform is always applied to the original full-resolution *tba_las*.

    Parameters
    ----------
    tba_las:
        To-be-aligned LAZ/LAS cloud (full resolution; the final coreg output
        preserves this cloud's original density).
    ref_las:
        Reference LAZ/LAS cloud (full resolution).
    output_dir:
        Root directory for all intermediate files and outputs.
    stable_ref_las:
        Pre-built stable-terrain reference LAZ for Stage 3.
        Build it with :func:`extract_stable_reference`, passing the
        pre-downsampled reference path so its density matches Stage 1 & 2.
        If *None*, the (downsampled) *ref_las* is used for all three stages.
    p2p_max_displacement:
        Stage-1 max correspondence distance [m].
    sp2p_max_displacement:
        Stage-2 max correspondence distance [m].
    m_sp2p_max_displacement:
        Stage-3 max correspondence distance [m].
    ref_downsample_factor:
        Fraction of reference cloud points to keep for ICP (0 < factor <= 1.0).
        Default 1.0 passes the full cloud to ASP.
    tba_downsample_factor:
        Fraction of TBA cloud points to keep for ICP.
        Defaults to ``ref_downsample_factor``.
    utm_epsg:
        When set, enables ECEF mode: downsampled clouds are converted
        UTM → ECEF before each ``pc_align`` call so ICP runs in true ECEF
        space.  The final transform is applied via UTM → ECEF → T → ECEF →
        UTM so the output cloud stays in UTM.  ``stable_ref_las`` must also
        be in UTM — the conversion is handled internally.
        When ``None`` (default), the pipeline runs in flat-UTM mode.
    verbose:
        Print each ``pc_align`` call and its stdout.

    Returns
    -------
    aligned_las : Path
        Co-registered point cloud written inside *output_dir*.
    transform_path : Path
        Final 4×4 transform matrix (ASP ``*-transform.txt`` format).
    """
    _check_asp()

    tba_las    = Path(tba_las)
    ref_las    = Path(ref_las)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _tba_ds = tba_downsample_factor if tba_downsample_factor is not None else ref_downsample_factor

    # ------------------------------------------------------------------
    # Save downsampled copies for ICP (if requested).
    # The final transform is always applied to the original tba_las.
    # ------------------------------------------------------------------
    ds_dir = output_dir / "downsampled"
    ds_dir.mkdir(parents=True, exist_ok=True)

    if ref_downsample_factor < 1.0:
        ref_ds_path = ds_dir / (ref_las.stem + f"_ds{ref_downsample_factor:.2f}.las")
        if not ref_ds_path.exists():
            print(f"  Saving downsampled reference ({ref_downsample_factor:.0%}) → {ref_ds_path}")
            save_las(load_las(ref_las, downsample_factor=ref_downsample_factor), ref_ds_path)
        ref_icp = ref_ds_path
    else:
        ref_icp = ref_las

    # When downsampling TBA, write a cached subset to disk so the same random
    # sample is reused across all three stages. When not downsampling, pass
    # *tba_las* straight to ASP — saves a redundant ~250 MB copy per day.
    # *tba_las* should be uncompressed .las (single-day BA now exports .las
    # directly so ASP doesn't have to decompress on every stage).
    if _tba_ds < 1.0:
        tba_ds_path = ds_dir / (tba_las.stem + f"_ds{_tba_ds:.2f}.las")
        if not tba_ds_path.exists():
            print(f"  Saving downsampled TBA ({_tba_ds:.0%}) → {tba_ds_path}")
            save_las(load_las(tba_las, downsample_factor=_tba_ds), tba_ds_path)
        tba_icp = tba_ds_path
    else:
        tba_icp = tba_las

    ref_stage3 = Path(stable_ref_las) if stable_ref_las else ref_icp

    # ------------------------------------------------------------------
    # ECEF mode: convert each cloud once and share across all three stages.
    # Ref ECEFs land next to their source (auto-tracks into _ref_cache/ecef/
    # when the orchestrator passes a shared-cache ref). TBA ECEF lives under
    # the per-day coreg output dir.
    # ------------------------------------------------------------------
    if utm_epsg is not None:
        ref_ecef_dir = ref_icp.parent / "ecef"
        ref_ecef_dir.mkdir(parents=True, exist_ok=True)
        ref_ecef = ref_ecef_dir / (ref_icp.stem + "_ecef.las")
        if not ref_ecef.exists():
            las_utm_to_ecef(ref_icp, utm_epsg, ref_ecef)
        ref_icp_pc = ref_ecef

        ref_stage3_ecef_dir = ref_stage3.parent / "ecef"
        ref_stage3_ecef_dir.mkdir(parents=True, exist_ok=True)
        ref_stage3_ecef = ref_stage3_ecef_dir / (ref_stage3.stem + "_ecef.las")
        if not ref_stage3_ecef.exists():
            las_utm_to_ecef(ref_stage3, utm_epsg, ref_stage3_ecef)
        ref_stage3_pc = ref_stage3_ecef

        tba_ecef_dir = output_dir / "ecef"
        tba_ecef_dir.mkdir(parents=True, exist_ok=True)
        tba_ecef = tba_ecef_dir / (tba_icp.stem + "_ecef.las")
        if not tba_ecef.exists():
            las_utm_to_ecef(tba_icp, utm_epsg, tba_ecef)
        tba_icp_pc = tba_ecef
    else:
        ref_icp_pc    = ref_icp
        ref_stage3_pc = ref_stage3
        tba_icp_pc    = tba_icp

    # ------------------------------------------------------------------
    # Stage 1: point-to-plane ICP
    # Corrects large initial position offsets (rigid body).
    # HSfM uses max_displacement = 2000 m for km-scale NAGAP errors.
    # ------------------------------------------------------------------
    print(f"  Stage 1 — point-to-plane ICP "
          f"(max_displacement={p2p_max_displacement} m)")
    t1 = pc_align_stage(
        tba_icp_pc, ref_icp_pc,
        output_dir / "stage1" / "run",
        alignment_method="point-to-plane",
        max_displacement=p2p_max_displacement,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # Stage 2: similarity-point-to-point ICP
    # Refines alignment and corrects systematic scale error.
    # t1 passed as --initial-transform; ASP outputs T2 ∘ T1.
    # HSfM uses max_displacement = 1000 m.
    # ------------------------------------------------------------------
    print(f"  Stage 2 — similarity-point-to-point ICP "
          f"(max_displacement={sp2p_max_displacement} m)")
    t2 = pc_align_stage(
        tba_icp_pc, ref_icp_pc,
        output_dir / "stage2" / "run",
        alignment_method="similarity-point-to-point",
        max_displacement=sp2p_max_displacement,
        initial_transform=t1,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # Stage 3: similarity-point-to-point on stable terrain only.
    # The stable reference excludes glacier and vegetation surfaces,
    # matching HSfM's mask_dem step (dem_mask.py --glaciers --nlcd).
    # t2 passed as --initial-transform; ASP outputs T3 ∘ T2 ∘ T1.
    # HSfM uses max_displacement = 100 m.
    # ------------------------------------------------------------------
    print(f"  Stage 3 — similarity-point-to-point ICP, stable terrain only "
          f"(max_displacement={m_sp2p_max_displacement} m)")
    t3 = pc_align_stage(
        tba_icp_pc, ref_stage3_pc,
        output_dir / "stage3" / "run",
        alignment_method="similarity-point-to-point",
        max_displacement=m_sp2p_max_displacement,
        initial_transform=t2,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # Apply the final composed transform T3 ∘ T2 ∘ T1 to the original
    # full-resolution tba_las.  Output stays in UTM for M3C2 compatibility.
    # utm_epsg=None: T applied directly in UTM (flat-Cartesian mode).
    # utm_epsg set:  each point goes UTM → ECEF → T → ECEF → UTM.
    # ------------------------------------------------------------------
    T = _read_asp_transform(t3)
    aligned_las = output_dir / (tba_las.stem + "_coreg_hsfm.las")
    print("  Applying final transform to full-resolution TBA cloud …")
    _apply_transform_to_las(tba_las, T, aligned_las, utm_epsg=utm_epsg)

    print(f"  Co-registered cloud → {aligned_las}")
    print(f"  Final transform     → {t3}")
    return aligned_las, t3


# ---------------------------------------------------------------------------
# Accuracy assessment (M3C2 on stable terrain)
# ---------------------------------------------------------------------------

def evaluate_coreg(
    ref_las: Path,
    tba_before_las: Path,
    tba_after_las: Path,
    ref_downsample_factor: float = 0.5,
    tba_downsample_factor: float = None,
    slope_threshold: float = 60.0,
    glacier_mask_path: Path = None,
    plot_dir: Path = None,
    stable_dir: Path = None,
    stable_ref_las: Path = None,
) -> dict:
    """Compute M3C2 distances on stable terrain before and after co-registration.

    Follows the same ``ref_downsample_factor`` / ``tba_downsample_factor`` convention
    as :func:`cntp.batch.coreg_pc`:
    ``ref_downsample_factor`` applies to the reference cloud;
    ``tba_downsample_factor`` applies to both TBA clouds and defaults to
    ``ref_downsample_factor`` when not set.

    Parameters
    ----------
    ref_las:
        Reference LAZ/LAS cloud.
    tba_before_las:
        Original (unaligned) TBA cloud.
    tba_after_las:
        Co-registered cloud returned by :func:`pc_align_p2p_sp2p`.
    ref_downsample_factor:
        Fraction of reference points to keep when loading.
    tba_downsample_factor:
        Fraction of TBA points to keep.  Defaults to ``ref_downsample_factor``.
    slope_threshold:
        Minimum slope angle [°] for stable-terrain detection.
    glacier_mask_path:
        Shapefile with glacier polygon(s). When provided, glacier points are
        removed from all three clouds before M3C2, matching the surfaces used
        for Stage 3 ICP in :func:`extract_stable_reference`.
    stable_dir:
        If provided, the stable-terrain TBA cloud after co-registration is
        written here as ``<tba_after_las.stem>_stable.las``.
    stable_ref_las:
        Pre-built stable-terrain reference cloud (output of
        :func:`extract_stable_reference`).  When provided, the ref cloud is
        not loaded or filtered — the cached file is used directly, skipping
        the glacier-mask and slope/NDWI KDTree passes for the reference.
        ``ref_las`` and ``ref_downsample_factor`` are ignored in this case.
        Reference diagnostic plots (NDWI + RGB) are generated at
        :func:`extract_stable_reference` time so this function only plots TBA.

    Returns
    -------
    dict with keys:
        med_before, std_before, dist_before,
        med_after,  std_after,  dist_after,
        stable_tba_after  (Nx9 array used for M3C2 after co-registration).
    """
    _tba_ds = tba_downsample_factor if tba_downsample_factor is not None else ref_downsample_factor

    before = load_las(tba_before_las, downsample_factor=_tba_ds)
    after  = load_las(tba_after_las,  downsample_factor=_tba_ds)

    if glacier_mask_path is not None:
        before = apply_glacier_mask(before, glacier_mask_path)
        after  = apply_glacier_mask(after,  glacier_mask_path)

    if stable_ref_las is not None:
        ref_stable = load_las(Path(stable_ref_las))
    else:
        ref = load_las(ref_las, downsample_factor=ref_downsample_factor)
        if glacier_mask_path is not None:
            ref = apply_glacier_mask(ref, glacier_mask_path)
        _, ref_stable = extract_stable_terrain(ref, slope_threshold=slope_threshold)

    before_slope, before_stable = extract_stable_terrain(before, slope_threshold=slope_threshold)
    after_slope,  after_stable  = extract_stable_terrain(after,  slope_threshold=slope_threshold)

    # Reference plots are generated once in extract_stable_reference; here we
    # only plot the (per-day) TBA cloud's NDWI + stable-terrain RGB.
    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        tba_plot_dir = plot_dir / "tba"
        tba_plot_dir.mkdir(parents=True, exist_ok=True)
        grayscale = np.mean(before_slope[:, 3:6], axis=1)
        ndwi = (before_slope[:, 5] - before_slope[:, 3]) / (before_slope[:, 3] + before_slope[:, 5])
        plot_stable_terrain_diagnostics(
            before_slope, before_stable, ndwi, grayscale,
            _NDWI_A, _NDWI_B, tba_plot_dir, title="tba",
        )

    def _to_epoch(pts: np.ndarray) -> py4dgeo.Epoch:
        return py4dgeo.Epoch(pts[:, :3])

    epoch_ref    = _to_epoch(ref_stable)
    epoch_before = _to_epoch(before_stable)
    epoch_after  = _to_epoch(after_stable)

    med_b, std_b, dist_b = run_m3c2(epoch_ref, epoch_before)
    med_a, std_a, dist_a = run_m3c2(epoch_ref, epoch_after)

    print(f"  Before — median M3C2: {med_b:.4f} m  std: {std_b:.4f} m")
    print(f"  After  — median M3C2: {med_a:.4f} m  std: {std_a:.4f} m")

    if stable_dir is not None:
        stable_dir = Path(stable_dir)
        stable_dir.mkdir(parents=True, exist_ok=True)
        tba_stable_out = stable_dir / (Path(tba_after_las).stem + "_stable.laz")
        save_las(after_stable, tba_stable_out)
        print(f"  Stable TBA (after)  : {tba_stable_out}")

    if plot_dir is not None:
        plot_m3c2_distances(dist_b, dist_a, plot_dir, title="ASP HSfM co-registration")

    return dict(
        med_before=med_b, std_before=std_b, dist_before=dist_b,
        med_after=med_a,  std_after=std_a,  dist_after=dist_a,
        stable_tba_after=after_stable,
    )


# ---------------------------------------------------------------------------
# DEM rasterisation
# ---------------------------------------------------------------------------

def point2dem(
    cloud_las: str | Path,
    out_prefix: str | Path = None,
    res: float = 1.0,
    utm_epsg: int = None,
    max_gap_pixels: int = 1,
    nodata: float = -9999.0,
    ref_las: str | Path = None,
    extra_args: tuple = (),
    verbose: bool = False,
) -> Path:
    """Wrap ASP's ``point2dem`` CLI to rasterise a LAS cloud into a DEM.

    Thin wrapper around the same Ames Stereo Pipeline binary that
    :func:`pc_align_p2p_sp2p` already requires. ``point2dem`` is streaming and
    multi-threaded: no Python-side RAM blow-up, no Delaunay triangulation, no
    Clough-Tocher overshoot. Per-pixel Gaussian distance-weighted average
    (point2dem's default ``weighted_average`` filter — what the HSfM workflow
    calls "IDW"), written straight to a GeoTIFF. Not used by the 4D-SfM pipeline
    (which rasterises
    via :func:`cntp.raster.build_dem_and_ortho`), but kept here as a faster,
    lower-memory alternative for ad-hoc DEM generation.

    Parameters
    ----------
    cloud_las : str | Path
        Input LAS/LAZ point cloud (e.g. a coregistered cloud).
    out_prefix : str | Path, optional
        ASP output prefix passed to ``-o``. The DEM lands at
        ``<out_prefix>-DEM.tif`` (ASP appends the suffix). Default
        ``<cloud_dir>/<cloud_stem>``.
    res : float
        Output pixel size (``--tr``). Default 1.0 m.
    utm_epsg : int, optional
        EPSG code for ``--t_srs``. When ``None``, read from ``cloud_las``'s
        LAS header.
    max_gap_pixels : int
        Maps to ASP ``--search-radius-factor``: pixels farther than
        ``max_gap_pixels * res`` from any cloud point produce nodata.
        Default 1 (same convention as
        :func:`cntp.raster.build_dem_and_ortho`).
    nodata : float
        ASP ``--nodata-value``. Default -9999.
    ref_las : str | Path, optional
        When set, the output grid is anchored to *ref_las*'s XY bounding box
        (read from header — no points loaded) via ASP
        ``--t_projwin xmin ymin xmax ymax`` (ASP convention — lower-left +
        upper-right, NOT GDAL's upper-left/lower-right). Mirrors
        :func:`cntp.raster.build_dem_and_ortho`'s ``ref_las`` argument so the
        ASP DEM lands on the same footprint as the scipy DEM and the two can
        be differenced without resampling. Default ``None`` ⇒ ASP derives the
        grid from the input cloud's own bounding box.
    extra_args : tuple
        Additional CLI flags appended verbatim, e.g.
        ``("--remove-outliers", "--max-valid-triangulation-error", "5")``.
    verbose : bool
        Print the assembled CLI and ASP's stdout.

    Returns
    -------
    Path
        Path to ``<out_prefix>-DEM.tif``.
    """
    if shutil.which("point2dem") is None:
        raise RuntimeError(
            "point2dem not found on PATH.\n"
            "Install NASA Ames Stereo Pipeline:\n"
            "  https://stereopipeline.readthedocs.io/en/latest/installation.html"
        )

    cloud_las = Path(cloud_las)
    if out_prefix is None:
        out_prefix = cloud_las.parent / cloud_las.stem
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if utm_epsg is None:
        with laspy.open(cloud_las) as f:
            crs = f.header.parse_crs()
        if crs is None or crs.to_epsg() is None:
            raise ValueError(
                f"No EPSG in {cloud_las.name} header; pass utm_epsg explicitly."
            )
        utm_epsg = crs.to_epsg()

    threads = os.cpu_count() or 1

    cmd = [
        "point2dem",
        "--threads", str(threads),
        "--tr", str(res),
        "--t_srs", f"EPSG:{utm_epsg}",
        "--search-radius-factor", str(max_gap_pixels),
        "--nodata-value", str(nodata),
        "-o", str(out_prefix),
    ]
    if ref_las is not None:
        ref_min, ref_max = read_las_bounds(ref_las)
        # Defensive min/max — the LAS spec doesn't require header.mins <=
        # header.maxs, so some writers store them reversed (we hit this with
        # Reference_UAV_TLC_PCS.laz, whose header has Y min > Y max → a
        # degenerate projwin → empty DEM).
        x0, x1 = float(ref_min[0]), float(ref_max[0])
        y0, y1 = float(ref_min[1]), float(ref_max[1])
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)
        # ASP convention: --t_projwin xmin ymin xmax ymax (lower-left +
        # upper-right). Different from GDAL's --projwin ulx uly lrx lry.
        cmd.extend(["--t_projwin", str(xmin), str(ymin), str(xmax), str(ymax)])
    cmd.extend(str(a) for a in extra_args)
    cmd.append(str(cloud_las))

    _run_command(cmd, verbose=verbose)

    dem_path = out_prefix.parent / f"{out_prefix.name}-DEM.tif"
    if not dem_path.exists():
        raise RuntimeError(
            f"point2dem completed but expected output {dem_path} not found."
        )
    if verbose:
        print(f"  ASP DEM → {dem_path}")
    return dem_path

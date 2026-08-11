"""Raw time-lapse image homogeniser — the "translator" step.

Camera time-lapse data arrives in a different on-disk mess for every field
team: nested EK-card subfolders, mixed extensions, timestamps only in EXIF.
The rest of :mod:`tlapse4d` only ever works with one *standard* filename::

    <camera>_<YYYY-MM-DD>_<HHMMSS>.JPG     e.g.  C7_2023-11-27_083000.JPG

:func:`homogenize_images` is the single adapter between those two worlds.
Point it at a site's raw camera tree and it copies every image to a standard
name derived from the photo's EXIF capture time, leaving the originals
untouched, and writes a manifest CSV recording exactly what mapped to what.

Already have standard-named images? You don't need this at all — point the
pipeline straight at your folder (see :func:`tlapse4d.metashape.discover_images`).

Requires ImageMagick's ``identify`` on PATH for EXIF extraction.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".JPG", ".JPEG")


def _exif_datetime(img_path: Path) -> str:
    """Return EXIF DateTimeOriginal as ``'YYYY:MM:DD HH:MM:SS'`` (raises if absent)."""
    result = subprocess.run(
        ["identify", "-format", "%[EXIF:DateTimeOriginal]", str(img_path)],
        capture_output=True, text=True, check=True,
    )
    dt = result.stdout.strip()
    if not dt:
        raise ValueError("no EXIF DateTimeOriginal")
    return dt


def _read_stamp(img_path: Path):
    """Read EXIF capture time → ``(date 'YYYY-MM-DD', time 'HHMMSS', error)``.

    Parallel-safe (read-only). On any failure returns ``(None, None, exc)`` so
    the caller can record a skip instead of aborting the whole run.
    """
    try:
        dt = _exif_datetime(img_path)                  # "2023:11:27 08:30:00"
        date_part, time_part = dt.split(" ")
        return (date_part.replace(":", "-"), time_part.replace(":", ""), None)
    except Exception as e:  # noqa: BLE001 — any failure → skip this image
        return (None, None, e)


def _copy_task(task):
    """Copy one ``(src, dst, stamp)`` task; return ``(task, error)``.

    Parallel-safe: each task writes a distinct, pre-assigned ``dst``, so there
    are no name races between workers.

    Uses ``copyfile`` (bytes only) rather than ``copy2``: copying file
    *metadata* (chmod/utime) fails with ``[Errno 1] Operation not permitted``
    on Windows/drvfs mounts like ``/mnt/g``. The EXIF capture time lives inside
    the JPEG, which the byte copy preserves, so nothing downstream is lost.
    """
    src, dst, _stamp = task
    try:
        shutil.copyfile(src, dst)
        return (task, None)
    except Exception as e:  # noqa: BLE001
        return (task, e)


def homogenize_images(
    base_dir: str | Path,
    output_dir: str | Path | None = None,
    cameras: list[str] | None = None,
    subfolders: list[str] | None = None,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    per_camera_subdir: bool = True,
    manifest: str | Path | None = None,
    n_jobs: int | None = None,
    verbose: bool = True,
) -> dict:
    """Copy raw camera images into the standard ``<cam>_<date>_<time>.JPG`` layout.

    The output is what the rest of the pipeline expects; the camera id is read
    from the *filename* downstream, so the output folder structure is only for
    tidiness.

    Parameters
    ----------
    base_dir : str | Path
        Root of the raw data. Expected to contain one sub-directory per camera
        (e.g. ``C6/``, ``C7/`` … or ``C6_renamed/`` …). Each camera folder may
        nest its images arbitrarily deep (EK-card subfolders); they are found
        recursively.
    output_dir : str | Path, optional
        Where the standardised copies are written. Default ``None`` creates a
        sibling folder next to *base_dir* named ``<base_dir>_renamed`` (e.g.
        ``…/TLCAM/ChangriNorth`` → ``…/TLCAM/ChangriNorth_renamed``) — kept at
        the same level as, and outside of, the raw tree, so it is never
        rescanned as a camera and never picked up by the raw side.
    cameras : list[str], optional
        Camera folder names to process. Default: auto-detect every immediate
        sub-directory of *base_dir*. The folder name becomes the camera id
        (``C6/`` → ``C6``).
    subfolders : list[str], optional
        Restrict the search inside each camera folder to these named
        sub-directories. Default ``None`` recurses through everything.
    extensions : tuple[str]
        Image extensions to consider (case-insensitive).
    per_camera_subdir : bool
        When True (default) write into ``output_dir/<cam>/`` (one folder per
        camera, named by the camera id — the ``_renamed`` marker is already on
        the parent *output_dir*); when False write all images flat into
        *output_dir*. Either way the pipeline finds them (camera id is read
        from the filename, not the folder).
    manifest : str | Path, optional
        Manifest CSV path. Default ``output_dir/manifest.csv``. Records
        ``camera, original_path, standard_name, datetime, status``.
    n_jobs : int, optional
        Worker threads for the EXIF reads and file copies (both are I/O-bound
        and embarrassingly parallel). Default ``None`` → ``min(8, cpu_count)``.
        Pass ``1`` for fully serial. Name assignment always stays serial, so
        the output is identical regardless of *n_jobs*.
    verbose : bool
        Print per-camera progress.

    Returns
    -------
    dict
        ``{camera: {"copied": int, "skipped": int}}`` plus a ``"manifest"`` key
        with the manifest path.
    """
    if shutil.which("identify") is None:
        raise RuntimeError(
            "ImageMagick's `identify` not found on PATH — needed to read EXIF "
            "timestamps. Install ImageMagick, or supply already-standard-named "
            "images and skip this step (see tlapse4d.metashape.discover_images)."
        )

    if n_jobs is None:
        n_jobs = min(8, os.cpu_count() or 1)
    n_jobs = max(1, n_jobs)

    base_dir   = Path(base_dir)
    output_dir = (
        Path(output_dir) if output_dir is not None
        else base_dir.parent / f"{base_dir.name}_renamed"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    exts = {e.lower() for e in extensions}

    def _run(fn, items):
        """Map *fn* over *items*, in parallel when n_jobs > 1 (order preserved)."""
        if n_jobs > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                return list(ex.map(fn, items))
        return [fn(x) for x in items]

    if cameras is None:
        cam_dirs = sorted(p for p in base_dir.iterdir() if p.is_dir())
    else:
        cam_dirs = [base_dir / c for c in cameras]

    manifest_path = Path(manifest) if manifest is not None else output_dir / "manifest.csv"
    manifest_rows: list[dict] = []
    summary: dict = {}

    for cam_dir in cam_dirs:
        cam = cam_dir.name
        if not cam_dir.is_dir():
            if verbose:
                print(f"[{cam}] not a directory — skipping")
            continue

        # Gather source images recursively (optionally limited to subfolders).
        search_dirs = [cam_dir / s for s in subfolders] if subfolders else [cam_dir]
        images: list[Path] = []
        for d in search_dirs:
            if d.is_dir():
                images.extend(
                    p for p in d.rglob("*")
                    if p.is_file() and p.suffix.lower() in exts
                )
        images = sorted(images)

        out_cam_dir = (output_dir / cam) if per_camera_subdir else output_dir
        out_cam_dir.mkdir(parents=True, exist_ok=True)

        copied = skipped = 0

        # Phase 1 — read EXIF capture times (parallel, read-only).
        stamps = _run(_read_stamp, images)

        # Phase 2 — assign unique standard names (serial → no name races).
        used: set[str] = set()
        copy_tasks: list[tuple[Path, Path, str]] = []
        for img, (date_clean, time_clean, err) in zip(images, stamps):
            if err is not None:
                if verbose:
                    print(f"  Skipped {img.name}: {err}")
                manifest_rows.append({
                    "camera": cam, "original_path": str(img),
                    "standard_name": "", "datetime": "", "status": f"skipped: {err}",
                })
                skipped += 1
                continue
            stem = f"{cam}_{date_clean}_{time_clean}"
            name = f"{stem}.JPG"
            # Guarantee a unique name even with 3+ collisions (multiple photos
            # in the same second). The old script overwrote on the 3rd clash.
            n = 1
            while name in used or (out_cam_dir / name).exists():
                name = f"{stem}_{n}.JPG"
                n += 1
            used.add(name)
            copy_tasks.append((img, out_cam_dir / name, f"{date_clean} {time_clean}"))

        # Phase 3 — copy files (parallel; each writes a distinct dst).
        for (src, dst, stamp), err in _run(_copy_task, copy_tasks):
            if err is not None:
                if verbose:
                    print(f"  Copy failed {dst.name}: {err}")
                manifest_rows.append({
                    "camera": cam, "original_path": str(src),
                    "standard_name": "", "datetime": stamp,
                    "status": f"skipped: copy failed: {err}",
                })
                skipped += 1
                continue
            manifest_rows.append({
                "camera": cam, "original_path": str(src),
                "standard_name": dst.name, "datetime": stamp, "status": "copied",
            })
            copied += 1

        summary[cam] = {"copied": copied, "skipped": skipped}
        if verbose:
            print(f"[{cam}] copied {copied}, skipped {skipped} → {out_cam_dir}")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["camera", "original_path", "standard_name",
                           "datetime", "status"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    if verbose:
        print(f"\nManifest → {manifest_path}  ({len(manifest_rows)} rows)")

    summary["manifest"] = manifest_path
    return summary


def ensure_standardized(
    image_dir: str | Path,
    cameras: list[str] | None = None,
    verbose: bool = True,
    **homogenize_kwargs,
) -> Path:
    """Return a folder of standard-named time-lapse images, homogenising only if needed.

    Idempotent one-call front door for a notebook setup cell. Decides what to do
    by *looking at the filenames*:

    1. If *image_dir* already holds standard-named images
       (``<cam>_<date>_<time>.<ext>``) → return *image_dir* unchanged. Nothing to do.
    2. Else if a sibling ``<image_dir>_renamed`` already exists **with a
       completed run** (its ``manifest.csv`` is present and it holds standard
       images) → return that. A previous homogenise is reused, so re-running the
       cell is cheap.
    3. Else → run :func:`homogenize_images` on *image_dir* and return the new
       ``<image_dir>_renamed``.

    Parameters
    ----------
    image_dir : str | Path
        Either an already-standard image folder, or a raw camera tree to convert.
    cameras, **homogenize_kwargs
        Forwarded to :func:`homogenize_images` when homogenisation is needed.

    Returns
    -------
    Path
        The directory to hand to the pipeline as ``tlcam_dir``.
    """
    from tlapse4d.metashape import discover_images  # lazy: avoid import cycle

    image_dir = Path(image_dir)

    # 1. Already standard-named in place?
    if discover_images(image_dir):
        if verbose:
            print(f"Images already standard-named → using {image_dir}")
        return image_dir

    # 2. A completed previous run sitting in the sibling folder?
    renamed = image_dir.parent / f"{image_dir.name}_renamed"
    if (renamed / "manifest.csv").exists() and discover_images(renamed):
        if verbose:
            print(f"Found completed standardised images → using {renamed}")
        return renamed

    # 3. Need to homogenise.
    if renamed.exists():
        print(
            f"WARNING: {renamed} exists but looks incomplete (no manifest.csv / "
            f"no standard images). Delete it and re-run to redo cleanly."
        )
    if verbose:
        print(f"No standard-named images found in {image_dir} — homogenising …")
    summary = homogenize_images(image_dir, cameras=cameras, verbose=verbose,
                                **homogenize_kwargs)
    return Path(summary["manifest"]).parent

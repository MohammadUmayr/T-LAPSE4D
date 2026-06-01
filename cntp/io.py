import rasterio
from rasterio.errors import RasterioIOError
import numpy as np
from scipy.spatial import cKDTree
import laspy
from pathlib import Path
import geopandas as gpd
import shapely


def read_las_bounds(path: str | Path) -> tuple:
    """Read the XYZ bounding box from a .las/.laz file header without loading points.

    Parameters
    ----------
    path : str | Path
        Path to the .las/.laz file.

    Returns
    -------
    min_bound : np.ndarray
        Minimum XYZ bounds (shape 3,).
    max_bound : np.ndarray
        Maximum XYZ bounds (shape 3,).
    """
    with laspy.open(Path(path)) as f:
        min_bound = np.array(f.header.mins, dtype=np.float64)
        max_bound = np.array(f.header.maxs, dtype=np.float64)
    return min_bound, max_bound


def load_las(path: str | Path, downsample_factor: float = 1.0) -> np.ndarray:
    """Load a .las/.laz file into an Nx9 array: X, Y, Z, R, G, B, NX, NY, NZ.

    RGB is scaled to 0-255 (from LAS uint16 0-65535).
    Normals are read from extra dimensions named 'normal x/y/z'.

    Parameters
    ----------
    path : str | Path
        Path to the .las/.laz file.
    downsample_factor : float
        Fraction of points to keep (0 < factor <= 1.0). Sampling is done
        chunk-by-chunk so the full cloud is never held in RAM. Default 1.0
        loads all points.
    """
    chunks = []
    chunk_size = 500_000
    rng = np.random.default_rng()

    with open(Path(path), "rb") as f, laspy.LasReader(f) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            n = len(chunk)
            if downsample_factor < 1.0:
                k = max(1, int(n * downsample_factor))
                idx = rng.choice(n, size=k, replace=False)
            else:
                idx = slice(None)

            x  = np.array(chunk.x[idx],     dtype=np.float64)
            y  = np.array(chunk.y[idx],     dtype=np.float64)
            z  = np.array(chunk.z[idx],     dtype=np.float64)
            r  = np.array(chunk.red[idx],   dtype=np.float64) / 257.0
            g  = np.array(chunk.green[idx], dtype=np.float64) / 257.0
            b  = np.array(chunk.blue[idx],  dtype=np.float64) / 257.0
            nx = np.array(chunk['normal x'][idx], dtype=np.float64)
            ny = np.array(chunk['normal y'][idx], dtype=np.float64)
            nz = np.array(chunk['normal z'][idx], dtype=np.float64)
            chunks.append(np.column_stack([x, y, z, r, g, b, nx, ny, nz]))

    return np.vstack(chunks)


def save_las(data: np.ndarray, path: str | Path, crs: int = None) -> None:
    """Save an Nx9 array (X, Y, Z, R, G, B, NX, NY, NZ) to a .las/.laz file.

    RGB values are expected in 0-255 range and stored as LAS uint16 (0-65535).
    Normals are stored as float32 extra dimensions named 'normal x/y/z'.

    Parameters
    ----------
    data : np.ndarray
        Nx9 point cloud (X, Y, Z, R, G, B, NX, NY, NZ).
    path : str | Path
        Output path (.las or .laz).
    crs : int, optional
        EPSG code to embed in the LAS header (written as GeoTIFF VLRs for
        LAS 1.2). When set, downstream readers can recover it via
        ``laspy.open(path).header.parse_crs()``. Default ``None`` writes the
        file without CRS metadata.
    """
    path = Path(path)
    header = laspy.LasHeader(point_format=2, version="1.2")
    # Derive scale so stored integers give ~0.1 mm precision in the cloud's
    # native units (works for both metric and geographic-degree coordinates).
    xyz_range = data[:, :3].max(axis=0) - data[:, :3].min(axis=0)
    header.scales  = np.maximum(xyz_range / 1_000_000, 1e-9)
    header.offsets = data[:, :3].min(axis=0)
    header.add_extra_dims([
        laspy.ExtraBytesParams(name="normal x", type=np.float32),
        laspy.ExtraBytesParams(name="normal y", type=np.float32),
        laspy.ExtraBytesParams(name="normal z", type=np.float32),
    ])
    if crs is not None:
        from pyproj import CRS as _CRS
        header.add_crs(_CRS.from_epsg(int(crs)))
    las = laspy.LasData(header=header)
    las.x = data[:, 0]
    las.y = data[:, 1]
    las.z = data[:, 2]
    las.red   = np.clip(data[:, 3] * 257, 0, 65535).astype(np.uint16)
    las.green = np.clip(data[:, 4] * 257, 0, 65535).astype(np.uint16)
    las.blue  = np.clip(data[:, 5] * 257, 0, 65535).astype(np.uint16)
    las['normal x'] = data[:, 6].astype(np.float32)
    las['normal y'] = data[:, 7].astype(np.float32)
    las['normal z'] = data[:, 8].astype(np.float32)
    las.write(str(path))



def apply_glacier_mask(cloud: np.ndarray, mask_path: str | Path) -> np.ndarray:
    """Remove points that fall inside the glacier polygon.

    Parameters
    ----------
    cloud : np.ndarray
        Nx9 array (X, Y, Z, ...). XY must be in the same CRS as the shapefile.
    mask_path : str | Path
        Path to a shapefile containing the glacier polygon(s).

    Returns
    -------
    np.ndarray
        Cloud with glacier points removed.
    """
    gdf = gpd.read_file(Path(mask_path))
    glacier_geom = gdf.geometry.union_all()
    inside = shapely.contains_xy(glacier_geom, cloud[:, 0], cloud[:, 1])
    return cloud[~inside]


def save_dem(array, filename, crs_epsg, transform):
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs_epsg,
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(array, 1)
    print("Saved:", filename)

def save_ortho(x, y, rgb, xi, yi, res, max_gap_pixels, filename, crs_epsg, transform):
        # Precompute DEM pixel centers
        pixels_xy = np.column_stack([xi.ravel(), yi.ravel()])

        # Build KD-tree in XY
        tree = cKDTree(np.column_stack([x, y]))

        # Nearest neighbour lookup
        dist, idx = tree.query(pixels_xy, k=1, distance_upper_bound=res * max_gap_pixels)
        # Mask out empty pixels
        mask = np.isfinite(dist)
        
        # Prepare output arrays
        r_img = np.zeros(idx.shape, dtype=np.uint8)
        g_img = np.zeros(idx.shape, dtype=np.uint8)
        b_img = np.zeros(idx.shape, dtype=np.uint8)

        r_img[mask] = rgb[idx[mask], 0]
        g_img[mask] = rgb[idx[mask], 1]
        b_img[mask] = rgb[idx[mask], 2]

        # Stack into a single image array
        H, W = xi.shape
        ortho = np.dstack([
            r_img.reshape(H, W),
            g_img.reshape(H, W),
            b_img.reshape(H, W)
        ])

        with rasterio.open(
            filename,
            "w",
            driver="GTiff",
            height=ortho.shape[0],
            width=ortho.shape[1],
            count=3,                           # three bands!
            dtype=ortho.dtype,
            crs=crs_epsg,
            transform=transform,
            nodata=0
        ) as dst:
            dst.write(ortho[:, :, 0], 1)  # Red
            dst.write(ortho[:, :, 1], 2)  # Green
            dst.write(ortho[:, :, 2], 3)  # Blue

        print("Saved orthoimage:", filename)


def build_dem_and_ortho(
    cloud_las: str | Path,
    ref_las: str | Path,
    out_dir: str | Path,
    name_stem: str,
    res: float = 1.0,
    max_gap_pixels: int = 1,
    utm_epsg: int = None,
    cloud_downsample: float = 1.0,
    overwrite: bool = False,
) -> tuple:
    """Build a DEM and orthoimage from a co-registered point cloud on the reference's grid.

    The output grid is anchored to the reference cloud's XY bounding box
    (read from the LAS header — no points loaded), so every day's DEM/ortho
    lands on the same pixel raster and stacks/differences directly.

    Both rasters are written in one pass — the cloud is loaded once and
    `(x, y)` is shared between :func:`cntp.coreg.interpolate_and_mask` (DEM,
    cubic griddata + KDTree gap mask) and :func:`save_ortho` (RGB nearest-
    neighbour).

    Parameters
    ----------
    cloud_las : str | Path
        Co-registered LAS/LAZ — typically ``aligned_las`` from Step 3 of
        :func:`cntp.pipeline_4dsfm.run_4dsfm_day`.
    ref_las : str | Path
        Original (non-downsampled) reference cloud. Only its header bbox is
        read — no points are loaded.
    out_dir : str | Path
        Destination directory. Files written are
        ``<name_stem>_dem.tif`` and ``<name_stem>_ortho.tif``.
    name_stem : str
        Filename stem (e.g. the date string ``"2023-09-15"``).
    res : float
        Pixel size in metres (default 1.0).
    max_gap_pixels : int
        Pixels further than ``max_gap_pixels * res`` from any cloud point are
        nulled (DEM) or left as nodata (ortho). Default 1.
    utm_epsg : int, optional
        EPSG code of the output raster CRS. When ``None``, parsed from
        ``cloud_las``'s LAS header (Metashape writes it on export).
    cloud_downsample : float
        Fraction of cloud points to keep (0 < f ≤ 1). Default 1.0. Pass the
        same value as ``tba_downsample`` in the coreg step for parity with
        the ICP basis.
    overwrite : bool
        When False (default), skip computation entirely if both
        ``<name_stem>_dem.tif`` and ``<name_stem>_ortho.tif`` already exist
        in *out_dir*. When True, force a rebuild.

    Returns
    -------
    (dem_path, ortho_path) : tuple
        Paths to the written GeoTIFFs. If a write fails with permission
        denied, that path is returned as ``None``.
    """
    from cntp.coreg import interpolate_and_mask

    cloud_las = Path(cloud_las)
    ref_las   = Path(ref_las)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dem_path   = out_dir / f"{name_stem}_dem.tif"
    ortho_path = out_dir / f"{name_stem}_ortho.tif"

    if not overwrite and dem_path.exists() and ortho_path.exists():
        print(f"  DEM + ortho cached → {dem_path.name}, {ortho_path.name}")
        return dem_path, ortho_path

    # ── 1. Grid from the reference cloud's header bbox (no points loaded) ──
    ref_min, ref_max = read_las_bounds(ref_las)
    xmin, xmax = float(ref_min[0]), float(ref_max[0])
    ymin, ymax = float(ref_min[1]), float(ref_max[1])

    xi = np.arange(xmin, xmax, res)
    yi = np.arange(ymax, ymin, -res)
    xi, yi = np.meshgrid(xi, yi)

    transform = rasterio.transform.from_origin(xmin, ymax, res, res)

    # ── 2. Resolve CRS (param wins, else read from cloud LAS header) ───────
    if utm_epsg is None:
        with laspy.open(cloud_las) as f:
            crs = f.header.parse_crs()
        if crs is None or crs.to_epsg() is None:
            raise ValueError(
                f"No EPSG in {cloud_las.name} header; pass utm_epsg explicitly."
            )
        utm_epsg = crs.to_epsg()
    crs_epsg = f"EPSG:{utm_epsg}"

    # ── 3. Load + XY-clip the slave cloud to the reference footprint ───────
    cloud = load_las(cloud_las, downsample_factor=cloud_downsample)
    in_box = ((cloud[:, 0] >= xmin) & (cloud[:, 0] <= xmax) &
              (cloud[:, 1] >= ymin) & (cloud[:, 1] <= ymax))
    cloud = cloud[in_box]
    print(f"  Loaded {len(cloud):,} pts after XY clip "
          f"(downsample={cloud_downsample}, grid {xi.shape[1]}×{xi.shape[0]} @ {res} m, "
          f"{crs_epsg})")

    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    rgb     = cloud[:, 3:6]

    # ── 4. DEM (cubic griddata + KDTree gap mask) ──────────────────────────
    zi = interpolate_and_mask(x, y, z, xi, yi, res, max_gap_pixels)
    try:
        save_dem(zi, dem_path, crs_epsg, transform)
    except RasterioIOError as e:
        if "Permission denied" in str(e):
            print(f"  Skipping DEM (permission denied): {dem_path}")
            dem_path = None
        else:
            raise

    # ── 5. Orthoimage (KDTree nearest-neighbour RGB) ───────────────────────
    try:
        save_ortho(x, y, rgb, xi, yi, res, max_gap_pixels,
                   ortho_path, crs_epsg, transform)
    except RasterioIOError as e:
        if "Permission denied" in str(e):
            print(f"  Skipping ortho (permission denied): {ortho_path}")
            ortho_path = None
        else:
            raise

    return dem_path, ortho_path


def build_reference_dem_and_ortho(
    ref_cloud_path: str | Path,
    cache_dir: str | Path,
    res: float = 1.0,
    max_gap_pixels: int = 1,
    utm_epsg: int = None,
    cloud_downsample: float = 1.0,
    overwrite: bool = False,
) -> tuple:
    """Build the reference DEM + orthoimage once, cache under *cache_dir*.

    Thin wrapper over :func:`build_dem_and_ortho` with ``cloud_las == ref_las``.
    The reference rasters live alongside the other shared reference artefacts
    in ``_ref_cache/`` and are reused across every day's processing — they
    only need to be computed once.

    Skips computation entirely if both output files already exist.

    Parameters
    ----------
    ref_cloud_path : str | Path
        Reference LAS/LAZ. Pass the original cloud for highest fidelity, or
        the pipeline's cached ``_ref_cache/<stem>_ds<f>.las`` for a faster
        build that's consistent with the ICP basis.
    cache_dir : str | Path
        Destination — typically ``output_new/_ref_cache/``.
    res, max_gap_pixels, utm_epsg :
        See :func:`build_dem_and_ortho`.
    cloud_downsample : float
        Additional downsample applied at load time on top of whatever the
        input file already represents (0 < f ≤ 1). Default 1.0 (no extra
        downsample). Use a smaller value when the input is large enough to
        OOM ``griddata``. The output grid spacing is set by ``res`` —
        downsampling further only thins the source points fed to the
        cubic interpolation.
    overwrite : bool
        When False (default), skip computation if both outputs already
        exist in *cache_dir*. When True, force a rebuild.

    Returns
    -------
    (dem_path, ortho_path) : tuple
        ``cache_dir/reference_dem.tif`` and ``cache_dir/reference_ortho.tif``.
        Filenames are fixed — if you change ``cloud_downsample`` or ``res``
        between runs, delete these manually (or set ``overwrite=True``) to
        regenerate.
    """
    return build_dem_and_ortho(
        cloud_las        = Path(ref_cloud_path),
        ref_las          = Path(ref_cloud_path),
        out_dir          = cache_dir,
        name_stem        = "reference",
        res              = res,
        max_gap_pixels   = max_gap_pixels,
        utm_epsg         = utm_epsg,
        cloud_downsample = cloud_downsample,
        overwrite        = overwrite,
    )


def build_dod(
    ref_dem_path: str | Path,
    day_dem_path: str | Path,
    out_path: str | Path = None,
    overwrite: bool = False,
) -> Path:
    """Compute a DEM of Difference: ``ref_dem - day_dem`` on the common grid.

    Both DEMs must share the same shape, affine transform, and CRS — which
    they do when both were built via :func:`build_dem_and_ortho` using the
    same reference cloud and the same ``res``. Pixels where *either* DEM
    holds nodata propagate to NaN in the output, so the DoD is naturally
    masked to the intersection of valid coverage (typically the day's
    footprint inside the reference's wider footprint).

    Sign convention: **positive = reference surface is higher than the day**
    (melt/scour if the day is later than the reference epoch, or
    accumulation that has since occurred relative to the reference baseline
    if the day is earlier).

    Parameters
    ----------
    ref_dem_path : str | Path
        Reference DEM GeoTIFF (e.g. ``_ref_cache/<stem>_dem.tif``).
    day_dem_path : str | Path
        Day's co-registered DEM GeoTIFF
        (e.g. ``<output_new>/<date>/single_day/<date>_dem.tif``).
    out_path : str | Path, optional
        Output GeoTIFF. Default ``<day_dem_path.parent>/DOD.tif``.
    overwrite : bool
        When False (default), skip if *out_path* exists.

    Returns
    -------
    Path
        Path to the written DoD GeoTIFF.
    """
    ref_dem_path = Path(ref_dem_path)
    day_dem_path = Path(day_dem_path)
    if out_path is None:
        out_path = day_dem_path.parent / "DOD.tif"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and out_path.exists():
        print(f"  DoD cached → {out_path.name}")
        return out_path

    with rasterio.open(ref_dem_path) as ref_src, rasterio.open(day_dem_path) as day_src:
        if ref_src.shape != day_src.shape:
            raise ValueError(
                f"DEM shapes differ — ref {ref_src.shape} vs day {day_src.shape}. "
                "Both must be built on the same reference cloud grid."
            )
        if ref_src.transform != day_src.transform:
            raise ValueError(
                f"DEM affine transforms differ — pixels are not aligned.\n"
                f"  ref : {ref_src.transform}\n  day : {day_src.transform}"
            )
        if ref_src.crs != day_src.crs:
            raise ValueError(
                f"DEM CRS differ: ref {ref_src.crs} vs day {day_src.crs}."
            )
        ref_arr   = ref_src.read(1).astype(np.float64)
        day_arr   = day_src.read(1).astype(np.float64)
        ref_nodata = ref_src.nodata
        day_nodata = day_src.nodata
        transform = day_src.transform
        crs_epsg  = day_src.crs.to_string()

    # Replace numeric nodata sentinels (e.g. ASP's -9999) with NaN so the
    # subtraction below propagates correctly. Sources already using NaN as
    # nodata are unaffected — `arr == NaN` is False everywhere, so np.where
    # is a no-op for them.
    if ref_nodata is not None and not np.isnan(ref_nodata):
        ref_arr = np.where(ref_arr == ref_nodata, np.nan, ref_arr)
    if day_nodata is not None and not np.isnan(day_nodata):
        day_arr = np.where(day_arr == day_nodata, np.nan, day_arr)

    # NaN propagates naturally — pixels with nodata in either source become NaN.
    dod = ref_arr - day_arr
    save_dem(dod, out_path, crs_epsg, transform)

    n_valid = int(np.isfinite(dod).sum())
    n_total = int(dod.size)
    print(f"  DoD valid pixels : {n_valid:,}/{n_total:,} "
          f"({100 * n_valid / n_total:.1f}%)")
    return out_path
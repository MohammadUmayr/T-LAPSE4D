from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
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
            idx: np.ndarray | slice
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


def save_las(data: np.ndarray, path: str | Path, crs: int | None = None) -> None:
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

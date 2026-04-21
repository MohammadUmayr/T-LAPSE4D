import rasterio
import numpy as np
from scipy.spatial import cKDTree
import laspy
from pathlib import Path


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


def save_las(data: np.ndarray, path: str | Path) -> None:
    """Save an Nx9 array (X, Y, Z, R, G, B, NX, NY, NZ) to a .las/.laz file.

    RGB values are expected in 0-255 range and stored as LAS uint16 (0-65535).
    Normals are stored as float32 extra dimensions named 'normal x/y/z'.
    """
    path = Path(path)
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.add_extra_dims([
        laspy.ExtraBytesParams(name="normal x", type=np.float32),
        laspy.ExtraBytesParams(name="normal y", type=np.float32),
        laspy.ExtraBytesParams(name="normal z", type=np.float32),
    ])
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
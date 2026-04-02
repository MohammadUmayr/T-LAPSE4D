import rasterio
import numpy as np
from scipy.spatial import cKDTree

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
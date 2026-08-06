import numpy as np
import py4dgeo

# NDWI vs intensity separation line constants
_NDWI_A = -150 / 0.25   # slope
_NDWI_B = 150            # intercept

def downsample_point_cloud(points: np.ndarray, downsample_factor: float) -> np.ndarray:
    """
    Downsample a point cloud by randomly selecting a subset of points.

    Args:
    - points (numpy.ndarray): Input point cloud (N x 3).
    - downsample_factor (float): Downsampling factor (0.0 to 1.0).

    Returns:
    - downsampled_points (numpy.ndarray): Downsampled point cloud.
    """
    num_points = points.shape[0]
    num_points_to_keep = int(num_points * downsample_factor)
    indices = np.random.choice(num_points, num_points_to_keep, replace=False)
    downsampled_points = points[indices]
    return downsampled_points

def filter_points_inside_box(points: np.ndarray, min_bound: np.ndarray, max_bound: np.ndarray) -> np.ndarray:
    """
    Filter points that fall inside a specified bounding box.

    Args:
    - points (numpy.ndarray): Input point cloud (N x 3).
    - min_bound (numpy.ndarray): Minimum bounds of the box (1 x 3).
    - max_bound (numpy.ndarray): Maximum bounds of the box (1 x 3).

    Returns:
    - filtered_points (numpy.ndarray): Filtered point cloud.
    """
    mask = np.all((points[:, :3] >= min_bound) & (points[:, :3] <= max_bound), axis=1)
    filtered_points = points[mask]
    return filtered_points

def filter_points_outside_box(points: np.ndarray, min_bound: np.ndarray, max_bound: np.ndarray) -> np.ndarray:
    """
    Filter points that fall outside a specified bounding box.

    Args:
    - points (numpy.ndarray): Input point cloud (N x 3).
    - min_bound (numpy.ndarray): Minimum bounds of the box (1 x 3).
    - max_bound (numpy.ndarray): Maximum bounds of the box (1 x 3).

    Returns:
    - filtered_points (numpy.ndarray): Filtered point cloud.
    """
    mask = ~np.all((points[:, :3] >= min_bound) & (points[:, :3] <= max_bound), axis=1)
    filtered_points = points[mask]
    return filtered_points

def filter_points(
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    criterium: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Filter points based on given criterium.

    Args:
    - points (numpy.ndarray): Input point cloud coordinates (N x 3).
    - colors (numpy.ndarray): RGB colors corresponding to points (N x 3).
    - normals (numpy.ndarray): Point normals corresponding to points (N x 3).
    - criterium (numpy.ndarray): Vector of values used for thresholding (N x 1).
    - threshold (float): Threshold value for color intensity filtering.

    Returns:
    - filtered_data (numpy.ndarray): Filtered point cloud data (N x 9).
    """
    mask = criterium < threshold  # Filter points based on grayscale intensity
    filtered_data = np.hstack((points[mask], colors[mask], normals[mask]))
    return filtered_data

def otsu_thresholding(vector: np.ndarray, bins_num: int) -> float:
    """
    Determine Otsu threshold of distribution

    Args:
    - vector (float64): input values (Nx1)
    - bins_num (int): number of bins in histogram

    Returns:
    threshold (int): Otsu threshold value
    """
     
    # Get the image histogram
    hist, bin_edges = np.histogram(vector, bins=bins_num)
     
    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
     
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
     
    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
     
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
     
    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)
     
    threshold = bin_mids[:-1][index_of_max_val]
    return threshold

def calculate_aspect_slope(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate aspect and slope from the normal vector.

    Args:
    - normal (numpy.ndarray): Normal vector of the point (Nx3).

    Returns:
    - aspect (numpy.ndarray): Aspect angle in degrees.
    - slope (numpy.ndarray): Slope angle in degrees.
    """
    # Extract x, y, z components of normal vector
    normal_x, normal_y, normal_z = normal[:, 0], normal[:, 1], normal[:, 2]

    # Calculate aspect angle
    aspect = np.arctan2(-normal_y, -normal_x) * 180 / np.pi

    # Calculate slope angle
    slope = np.arctan(np.sqrt(normal_x**2 + normal_y**2) / normal_z) * 180 / np.pi

    return aspect, slope

def extract_stable_terrain(cloud: np.ndarray, slope_threshold: float = 60) -> tuple:
    """Extract stable terrain points from a point cloud using slope and NDWI filters.

    Parameters
    ----------
    cloud : np.ndarray
        Nx9 array (X, Y, Z, R, G, B, NX, NY, NZ) with RGB in 0-255.
    slope_threshold : float
        Minimum slope angle in degrees to consider a point geometrically stable.
        Default is 60°.

    Returns
    -------
    stable_slope : np.ndarray
        Points passing the slope filter only (used for geometry plot and NDWI scatter).
    stable_final : np.ndarray
        Points passing both slope and NDWI filters (used for RGB plot and ICP).
    """
    _, slope = calculate_aspect_slope(cloud[:, 6:])
    stable_slope = cloud[slope > slope_threshold]

    grayscale = np.mean(stable_slope[:, 3:6], axis=1)
    ndwi = (stable_slope[:, 5] - stable_slope[:, 3]) / (stable_slope[:, 3] + stable_slope[:, 5])
    mask = grayscale - (ndwi * _NDWI_A + _NDWI_B) < 0
    stable_final = stable_slope[mask]

    return stable_slope, stable_final


def run_m3c2(epoch_ref: py4dgeo.Epoch, epoch_tba: py4dgeo.Epoch,
             normal_radii: float = 2.5, cyl_radius: float = 2.5,
             max_distance: float = 30) -> tuple:
    """Run M3C2 distance computation between two epochs.

    Parameters
    ----------
    epoch_ref : py4dgeo.Epoch
        Reference epoch.
    epoch_tba : py4dgeo.Epoch
        To-be-aligned epoch.
    normal_radii : float
        Radius for normal estimation (default 2.5).
    cyl_radius : float
        Cylinder radius for distance computation (default 2.5).
    max_distance : float
        Maximum search distance (default 30).

    Returns
    -------
    median : float
    nmad : float
        1.4826 × median(|d − median(d)|) — robust spread, not inflated by outliers.
    std : float
        Standard deviation — sensitive to outliers; kept for comparison.
    distances : np.ndarray
    """
    m3c2 = py4dgeo.M3C2(
        epochs=(epoch_ref, epoch_tba),
        corepoints=epoch_ref.cloud[:],
        normal_radii=(normal_radii,),
        cyl_radius=cyl_radius,
        max_distance=max_distance,
    )
    distances, _ = m3c2.run()
    med  = float(np.nanmedian(distances))
    nmad = float(1.4826 * np.nanmedian(np.abs(distances - med)))
    std  = float(np.nanstd(distances))
    return med, nmad, std, distances

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

def downsample_point_cloud(points, downsample_factor):
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

def filter_points_inside_box(points, min_bound, max_bound):
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

def filter_points_outside_box(points, min_bound, max_bound):
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

def filter_points(points, colors, normals, criterium, threshold):
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

def otsu_thresholding(vector, bins_num):
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

def calculate_aspect_slope(normal):
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

def interpolate_and_mask(x, y, z, xi, yi, res, max_gap_pixels):
    # Interpolation (cubic)
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    # Distance to nearest real point
    tree = cKDTree(np.column_stack((x, y)))
    dist, _ = tree.query(np.column_stack((xi.ravel(), yi.ravel())), k=1)
    dist = dist.reshape(xi.shape)

    # Mask far pixels
    mask = dist > (max_gap_pixels * res)
    zi_masked = np.where(mask, np.nan, zi)

    return zi_masked
import csv
import numpy as np
import py4dgeo
import rasterio
from rasterio.errors import RasterioIOError
from pathlib import Path
from tqdm.auto import tqdm

import cntp.coreg
from cntp.coreg import downsample_point_cloud, filter_points_inside_box, filter_points_outside_box, filter_points, calculate_aspect_slope, interpolate_and_mask
from cntp.io import save_dem, save_ortho
from cntp.plot import plot_stable_terrain_geometry, plot_ndwi_vs_intensity, plot_stable_terrain_rgb



def coreg_pc(ref_cloud_path: Path, 
             tba_cloud_dir: Path, 
             dem_dir: Path, 
             PC_dir: str | Path,
             plot_dir: str | Path,
             min_bound: np.ndarray,
             max_bound: np.ndarray,
             crs_epsg: str = "EPSG:2154",
             overwrite: bool = False,
             overwrite_plots: bool = False,
             ):
    """Co-register tba point clouds to a reference cloud using ICP via py4dgeo.

    Parameters
    ----------
    ref_cloud_path : Path
        Path to the reference point cloud text file.
    tba_cloud_dir : Path
        Directory containing tba point cloud files (PC*.txt).
    dem_dir : Path
        Output directory for DEMs and orthoimages.
    PC_dir : str | Path
        Output directory for co-registered point cloud text files.
    plot_dir : str | Path
        Output directory for diagnostic plots.
    min_bound : np.ndarray
        Minimum XYZ bounds for initial bounding box filter (shape 3,).
    max_bound : np.ndarray
        Maximum XYZ bounds for initial bounding box filter (shape 3,).
    crs_epsg : str
        EPSG code for the coordinate reference system (default: "EPSG:2154").
    overwrite : bool
        If False (default), skip point clouds whose outputs already exist.
    overwrite_plots : bool
        If False (default), skip plots that already exist.
    """
    # Ensure output directories exist
    Path(dem_dir).mkdir(parents=True, exist_ok=True)
    Path(PC_dir).mkdir(parents=True, exist_ok=True)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Load reference cloud once
    ref_cloud = np.loadtxt(ref_cloud_path)

    # Filter points inside the bounding box
    ref_cloud = filter_points_inside_box(ref_cloud[:, :], min_bound, max_bound)

    # prepare ref point cloud
    downsample_factor = 1
    downsampled_data_ref = downsample_point_cloud(ref_cloud[:, :], downsample_factor)

    coordinates_ref = downsampled_data_ref[:, :3]
    rgb_color_ref = downsampled_data_ref[:, 3:6]
    point_normals_ref = downsampled_data_ref[:, 6:]

    aspect_ref, slope_ref = calculate_aspect_slope(point_normals_ref)


    # remove all points less steep than 60° 
    mask = slope_ref > 60  # Filter points based on grayscale intensity
    stable_points_ref = downsampled_data_ref[mask]

    min_bound = np.array([1.0107e6, 6.54465e6, 2700])  # Minimum bounds (adjust as needed)
    max_bound = np.array([1.0112e6, 6.5452e6, 3100])  # Maximum bounds (adjust as needed)
    stable_points_ref = filter_points_outside_box(stable_points_ref[:, :], min_bound, max_bound)
    min_bound = np.array([1.0106e6, 6.5445e6, 2900])  # Minimum bounds (adjust as needed)
    max_bound = np.array([1.0112e6, 6.5452e6, 3300])  # Maximum bounds (adjust as needed)
    stable_points_ref = filter_points_inside_box(stable_points_ref[:, :], min_bound, max_bound)
    min_bound = np.array([1.01105e6, 6.5444e6, 2600])  # Minimum bounds (adjust as needed)
    max_bound = np.array([1.0113e6, 6.5452e6, 3300])  # Maximum bounds (adjust as needed)
    stable_points_ref = filter_points_outside_box(stable_points_ref[:, :], min_bound, max_bound)

    _plot_dir = Path(plot_dir)
    _geom_plot = _plot_dir / "stable_terrain_geometry.png"
    if overwrite_plots or not _geom_plot.exists():
        print(f"Plotting stable terrain geometry at {_geom_plot}")
        plot_stable_terrain_geometry(stable_points_ref, plot_dir)
    else:
        print(f"Skipping plot (already exists): {_geom_plot}")

    grayscale_intensity_ref = np.mean(stable_points_ref[:, 3:6], axis=1)
    ndwi_ref = (stable_points_ref[:, 5]-stable_points_ref[:, 3])/(stable_points_ref[:, 3]+stable_points_ref[:, 5])

    # Separation line
    a = -150/0.25
    b = 150
    x_values = np.linspace(min(ndwi_ref), max(ndwi_ref), 100)
    y_values = a * x_values + b

    _ndwi_plot = _plot_dir / "ndwi_vs_intensity.png"
    if overwrite_plots or not _ndwi_plot.exists():
        print(f"Plotting NDWI vs intensity at {_ndwi_plot}")
        plot_ndwi_vs_intensity(ndwi_ref, grayscale_intensity_ref, stable_points_ref[:, 3:6], x_values, y_values, plot_dir)
    else:
        print(f"Skipping plot (already exists): {_ndwi_plot}")

    mask = grayscale_intensity_ref-(ndwi_ref*a+b)<0
    stable_points_ref = stable_points_ref[mask]

    _ref_rgb_plot = _plot_dir / "stable_terrain_ref_rgb.png"
    if overwrite_plots or not _ref_rgb_plot.exists():
        print(f"Plotting stable terrain ref RGB at {_ref_rgb_plot}")
        plot_stable_terrain_rgb(stable_points_ref, plot_dir, title='Zone de terrain stable - nuage ref', filename='stable_terrain_ref_rgb.png')
    else:
        print(f"Skipping plot (already exists): {_ref_rgb_plot}")

    epoch_all_ref = py4dgeo.Epoch(downsampled_data_ref[:,:3])
    epoch_stable_ref = py4dgeo.Epoch(stable_points_ref[:,:3])

    # for export
    res = 1.0 # Define grid resolution in meters
    x_ref,y_ref,z_ref=downsampled_data_ref[:, 0],downsampled_data_ref[:, 1],downsampled_data_ref[:, 2]

    # Define grid bounds
    xmin, xmax = np.min(x_ref), np.max(x_ref)
    ymin, ymax = np.min(y_ref), np.max(y_ref)

    # Create a regular grid
    xi_ref = np.arange(xmin, xmax, res)
    yi_ref = np.arange(ymax, ymin, -res)
    xi_ref, yi_ref = np.meshgrid(xi_ref, yi_ref)

    # interpolate also zones with gaps within one pixel of pixels with values
    max_gap_pixels = 1

    # For saving as GeoTIFF
    transform = rasterio.transform.from_origin(xmin, ymax, res, res)
    
    stats_rows = []

    # Loop over all PC*.txt files
    tba_cloud_paths = sorted(tba_cloud_dir.glob("PC*.txt"))
    for tba_cloud_path in tqdm(tba_cloud_paths, desc="Co-registering point clouds"):
        tba_cloud_name = tba_cloud_path.stem  # e.g. "PC_TL2025-04-10_104622"

        # Check if outputs already exist
        pc_output = Path(PC_dir) / f"{tba_cloud_name}_coreg_TL.txt"
        dem_output = Path(dem_dir) / f"DEM_{tba_cloud_name}_coreg_RGF93.tif"
        ortho_output = Path(dem_dir) / f"ORTHO_{tba_cloud_name}_coreg_RGF93.tif"
        if not overwrite and pc_output.exists() and dem_output.exists() and ortho_output.exists():
            tqdm.write(f"Skipping {tba_cloud_name} (outputs already exist)")
            continue

        tba_cloud = np.loadtxt(tba_cloud_path)

        tqdm.write(f"Processing {tba_cloud_name}")

        # Define the minimum and maximum bounds of the bounding box
        min_bound = np.array([1.0106e6, 6.5445e6, 2750])  # Minimum bounds (adjust as needed)
        max_bound = np.array([1.0112e6, 6.5452e6, 3300])  # Maximum bounds (adjust as needed)

        # Filter points inside the bounding box
        tba_cloud = filter_points_inside_box(tba_cloud[:, :], min_bound, max_bound)

        # # Downsample the point cloud (adjust the downsample_factor as needed)
        # downsample_factor = 1  # Adjust this value based on your preference between 0 (no points) and 1 (all points)
        # downsampled_data_ref = downsample_point_cloud(ref_cloud[:, :], downsample_factor)
        downsampled_data_tba = downsample_point_cloud(tba_cloud[:, :], downsample_factor)

        # Split the data into coordinates (x, y, z) and RGB color
        coordinates_tba = downsampled_data_tba[:, :3]
        rgb_color_tba = downsampled_data_tba[:, 3:6]
        point_normals_tba = downsampled_data_tba[:, 6:]

        aspect_tba, slope_tba = calculate_aspect_slope(point_normals_tba)

        ######## EXTRACT STABLE TERRAIN BASED ON SLOPE & COLOR ##########

        # remove all points less steep than 60° 
        mask = slope_tba > 60
        stable_points_tba = downsampled_data_tba[mask]

        # remove all points which are on the cone
        min_bound = np.array([1.0107e6, 6.54465e6, 2700])  # Minimum bounds (adjust as needed)
        max_bound = np.array([1.0112e6, 6.5452e6, 3100])  # Maximum bounds (adjust as needed)
        stable_points_tba = filter_points_outside_box(stable_points_tba[:, :], min_bound, max_bound)
        min_bound = np.array([1.0106e6, 6.5445e6, 2900])  # Minimum bounds (adjust as needed)
        max_bound = np.array([1.0112e6, 6.5452e6, 3300])  # Maximum bounds (adjust as needed)
        stable_points_tba = filter_points_inside_box(stable_points_tba[:, :], min_bound, max_bound)
        min_bound = np.array([1.01105e6, 6.5444e6, 2600])  # Minimum bounds (adjust as needed)
        max_bound = np.array([1.0113e6, 6.5452e6, 3300])  # Maximum bounds (adjust as needed)
        stable_points_tba = filter_points_outside_box(stable_points_tba[:, :], min_bound, max_bound)

        # remove all points which are more blue than red
        grayscale_intensity_tba = np.mean(stable_points_tba[:, 3:6], axis=1)
        ndwi_tba = (stable_points_tba[:, 5]-stable_points_tba[:, 3])/(stable_points_tba[:, 3]+stable_points_tba[:, 5])

        mask = grayscale_intensity_tba-(ndwi_tba*a+b)<0
        stable_points_tba = stable_points_tba[mask]

        _tba_rgb_plot = _plot_dir / f'stable_terrain_{tba_cloud_name}_rgb.png'
        if overwrite_plots or not _tba_rgb_plot.exists():
            tqdm.write(f"Plotting stable terrain RGB at {_tba_rgb_plot}")
            plot_stable_terrain_rgb(stable_points_tba, plot_dir, title=f'Zone de terrain stable - {tba_cloud_name}', filename=f'stable_terrain_{tba_cloud_name}_rgb.png')
        else:
            tqdm.write(f"Skipping plot (already exists): {_tba_rgb_plot}")

        ####### COMPARE POINT CLOUDS (M3C2 ALGORITHM) #####################

        # Load point clouds into py4dgeo objects
        epoch_all_tba = py4dgeo.Epoch(downsampled_data_tba[:,:3])
        epoch_stable_tba = py4dgeo.Epoch(stable_points_tba[:,:3])

        # Instantiate and parametrize the M3C2 algorithm object
        m3c2 = py4dgeo.M3C2(
            epochs=(epoch_stable_ref, epoch_stable_tba),
            corepoints=epoch_stable_ref.cloud[::10],
            normal_radii=(2.5,),
            cyl_radius=2.5,
            max_distance=30,
            #registration_error=(0.0),
        )

        # Run the distance computation
        m3c2_distances_stableparts, uncertainties_stableparts = m3c2.run()

        med_before = float(np.nanmedian(m3c2_distances_stableparts))
        std_before = float(np.nanstd(m3c2_distances_stableparts))
        tqdm.write(f"Median M3C2 distances STABLE: {med_before:.3f} m")
        tqdm.write(f"Std. dev. of M3C2 distances STABLE: {std_before:.3f} m")


        ########## COREGISTRATION #############

        # rotations allowed relative to camera origin
        CAM_coordinates = np.array([1011523.41,	6545566.751,	2985.720033])

        trafo = py4dgeo.iterative_closest_point(
            epoch_stable_ref, epoch_stable_tba, reduction_point=CAM_coordinates
        )
        epoch_all_tba_coreg = epoch_all_tba
        epoch_stable_tba_coreg = epoch_stable_tba

        epoch_all_tba_coreg.transform(trafo)
        epoch_stable_tba_coreg.transform(trafo)

        ########## RECALCULATE DIFFERENCES ##########

        # Instantiate and parametrize the M3C2 algorithm object
        m3c2 = py4dgeo.M3C2(
            epochs=(epoch_stable_ref, epoch_stable_tba_coreg),
            corepoints=epoch_stable_ref.cloud[::10],
            normal_radii=(2.5,),
            cyl_radius=2.5,
            max_distance=30,
            #registration_error=(0.0),
        )

        # Run the distance computation
        m3c2_distances_stableparts, uncertainties_stableparts = m3c2.run()

        med_after = float(np.nanmedian(m3c2_distances_stableparts))
        std_after = float(np.nanstd(m3c2_distances_stableparts))
        tqdm.write(f"Median M3C2 distances STABLE: {med_after:.3f} m")
        tqdm.write(f"Std. dev. of M3C2 distances STABLE: {std_after:.3f} m")

        stats_rows.append([tba_cloud_name, med_before, std_before, med_after, std_after])


        ######## EXPORT COREGISTERED POINT CLOUD ###########
        coordinates_tba_coreg = epoch_all_tba_coreg.cloud
        rgb_color_tba = downsampled_data_tba[:, 3:6]
        point_normals_tba = downsampled_data_tba[:, 6:]

        tba_cloud_coreg = np.column_stack((coordinates_tba_coreg, rgb_color_tba, point_normals_tba))

        np.savetxt(PC_dir+tba_cloud_name+"_coreg_TL.txt", tba_cloud_coreg, fmt='%.2f %.2f %.2f %d %d %d %.6f %.6f %.6f')

        ######## CONVERT TO DEMs & EXPORT
        x_tba1,y_tba1,z_tba1=coordinates_tba_coreg[:, 0],coordinates_tba_coreg[:, 1],coordinates_tba_coreg[:, 2]

        zi_tba1_masked = interpolate_and_mask(x_tba1, y_tba1, z_tba1, xi_ref, yi_ref, res, max_gap_pixels)

        save_dem(zi_tba1_masked,   dem_dir / f"DEM_{tba_cloud_name}_coreg_RGF93.tif", crs_epsg, transform)

        ######## EXPORT ORTHOIMAGES
        save_ortho(x_tba1, y_tba1, rgb_color_tba, xi_ref, yi_ref, res, max_gap_pixels, 
            dem_dir / f"ORTHO_{tba_cloud_name}_coreg_RGF93.tif", crs_epsg, transform)

        tqdm.write('Outputs saved')
        
        # break

    # Write coregistration statistics to CSV
    csv_path = Path(dem_dir) / "coreg_stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tba_cloud_name", "med_before_coreg", "std_before_coreg", "med_after_coreg", "std_after_coreg"])
        writer.writerows(stats_rows)
    print(f"Stats saved to {csv_path}")
        
    print('DONE')
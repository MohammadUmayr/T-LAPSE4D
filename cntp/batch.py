import csv
import numpy as np
import py4dgeo
from pathlib import Path
from tqdm.auto import tqdm

from cntp.coreg import filter_points_inside_box, extract_stable_terrain, coreg_pc as _coreg_single, _NDWI_A, _NDWI_B
from cntp.io import load_las, save_las, read_las_bounds
from cntp.plot import plot_stable_terrain_diagnostics


def coreg_pc(ref_cloud_path: str | Path,
                tba_cloud: str | Path,
                output_dir: str | Path,
                min_bound: np.ndarray = None,
                max_bound: np.ndarray = None,
                crs_epsg: str = "EPSG:2154",
                downsample_factor: float = 0.5,
                cam_coordinates: np.ndarray = None,
                overwrite: bool = False,
                overwrite_plots: bool = False,
                ):
    """Batch co-register TBA point clouds to a reference cloud.

    Output layout created automatically
    ------------------------------------
    output_dir/
        dem_dir/               # trafo txts, coreg_stats.csv
        coreg_PC_dir/          # co-registered point clouds (.las)
        plot_dir/
            reference/         # stable_terrain_geometry, ndwi_vs_intensity, stable_terrain_rgb
            <tba_cloud_name>/  # same three plots for each TBA cloud
            ...

    Parameters
    ----------
    ref_cloud_path : str | Path
        Path to the reference point cloud (.las/.laz).
    tba_cloud : str | Path
        Either a single .las/.laz file or a directory containing .las/.laz files.
    output_dir : str | Path
        Root directory where all outputs are written.
    min_bound : np.ndarray, optional
        Minimum XYZ bounds for bounding box filter (shape 3,).
        If None, extracted automatically from the reference cloud header.
    max_bound : np.ndarray, optional
        Maximum XYZ bounds for bounding box filter (shape 3,).
        If None, extracted automatically from the reference cloud header.
    crs_epsg : str
        EPSG code for the coordinate reference system (default: "EPSG:2154").
    downsample_factor : float
        Fraction of points to keep during processing (0 < factor <= 1.0).
    cam_coordinates : np.ndarray
        Camera origin used as the ICP reduction point (default: [0, 0, 0]).
    overwrite : bool
        If False (default), skip point clouds whose outputs already exist.
    overwrite_plots : bool
        If False (default), skip plots that already exist.
    """
    if cam_coordinates is None:
        cam_coordinates = np.array([0, 0, 0])

    if min_bound is None or max_bound is None:
        min_bound, max_bound = read_las_bounds(ref_cloud_path)
        print(f"Bounds from reference header: min={min_bound}, max={max_bound}")

    output_dir = Path(output_dir)
    dem_dir      = output_dir / "dem_dir"
    PC_dir       = output_dir / "coreg_PC_dir"
    plot_dir     = output_dir / "plot_dir"
    ref_plot_dir = plot_dir / "reference"

    for d in (dem_dir, PC_dir, ref_plot_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Resolve TBA cloud list
    tba_cloud = Path(tba_cloud)
    if tba_cloud.is_file():
        tba_cloud_paths = [tba_cloud]
    elif tba_cloud.is_dir():
        tba_cloud_paths = sorted(
            list(tba_cloud.glob("*.las")) + list(tba_cloud.glob("*.laz"))
        )
    else:
        raise ValueError(f"tba_cloud must be an existing file or directory, got: {tba_cloud}")

    # Filter out already-processed clouds when not overwriting
    if not overwrite:
        tba_cloud_paths = [p for p in tba_cloud_paths
                           if not (PC_dir / f"{p.stem}_coreg_TL.las").exists()]
        if not tba_cloud_paths and not overwrite_plots:
            print("All outputs already exist, nothing to do.")
            return

    # ------------------------------------------------------------------
    # Load and prepare reference cloud
    # ------------------------------------------------------------------
    ref_cloud = load_las(ref_cloud_path, downsample_factor=downsample_factor)
    ref_cloud = filter_points_inside_box(ref_cloud, min_bound, max_bound)

    stable_slope_ref, stable_final_ref = extract_stable_terrain(ref_cloud)

    grayscale_ref = np.mean(stable_slope_ref[:, 3:6], axis=1)
    ndwi_ref = (stable_slope_ref[:, 5] - stable_slope_ref[:, 3]) / (stable_slope_ref[:, 3] + stable_slope_ref[:, 5])

    plot_stable_terrain_diagnostics(
        stable_slope_ref, stable_final_ref, ndwi_ref, grayscale_ref,
        _NDWI_A, _NDWI_B, ref_plot_dir, title='nuage ref', overwrite=overwrite_plots,
    )

    epoch_stable_ref = py4dgeo.Epoch(stable_final_ref[:, :3])

    # ------------------------------------------------------------------
    # Process each TBA cloud
    # ------------------------------------------------------------------
    stats_rows = []

    for tba_cloud_path in tqdm(tba_cloud_paths, desc="Co-registering point clouds"):
        tba_cloud_name = tba_cloud_path.stem

        pc_output    = PC_dir  / f"{tba_cloud_name}_coreg_TL.las"
        tba_plot_dir = plot_dir / tba_cloud_name
        tba_plot_dir.mkdir(parents=True, exist_ok=True)

        if not overwrite and pc_output.exists():
            tqdm.write(f"Skipping {tba_cloud_name} (outputs already exist)")
            continue

        tqdm.write(f"Processing {tba_cloud_name}")

        tba_data = load_las(tba_cloud_path, downsample_factor=downsample_factor)
        tba_data = filter_points_inside_box(tba_data, min_bound, max_bound)

        result = _coreg_single(epoch_stable_ref, tba_data, cam_coordinates)

        tqdm.write(f"Before coreg — median: {result['med_before']:.3f} m  std: {result['std_before']:.3f} m")
        tqdm.write(f"After  coreg — median: {result['med_after']:.3f} m  std: {result['std_after']:.3f} m")

        plot_stable_terrain_diagnostics(
            result["stable_slope"], result["stable_final"],
            result["ndwi"], result["grayscale"],
            _NDWI_A, _NDWI_B, tba_plot_dir, title=tba_cloud_name, overwrite=overwrite_plots,
        )

        stats_rows.append([tba_cloud_name,
                           result["med_before"], result["std_before"],
                           result["med_after"],  result["std_after"]])

        # Save transformation
        trafo = result["trafo"]
        trafo_path = dem_dir / f"trafo_{tba_cloud_name}.txt"
        with open(trafo_path, "w") as trafo_file:
            trafo_file.write("affine_transformation (4x4):\n")
            np.savetxt(trafo_file, trafo.affine_transformation)
            trafo_file.write("reduction_point:\n")
            np.savetxt(trafo_file, trafo.reduction_point.reshape(1, -1))
        tqdm.write(f"Trafo saved to {trafo_path}")

        save_las(result["cloud_coreg"], pc_output)
        tqdm.write(f"Saved co-registered cloud to {pc_output}")

    # Write co-registration statistics
    csv_path = dem_dir / "coreg_stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tba_cloud_name", "med_before_coreg", "std_before_coreg",
                         "med_after_coreg", "std_after_coreg"])
        writer.writerows(stats_rows)
    print(f"Stats saved to {csv_path}")
    print("DONE")

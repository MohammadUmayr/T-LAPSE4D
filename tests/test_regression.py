"""
Regression tests against the locked baseline in coreg_stats.csv.

Tolerance-based metric comparisons (med_before, med_after, std_after) are
skipped until the right tolerances are defined — run the notebook ~10 times
with downsample_factor=0.15 and measure the actual run-to-run spread first.
"""
import csv
import numpy as np
from pathlib import Path
import laspy
import pytest

STATS_CSV = Path(__file__).parent.parent / "contributors/umayr/output/dem_dir/coreg_stats.csv"

# Baseline values captured from the current run (2026-04-21).
# med/std in metres.
BASELINE = {
    "Reference_UAV_TLC": {
        "med_before": 1.49e-05,
        "std_before": 4.75e-04,
        "med_after":  4.73e-05,
        "std_after":  3.04e-04,
    },
    "TL": {
        "med_before": -4.42e-05,
        "std_before":  1.41e-03,
        "med_after":  -1.20e-05,
        "std_after":   1.46e-03,
    },
    "TL_UAV": {
        "med_before":  4.37e-09,
        "std_before":  2.19e-04,
        "med_after":   3.16e-05,
        "std_after":   2.05e-04,
    },
}


@pytest.fixture(scope="module")
def stats():
    if not STATS_CSV.exists():
        pytest.skip(f"Stats CSV not found: {STATS_CSV}")
    with open(STATS_CSV) as f:
        return {row["tba_cloud_name"]: row for row in csv.DictReader(f)}


def test_all_expected_clouds_present(stats):
    """Every cloud in the baseline must appear in the CSV."""
    missing = [name for name in BASELINE if name not in stats]
    assert not missing, f"Missing clouds in stats CSV: {missing}"


@pytest.mark.skip(reason="Tolerance not yet defined — measure run-to-run spread first")
@pytest.mark.parametrize("cloud_name", list(BASELINE.keys()))
def test_med_before_matches_baseline(stats, cloud_name):
    actual = float(stats[cloud_name]["med_before_coreg"])
    expected = BASELINE[cloud_name]["med_before"]
    # TODO: replace ... with a measured tolerance
    assert abs(actual - expected) < ..., (
        f"{cloud_name} med_before changed: {actual:.2e} vs baseline {expected:.2e}"
    )


@pytest.mark.skip(reason="Tolerance not yet defined — measure run-to-run spread first")
@pytest.mark.parametrize("cloud_name", list(BASELINE.keys()))
def test_med_after_matches_baseline(stats, cloud_name):
    actual = float(stats[cloud_name]["med_after_coreg"])
    expected = BASELINE[cloud_name]["med_after"]
    # TODO: replace ... with a measured tolerance
    assert abs(actual - expected) < ..., (
        f"{cloud_name} med_after changed: {actual:.2e} vs baseline {expected:.2e}"
    )


@pytest.mark.skip(reason="Tolerance not yet defined — measure run-to-run spread first")
@pytest.mark.parametrize("cloud_name", list(BASELINE.keys()))
def test_std_after_matches_baseline(stats, cloud_name):
    actual = float(stats[cloud_name]["std_after_coreg"])
    expected = BASELINE[cloud_name]["std_after"]
    # TODO: replace ... with a measured tolerance
    assert abs(actual - expected) < ..., (
        f"{cloud_name} std_after changed: {actual:.2e} vs baseline {expected:.2e}"
    )


@pytest.mark.skip(reason="Tolerance not yet defined — measure run-to-run spread first")
@pytest.mark.parametrize("cloud_name", list(BASELINE.keys()))
def test_std_before_matches_baseline(stats, cloud_name):
    actual = float(stats[cloud_name]["std_before_coreg"])
    expected = BASELINE[cloud_name]["std_before"]
    # TODO: replace ... with a measured tolerance
    assert abs(actual - expected) < ..., (
        f"{cloud_name} std_before changed: {actual:.2e} vs baseline {expected:.2e}"
    )


# ---------------------------------------------------------------------------
# Transformation matrix tests
# ---------------------------------------------------------------------------

DEM_DIR = Path(__file__).parent.parent / "contributors/umayr/output/dem_dir"
PC_DIR  = Path(__file__).parent.parent / "contributors/umayr/output/coreg_PC_dir"

# Reference spatial bounds read from the UAV.laz header (EPSG:4326).
REF_MIN = np.array([ 86.7703,  27.9754, 5312.04])
REF_MAX = np.array([ 86.7816,  27.9847, 5745.71])


def _parse_trafo(path: Path) -> np.ndarray:
    """Parse a trafo_*.txt file and return the 4×4 affine matrix."""
    lines = path.read_text().splitlines()
    # Line 0: "affine_transformation (4x4):"
    # Lines 1-4: the four rows
    matrix_lines = [l for l in lines[1:5] if l.strip()]
    return np.array([list(map(float, l.split())) for l in matrix_lines])


@pytest.mark.parametrize("cloud_name", list(BASELINE.keys()))
def test_trafo_file_exists(cloud_name):
    """Transformation file must be written for every processed cloud."""
    path = DEM_DIR / f"trafo_{cloud_name}.txt"
    assert path.exists(), f"Missing trafo file: {path}"


@pytest.mark.parametrize("cloud_name", list(BASELINE.keys()))
def test_trafo_is_valid_rigid_body(cloud_name):
    """Affine matrix must be a valid rigid-body transform: det(R) ≈ 1, last row = [0,0,0,1]."""
    path = DEM_DIR / f"trafo_{cloud_name}.txt"
    if not path.exists():
        pytest.skip(f"Trafo file missing: {path}")

    A = _parse_trafo(path)

    assert A.shape == (4, 4), f"Expected 4×4 matrix, got {A.shape}"

    R = A[:3, :3]
    det = np.linalg.det(R)
    np.testing.assert_allclose(det, 1.0, atol=1e-6,
        err_msg=f"{cloud_name}: det(R) = {det:.8f}, expected 1.0")

    np.testing.assert_allclose(A[3], [0, 0, 0, 1], atol=1e-10,
        err_msg=f"{cloud_name}: last row of affine matrix is not [0,0,0,1]: {A[3]}")


# ---------------------------------------------------------------------------
# Co-registered .las file tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cloud_name", list(BASELINE.keys()))
def test_coreg_las_exists(cloud_name):
    """Co-registered .las file must be written for every processed cloud."""
    path = PC_DIR / f"{cloud_name}_coreg_TL.las"
    assert path.exists(), f"Missing co-registered cloud: {path}"


@pytest.mark.parametrize("cloud_name", list(BASELINE.keys()))
def test_coreg_las_is_non_empty(cloud_name):
    """Co-registered .las must contain at least one point."""
    path = PC_DIR / f"{cloud_name}_coreg_TL.las"
    if not path.exists():
        pytest.skip(f"File missing: {path}")

    with laspy.open(path) as f:
        count = f.header.point_count

    assert count > 0, f"{cloud_name}: co-registered cloud has 0 points"


@pytest.mark.parametrize("cloud_name", list(BASELINE.keys()))
def test_coreg_las_bounds_within_reference(cloud_name):
    """Co-registered cloud must not be far outside the reference spatial extent.

    ICP is applied after the bounding box filter, so points near the edges can
    legitimately drift slightly outside the reference bounds. The margin here is
    set to catch runaway ICP (cloud ending up kilometers away), not minor edge drift:
      - XY: 0.05 degrees (~5 km at this latitude)
      - Z:  500 m
    """
    path = PC_DIR / f"{cloud_name}_coreg_TL.las"
    if not path.exists():
        pytest.skip(f"File missing: {path}")

    with laspy.open(path) as f:
        mins = np.array(f.header.mins)
        maxs = np.array(f.header.maxs)

    margin = np.array([0.05, 0.05, 500.0])

    assert np.all(mins >= REF_MIN - margin), (
        f"{cloud_name}: cloud min {mins} is far outside reference min {REF_MIN}"
    )
    assert np.all(maxs <= REF_MAX + margin), (
        f"{cloud_name}: cloud max {maxs} is far outside reference max {REF_MAX}"
    )

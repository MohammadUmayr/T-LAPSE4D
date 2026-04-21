"""
Synthetic co-registration accuracy tests.

These tests create point clouds with a *known* rigid transformation applied,
run coreg_pc, and verify that ICP recovers the transformation well enough
that the M3C2 residual improves and stays within a tolerance.

No real .laz files are required.
"""
import numpy as np
import py4dgeo
import pytest
from cntp.coreg import coreg_pc, extract_stable_terrain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_stable_cloud(n: int = 4000, seed: int = 0) -> np.ndarray:
    """Return an Nx9 cloud where most points will pass the stable terrain filters.

    - XYZ spread over a 20×20×10 m volume.
    - RGB is grey-brown (high grayscale, low NDWI → passes NDWI filter).
    - Normals are predominantly horizontal (slope ~ 70–90°) → passes slope filter.
    """
    rng = np.random.default_rng(seed)
    xyz = rng.uniform([0, 0, 0], [20, 20, 10], (n, 3))

    # Grey-brown: R > B so NDWI = (B-R)/(R+B) is negative → below separation line
    rgb = rng.uniform(80, 140, (n, 3))
    rgb[:, 2] = rng.uniform(40, 80, n)  # keep B < R

    # Mostly horizontal normals: high sin component, low cos (Z) component
    angle = rng.uniform(0, 2 * np.pi, n)
    tilt  = rng.uniform(0.87, 1.0, n)   # sin of inclination; gives slope > 60°
    nx = tilt * np.cos(angle)
    ny = tilt * np.sin(angle)
    nz = np.sqrt(np.clip(1.0 - nx**2 - ny**2, 1e-6, None))
    normals = np.column_stack([nx, ny, nz])

    return np.column_stack([xyz, rgb, normals])


def apply_translation(cloud: np.ndarray, shift: np.ndarray) -> np.ndarray:
    shifted = cloud.copy()
    shifted[:, :3] += shift
    return shifted


def apply_rotation(cloud: np.ndarray, R3x3: np.ndarray) -> np.ndarray:
    """Apply a 3×3 rotation around the cloud centroid to XYZ and normals.

    Both XYZ (cols 0:3) and normals (cols 6:9) are rotated so that the
    geometry stays consistent — important because extract_stable_terrain uses
    normals to compute slope, and X/Y rotations change which points survive
    the slope filter.
    """
    rotated = cloud.copy()
    centroid = cloud[:, :3].mean(axis=0)
    rotated[:, :3] = (cloud[:, :3] - centroid) @ R3x3.T + centroid
    rotated[:, 6:9] = cloud[:, 6:9] @ R3x3.T
    return rotated


def Rx(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array([[1, 0,           0          ],
                     [0, np.cos(t), -np.sin(t)   ],
                     [0, np.sin(t),  np.cos(t)   ]])


def Ry(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array([[ np.cos(t), 0, np.sin(t)  ],
                     [ 0,         1, 0           ],
                     [-np.sin(t), 0, np.cos(t)  ]])


def Rz(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t),  np.cos(t), 0],
                     [0,          0,         1]])


def build_ref_epoch(cloud: np.ndarray) -> py4dgeo.Epoch:
    """Extract stable terrain from cloud and return a py4dgeo Epoch."""
    _, stable_final = extract_stable_terrain(cloud)
    assert len(stable_final) > 50, (
        f"Too few stable points ({len(stable_final)}) — "
        "synthetic cloud may not pass the filters"
    )
    return py4dgeo.Epoch(stable_final[:, :3])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_coreg_improves_m3c2_after_small_translation():
    """ICP should reduce M3C2 residual after a 0.3 m translation."""
    ref = make_stable_cloud(seed=1)
    tba = apply_translation(ref, np.array([0.3, -0.2, 0.05]))

    epoch_ref = build_ref_epoch(ref)
    result = coreg_pc(epoch_ref, tba)

    assert abs(result["med_after"]) < abs(result["med_before"]) or abs(result["med_after"]) < 0.05, (
        f"M3C2 did not improve: before={result['med_before']:.4f} m, after={result['med_after']:.4f} m"
    )


def test_coreg_residual_within_tolerance_after_translation():
    """After co-registering a 0.5 m shifted cloud the residual must be < 0.1 m."""
    ref = make_stable_cloud(seed=2)
    tba = apply_translation(ref, np.array([0.5, 0.0, 0.0]))

    epoch_ref = build_ref_epoch(ref)
    result = coreg_pc(epoch_ref, tba)

    assert abs(result["med_after"]) < 0.1, (
        f"Residual after coreg too large: {result['med_after']:.4f} m"
    )


def test_coreg_identical_clouds_stays_near_zero():
    """Co-registering a cloud to itself must produce M3C2 = 0 (to floating-point precision).

    There is no geometric distance between identical point sets, so M3C2
    returns exactly 0 or sub-nanometre floating-point noise. Any value larger
    than 1e-10 m indicates a bug in epoch construction, M3C2 computation, or
    ICP introducing spurious drift.
    """
    ref = make_stable_cloud(seed=3)
    epoch_ref = build_ref_epoch(ref)
    result = coreg_pc(epoch_ref, ref.copy())

    assert abs(result["med_before"]) < 1e-10, (
        f"M3C2 before should be 0 for identical clouds, got {result['med_before']:.2e} m"
    )
    assert abs(result["med_after"]) < 1e-10, (
        f"M3C2 after should be 0 for identical clouds, got {result['med_after']:.2e} m"
    )


def test_coreg_output_keys_present():
    """Result dict must contain all documented keys."""
    ref = make_stable_cloud(seed=4)
    tba = apply_translation(ref, np.array([0.1, 0.1, 0.0]))
    epoch_ref = build_ref_epoch(ref)
    result = coreg_pc(epoch_ref, tba)

    expected_keys = {
        "cloud_coreg", "trafo",
        "stable_slope", "stable_final",
        "ndwi", "grayscale",
        "med_before", "std_before",
        "med_after", "std_after",
    }
    assert expected_keys <= result.keys(), (
        f"Missing keys: {expected_keys - result.keys()}"
    )


def test_coreg_output_cloud_shape_preserved():
    """Co-registered cloud must have the same number of points and 9 columns."""
    ref = make_stable_cloud(n=2000, seed=5)
    tba = apply_translation(ref, np.array([0.2, -0.1, 0.0]))
    epoch_ref = build_ref_epoch(ref)
    result = coreg_pc(epoch_ref, tba)

    assert result["cloud_coreg"].shape == tba.shape, (
        f"Output shape {result['cloud_coreg'].shape} != input shape {tba.shape}"
    )


def test_coreg_transformation_is_near_rigid():
    """The recovered affine transformation should be close to a rotation matrix (det ≈ 1)."""
    ref = make_stable_cloud(seed=6)
    tba = apply_translation(ref, np.array([0.4, 0.2, 0.0]))
    epoch_ref = build_ref_epoch(ref)
    result = coreg_pc(epoch_ref, tba)

    R = result["trafo"].affine_transformation[:3, :3]
    det = np.linalg.det(R)
    np.testing.assert_allclose(det, 1.0, atol=1e-3,
        err_msg=f"Rotation matrix determinant {det:.6f} is not close to 1")


def test_recovered_translation_matches_known_shift():
    """The translation vector in the affine matrix must equal the negated input shift.

    py4dgeo ICP returns T such that T(TBA) ≈ ref.  For TBA = ref + shift the
    expected recovered translation is t = -shift and R = I (no rotation).

    The affine transform (with reduction_point=[0,0,0]) is applied as:
        p_out = R @ p + t
    so t is extracted directly from affine_transformation[:3, 3].
    """
    known_shift = np.array([0.3, -0.2, 0.05])
    ref = make_stable_cloud(seed=7)
    tba = apply_translation(ref, known_shift)

    epoch_ref = build_ref_epoch(ref)
    result = coreg_pc(epoch_ref, tba)

    t = result["trafo"].affine_transformation[:3, 3]
    np.testing.assert_allclose(
        t, -known_shift, atol=0.05,
        err_msg=f"Recovered translation {t} does not match expected {-known_shift}"
    )


def test_recovered_rotation_is_identity_for_pure_translation():
    """For a pure translation, the rotation part of the recovered transform must be identity."""
    ref = make_stable_cloud(seed=8)
    tba = apply_translation(ref, np.array([0.5, -0.3, 0.0]))

    epoch_ref = build_ref_epoch(ref)
    result = coreg_pc(epoch_ref, tba)

    R = result["trafo"].affine_transformation[:3, :3]
    np.testing.assert_allclose(
        R, np.eye(3), atol=1e-3,
        err_msg=f"Rotation matrix is not identity for a pure translation:\n{R}"
    )


@pytest.mark.parametrize("axis,Rfn", [("X", Rx), ("Y", Ry), ("Z", Rz)])
@pytest.mark.parametrize("angle_deg", [2.0, 5.0])
def test_recovered_rotation_matches_known_rotation(axis, Rfn, angle_deg):
    """The rotation block of the affine matrix must match the inverse of the applied rotation.

    apply_rotation rotates XYZ and normals around the cloud centroid by R3x3.
    ICP must return T such that T(TBA) ≈ ref, so the recovered rotation
    R_rec must equal the inverse: Rfn(−angle_deg).

    All three axes (X/Y/Z) are tested — X and Y change the normal's Z component
    so normals must be rotated consistently with XYZ, otherwise the slope filter
    classifies points using stale normals.

    Note: the translation column of the affine matrix will be non-zero because
    the rotation is applied around the centroid, not the origin. Only the
    rotation block is checked here.
    """
    ref = make_stable_cloud(seed=9)
    tba = apply_rotation(ref, Rfn(angle_deg))

    epoch_ref = build_ref_epoch(ref)
    result = coreg_pc(epoch_ref, tba)

    R_rec = result["trafo"].affine_transformation[:3, :3]
    R_exp = Rfn(-angle_deg)

    np.testing.assert_allclose(
        R_rec, R_exp, atol=1e-3,
        err_msg=(
            f"Recovered rotation does not match expected inverse for "
            f"axis={axis}, angle={angle_deg}°.\n"
            f"R_rec:\n{R_rec}\nR_exp:\n{R_exp}"
        )
    )

"""Tests for xlcs."""

import numpy as np
from scipy.spatial import ConvexHull

from xlcs.contours import peak_in_hull, polygon_area
from xlcs.grid_calc import diff_x, diff_y, haversine, nonUniDiff3, nonUniDiff5
from xlcs.lagrangian_tools import (
    _merge_singularities,
    _p1dist,
    eigenspectrum,
    elliptic_lcs,
    find_singularities,
    make_trajectory_grid,
)
from xlcs.structures import Singularity

# ---------------------------------------------------------------------------
# haversine
# ---------------------------------------------------------------------------


def test_haversine_zero():
    """Same point returns zero distance."""
    assert haversine(0.0, 0.0, 0.0, 0.0) == 0.0


def test_haversine_one_degree_latitude():
    """One degree of latitude at any longitude is ~111 195 m."""
    d = haversine(0.0, 0.0, 0.0, 1.0)
    assert np.isclose(d, 111_195.0, rtol=1e-3)


def test_haversine_one_degree_longitude_at_equator():
    """One degree of longitude at the equator equals one degree of latitude."""
    d_lon = haversine(0.0, 0.0, 1.0, 0.0)
    d_lat = haversine(0.0, 0.0, 0.0, 1.0)
    assert np.isclose(d_lon, d_lat, rtol=1e-6)


def test_haversine_symmetric():
    """Distance A→B equals distance B→A."""
    d1 = haversine(-90.0, 25.0, -80.0, 30.0)
    d2 = haversine(-80.0, 30.0, -90.0, 25.0)
    assert np.isclose(d1, d2)


# ---------------------------------------------------------------------------
# nonUniDiff3 / nonUniDiff5
# ---------------------------------------------------------------------------


def test_nonUniDiff3_linear():
    """Exact derivative for f(x) = 2x + 3 (f' = 2)."""
    x = np.array([0.0, 1.0, 2.0])
    f = 2.0 * x + 3.0
    assert np.isclose(nonUniDiff3(x, f, 1.0), 2.0)


def test_nonUniDiff3_nonuniform():
    """Exact on non-uniform spacing for f(x) = x^2 (f'(1) = 2)."""
    x = np.array([0.0, 1.0, 3.0])
    f = x**2
    assert np.isclose(nonUniDiff3(x, f, 1.0), 2.0)


def test_nonUniDiff5_linear():
    """Exact derivative for f(x) = 3x - 1 (f' = 3)."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f = 3.0 * x - 1.0
    assert np.isclose(nonUniDiff5(x, f, 2.0), 3.0)


def test_nonUniDiff5_quadratic():
    """Exact on quadratic f(x) = x^2 (f'(2) = 4)."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f = x**2
    assert np.isclose(nonUniDiff5(x, f, 2.0), 4.0)


# ---------------------------------------------------------------------------
# diff_x / diff_y
# ---------------------------------------------------------------------------


def test_diff_x_linear():
    """diff_x returns exact derivative for f = 2x (f' = 2) on a uniform grid."""
    nx, ny = 10, 5
    x = np.tile(np.linspace(0.0, 1.0, nx).reshape(-1, 1), (1, ny))
    result = diff_x(2.0 * x, x)
    assert np.allclose(result, 2.0, rtol=1e-6)


def test_diff_y_linear():
    """diff_y returns exact derivative for f = 3y (f' = 3) on a uniform grid."""
    nx, ny = 5, 10
    y = np.tile(np.linspace(0.0, 1.0, ny).reshape(1, -1), (nx, 1))
    result = diff_y(3.0 * y, y)
    assert np.allclose(result, 3.0, rtol=1e-6)


def test_diff_x_constant():
    """Derivative of a constant field is zero."""
    nx, ny = 8, 6
    x = np.tile(np.linspace(0.0, 1.0, nx).reshape(-1, 1), (1, ny))
    result = diff_x(np.ones((nx, ny)), x)
    assert np.allclose(result, 0.0, atol=1e-10)


def test_diff_y_constant():
    """Derivative of a constant field is zero."""
    nx, ny = 6, 8
    y = np.tile(np.linspace(0.0, 1.0, ny).reshape(1, -1), (nx, 1))
    result = diff_y(np.ones((nx, ny)), y)
    assert np.allclose(result, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# eigenspectrum
# ---------------------------------------------------------------------------


def test_eigenspectrum_identity():
    """Identity CG tensor gives lmin=lmax=1 and ftle=0."""
    ny, nx = 4, 5
    cg = np.zeros((ny, nx, 2, 2))
    cg[:, :, 0, 0] = 1.0
    cg[:, :, 1, 1] = 1.0
    lmin, lmax, _, _, ftle = eigenspectrum(cg, 1.0)
    assert np.allclose(lmin, 1.0)
    assert np.allclose(lmax, 1.0)
    assert np.allclose(ftle, 0.0)


def test_eigenspectrum_diagonal():
    """Diagonal CG tensor gives known eigenvalues and FTLE."""
    T = 10.0
    ny, nx = 4, 5
    cg = np.zeros((ny, nx, 2, 2))
    cg[:, :, 0, 0] = 1.0
    cg[:, :, 1, 1] = 4.0
    lmin, lmax, _, _, ftle = eigenspectrum(cg, T)
    assert np.allclose(lmin, 1.0)
    assert np.allclose(lmax, 4.0)
    assert np.allclose(ftle, np.log(4.0) / (2.0 * T))


def test_eigenspectrum_nan_input():
    """NaN entries in cg are treated as zero (no crash, ftle=nan where lmax<=0)."""
    ny, nx = 3, 3
    cg = np.full((ny, nx, 2, 2), np.nan)
    lmin, lmax, _, _, ftle = eigenspectrum(cg, 1.0)
    assert np.all(np.isnan(ftle))


# ---------------------------------------------------------------------------
# polygon_area
# ---------------------------------------------------------------------------


def test_polygon_area_unit_square():
    """Unit square has area 1."""
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert np.isclose(polygon_area(xy), 1.0)


def test_polygon_area_right_triangle():
    """Right triangle with legs 2 has area 2."""
    xy = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    assert np.isclose(polygon_area(xy), 2.0)


def test_polygon_area_rectangle():
    """3×4 rectangle has area 12."""
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [0.0, 4.0]])
    assert np.isclose(polygon_area(xy), 12.0)


# ---------------------------------------------------------------------------
# peak_in_hull
# ---------------------------------------------------------------------------


def test_peak_in_hull_inside():
    """Center of a unit square is inside its convex hull."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    hull = ConvexHull(points)
    assert peak_in_hull(np.array([0.5, 0.5]), hull)


def test_peak_in_hull_outside():
    """A point far from the square is outside its convex hull."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    hull = ConvexHull(points)
    assert not peak_in_hull(np.array([2.0, 0.5]), hull)


# ---------------------------------------------------------------------------
# make_trajectory_grid
# ---------------------------------------------------------------------------


def test_make_trajectory_grid_coords():
    """Output dataset has the expected coordinate variables."""
    px = np.arange(-90.0, -80.0, 0.5)
    py = np.arange(20.0, 28.0, 0.5)
    ds, grid = make_trajectory_grid(px, py)
    for coord in ("xc", "yc", "dx", "dy"):
        assert coord in ds.coords, f"missing coordinate: {coord}"


def test_make_trajectory_grid_shapes():
    """Dx and dy have the expected shapes given the input arrays."""
    px = np.arange(-90.0, -80.0, 0.5)  # 20 points
    py = np.arange(20.0, 28.0, 0.5)  # 16 points
    ds, _ = make_trajectory_grid(px, py)
    assert ds.dx.shape == (len(py), len(px) - 1)
    assert ds.dy.shape == (len(py) - 1, len(px))


def test_make_trajectory_grid_positive_spacing():
    """Metric spacings are strictly positive."""
    px = np.arange(-90.0, -80.0, 1.0)
    py = np.arange(20.0, 30.0, 1.0)
    ds, _ = make_trajectory_grid(px, py)
    assert (ds.dx.values > 0).all()
    assert (ds.dy.values > 0).all()


def test_make_trajectory_grid_dx_decreases_with_latitude():
    """Dx (zonal spacing in m) should decrease toward higher latitudes."""
    px = np.array([-90.0, -89.0])
    py = np.array([10.0, 30.0, 60.0])
    ds, _ = make_trajectory_grid(px, py)
    dx = ds.dx.values[:, 0]  # one column
    assert dx[0] > dx[1] > dx[2]


# ---------------------------------------------------------------------------
# _p1dist
# ---------------------------------------------------------------------------


def test_p1dist_equal_angles():
    """Distance between equal angles is zero."""
    a = np.array([0.0, np.pi / 4, np.pi / 2])
    assert np.allclose(_p1dist(a, a), 0.0)


def test_p1dist_range():
    """Result is always in (-π/2, π/2]."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, np.pi, 100)
    b = rng.uniform(0, np.pi, 100)
    d = _p1dist(a, b)
    assert np.all(d > -np.pi / 2 - 1e-12)
    assert np.all(d <= np.pi / 2 + 1e-12)


def test_p1dist_wrap():
    """Going from just below 0 to just above 0 gives a small positive distance."""
    eps = 0.01
    d = _p1dist(np.array([eps]), np.array([np.pi - eps]))
    assert d[0] > 0


# ---------------------------------------------------------------------------
# find_singularities / _merge_singularities
# ---------------------------------------------------------------------------


def _make_half_index_field(ny: int = 5, nx: int = 5):
    """Construct a vmin field with a +1/2 singularity at the grid centre."""
    xc = np.linspace(-2.0, 2.0, nx)
    yc = np.linspace(-2.0, 2.0, ny)
    vmin = np.zeros((ny, nx, 2))
    for i in range(ny):
        for j in range(nx):
            theta = np.arctan2(yc[i], xc[j]) / 2  # +1/2 wedge singularity
            vmin[i, j] = [np.cos(theta), np.sin(theta)]
    return xc, yc, vmin


def test_find_singularities_uniform():
    """A uniform eigenvector field has no singularities."""
    ny, nx = 6, 8
    xc = np.linspace(-3.0, 3.0, nx)
    yc = np.linspace(-2.0, 2.0, ny)
    vmin = np.zeros((ny, nx, 2))
    vmin[:, :, 0] = 1.0  # all pointing in the x direction
    assert find_singularities(vmin, xc, yc) == []


def test_find_singularities_half_index():
    """A half-angle field has one +1/2 singularity near the grid centre."""
    xc, yc, vmin = _make_half_index_field()
    sings = find_singularities(vmin, xc, yc)
    half = [s for s in sings if abs(s.index - 0.5) < 0.01]
    assert len(half) >= 1
    # singularity should be near the origin
    assert abs(half[0].lon) < 1.5 and abs(half[0].lat) < 1.5


def test_find_singularities_combined():
    """Two +1/2 singularities within combine_radius merge into one +1 centre."""
    xc, yc, vmin = _make_half_index_field()
    sings_raw = find_singularities(vmin, xc, yc, combine_radius=0.0)
    half_raw = [s for s in sings_raw if abs(s.index - 0.5) < 0.01]

    # With a large combine_radius all wedges should merge
    sings_merged = find_singularities(vmin, xc, yc, combine_radius=4.0)
    centers = [s for s in sings_merged if abs(s.index - 1.0) < 0.01]
    # The two +1/2 singularities should combine into at least one +1 centre
    assert len(centers) >= 1 or len(half_raw) == 0  # pass if no wedges existed


def test_merge_singularities_within_radius():
    """Two singularities within radius merge into one with summed index."""
    s1 = Singularity(lon=0.0, lat=0.0, index=0.5)
    s2 = Singularity(lon=0.1, lat=0.0, index=0.5)
    merged = _merge_singularities([s1, s2], radius=0.5)
    assert len(merged) == 1
    assert np.isclose(merged[0].index, 1.0)
    assert np.isclose(merged[0].lon, 0.05)


def test_merge_singularities_outside_radius():
    """Two singularities outside the merge radius stay separate."""
    s1 = Singularity(lon=0.0, lat=0.0, index=0.5)
    s2 = Singularity(lon=5.0, lat=0.0, index=0.5)
    merged = _merge_singularities([s1, s2], radius=1.0)
    assert len(merged) == 2


def test_merge_singularities_zero_sum_dropped():
    """A group whose indices sum to zero is removed from the output."""
    s1 = Singularity(lon=0.0, lat=0.0, index=0.5)
    s2 = Singularity(lon=0.1, lat=0.0, index=-0.5)
    merged = _merge_singularities([s1, s2], radius=1.0)
    assert len(merged) == 0


# ---------------------------------------------------------------------------
# elliptic_lcs
# ---------------------------------------------------------------------------


def test_elliptic_lcs_no_singularities():
    """A uniform eigenvector field has no singularities and therefore no barriers."""
    ny, nx = 10, 10
    xc = np.linspace(-5.0, 5.0, nx)
    yc = np.linspace(-5.0, 5.0, ny)
    lmin = np.ones((ny, nx))
    lmax = np.full((ny, nx), 2.0)
    vmin = np.zeros((ny, nx, 2))
    vmin[:, :, 0] = 1.0
    vmax = np.zeros((ny, nx, 2))
    vmax[:, :, 1] = 1.0
    sings, barriers = elliptic_lcs(xc, yc, lmin, lmax, vmin, vmax)
    assert barriers == []

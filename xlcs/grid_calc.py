import numba as nb
import numpy as np
from typing import Tuple

@nb.njit
def haversine(
    lon1: np.array, lat1: np.array, lon2: np.array, lat2: np.array
) -> np.array:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.
    """
    lon1, lat1 = np.radians(lon1), np.radians(lat1)
    lon2, lat2 = np.radians(lon2), np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    # distance
    earthRadius = 6.371e6  # m
    d = 2 * np.arcsin(np.sqrt(a)) * earthRadius  # km
    return d


@nb.njit
def grid_to_m(
    lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform a rectilinear grids defined by two 1D vector for longitude and latitude
    to meters

    Args:
        lon: meridional coordinates of the grid in degrees
        lat: zonal coordinates of the grid in degrees
        lon0: reference longitude
        lat0: reference latitude

    Returns:
        x: meridional coordinates of the grid in meters
        y: zonal coordinates of the grid in meters
    """
    nx, ny = lon.shape
    x = np.zeros((nx, ny))
    y = np.zeros((nx, ny))

    for i in range(0, len(x)):
        for j in range(0, len(x[1])):
            x[i, j] = haversine(lon[i, j], lat[i, j], lon0, lat[i, j])
            y[i, j] = haversine(lon[i, j], lat[i, j], lon[i, j], lat0)
    return x, y


@nb.njit
def nonUniDiff5(x, f, x0):
    """
    Implementation of the first derivative 5 points formula from:
    Derivative formulae and errors for non-uniformly spaced points
    by M.K. Bowen and Ronald Smith doi:10.1098/rspa.2004.1430

    As the name indicate, the five points do not have to be on a
    regular grid.

    Args:
        x: sequence of 5 points
        f: f(x) value of a function at x
        x0: point where to evaluate the derivates
    Return:
        f'(x0): first derivative evaluate at x0

    """
    a1 = x[0] - x0
    a2 = x[1] - x0
    a3 = x[2] - x0
    a4 = x[3] - x0
    a5 = x[4] - x0

    value = (
        -((a2 * a3 * a4 + a2 * a3 * a5 + a2 * a4 * a5 + a3 * a4 * a5) * f[0])
        / ((a1 - a2) * (a1 - a3) * (a1 - a4) * (a1 - a5))
        - ((a1 * a3 * a4 + a1 * a3 * a5 + a1 * a4 * a5 + a3 * a4 * a5) * f[1])
        / ((a2 - a1) * (a2 - a3) * (a2 - a4) * (a2 - a5))
        - ((a1 * a2 * a4 + a1 * a2 * a5 + a1 * a4 * a5 + a2 * a4 * a5) * f[2])
        / ((a3 - a1) * (a3 - a2) * (a3 - a4) * (a3 - a5))
        - ((a1 * a2 * a3 + a1 * a2 * a5 + a1 * a3 * a5 + a2 * a3 * a5) * f[3])
        / ((a4 - a1) * (a4 - a2) * (a4 - a3) * (a4 - a5))
        - ((a1 * a2 * a3 + a1 * a2 * a4 + a1 * a3 * a4 + a2 * a3 * a4) * f[4])
        / ((a5 - a1) * (a5 - a2) * (a5 - a3) * (a5 - a4))
    )
    return value


@nb.njit
def nonUniDiff3(x, f, x0):
    """
    Implementation of the first derivative 3 points formula from:
    Derivative formulae and errors for non-uniformly spaced points
    by M.K. Bowen and Ronald Smith doi:10.1098/rspa.2004.1430

    As the name indicate, the five points do not have to be on a
    regular grid.

    Args:
        x: sequence of 5 points
        f: f(x) value of a function at x
        x0: point where to evaluate the derivates
    Return:
        f'(x0): first derivative evaluate at x0

    """
    a1 = x[0] - x0
    a2 = x[1] - x0
    a3 = x[2] - x0

    value = (
        -(a2 + a3) * f[0] / ((a1 - a2) * (a1 - a3))
        - (a1 + a3) * f[1] / ((a2 - a1) * (a2 - a3))
        - (a1 + a2) * f[2] / ((a3 - a1) * (a3 - a2))
    )
    return value


@nb.njit
def diff_x(var: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    First derivatives are calculated using 3 points using finite difference schemes that can be used on non-uniform grid
    Args:
        var: variable to derivate [nx,ny]
        x: zonal grid [nx,ny]
    Return:
        dx: first derivative of variable [nx, ny]
    """
    nx, ny = len(x), len(x[0])
    dx = np.zeros((nx, ny))

    for j in range(0, ny):
        # Forward diff
        p = [x[0, j], x[1, j], x[2, j]]
        f = [var[0, j], var[1, j], var[2, j]]
        dx[0, j] = nonUniDiff3(p, f, x[0, j])

        # Central diff
        for i in range(1, nx - 1):
            p = [x[i - 1, j], x[i, j], x[i + 1, j]]
            f = [var[i - 1, j], var[i, j], var[i + 1, j]]
            dx[i, j] = nonUniDiff3(p, f, x[i, j])

        # Backward diff
        p = [x[nx - 3, j], x[nx - 2, j], x[nx - 1, j]]
        f = [
            var[nx - 3, j],
            var[nx - 2, j],
            var[nx - 1, j],
        ]
        dx[nx - 1, j] = nonUniDiff3(p, f, x[nx - 1, j])
    return dx


@nb.njit
def diff_y(var: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    First derivatives are calculated using 5 points using finite difference schemes that can be used on non-uniform grid
    Args:
        var: variable to derivate [nx,ny]
        y: zonal grid [nx,ny]
    Return:
        dy: first derivative of variable [nx, ny]
    """
    nx, ny = len(y), len(y[0])
    dy = np.zeros((nx, ny))

    for i in range(0, nx):
        # Forward diff
        p = [y[i, 0], y[i, 1], y[i, 2]]
        f = [var[i, 0], var[i, 1], var[i, 2]]
        dy[i, 0] = nonUniDiff3(p, f, y[i, 0])

        # Central diff
        for j in range(1, ny - 1):
            p = [y[i, j - 1], y[i, j], y[i, j + 1]]
            f = [var[i, j - 2], var[i, j - 1], var[i, j], var[i, j + 1], var[i, j + 2]]
            dy[i, j] = nonUniDiff3(p, f, y[i, j])

        # Backward diff
        p = [y[i, ny - 3], y[i, ny - 2], y[i, ny - 1]]
        f = [
            var[i, ny - 3],
            var[i, ny - 2],
            var[i, ny - 1],
        ]
        dy[i, ny - 1] = nonUniDiff3(p, f, y[i, ny - 1])
    return dy


@nb.njit
def diff_x_5p(var: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    First derivatives are calculated using 5 points using finite difference schemes that can be used on non-uniform grid
    Args:
        var: variable to derivate [nx,ny]
        x: zonal grid [nx,ny]
    Return:
        dx: first derivative of variable [nx, ny]
    """
    nx, ny = len(x), len(x[0])
    dx = np.zeros((nx, ny))

    for j in range(0, ny):
        # Forward diff
        p = [x[0, j], x[1, j], x[2, j], x[3, j], x[4, j]]
        f = [var[0, j], var[1, j], var[2, j], var[3, j], var[4, j]]
        dx[0, j] = nonUniDiff5(p, f, x[0, j])

        p = [x[0, j], x[1, j], x[2, j], x[3, j], x[4, j]]
        f = [var[0, j], var[1, j], var[2, j], var[3, j], var[4, j]]
        dx[1, j] = nonUniDiff5(p, f, x[1, j])

        # Central diff
        for i in range(2, nx - 2):
            p = [x[i - 2, j], x[i - 1, j], x[i, j], x[i + 1, j], x[i + 2, j]]
            f = [var[i - 2, j], var[i - 1, j], var[i, j], var[i + 1, j], var[i + 2, j]]
            dx[i, j] = nonUniDiff5(p, f, x[i, j])

        # Backward diff
        p = [x[nx - 5, j], x[nx - 4, j], x[nx - 3, j], x[nx - 2, j], x[nx - 1, j]]
        f = [
            var[nx - 5, j],
            var[nx - 4, j],
            var[nx - 3, j],
            var[nx - 2, j],
            var[nx - 1, j],
        ]
        dx[nx - 2, j] = nonUniDiff5(p, f, x[nx - 2, j])

        p = [x[nx - 5, j], x[nx - 4, j], x[nx - 3, j], x[nx - 2, j], x[nx - 1, j]]
        f = [
            var[nx - 5, j],
            var[nx - 4, j],
            var[nx - 3, j],
            var[nx - 2, j],
            var[nx - 1, j],
        ]
        dx[nx - 1, j] = nonUniDiff5(p, f, x[nx - 1, j])
    return dx


@nb.njit
def diff_y_5p(var: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    First derivatives are calculated using 5 points using finite difference schemes that can be used on non-uniform grid
    Args:
        var: variable to derivate [nx,ny]
        y: zonal grid [nx,ny]
    Return:
        dy: first derivative of variable [nx, ny]
    """
    nx, ny = len(y), len(y[0])
    dy = np.zeros((nx, ny))

    for i in range(0, nx):
        # Forward diff
        p = [y[i, 0], y[i, 1], y[i, 2], y[i, 3], y[i, 4]]
        f = [var[i, 0], var[i, 1], var[i, 2], var[i, 3], var[i, 4]]
        dy[i, 0] = nonUniDiff5(p, f, y[i, 0])

        p = [y[i, 0], y[i, 1], y[i, 2], y[i, 3], y[i, 4]]
        f = [var[i, 0], var[i, 1], var[i, 2], var[i, 3], var[i, 4]]
        dy[i, 1] = nonUniDiff5(p, f, y[i, 1])

        # Central diff
        for j in range(2, ny - 2):
            p = [y[i, j - 2], y[i, j - 1], y[i, j], y[i, j + 1], y[i, j + 2]]
            f = [var[i, j - 2], var[i, j - 1], var[i, j], var[i, j + 1], var[i, j + 2]]
            dy[i, j] = nonUniDiff5(p, f, y[i, j])

        # Backward diff
        p = [y[i, ny - 5], y[i, ny - 4], y[i, ny - 3], y[i, ny - 2], y[i, ny - 1]]
        f = [
            var[i, ny - 5],
            var[i, ny - 4],
            var[i, ny - 3],
            var[i, ny - 2],
            var[i, ny - 1],
        ]
        dy[i, ny - 2] = nonUniDiff5(p, f, y[i, ny - 2])

        p = [y[i, ny - 5], y[i, ny - 4], y[i, ny - 3], y[i, ny - 2], y[i, ny - 1]]
        f = [
            var[i, ny - 5],
            var[i, ny - 4],
            var[i, ny - 3],
            var[i, ny - 2],
            var[i, ny - 1],
        ]
        dy[i, ny - 1] = nonUniDiff5(p, f, y[i, ny - 1])
    return dy


@nb.njit
def vorticity(x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Calculate vorticity for all velocity fields
    """
    x_m, y_m = grid_to_m(x, y, x[0, 0], y[0, 0])
    vort = np.zeros_like(u)
    for i in range(0, len(vort)):
        vort[i] = diff_x(v[i], x_m) - diff_y(u[i], y_m)
    return vort

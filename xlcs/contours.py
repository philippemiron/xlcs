"""Contour extraction for identifying Lagrangian vortex boundaries from LAVD fields."""

import numba as nb
import numpy as np
import xarray as xr
from scipy.spatial import ConvexHull
from skimage.feature import peak_local_max
from skimage.measure import find_contours


@nb.njit
def polygon_area(xy: np.array):
    """Return the area of a polygon defined by the points xy.

    Args:
        xy: polygon vertices [N, 2]

    Returns:
        area: scalar polygon area

    """
    x, y = np.ascontiguousarray(xy[:, 0]), np.ascontiguousarray(xy[:, 1])
    correction = x[-1] * y[0] - y[-1] * x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5 * np.abs(main_area + correction)


def peak_in_hull(p, hull, tol=1e-12):
    """Return True if point p lies inside the convex hull."""
    hq = hull.equations
    return np.all(hq[:, :-1] @ p + hq[:, -1] <= tol)


def peaks_in_hull(p, hull, tol=1e-12):
    """Return a boolean mask indicating which points in p lie inside the convex hull."""
    hq = hull.equations
    return np.all(hq[:, :-1] @ p.T + np.repeat(hq[:, -1][None, :], len(p), axis=0).T <= tol, 0)


def _check_contour_level(
    sub_lavd: np.ndarray,
    level: float,
    sub_pxy: np.ndarray,
    defTol: float,
) -> np.ndarray | None:
    """Return the first contour at level that encloses sub_pxy and passes convexity, or None."""
    for c in find_contours(sub_lavd, level):
        try:
            hull = ConvexHull(c, qhull_options="QJ")
            if peak_in_hull(sub_pxy, hull):
                area = polygon_area(c)
                if area > 0 and abs(area - hull.volume) / area * 100 < defTol:
                    return c
        except Exception:
            pass
    return None


def extract_contours(
    ds: xr.Dataset,
    defTol: float = 0.075,
    max_radius: float = 3.0,
    number_levels: int = 50,
) -> tuple[np.array, np.array]:
    """Extract closed convex contours around LAVD peaks to identify coherent vortices.

    Args:
        ds: Dataset with lavd [yc, xc] and xc/yc coordinate arrays
        defTol: convexity tolerance; closer to 0 means more circular contours
        max_radius: search radius around each peak in the same units as xc/yc
        number_levels: number of contour levels tested between 0 and the peak LAVD value

    Returns:
        peaks_xy: peak locations as grid indices [N, 2]
        contours: boundary contour in degrees for each peak (None if no valid contour found)

    """
    peaks_xy = peak_local_max(ds["lavd"].values, min_distance=20)  # indices
    peaks_value = ds["lavd"].values[peaks_xy[:, 0], peaks_xy[:, 1]]
    contours = np.empty_like(peaks_value, dtype="object")

    dx, dy = np.mean(np.diff(ds["xc"].values)), np.mean(np.diff(ds["yc"].values))
    xc_arr = ds["xc"].values
    yc_arr = ds["yc"].values
    xc_idx = np.arange(len(xc_arr), dtype=float)
    yc_idx = np.arange(len(yc_arr), dtype=float)

    n = 0
    for j in range(len(peaks_xy)):
        print(f"{j + 1}/{len(peaks_xy)} (Found {n} {'eddies' if n > 1 else 'eddy'})", end="\r")

        pxy = peaks_xy[j]
        i0 = max(0, int(pxy[0] - np.ceil(max_radius / dy)))
        i1 = min(ds.sizes["yc"] - 1, int(pxy[0] + np.ceil(max_radius / dy)))
        j0 = max(0, int(pxy[1] - np.ceil(max_radius / dx)))
        j1 = min(ds.sizes["xc"] - 1, int(pxy[1] + np.ceil(max_radius / dx)))

        sub_lavd = ds["lavd"].values[i0:i1, j0:j1]
        sub_pxy = pxy - np.array([i0, j0])

        c_levels = np.linspace(np.min(sub_lavd), peaks_value[j], number_levels)

        # Binary search for the outermost (lowest level) valid contour.
        # Convexity improves monotonically as level rises toward the peak, so the
        # transition from invalid → valid happens once; binary search finds it in
        # O(log number_levels) find_contours calls instead of O(number_levels).
        lo, hi, best = 0, len(c_levels) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            result = _check_contour_level(sub_lavd, c_levels[mid], sub_pxy, defTol)
            if result is not None:
                best = result.copy()
                hi = mid - 1  # try lower level (larger/outermost contour)
            else:
                lo = mid + 1  # need higher level for a valid contour

        if best is not None:
            best[:, 0] += i0
            best[:, 1] += j0
            contours[j] = np.column_stack((
                np.interp(best[:, 0], yc_idx, yc_arr),
                np.interp(best[:, 1], xc_idx, xc_arr),
            ))
            n += 1

    return peaks_xy, contours

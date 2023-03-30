import numba as nb
import numpy as np
from typing import Tuple
from scipy.interpolate import interp1d
from skimage.feature import peak_local_max
from skimage.measure import find_contours
from scipy.spatial import ConvexHull, convex_hull_plot_2d


@nb.njit
def polygon_area(xy: np.array):
    """
    Area of a polygon defined by the points xy

    Args:
        xy [N,2]: coordinates defining the polygon

    Returns:
        [floats]: area
    """
    x, y = xy[:, 0], xy[:, 1]
    correction = x[-1] * y[0] - y[-1] * x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5 * np.abs(main_area + correction)


def peak_in_hull(p, hull, tol=1e-12):
    """
    Test in a point p are in the convex hull (scipy.spatial.ConvexHull)
    """
    hq = hull.equations
    return np.all(hq[:, :-1] @ p + hq[:, -1] <= tol)


# not used now but I want to rethink the algorithm
# and reorder some calculations for speed up
def peaks_in_hull(p, hull, tol=1e-12):
    """
    Test in points p are in the convex hull (scipy.spatial.ConvexHull)
    """
    hq = hull.equations
    return np.all(
        hq[:, :-1] @ p.T + np.repeat(hq[:, -1][None, :], len(p), axis=0).T <= tol, 0
    )


def extract_contours(
    lon: np.array,
    lat: np.array,
    lavd: np.array,
    defTol: float = 0.075,
    max_radius: float = 3.0,
    number_levels: int = 50,
) -> Tuple[np.array, np.array]:
    """

    Args:
        lon: meridional coordinates of the grid in degrees [nx, ny]
        lat: zonal coordinates of the grid in degrees [nx, ny]
        lavd: Lagrangian averaged vorticity deviation [nx, ny]
        defTol: control the deficiency of the loop closer to 0 means perfectly convex (~circular)
        max_radius: Max radius of eddies control area where contours are extracted
                    Note: assumed in the same units has the longitude and latitude.
        number_levels: contour levels between [0, peak_lavd_value]

    Returns:
        peaks_xy: centers of extracted vortices [N, 2]
        contours: contours of extracted vortices [N, 2]

    """

    # peaks and data structures
    peaks_xy = peak_local_max(lavd, min_distance=20)  # indices
    peaks_value = lavd[peaks_xy[:, 0], peaks_xy[:, 1]]
    contours = np.empty_like(peaks_value, dtype="object")

    # coordinates are converted to indices by `find_contour`
    # this is used to bring back to degrees
    dx, dy = np.mean(np.diff(lon)), np.mean(np.diff(lat))
    flon = interp1d(np.arange(0, len(lon)), lon)
    flat = interp1d(np.arange(0, len(lat)), lat)

    n = 0  # only for counting
    for j in range(0, len(peaks_xy)):
        print(
            f"{j+1}/{len(peaks_xy)} (Found {n} {'eddies' if n>1 else 'eddy'})", end="\r"
        )
        
        # current peak and subdomain indices
        pxy = peaks_xy[j]
        i0, i1 = pxy[0] - np.ceil(max_radius/dx), pxy[0] + np.ceil(max_radius/dx)
        j0, j1 = pxy[1] - np.ceil(max_radius/dy), pxy[1] + np.ceil(max_radius/dy)
        
        # make sure not over the domain
        i0, i1 = max(0, int(i0)), min(len(lon)-1, int(i1))
        j0, j1 = max(0, int(j0)), min(len(lat)-1, int(j1))
        
        # lavd around the center
        sub_lavd = lavd[slice(i0,i1), slice(j0,j1)]
        sub_pxy = pxy - np.array([i0, j0])
        
        c_levels = np.linspace(np.min(sub_lavd), peaks_value[j], number_levels)

        # loop from the lowest value to the peak value
        #  - If we find a contour respecting the criteria 
        #    we stop because we want to keep the largest
        for c_level in c_levels:
            if contours[j] is None:              
                # this also returns (i,j) indices
                cs = find_contours(sub_lavd, c_level)
                for c in cs:
                    try:  # prevent error when ConvexHull fails on weirdly shaped contour
                        hull = ConvexHull(c)
                        if peak_in_hull(sub_pxy, hull):
                            areaPoly = polygon_area(c)
                            if (
                                abs(areaPoly - hull.volume) / areaPoly * 100 < defTol
                            ):  # in 2D hull.volume returns the area (!)
                                
                                # translate back to the full domain
                                c[:,0] += i0
                                c[:,1] += j0
                                
                                contours[j] = np.column_stack((flon(c[:, 0]), flat(c[:, 1])))
                                n += 1
                                break
                    except:
                        pass

    return peaks_xy, contours
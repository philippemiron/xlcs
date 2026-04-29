"""Lagrangian tools: flowmaps, Cauchy-Green tensors, FTLE, LAVD, and elliptic LCS."""

import os
import tempfile
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import gsw
import numba as nb
import numpy as np
import xarray as xr
import xgcm
from numpy import ndarray
from parcels import AdvectionRK4, FieldSet, JITParticle, ParticleSet
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from xarray import Dataset

from xlcs.kernels import OutOfBound, SampleVorticity, _eta_poincare_kernel
from xlcs.particle import EtaParticle, LAVDParticle
from xlcs.structures import Singularity


def make_trajectory_grid(
    lon: np.ndarray,
    lat: np.ndarray,
) -> tuple[xr.Dataset, xgcm.Grid]:
    """Create a particle release grid with metric spacings from 1D coordinate arrays.

    Args:
        lon: 1D array of longitude values in degrees (xc centers)
        lat: 1D array of latitude values in degrees (yc centers)

    Returns:
        ds: Dataset with xc/yc center coordinates and xg/yg inner-face coordinates,
            plus dx/dy metric spacings in meters
        grid: corresponding xgcm.Grid object

    """
    px, py = np.asarray(lon), np.asarray(lat)
    px_grid, py_grid = np.meshgrid(px, py)

    ds = xr.Dataset().assign_coords(
        {
            "dx": xr.DataArray(
                gsw.distance(px_grid, py_grid, axis=1),
                dims=["yc", "xg"],
                coords={"yc": py, "xg": 0.5 * (px[1:] + px[:-1])},
            ),
            "dy": xr.DataArray(
                gsw.distance(px_grid, py_grid, axis=0),
                dims=["yg", "xc"],
                coords={"yg": 0.5 * (py[1:] + py[:-1]), "xc": px},
            ),
        }
    )

    coords = {
        "X": {"center": "xc", "inner": "xg"},
        "Y": {"center": "yc", "inner": "yg"},
    }
    grid = xgcm.Grid(ds, periodic=[], coords=coords, autoparse_metadata=False)
    return ds, grid


def flowmap(
    ds: Dataset,
    filename: str,
    fs: FieldSet,
    t0: datetime,
    T: timedelta,
    dt: timedelta,
    lavd: bool | None = False,
) -> tuple[Any, ndarray]:
    """Calculate the flowmap from T-long trajectories initialized at ds.xc, ds.yc.

    When lavd=True the FieldSet fs must contain a vorticity field.

    Args:
        ds: Dataset with xc/yc coordinates for particle release positions
        filename: output zarr path for trajectory data
        fs: parcels FieldSet (must include vorticity field when lavd=True)
        t0: initial time
        T: integration duration
        dt: integration timestep (negative for backward integration)
        lavd: if True, sample vorticity along the trajectory

    Returns:
        pset: OceanParcels ParticleSet after integration
        origin_id: ID of the first particle, needed to index the output zarr

    """
    mx, my = np.meshgrid(ds.xc, ds.yc)
    if isinstance(t0, datetime):
        mt = np.full_like(mx, t0, dtype="datetime64[ms]")
    else:
        mt = np.full_like(mx, t0)

    pset = ParticleSet.from_list(
        fieldset=fs,
        pclass=LAVDParticle if lavd else JITParticle,
        lon=mx.flatten(),
        lat=my.flatten(),
        time=mt.flatten(),
    )

    origin_id = np.copy(pset.id[0])

    output_file = pset.ParticleFile(
        name=filename, outputdt=timedelta(seconds=int(abs(dt.total_seconds())))
    )

    kernels = pset.Kernel(OutOfBound) + pset.Kernel(AdvectionRK4)
    if lavd:
        kernels += pset.Kernel(SampleVorticity)

    pset.execute(
        kernels,
        runtime=T,
        dt=dt,
        output_file=output_file,
        verbose_progress=True,
    )

    pid = pset.id - origin_id
    ds["phi_x"] = (("yc", "xc"), np.zeros_like(mx))
    ds["phi_y"] = (("yc", "xc"), np.zeros_like(mx))
    ds["phi_x"].values[np.unravel_index(pid, mx.shape)] = pset.lon
    ds["phi_y"].values[np.unravel_index(pid, mx.shape)] = pset.lat
    ds["phi_x"].values[np.where(ds["phi_x"] == 0)] = np.nan
    ds["phi_y"].values[np.where(ds["phi_y"] == 0)] = np.nan

    return pset, origin_id


def cauchygreen(ds: xr.Dataset, grid: xgcm.Grid, keep_intermediates: bool = False):
    """Compute the Cauchy-Green deformation tensor from the flowmap stored in ds.

    Adds the four Jacobian partial derivatives and cg to ds.

    Args:
        ds: Dataset with phi_x, phi_y flowmap variables and dx/dy grid spacings
        grid: xgcm.Grid corresponding to ds
        keep_intermediates: if True, retain partial-derivative variables in ds

    """
    R = 6.3781e6  # WGS84 mean Earth radius in meters

    # Signed metric displacements of final positions along x direction → (yc, xg)
    # dFx = zonal component; dFy = meridional component
    lat_avg_x = 0.5 * (ds["phi_y"].values[:, 1:] + ds["phi_y"].values[:, :-1])
    dfx_dx = (
        R
        * np.cos(np.deg2rad(lat_avg_x))
        * np.deg2rad(np.diff(ds["phi_x"].values, axis=1))
        / ds["dx"].values
    )
    dfy_dx = R * np.deg2rad(np.diff(ds["phi_y"].values, axis=1)) / ds["dx"].values

    # Along y direction → (yg, xc)
    lat_avg_y = 0.5 * (ds["phi_y"].values[1:, :] + ds["phi_y"].values[:-1, :])
    dfx_dy = (
        R
        * np.cos(np.deg2rad(lat_avg_y))
        * np.deg2rad(np.diff(ds["phi_x"].values, axis=0))
        / ds["dy"].values
    )
    dfy_dy = R * np.deg2rad(np.diff(ds["phi_y"].values, axis=0)) / ds["dy"].values

    # Interpolate Jacobian elements to cell centers
    yc_c, xg_c = ds.coords["yc"], ds.coords["xg"]
    yg_c, xc_c = ds.coords["yg"], ds.coords["xc"]
    ds["phi_x_dx"] = grid.interp(
        xr.DataArray(dfx_dx, dims=["yc", "xg"], coords={"yc": yc_c, "xg": xg_c}),
        "X",
        boundary="extend",
    )
    ds["phi_y_dx"] = grid.interp(
        xr.DataArray(dfy_dx, dims=["yc", "xg"], coords={"yc": yc_c, "xg": xg_c}),
        "X",
        boundary="extend",
    )
    ds["phi_x_dy"] = grid.interp(
        xr.DataArray(dfx_dy, dims=["yg", "xc"], coords={"yg": yg_c, "xc": xc_c}),
        "Y",
        boundary="extend",
    )
    ds["phi_y_dy"] = grid.interp(
        xr.DataArray(dfy_dy, dims=["yg", "xc"], coords={"yg": yg_c, "xc": xc_c}),
        "Y",
        boundary="extend",
    )

    # C = J^T J
    cg_data = np.moveaxis(
        np.array(
            [
                [
                    ds["phi_x_dx"] * ds["phi_x_dx"] + ds["phi_y_dx"] * ds["phi_y_dx"],
                    ds["phi_x_dx"] * ds["phi_x_dy"] + ds["phi_y_dx"] * ds["phi_y_dy"],
                ],
                [
                    ds["phi_x_dx"] * ds["phi_x_dy"] + ds["phi_y_dx"] * ds["phi_y_dy"],
                    ds["phi_x_dy"] * ds["phi_x_dy"] + ds["phi_y_dy"] * ds["phi_y_dy"],
                ],
            ]
        ),
        [2, 3],
        [0, 1],
    )
    ds["cg"] = xr.DataArray(cg_data, dims=["yc", "xc", "dim0", "dim1"])

    if not keep_intermediates:
        for var in ["phi_x_dx", "phi_x_dy", "phi_y_dx", "phi_y_dy"]:
            del ds[var]


@nb.njit(parallel=True)
def eigenspectrum(cg, ref_time) -> tuple[np.array, np.array, np.array, np.array, np.array]:
    """Calculate the eigenspectrum and FTLE of the Cauchy-Green tensor.

    Args:
        cg: Cauchy-Green tensor [ny, nx, 2, 2]; NaN entries are treated as zero
        ref_time: integration time in seconds to scale the FTLE

    Returns:
        lmin: smallest eigenvalue [ny, nx]
        lmax: largest eigenvalue [ny, nx]
        vmin: eigenvector associated with the smallest eigenvalue [ny, nx, 2]
        vmax: eigenvector associated with the largest eigenvalue [ny, nx, 2]
        ftle: finite-time Lyapunov exponent [ny, nx]

    """
    cg = np.nan_to_num(cg)
    nlon = cg.shape[0]
    nlat = cg.shape[1]
    lmin = np.zeros((nlon, nlat))
    lmax = np.zeros((nlon, nlat))
    vmin = np.zeros((nlon, nlat, 2))
    vmax = np.zeros((nlon, nlat, 2))
    ftle = np.zeros((nlon, nlat))

    for k in nb.prange(0, nlon * nlat):
        i, j = k // nlat, k % nlat  # np.unravel_index() is not supported in numba
        # Symmetrize before decomposition: floating-point rounding in cauchygreen
        # can make c[0,1] ≠ c[1,0] by tiny amounts; eigh requires exact symmetry
        # and is more numerically stable than eig for symmetric matrices.
        c = cg[i, j]
        c_sym = 0.5 * (c + c.T)
        eig, v = np.linalg.eigh(c_sym)
        # eigh returns eigenvalues in ascending order — no argsort needed
        vmin[i, j, :], vmax[i, j, :] = v[:, 0], v[:, 1]

        if eig[1] <= 0:
            lmax[i, j], lmin[i, j], ftle[i, j] = 0.0, 0.0, np.nan
        else:
            lmin[i, j], lmax[i, j] = eig[0], eig[1]
            ftle[i, j] = 1.0 / (2.0 * ref_time) * np.log(lmax[i, j])
    return lmin, lmax, vmin, vmax, ftle


def lavd(ds, mean_vorticity, dt, origin_id, shape_p):
    """Calculate the Lagrangian averaged vorticity deviation (LAVD).

    Args:
        ds: xarray Dataset from the trajectory zarr output (contains vorticity variable)
        mean_vorticity: spatial mean vorticity to subtract as the reference
        dt: output timestep in seconds
        origin_id: ID of the first particle from flowmap()
        shape_p: 2D shape (ny, nx) of the particle release grid

    Returns:
        value: LAVD field on the particle release grid [ny, nx]

    """
    # mean vorticity should ideally be calculated per time step;
    # for large domains it is ~zero so a single scalar is a good approximation
    vorticity_deviation = np.trapezoid(np.abs(np.nan_to_num(ds.vorticity) - mean_vorticity), dx=dt)
    value = np.zeros(shape_p)
    pid = (ds.trajectory.values - origin_id).astype("int")
    value[np.unravel_index(pid, shape_p)] = vorticity_deviation
    return value


def _p1dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed angular distance on the half-circle (mod π).

    Implements the P1Dist function from CoherentStructures.jl.

    Args:
        a: angle array (radians)
        b: angle array (radians)

    Returns:
        Signed distance in (-π/2, π/2].

    """
    d = (a - b) % np.pi
    return np.where(d > np.pi / 2, d - np.pi, d)


def _merge_singularities(sings: list[Singularity], radius: float) -> list[Singularity]:
    """Merge singularities within radius, averaging coords and summing indices.

    Uses a KD-tree + sparse connected components instead of BFS so the cost is
    O(n log n) rather than O(n²).

    Implements the Combine(dist) heuristic from CoherentStructures.jl.

    Args:
        sings: list of Singularity objects
        radius: merge distance threshold (same units as lon/lat)

    Returns:
        Reduced list of Singularity objects; groups with zero total index are dropped.

    """
    n = len(sings)
    coords = np.array([[s.lon, s.lat] for s in sings])
    indices = np.array([s.index for s in sings])

    pairs = cKDTree(coords).query_pairs(radius, output_type="ndarray")
    if len(pairs):
        r, c = pairs[:, 0], pairs[:, 1]
        adj = csr_matrix(
            (np.ones(2 * len(r), dtype=np.int8), (np.r_[r, c], np.r_[c, r])),
            shape=(n, n),
        )
    else:
        adj = csr_matrix((n, n), dtype=np.int8)

    _, labels = connected_components(adj, directed=False)

    merged: list[Singularity] = []
    for comp in range(labels.max() + 1):
        mask = labels == comp
        total = float(indices[mask].sum())
        if abs(total) > 1e-9:
            merged.append(
                Singularity(
                    lon=float(coords[mask, 0].mean()),
                    lat=float(coords[mask, 1].mean()),
                    index=total,
                )
            )
    return merged


def find_singularities(
    vmin: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    combine_radius: float = 0.0,
) -> list[Singularity]:
    """Find singularities of the ξ₁ eigenvector line field.

    Uses the discrete winding-number method: for each grid cell, accumulate
    the P1-distance (signed angle on the half-circle) going counterclockwise
    around the four corners. A non-zero total identifies a singularity with
    topological index = winding_number / 2.

    Implements ``compute_singularities`` + ``singularity_detection`` from
    CoherentStructures.jl.

    Args:
        vmin: smallest CG eigenvector field [ny, nx, 2]
        xc: 1-D longitude coordinate array [nx]
        yc: 1-D latitude coordinate array [ny]
        combine_radius: merge singularities within this radius (units of xc/yc)

    Returns:
        List of Singularity objects.

    """
    ny, nx = vmin.shape[:2]
    print(f"  find_singularities: {ny}×{nx} grid ({(ny - 1) * (nx - 1):,} cells) …", flush=True)

    theta = np.arctan2(vmin[:, :, 1], vmin[:, :, 0]) % np.pi

    # counterclockwise traversal: right → up → left → down
    bl, br = theta[:-1, :-1], theta[:-1, 1:]
    tl, tr = theta[1:, :-1], theta[1:, 1:]
    winding = np.round(
        (_p1dist(br, bl) + _p1dist(tr, br) + _p1dist(tl, tr) + _p1dist(bl, tl)) / np.pi
    ).astype(int)

    rows, cols = np.where(winding != 0)
    sings: list[Singularity] = [
        Singularity(
            lon=0.5 * (xc[j] + xc[j + 1]),
            lat=0.5 * (yc[i] + yc[i + 1]),
            index=winding[i, j] / 2,
        )
        for i, j in zip(rows, cols, strict=True)
    ]

    counts = Counter(s.index for s in sings)
    summary = ", ".join(f"{v}×(idx={k:+g})" for k, v in sorted(counts.items()))
    print(f"  raw singularities: {len(sings)} [{summary}]", flush=True)

    if combine_radius > 0 and len(sings) >= 2:
        merged = _merge_singularities(sings, combine_radius)
        counts_m = Counter(s.index for s in merged)
        summary_m = ", ".join(f"{v}×(idx={k:+g})" for k, v in sorted(counts_m.items()))
        print(
            f"  after merge (r={combine_radius}°): {len(merged)} [{summary_m}]",
            flush=True,
        )
        return merged
    return sings


def _integrate_orbit(
    p: float,
    sign: int,
    seed: np.ndarray,
    cx: float,
    cy: float,
    fs: FieldSet,
    max_orbit_length: float,
    arc_dt: float,
) -> np.ndarray | None:
    """Integrate the closed orbit of η±p with Parcels and return its boundary points.

    Args:
        p: λ parameter at which the orbit closes
        sign: +1 or -1 for η⁺ or η⁻
        seed: Poincaré section seed [lon, lat]
        cx: vortex centre longitude
        cy: vortex centre latitude
        fs: Parcels FieldSet from _build_cg_fieldset
        max_orbit_length: maximum integration arc length
        arc_dt: arc-length step size

    Returns:
        [N, 2] array of (lat, lon) orbit points, or None if integration failed.

    """
    pset = ParticleSet.from_list(
        fieldset=fs,
        pclass=EtaParticle,
        lon=[float(seed[0])],
        lat=[float(seed[1])],
        time=[0.0],
        cx=[float(cx)],
        cy=[float(cy)],
        seed_lon=[float(seed[0])],
        seed_lat=[float(seed[1])],
        lat_prev=[float(seed[1])],
        p=[float(p)],
        eta_sign=[float(sign)],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, "orbit.zarr")
        output = pset.ParticleFile(outpath, outputdt=arc_dt)
        pset.execute(
            _eta_poincare_kernel,
            runtime=max_orbit_length,
            dt=arc_dt,
            output_file=output,
            verbose_progress=False,
        )

        if pset[0].has_returned != 1.0:
            return None

        ds = xr.open_zarr(outpath)
        traj = ds.isel(trajectory=0)
        lons = traj.lon.values
        lats = traj.lat.values

    valid = np.isfinite(lons) & np.isfinite(lats)
    lons = lons[valid]
    lats = lats[valid]
    if len(lons) < 3:
        return None

    # Trim trailing frozen positions (particle is frozen after Poincaré return)
    diffs = np.diff(lons) ** 2 + np.diff(lats) ** 2
    last_move = np.flatnonzero(diffs > 1e-14)
    if len(last_move) > 0:
        lons = lons[: last_move[-1] + 2]
        lats = lats[: last_move[-1] + 2]

    return np.column_stack([lats, lons])


def _build_cg_fieldset(
    xc: np.ndarray,
    yc: np.ndarray,
    lmin: np.ndarray,
    lmax: np.ndarray,
    vmin: np.ndarray,
    vmax: np.ndarray,
) -> Any:
    """Build a static Parcels FieldSet from CG eigenvalues and eigenvectors.

    All six scalar fields (lmin, lmax, vmx, vmy, vMx, vMy) are stored on the
    (yc, xc) grid. Cells where lmax ≤ 0 are zeroed so the kernel can detect
    them via the dl < 1e-12 guard. Domain-bound constants are added so the
    kernel can reject out-of-domain particles without triggering a Parcels error.

    mesh="flat" keeps degree-space velocities unchanged (no cos-lat correction
    from Parcels — the kernel applies it explicitly).

    """
    ny, nx = lmin.shape

    def _safe(arr: np.ndarray) -> np.ndarray:
        return np.nan_to_num(np.where(lmax > 0, arr, 0.0), nan=0.0)[np.newaxis, :, :]

    dims = {"lon": xc, "lat": yc, "time": np.array([0.0])}
    data = {
        "U": np.zeros((1, ny, nx)),
        "V": np.zeros((1, ny, nx)),
        "lmin": _safe(lmin),
        "lmax": _safe(lmax),
        "vmx": np.nan_to_num(vmin[:, :, 0], nan=0.0)[np.newaxis, :, :],
        "vmy": np.nan_to_num(vmin[:, :, 1], nan=0.0)[np.newaxis, :, :],
        "vMx": np.nan_to_num(vmax[:, :, 0], nan=0.0)[np.newaxis, :, :],
        "vMy": np.nan_to_num(vmax[:, :, 1], nan=0.0)[np.newaxis, :, :],
    }
    fs = FieldSet.from_data(data, dims, mesh="flat", allow_time_extrapolation=True)
    fs.add_constant("lon_min", float(xc[0]))
    fs.add_constant("lon_max", float(xc[-1]))
    fs.add_constant("lat_min", float(yc[0]))
    fs.add_constant("lat_max", float(yc[-1]))
    return fs


def _run_eta_poincare(
    lons: np.ndarray,
    lats: np.ndarray,
    cxs: np.ndarray,
    cys: np.ndarray,
    ps: np.ndarray,
    signs: np.ndarray,
    fs: Any,
    max_arc: float,
    arc_dt: float,
) -> np.ndarray:
    """Run all η±λ Poincaré integrations simultaneously in one Parcels execution.

    Args:
        lons: seed longitudes, one per particle
        lats: seed latitudes, one per particle
        cxs: vortex-centre longitudes, one per particle
        cys: vortex-centre latitudes, one per particle
        ps: λ parameter values, one per particle
        signs: η sign (+1 or -1), one per particle
        fs: Parcels FieldSet from _build_cg_fieldset
        max_arc: maximum arc length to integrate
        arc_dt: arc-length step size

    Returns:
        1-D array of longitude displacements at the Poincaré return (nan where
        the orbit did not close within max_arc).

    """
    n = len(lons)
    if n == 0:
        return np.empty(0)

    pset = ParticleSet.from_list(
        fieldset=fs,
        pclass=EtaParticle,
        lon=list(lons),
        lat=list(lats),
        time=[0.0] * n,
        cx=list(cxs),
        cy=list(cys),
        seed_lon=list(lons),
        seed_lat=list(lats),
        lat_prev=list(lats),
        p=list(ps),
        eta_sign=list(signs),
    )

    pset.execute(
        _eta_poincare_kernel,
        runtime=max_arc,
        dt=arc_dt,
        verbose_progress=True,
    )

    return np.array([p.return_dev if p.has_returned == 1.0 else np.nan for p in pset])


def _find_elliptic_barriers_parcels(
    centers: list[tuple[float, float]],
    xc: np.ndarray,
    yc: np.ndarray,
    lmin: np.ndarray,
    lmax: np.ndarray,
    vmin: np.ndarray,
    vmax: np.ndarray,
    itps: tuple,
    pmin: float,
    pmax: float,
    n_seeds: int,
    boxradius: float,
    n_bisection: int,
    max_orbit_length: float,
    _tolerance: float,
) -> list[np.ndarray | None]:
    """Find the outermost elliptic barrier for every centre simultaneously.

    All centres, seeds, and signs are batched into a single Parcels ParticleSet
    and advanced together at each bisection step. The Poincaré section is the
    horizontal ray to the right of each centre; bisection finds the λ* where
    the orbit closes. The outermost converged orbit per centre is returned.

    Implements ``compute_closed_orbits`` from CoherentStructures.jl.

    Args:
        centers: list of (lon, lat) for each elliptic centre
        xc: 1-D longitude coordinate array
        yc: 1-D latitude coordinate array
        lmin: smallest CG eigenvalue field [ny, nx]
        lmax: largest CG eigenvalue field [ny, nx]
        vmin: smallest CG eigenvector field [ny, nx, 2]
        vmax: largest CG eigenvector field [ny, nx, 2]
        itps: (lmin_itp, lmax_itp) interpolators for centre-point eigenvalue queries
        pmin: global lower λ bound
        pmax: global upper λ bound
        n_seeds: number of seed radii per centre
        boxradius: maximum search radius from centre (degrees)
        n_bisection: bisection iterations
        max_orbit_length: maximum arc length per orbit integration
        arc_dt: arc-length step size for orbit integration

    Returns:
        List of [N, 2] arrays (lat, lon) — one per centre, None if no barrier.

    """
    if not centers:
        return []

    fs = _build_cg_fieldset(xc, yc, lmin, lmax, vmin, vmax)
    arc_dt = max_orbit_length / 200.0

    radii = list(reversed(np.linspace(boxradius / n_seeds, boxradius, n_seeds)))

    # Build flattened particle table: all centres × seeds × signs, outermost first.
    # λ bracket is determined from the centre's eigenvalues, not each seed's.
    all_lons: list[float] = []
    all_lats: list[float] = []
    all_cxs: list[float] = []
    all_cys: list[float] = []
    all_signs: list[float] = []
    p_lo: list[float] = []
    p_hi: list[float] = []
    info: list[tuple[int, int]] = []  # (centre_idx, sign)

    for ci, (cx, cy) in enumerate(centers):
        pt = np.array([[cy, cx]])
        l1_c = itps[0](pt).item()
        l2_c = itps[1](pt).item()
        if not (np.isfinite(l1_c) and np.isfinite(l2_c)):
            continue
        pm_lo = max(pmin, l1_c)
        pm_hi = min(pmax, l2_c)
        if pm_hi <= pm_lo:
            continue
        for r in radii:
            for sign in (-1.0, 1.0):
                all_lons.append(cx + r)
                all_lats.append(cy)
                all_cxs.append(cx)
                all_cys.append(cy)
                all_signs.append(sign)
                p_lo.append(pm_lo)
                p_hi.append(pm_hi)
                info.append((ci, int(sign)))

    if not all_lons:
        return [None] * len(centers)

    lons = np.array(all_lons)
    lats = np.array(all_lats)
    cxs = np.array(all_cxs)
    cys = np.array(all_cys)
    signs = np.array(all_signs)
    p_lo_arr = np.array(p_lo)
    p_hi_arr = np.array(p_hi)

    n_particles = len(lons)
    print(
        f"  bisection: {n_particles} particles "
        f"({len(centers)} centres × ~{n_seeds} seeds × 2 signs)",
        flush=True,
    )

    # Evaluate Poincaré return at both ends of the λ bracket
    print("  bisection step 0/2 (p_lo) …", flush=True)
    f_lo = _run_eta_poincare(lons, lats, cxs, cys, p_lo_arr, signs, fs, max_orbit_length, arc_dt)
    print("  bisection step 1/2 (p_hi) …", flush=True)
    f_hi = _run_eta_poincare(lons, lats, cxs, cys, p_hi_arr, signs, fs, max_orbit_length, arc_dt)

    # Active particles: bracket straddles zero (sign-change condition)
    active = np.isfinite(f_lo) & np.isfinite(f_hi) & (f_lo * f_hi < 0)
    print(
        f"  bisection: {active.sum()} / {n_particles} particles have a bracket sign change",
        flush=True,
    )

    # Bisection over λ — all active particles advance together each step
    f_lo = np.where(active, f_lo, np.nan)
    for i in range(n_bisection):
        n_active = int(active.sum())
        if n_active == 0:
            print(
                f"  bisection step {i + 2}/{n_bisection + 2}: no active particles, stopping early",
                flush=True,
            )
            break
        print(
            f"  bisection step {i + 2}/{n_bisection + 2}: {n_active} active particles …",
            flush=True,
        )
        p_mid = 0.5 * (p_lo_arr + p_hi_arr)
        f_mid = _run_eta_poincare(lons, lats, cxs, cys, p_mid, signs, fs, max_orbit_length, arc_dt)
        same_sign = active & np.isfinite(f_mid) & (f_lo * f_mid > 0)
        p_lo_arr = np.where(same_sign, p_mid, p_lo_arr)
        f_lo = np.where(same_sign, f_mid, f_lo)
        p_hi_arr = np.where(active & ~same_sign & np.isfinite(f_mid), p_mid, p_hi_arr)
        active &= np.isfinite(f_mid) & ((p_hi_arr - p_lo_arr) > 1e-7)

    n_converged = int(active.sum())
    print(f"  bisection done: {n_converged} converged particle(s)", flush=True)

    # For each centre, take the outermost converged particle and integrate orbit
    barriers = [None] * len(centers)
    for idx, (ci, sign) in enumerate(info):
        if not active[idx]:
            continue
        if barriers[ci] is not None:
            continue  # outermost already found for this centre
        pm = 0.5 * (p_lo_arr[idx] + p_hi_arr[idx])
        seed = np.array([lons[idx], lats[idx]])
        print(
            f"  integrating final orbit for centre {ci} (λ*={pm:.6f}, sign={sign:+d}) …",
            flush=True,
        )
        orbit = _integrate_orbit(
            pm, float(signs[idx]), seed, cxs[idx], cys[idx], fs, max_orbit_length, arc_dt
        )
        if orbit is not None:
            barriers[ci] = orbit

    return barriers


def elliptic_lcs(
    xc: np.ndarray,
    yc: np.ndarray,
    lmin: np.ndarray,
    lmax: np.ndarray,
    vmin: np.ndarray,
    vmax: np.ndarray,
    pmin: float = 0.7,
    pmax: float = 2.0,
    n_seeds: int = 40,
    boxradius: float = 2.0,
    combine_radius: float | None = None,
    n_bisection: int = 15,
    max_orbit_length: float = 20.0,
    tolerance: float = 1e-7,
) -> tuple[list[Singularity], list[np.ndarray]]:
    """Compute elliptic LCS as closed orbits of the η±λ direction field.

    For each elliptic singularity (topological index +1) of the ξ₁ line field,
    find the outermost closed orbit of the parametric direction field:

        η±λ = √((λ₂-λ)/(λ₂-λ₁))·ξ₁ ± √((λ-λ₁)/(λ₂-λ₁))·ξ₂

    using a Poincaré section and bisection over λ. Each closed orbit is a
    Lagrangian vortex boundary (material barrier to transport).

    Implements ``ellipticLCS`` from CoherentStructures.jl
    (Haller et al., J. Fluid Mech. 795, 2016).

    Args:
        xc: 1-D longitude coordinate array [nx]
        yc: 1-D latitude coordinate array [ny]
        lmin: smallest CG eigenvalue [ny, nx]
        lmax: largest CG eigenvalue [ny, nx]
        vmin: smallest CG eigenvector [ny, nx, 2]
        vmax: largest CG eigenvector [ny, nx, 2]
        pmin: lower bound on the λ parameter in η±λ
        pmax: upper bound on the λ parameter in η±λ
        n_seeds: number of seed points along the Poincaré section
        boxradius: half-width of the search area around each singularity (degrees)
        combine_radius: radius for merging raw singularities (default: boxradius/10)
        n_bisection: maximum bisection steps when solving for λ*
        max_orbit_length: maximum integration arc length when following an orbit
        tolerance: unused, kept for API compatibility

    Returns:
        singularities: list of all detected Singularity objects
        barriers: list of [N, 2] arrays (columns: lat, lon) for each closed barrier

    """
    if combine_radius is None:
        combine_radius = boxradius / 10

    singularities = find_singularities(vmin, xc, yc, combine_radius=combine_radius)
    centers = [(s.lon, s.lat) for s in singularities if abs(s.index - 1.0) < 0.1]

    # mask degenerate cells where particles left the domain (lmax == 0)
    lmin_m = np.where(lmax > 0, lmin, np.nan)
    lmax_m = np.where(lmax > 0, lmax, np.nan)

    kw: dict = {"bounds_error": False, "fill_value": np.nan}
    itps = (
        RegularGridInterpolator((yc, xc), lmin_m, **kw),
        RegularGridInterpolator((yc, xc), lmax_m, **kw),
    )

    print(
        f"  searching {len(centers)} centre(s) with {n_seeds} seeds × 2 signs "
        f"over {n_bisection} bisection steps …",
        flush=True,
    )
    barrier_list = _find_elliptic_barriers_parcels(
        centers,
        xc,
        yc,
        lmin,
        lmax,
        vmin,
        vmax,
        itps,
        pmin,
        pmax,
        n_seeds,
        boxradius,
        n_bisection,
        max_orbit_length,
        tolerance,
    )

    barriers: list[np.ndarray] = []
    for i, ((cx, cy), barrier) in enumerate(zip(centers, barrier_list, strict=True)):
        status = "found" if barrier is not None else "none"
        print(f"  [{i + 1}/{len(centers)}] centre ({cx:.2f}, {cy:.2f}) → {status}")
        if barrier is not None:
            barriers.append(barrier)

    return singularities, barriers

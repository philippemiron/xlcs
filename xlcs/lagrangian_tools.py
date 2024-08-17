from typing import Tuple, Any
import numpy as np
import xarray as xr
import numba as nb
from numpy import ndarray
from xarray import Dataset
from datetime import datetime, timedelta
from kernels import OutOfBound, SampleVorticity
from particle import LAVDParticle
import xgcm
import gsw
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4
from typing import Optional


def flowmap(
    ds: Dataset,
    filename: str,
    fs: FieldSet,
    t0: datetime,
    T: timedelta,
    dt: timedelta,
    lavd: Optional[bool] = False,
) -> Tuple[Any, ndarray]:
    """
    Calculate the flowmap from T-long trajectories initialized a coordinates x-y. If we are calculating LAVD (lavd=True), we need to sample the vorticity along the trajectory. In this case, we assume that the FieldSet fs contains a vorticity field.

    Args:
        filename: output filename
        fs: parcels.fieldset.FieldSet object to track this particle on (assume it contains vorticity field!)
        x: list of initial meridional coordinates
        y: list of initial zonal coordinates
        t0: initial time
        T: integration time
        dt: integration timestep
        lavd: if True, interpolate vorticity along the trajectory

    Returns:
        ds: xarray.Dataset augmented with the flowmap information

    """
    mx, my = np.meshgrid(ds.xc, ds.yc)
    # time can be datetime or seconds from the origin_time
    if isinstance(t0, datetime):
        mt = np.full_like(mx, t0, dtype="datetime64[ms]")
    else:
        mt = np.full_like(mx, t0)

    # initialize the particles
    pset = ParticleSet.from_list(
        fieldset=fs,  # velocity field
        pclass=LAVDParticle if lavd else JITParticle,  # type of particles
        lon=mx.flatten(),  # release lon
        lat=my.flatten(),  # release lat
        time=mt.flatten(),  # release time
    )

    origin_id = np.copy(pset.id[0])  # copy first id

    # define the output file
    output_file = pset.ParticleFile(
        name=filename, outputdt=timedelta(seconds=int(abs(dt.total_seconds())))
    )

    # integration
    kernels = pset.Kernel(OutOfBound) + pset.Kernel(AdvectionRK4)
    if lavd:
        kernels += pset.Kernel(SampleVorticity)

    pset.execute(
        kernels,  # the kernels (define how particles move)
        runtime=T,  # the total length of the run
        dt=dt,  # the timestep of the kernel
        output_file=output_file,
        verbose_progress=True,
    )

    # reshape flowmap to a two-dimensional grid
    pid = pset.id - origin_id
    ds["phi_x"] = (("yc", "xc"), np.zeros_like(mx))
    ds["phi_y"] = (("yc", "xc"), np.zeros_like(mx))
    ds["phi_x"].values[np.unravel_index(pid, mx.shape)] = pset.lon
    ds["phi_y"].values[np.unravel_index(pid, mx.shape)] = pset.lat

    # 0 are replace with nan
    ds["phi_x"].values[np.where(ds["phi_x"] == 0)] = np.nan
    ds["phi_y"].values[np.where(ds["phi_y"] == 0)] = np.nan

    return (
        pset,
        origin_id,
    )


def cauchygreen(ds: xr.Dataset, grid: xgcm.Grid):
    """_summary_

    Args:
        ds (xr.Dataset): _description_
        grid (xgcm.Grid): _description_
    """
    ds["phi_x_m"] = (
        ("yc", "xg"),
        gsw.distance(ds["phi_x"].data, ds["phi_y"].data, axis=1),
    )
    ds["phi_y_m"] = (
        ("yg", "xc"),
        gsw.distance(ds["phi_x"].data, ds["phi_y"].data, axis=0),
    )

    ds["phi_x_dx"] = grid.diff(ds["phi_x_m"], "X", boundary="extend") / ds.xc
    ds["phi_x_dy"] = grid.interp(
        grid.diff(ds["phi_x_m"], "Y", boundary="extend") / ds.yg,
        ["X", "Y"],
        to="center",
        boundary="extend",
    )
    ds["phi_y_dx"] = grid.interp(
        grid.diff(ds["phi_y_m"], "X", boundary="extend") / ds.xg,
        ["X", "Y"],
        to="center",
        boundary="extend",
    )
    ds["phi_y_dy"] = grid.diff(ds["phi_y_m"], "Y", boundary="extend") / ds.yc

    # form the Cauchy-Green tensor
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
    ds["cg"] = xr.DataArray(cg_data, dims=["yc", "xc", "dim", "dim"])


@nb.njit(parallel=True)
def eigenspectrum(
    cg, ref_time
) -> tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Calculate the eigensepctrum and the Finite-Time Lyapunov Exponent (FTLE) of the Cauchy-Green tensor
    Args:
        cg: Cauchy-Green tensor [nx, ny]
        ref_time: time to scale the FTLE

    Returns:
        lmin: smallest eigenvalue
        lmax: largest eigenvalues
        vmin: eigenvector associated with the smallest eigenvalue
        vmax: eigenvector associated with the largest eigenvalue
        ftle: finite-time lyapunov exponent

    """
    nlon = cg.shape[0]
    nlat = cg.shape[1]
    lmin = np.zeros((nlon, nlat))
    lmax = np.zeros((nlon, nlat))
    vmin = np.zeros((nlon, nlat, 2))
    vmax = np.zeros((nlon, nlat, 2))
    ftle = np.zeros((nlon, nlat))

    for k in nb.prange(0, nlon * nlat):
        i, j = k // nlat, k % nlat  # np.unravel_index() is not numba supported
        eig, v = np.linalg.eig(cg[i, j])
        idx = eig.argsort()
        eig, v = eig[idx], v[:, idx]
        vmin[i, j, :], vmax[i, j, :] = v[:, 0], v[:, 1]

        if eig[1] <= 0:
            lmax[i, j], lmin[i, j], ftle[i, j] = 0.0, 0.0, np.nan
        else:
            lmin[i, j], lmax[i, j] = eig[0], eig[1]
            ftle[i, j] = 1.0 / (2.0 * ref_time) * np.log(lmax[i, j])
    return lmin, lmax, vmin, vmax, ftle


def lavd(ds, mean_vorticity, dt, origin_id, shape_p):
    """
    Calculate the Lagrangian averaged vorticity deviation (LAVD) from the trajectories output
    Args:
        ds:
        mean_vorticity:
        dt:
        origin_id:
        shape_p:

    Returns:
        lavd: Lagrangian averaged vorticity deviation
    """
    # TODO: mean vorticity should be calculated per time steps
    # in large domain the vorticity is ~zero so it shouldn't affect much

    # integration along the trajectory
    vorticity_deviation = np.trapz(
        np.abs(np.nan_to_num(ds.vorticity) - mean_vorticity), dx=dt
    )
    value = np.zeros(shape_p)
    pid = (ds.trajectory.values - origin_id).astype("int")
    value[np.unravel_index(pid, shape_p)] = vorticity_deviation
    return value

from typing import Tuple, Any
import math
import numpy as np
import xarray as xr
import numba as nb
from numpy import ndarray
from xarray import Dataset
import grid_calc
from datetime import datetime, timedelta

from parcels import (
    FieldSet,
    ParticleSet,
    JITParticle,
    AdvectionRK4,
    ErrorCode,
    Variable,
)


def DeleteParticle(particle, fieldset, time):
    """
    Remove particles that go out of bound

    Args:
        particle: object that defines custom particle
        fieldset: parcels.fieldset.FieldSet object to track this particle on
        time: current time
    """
    particle.delete()


def RemoveOnLand(particle, fieldset, time):
    """
    Kernel to remove initial particles where there is no velocity.
    This kernel is execute once before the regular advection.

    Args:
        particle: object that defines custom particle
        fieldset: parcels.fieldset.FieldSet object to track this particle on
        time: current time
    """
    u, v = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    if math.fabs(u) < 1e-12:
        particle.delete()


def flowmap(
    filename: str,
    fs: FieldSet,
    x: np.array,
    y: np.array,
    t0: datetime,
    T: timedelta,
    dt: timedelta,
) -> Tuple[Any, ndarray, Dataset]:
    """
    Calculate the flowmap from T-long trajectories initialized a coordinates x-y.
    Args:
        filename: output filename
        fs: parcels.fieldset.FieldSet object to track this particle on
        x: list of initial meridional coordinates
        y: list of initial zonal coordinates
        t0: initial time
        T: integration time
        dt: integration timestep

    Returns:
        pset: particles id and final position
        origin_id: the index of the first particle to reconstruct the 2d grid from 1d vector
        Dataset: contains the flowmap at the initial locations of the particles
    """
    mx, my = np.meshgrid(x, y, indexing="ij")
    # time can be datetime or seconds from the origin_time
    if type(t0) == datetime:
        mt = np.full_like(mx, t0, dtype='datetime64[ms]')
    else:
        mt = np.full_like(mx, t0)

    # initialize the particles
    pset = ParticleSet.from_list(
        fieldset=fs,  # velocity field
        pclass=JITParticle,  # type of particles
        lon=mx.flatten(),  # release lon
        lat=my.flatten(),  # release lat
        time=mt.flatten(),  # release time
    )

    origin_id = np.copy(pset.id[0])  # copy first id

    # remove particle on land
    pset.execute(
        RemoveOnLand, dt=0, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}
    )

    output_file = pset.ParticleFile(
        name=filename, outputdt=timedelta(seconds=int(abs(dt.total_seconds())))
    )

    # integration
    pset.execute(
        pset.Kernel(AdvectionRK4),
        runtime=T,
        dt=dt,
        output_file=output_file,
        verbose_progress=True,
        recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
    )

    # reshape flowmap to a two-dimensional grid
    pid = pset.id - origin_id
    phi_x = np.zeros_like(mx)
    phi_y = np.zeros_like(mx)
    phi_x[np.unravel_index(pid, mx.shape)] = pset.lon
    phi_y[np.unravel_index(pid, mx.shape)] = pset.lat

    return (
        pset,
        origin_id,
        xr.Dataset(
            data_vars=dict(
                phi_x=(["XG", "YG"], phi_x),
                phi_y=(["XG", "YG"], phi_y),
            ),
            coords=dict(
                longitude=(["lon", "lat"], mx),
                latitude=(["lon", "lat"], my),
            ),
            attrs=dict(description=f"Flowmap from {t0} with T = {T.days} days."),
        ),
    )


# Custom particle and kernel to sample the vorticity along the trajectory
class LAVDParticle(JITParticle):
    """
    Custom particles definition to interpolate vorticity along the trajectory
    """

    vorticity = Variable("vorticity", initial=0.0)


def SampleVorticity(particle, fieldset, time):
    particle.vorticity = fieldset.vorticity[
        time, particle.depth, particle.lat, particle.lon
    ]


def flowmap_lavd(
    filename: str,
    fs: FieldSet,
    x: np.array,
    y: np.array,
    t0: datetime,
    T: timedelta,
    dt: timedelta,
) -> Tuple[Any, ndarray, Dataset]:
    """
    Calculate the flowmap from T-long trajectories initialized a coordinates x-y. Because we need to sample the
    vorticity along the trajectories, this version of the flowmap uses a special Particle. Otherwise, it is similar
    to the flowmap function part of this module.

    Args:
        filename: output filename
        fs: parcels.fieldset.FieldSet object to track this particle on (assume it contains vorticity field!)
        x: list of initial meridional coordinates
        y: list of initial zonal coordinates
        t0: initial time
        T: integration time
        dt: integration timestep

    Returns:

    """
    mx, my = np.meshgrid(x, y)
    # time can be datetime or seconds from the origin_time
    if type(t0) == datetime:
        mt = np.full_like(mx, t0, dtype='datetime64[ms]')
    else:
        mt = np.full_like(mx, t0)

    # initialize the particles
    pset = ParticleSet.from_list(
        fieldset=fs,  # velocity field
        pclass=LAVDParticle,  # type of particles
        lon=mx.flatten(),  # release lon
        lat=my.flatten(),  # release lat
        time=mt.flatten(),  # release time
    )

    origin_id = np.copy(pset.id[0])  # copy first id

    # remove particle on land
    pset.execute(
        RemoveOnLand, dt=0, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}
    )

    output_file = pset.ParticleFile(
        name=filename, outputdt=timedelta(seconds=int(abs(dt.total_seconds())))
    )

    # integration
    kernels = pset.Kernel(SampleVorticity) + pset.Kernel(AdvectionRK4)
    pset.execute(
        kernels,  # the kernels (define how particles move)
        runtime=T,  # the total length of the run
        dt=dt,  # the timestep of the kernel
        output_file=output_file,
        verbose_progress=True,
        recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
    )

    # reshape flowmap to a two-dimensional grid
    pid = pset.id - origin_id
    phi_x = np.zeros_like(mx)
    phi_y = np.zeros_like(mx)
    phi_x[np.unravel_index(pid, mx.shape)] = pset.lon
    phi_y[np.unravel_index(pid, mx.shape)] = pset.lat

    return (
        pset,
        origin_id,
        xr.Dataset(
            data_vars=dict(
                phi_x=(["YG", "XG"], phi_x),
                phi_y=(["YG", "XG"], phi_y),
            ),
            coords=dict(
                longitude=(["YG", "XG"], mx),
                latitude=(["YG", "XG"], my),
            ),
            attrs=dict(description=f"Flowmap from {t0} with T = {T.days} days."),
        ),
    )


def cauchygreen(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate the Cauchy-Green tensor from the flowmap
    Args:
        ds:

    Returns:

    """
    # reference (lon, lat) to convert to meters
    lon0 = np.min(ds.longitude.data)
    lat0 = np.min(ds.latitude.data)

    # convert the grid from [deg] to [m]
    x_m, y_m = grid_calc.grid_to_m(ds.longitude.data, ds.latitude.data, lon0, lat0)
    ds["x_m"] = (("lon", "lat"), x_m)
    ds["y_m"] = (("lon", "lat"), y_m)
    ds = ds.set_coords(["x_m", "y_m"])

    # and the flowmap from [deg] to [m]
    phi_x_m, phi_y_m = grid_calc.grid_to_m(ds.phi_x.data, ds.phi_y.data, lon0, lat0)
    ds["phi_x_m"] = (("lon", "lat"), phi_x_m)
    ds["phi_y_m"] = (("lon", "lat"), phi_y_m)

    # flowmap derivatives
    ds["phi_x_dx"] = (("lon", "lat"), grid_calc.diff_x(phi_x_m.data, ds["x_m"].data))
    ds["phi_x_dy"] = (("lon", "lat"), grid_calc.diff_y(phi_x_m.data, ds["y_m"].data))
    ds["phi_y_dx"] = (("lon", "lat"), grid_calc.diff_x(phi_y_m.data, ds["x_m"].data))
    ds["phi_y_dy"] = (("lon", "lat"), grid_calc.diff_y(phi_y_m.data, ds["y_m"].data))

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
    ds["cg"] = xr.DataArray(cg_data, dims=["lon", "lat", "dim", "dim"])
    return ds


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
    Calculate the Lagrangian averaged vorticity deviation (LAVD) from the v
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
        np.abs(np.ma.masked_invalid(ds.vorticity) - mean_vorticity), dx=dt
    )
    value = np.zeros(shape_p)
    pid = (ds.trajectory.values - origin_id).astype("int")
    value[np.unravel_index(pid, shape_p)] = vorticity_deviation
    return value

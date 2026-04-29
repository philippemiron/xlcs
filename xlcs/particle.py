"""Custom OceanParcels particle classes."""

from parcels import JITParticle, Variable


class LAVDParticle(JITParticle):
    """JITParticle with a vorticity variable for LAVD computation."""

    vorticity = Variable("vorticity", initial=0.0)


class EtaParticle(JITParticle):
    """JITParticle for η±λ orbit integration with Poincaré section detection.

    Carries the vortex-centre coordinates, λ parameter, direction sign, and the
    full state needed to detect when the orbit crosses back through the seed
    latitude (Poincaré section).

    Attributes:
        cx, cy              : vortex-centre longitude and latitude (degrees)
        seed_lon/lat        : Poincaré section seed point
        lat_prev            : latitude at the previous integration step
        max_lat_dev         : largest absolute departure from seed_lat seen so far
        return_dev          : longitude displacement when the orbit first crosses back
        has_returned        : 0 running / 1 crossed section / 2 invalid or OOB
        has_crossed_center  : 1 once the orbit has passed west of the vortex centre
        p                   : λ parameter value for η±λ
        eta_sign            : +1 for η⁺, -1 for η⁻

    """

    cx = Variable("cx", initial=0.0)
    cy = Variable("cy", initial=0.0)
    seed_lon = Variable("seed_lon", initial=0.0)
    seed_lat = Variable("seed_lat", initial=0.0)
    lat_prev = Variable("lat_prev", initial=0.0)
    max_lat_dev = Variable("max_lat_dev", initial=0.0)
    return_dev = Variable("return_dev", initial=0.0)
    has_returned = Variable("has_returned", initial=0.0)
    has_crossed_center = Variable("has_crossed_center", initial=0.0)
    p = Variable("p", initial=1.0)
    eta_sign = Variable("eta_sign", initial=1.0)

"""OceanParcels kernel functions for particle advection and sampling."""

import math

from parcels import StatusCode


def OutOfBound(particle, fieldset, time):
    """Delete particles that leave the domain boundary.

    Args:
        particle: OceanParcels particle object
        fieldset: parcels FieldSet the particle is advected on
        time: current simulation time

    """
    if particle.state == StatusCode.ErrorOutOfBounds:
        particle.delete()


def SampleVorticity(particle, fieldset, time):
    """Sample vorticity from the fieldset at the current particle location.

    Args:
        particle: OceanParcels particle object (must have a vorticity attribute)
        fieldset: parcels FieldSet containing a vorticity field
        time: current simulation time

    """
    particle.vorticity = fieldset.vorticity[time, particle.depth, particle.lat, particle.lon]


def _eta_poincare_kernel(particle, fieldset, time):
    """Advance a η±λ particle one arc-length step and detect the Poincaré return.

    Implements one Euler step of the normalised η±λ = α·ξ₁ ± β·ξ₂ direction
    field and checks for re-crossing of the seed latitude (Poincaré section).

    The FieldSet must expose scalar fields lmin, lmax, vmx, vmy, vMx, vMy, and
    FieldSet constants lon_min, lon_max, lat_min, lat_max for domain bounds.

    Operates on EtaParticle objects.

    """
    if particle.has_returned < 0.5:
        # Hard domain-boundary check (nested to avoid bare return)
        if particle.lon < fieldset.lon_min:
            particle.has_returned = 2.0
        elif particle.lon > fieldset.lon_max:
            particle.has_returned = 2.0
        elif particle.lat < fieldset.lat_min:
            particle.has_returned = 2.0
        elif particle.lat > fieldset.lat_max:
            particle.has_returned = 2.0
        else:
            # Sample raw CG eigenvalues and eigenvectors
            # Use e1/e2 names to avoid collision with CField * kernel parameters
            l1 = fieldset.lmin[time, particle.depth, particle.lat, particle.lon]
            l2 = fieldset.lmax[time, particle.depth, particle.lat, particle.lon]
            e1x = fieldset.vmx[time, particle.depth, particle.lat, particle.lon]
            e1y = fieldset.vmy[time, particle.depth, particle.lat, particle.lon]
            e2x = fieldset.vMx[time, particle.depth, particle.lat, particle.lon]
            e2y = fieldset.vMy[time, particle.depth, particle.lat, particle.lon]

            dl = l2 - l1
            if dl < 1e-12:
                particle.has_returned = 2.0
            else:
                # Orient ξ₁ counterclockwise around the vortex centre
                sx = particle.lon - particle.cx
                sy = particle.lat - particle.cy
                if sx * e1y - sy * e1x < 0.0:
                    e1x = -e1x
                    e1y = -e1y

                # Orient ξ₂ radially outward from the vortex centre
                if sx * e2x + sy * e2y < 0.0:
                    e2x = -e2x
                    e2y = -e2y

                # Compute α and β
                p = particle.p
                a_sq = (l2 - p) / dl
                b_sq = (p - l1) / dl
                if a_sq < 0.0:
                    a_sq = 0.0
                if b_sq < 0.0:
                    b_sq = 0.0
                a = math.sqrt(a_sq)
                b = math.sqrt(b_sq)

                # η in physical (metric) space
                ex = a * e1x + particle.eta_sign * b * e2x
                ey = a * e1y + particle.eta_sign * b * e2y

                # degree-space: dlon/dt = η_x / cos(lat)
                coslat = math.cos(particle.lat * math.pi / 180.0)
                if coslat < 1e-10:
                    coslat = 1e-10
                u = ex / coslat
                v = ey

                # Normalise for arc-length parameterisation
                mag = math.sqrt(u * u + v * v)
                if mag < 1e-12:
                    particle.has_returned = 2.0
                else:
                    u = u / mag
                    v = v / mag

                    # Euler step via delta variables (Parcels 3+ API)
                    dlon = u * particle.dt
                    dlat = v * particle.dt
                    particle_dlon += dlon  # noqa: F821, F841
                    particle_dlat += dlat  # noqa: F821, F841

                    # New position (particle.lon/lat still hold old values here)
                    new_lon = particle.lon + dlon
                    new_lat = particle.lat + dlat

                    # Track maximum departure from seed latitude
                    adev = new_lat - particle.seed_lat
                    if adev < 0.0:
                        adev = -adev
                    if adev > particle.max_lat_dev:
                        particle.max_lat_dev = adev

                    # Track whether the orbit has passed west of the vortex centre;
                    # prevents detecting a half-revolution crossing as a return.
                    if new_lon < particle.cx:
                        particle.has_crossed_center = 1.0

                    # Poincaré section: seed_lat crossing on the east side of the
                    # centre, only after the orbit has gone past the centre (west side).
                    if particle.has_crossed_center > 0.5 and particle.max_lat_dev > 0.01:
                        prev_dev = particle.lat_prev - particle.seed_lat
                        curr_dev = new_lat - particle.seed_lat
                        if prev_dev * curr_dev < 0.0 and new_lon > particle.cx:
                            particle.return_dev = new_lon - particle.seed_lon
                            particle.has_returned = 1.0

                    particle.lat_prev = new_lat

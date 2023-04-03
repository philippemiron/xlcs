import math


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
    if math.fabs(u) < 1e-12 and math.fabs(v) < 1e-12:
        particle.delete()


def SampleVorticity(particle, fieldset, time):
    """Interpolate vorticity at the particle location

    Args:
        particle: particle object
        fieldset: advection fieldset
        time: current time
    """
    particle.vorticity = fieldset.vorticity[
        time, particle.depth, particle.lat, particle.lon
    ]

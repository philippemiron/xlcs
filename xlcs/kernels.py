from parcels import StatusCode


def OutOfBound(particle, fieldset, time):
    """
    Remove particles that go out of bound

    Args:
        particle: object that defines custom particle
        fieldset: parcels.fieldset.FieldSet object to track this particle on
        time: current time
    """
    if particle.state == StatusCode.ErrorOutOfBounds:
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

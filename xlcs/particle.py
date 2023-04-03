from parcels import JITParticle, Variable


class LAVDParticle(JITParticle):
    """
    Particle class with with vorticity variable
    """

    vorticity = Variable("vorticity", initial=0.0)

"""Data-container types for xlcs results."""

from dataclasses import dataclass


@dataclass
class Singularity:
    """Singularity of the ξ₁ eigenvector line field with its topological index.

    Attributes:
        lon: longitude of the singularity in degrees
        lat: latitude of the singularity in degrees
        index: topological index (+1/2 wedge, -1/2 trisector, +1 elliptic centre, …)

    """

    lon: float
    lat: float
    index: float

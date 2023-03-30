# xlcs
![CI](https://github.com/philippemiron/xlcs/workflows/CI/badge.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/philippemiron/xlcs/main?labpath=examples)
[![Available on conda-forge](https://anaconda.org/conda-forge/xlcs/badges/version.svg?style=flat-square)](https://anaconda.org/conda-forge/xlcs/)
[![Available on pypi](https://img.shields.io/pypi/v/xlcs.svg?style=flat-square&color=blue)](https://pypi.org/project/xlcs/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fphilippemiron%2Fxlcs&count_bg=%232EE352&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=true)](https://hits.seeyoufarm.com)

## Summary
python package to generate Lagrangian Coherent Structures from a collection of methods

The package consumes and outputs [xarray](https://github.com/pydata/xarray) Datasets and uses [OceanParcels](https://github.com/oceanparcels/parcels/) to calculate trajectories.

The goal of the package is to make it has easy as possible to calculate Lagrangian Coherent Structures any velocity fields that can be loaded into an `xarray.Dataset()`.


## Initial Release

Coming to github very soon!

The release will include methods to calculate:

- Finite Time Lyapunov exponenent calculation ([Haller 2001](https://www.sciencedirect.com/science/article/abs/pii/S0167278900001998))
- Lagrangian averaged vorticity deviation ([G. Haller, A. Hadjighasem, M. Farazmand and F. Huhn ](http://www.georgehaller.com/reprints/LAVD.pdf))
- Geodesic elliptic material vortices ([Haller & Beron 2013](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/coherent-lagrangian-vortices-the-black-holes-of-turbulence/3B50A4590B35E5637280F01A58502258), [Karrasch, Huhn, and Haller, 2015](https://royalsocietypublishing.org/doi/10.1098/rspa.2014.0639))

used for the identification of eddies in velocity fields.

## Getting started


## Found an issue or need help?

Please create a new issue [here](https://github.com/philippemiron/xlcs/issues) and provide as much detail as possible about your problem or question.
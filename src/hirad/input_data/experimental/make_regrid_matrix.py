# Adapted from anemoi-transform/src/anemoi/transform/commands/make-regrid-matrix.py to support anemoi dataset as input

import yaml
from anemoi.datasets import open_dataset
import numpy as np
from earthkit.regrid.utils.mir import mir_make_matrix

ERA_CONFIG = 'era.yaml'
COSMO_CONFIG = 'cosmo.yaml'

with open(ERA_CONFIG) as era_file:
    era_config = yaml.safe_load(era_file)
era = open_dataset(era_config, start=20160101, end=20160101)


with open(COSMO_CONFIG) as cosmo_file:
    cosmo_config = yaml.safe_load(cosmo_file)
cosmo = open_dataset(cosmo_config, start=20160101, end=20160101)

lat1 = era.latitudes
lon1 = era.longitudes
lat2 = cosmo.latitudes
lon2 = cosmo.longitudes

sparse_array = mir_make_matrix(lat1, lon1, lat2, lon2, output=None) # mir=args.mir, **kwargs

np.savez("regrid-matrix.npz",
         matrix_data=sparse_array.data,
         matrix_indices=sparse_array.indices,
         matrix_indptr=sparse_array.indptr,
         matrix_shape=sparse_array.shape,
         in_latitudes=lat1,
         in_longitudes=lon1,
         out_latitudes=lat2,
         out_longitudes=lon2,
)
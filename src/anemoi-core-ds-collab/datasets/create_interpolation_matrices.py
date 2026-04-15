## NOT YET TESTED
# Currently trying with building via ecbundle. No idea how this works.
#export PATH=/capstor/scratch/cscs/mmcgloho/mirbuild/build/bin/:$PATH
# also using earthkit regrid built from source. not sure if it'll work but the release version 0.5.1 doesn't.
#export EARTHKIT_DATA_CACHE_POLICY="user"

import os
import numpy as np
from earthkit.regrid.utils.mir import mir_make_matrix
import earthkit.data

earthkit.data.config.set({
    "cache-policy": "user",
    "user-cache-directory": "/users/mmcgloho/.cache/earthkit-regrid/"
})

DATA_DIR = '/capstor/store/cscs/pasc/c38/anemoi-downscaling-data/'

cosmo_grid=np.load(os.path.join(DATA_DIR,'cosmo_grid.npz'))
era_grid=np.load(os.path.join(DATA_DIR,'era_cosmo_cropped_grid.npz'))

cosmo_lats=cosmo_grid['latitudes']
cosmo_lons=cosmo_grid['longitudes']
era_lats=era_grid['latitudes']
era_lons=era_grid['longitudes']



era2cosmo_sparse_array = mir_make_matrix(in_lat=era_lats, in_lon=era_lons, out_lat=cosmo_lats, out_lon=cosmo_lons, output=None) # mir=args.mir, **kwargs
#era2cosmo_sparse_array = mir_make_matrix(in_grid=era_grid, out_grid=cosmo_grid)

np.savez(os.path.join(DATA_DIR,"eracrop_to_cosmo_interpolation_linear.npz"),
         matrix_data=era2cosmo_sparse_array.data,
         matrix_indices=era2cosmo_sparse_array.indices,
         matrix_indptr=era2cosmo_sparse_array.indptr,
         matrix_shape=era2cosmo_sparse_array.shape,
         in_latitudes=era_lats,
         in_longitudes=era_lons,
         out_latitudes=cosmo_lats,
         out_longitudes=cosmo_lons,
)

# TODO: The reverse one if this works.
np.savez(os.path.join(DATA_DIR,"cosmo_to_eracrop_interpolation_linear.npz"),
         matrix_data=cosmo2era_sparse_array.data,
         matrix_indices=cosmo2era_sparse_array.indices,
         matrix_indptr=cosmo2era_sparse_array.indptr,
         matrix_shape=cosmo2era_sparse_array.shape,
         in_latitudes=cosmo_lats,
         in_longitudes=cosmo_lons,
         out_latitudes=era_lats,
         out_longitudes=era_lons,
)